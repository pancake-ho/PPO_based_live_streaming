#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

current_dir = os.path.dirname(os.path.abspath(__file__))
proposed_dir = os.path.join(current_dir, '../proposed')
sys.path.append(proposed_dir)

try:
    from proposed import DynamicChannelModel
except ImportError:
    print("오류: proposed.py 파일을 찾거나 불러올 수 없습니다.")
    print("파일명을 확인하거나 구체적인 경로를 확인해주세요.")
    exit()


# GPU 가용성 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- TCLiVi 베이스라인 (규칙 기반 속도 제어 및 자동 테스트 포함) ---")
print(f"사용 장치: {device}")

# 디렉토리 생성 코드는 변동 없음
if not os.path.exists('results'): os.makedirs('results')
if not os.path.exists('models'): os.makedirs('models')

#################################
# 환경 부분 (TCLiVi 베이스라인 - 속도 제어 포함)
#################################

class VideoStreamingEnv_TCLiVi:
    """환경 클래스 코드는 이전 수정과 완전히 동일하며, 완전성을 위해 이곳에 유지함"""
    def __init__(self, num_chunks=300, t_buffer=4, t_i=2,
                 W1=1, W2=300.0, W3=100.0, W_tau=1.0,
                 history_len=10, verbose=False, snr_db=35.0):
        
        self.num_chunks = num_chunks
        self.t_buffer = t_buffer
        self.t_i = t_i

        self.quality_levels = list(range(1, 21))
        self.max_quality = max(self.quality_levels)
        self.W1, self.W2, self.W3, self.W_tau = W1, W2, W3, W_tau

        self.playback_speeds = [0.9, 1.0, 1.1]
        self.num_speed_levels = len(self.playback_speeds)

        self.target_buffers = [2.0, 3.0]
        self.action_map = {}
        action_id = 0
        self.bitrate_map = {
            1: 0.045,   2: 0.089,   3: 0.131,   4: 0.178,   5: 0.221,
            6: 0.262,   7: 0.334,   8: 0.396,   9: 0.522,   10: 0.595,
            11: 0.791,  12: 1.032,  13: 1.244,  14: 1.546,  15: 2.133,
            16: 2.484,  17: 3.078,  18: 3.526,  19: 3.840,  20: 4.219
        }
        self.history_len = history_len
        self.num_history_features = 6
        self.chunk_history = deque(maxlen=self.history_len)

        for quality in self.quality_levels:
            for buf in self.target_buffers:
                self.action_map[action_id] = (quality, buf)
                action_id += 1

        self.speed_rule_margin = 1.0
        self.channel_model = DynamicChannelModel()

        self.verbose = verbose
        self.reset()

    def reset(self):
        self.interval_capacities, self.actual_completion_times, self.chunk_actions = {}, {}, {}
        self.generated_chunk_times = {i: (i + 1) * self.t_i for i in range(self.num_chunks)}
        self.u_i, self.p_i, self.d_i, self.k_j, self.l_i, self.speed_j, self.tau_i, self.x_j = [], [], [], [], [], [], [], []
        self.chunk_history.clear()
        for _ in range(self.history_len):
            self.chunk_history.append(np.zeros(self.num_history_features, dtype=np.float32))
        self.state = np.array(self.chunk_history, dtype=np.float32)

        return self.state
    
    def calculate_remaining_play_time(self, calculation_time, decision_chunk_idx):
        """
        计算缓冲区剩余播放时间 (Combined Version)
        需要考虑跳过的块 (x=1) 不贡献缓冲时间。
        """
        remaining_current_chunk_time = 0.0
        future_buffered_time = 0.0
        currently_playing_chunk_idx = -1
        current_chunk_speed = 1.0

        # 1. Find currently playing chunk (must not be skipped)
        for j in range(decision_chunk_idx):
            action_info = self.chunk_actions.get(j)
            # Check if chunk j was played (had action, NOT skipped, valid play start time)
            if action_info and action_info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                play_start_time = self.p_i[j]
                chunk_speed = action_info.get('speed', 1.0)
                play_duration = self.t_i / chunk_speed
                play_end_time = play_start_time + play_duration

                if play_start_time <= calculation_time < play_end_time:
                    remaining_current_chunk_time = max(0, play_end_time - calculation_time)
                    currently_playing_chunk_idx = j
                    current_chunk_speed = chunk_speed
                    break

        # 2. Calculate future buffered time (must not be skipped, received, not started)
        start_index_for_future = currently_playing_chunk_idx + 1
        for j in range(start_index_for_future, decision_chunk_idx):
            action_info = self.chunk_actions.get(j)
            # Check if chunk j is: (1) has action, (2) NOT skipped, (3) is received, (4) not started playing
            if action_info and action_info.get('x', 0) == 0: # (1, 2) Has action and not skipped
                completion_time = self.actual_completion_times.get(j, float('inf'))
                if completion_time <= calculation_time: # (3) Is received
                     play_start = self.p_i[j] if j < len(self.p_i) else float('inf')
                     if play_start > calculation_time : # (4) Not started playing
                         chunk_speed = action_info.get('speed', 1.0)
                         future_buffered_time += self.t_i / chunk_speed # Use speed-adjusted duration

        total_remaining_time = remaining_current_chunk_time + future_buffered_time
        return total_remaining_time
    
    def _get_capacity_for_interval(self, interval_index):
        """获取或生成指定时间间隔的信道容量 (bytes/sec)"""
        if interval_index < 0: # 处理可能的边界情况
             return 0 # 或者返回一个默认值？取决于逻辑
        if interval_index not in self.interval_capacities:
            # 为这个新的时间间隔生成信道容量
            capacity = self.channel_model.calculate_capacity(regenerate_channel=True)
            self.interval_capacities[interval_index] = capacity

            if self.verbose:
                channel_details = self.channel_model.get_channel_details()
                start_t = interval_index * self.t_i
                end_t = (interval_index + 1) * self.t_i

        return self.interval_capacities[interval_index]
    
    def _simulate_chunk_transmission(self, chunk_idx, k_i, start_time):
        """
        模拟单个块的传输过程，考虑跨越多个信道间隔。
        返回实际传输时间和完成时间。
        """
        bitrate_mbps = self.bitrate_map[k_i]

        # Formula: Size (Bytes) = (Bitrate (Mbps) * 1,000,000 bits/Mb * Duration (s)) / 8 bits/byte

        size_to_transmit = (bitrate_mbps * 1_000_000 * self.t_i) / 8

        if size_to_transmit <= 1e-9: # 如果块大小为0
            return 0.0, start_time

        transmitted_data = 0.0
        current_time = start_time
        total_transmission_time = 0.0

        max_simulation_time = start_time + 10 * self.num_chunks * self.t_i # 设置一个模拟时间上限，防止无限循环

        while transmitted_data < size_to_transmit and current_time < max_simulation_time:
            interval_idx = int(current_time // self.t_i)
            capacity = self._get_capacity_for_interval(interval_idx)

            # 计算当前时间点到当前间隔结束还剩多少时间
            interval_end_time = (interval_idx + 1) * self.t_i
            time_left_in_interval = interval_end_time - current_time

            if capacity <= 1e-9: # 信道容量几乎为0
                # 如果容量为0且还有数据未传输，理论上会无限期卡住
                # 在模拟中，我们可以选择跳过这个间隔或者认为传输失败
                # 这里选择跳到下一个间隔开始处继续尝试
                time_step = time_left_in_interval + 1e-9 # 跳过这个零容量间隔
                total_transmission_time += time_step
                current_time += time_step
                continue # 继续下一个循环

            # 在当前容量下，传完剩余数据需要多少时间
            data_needed = size_to_transmit - transmitted_data
            time_needed_at_current_capacity = data_needed / capacity

            # 实际能在这个间隔中花费的时间，是剩余时间和所需时间中的较小者
            time_spent_this_interval = min(time_left_in_interval, time_needed_at_current_capacity)

            # 在这段时间内传输了多少数据
            data_transmitted_this_step = capacity * time_spent_this_interval

            # 更新状态
            transmitted_data += data_transmitted_this_step
            total_transmission_time += time_spent_this_interval
            current_time += time_spent_this_interval

            # 防止浮点数误差导致无法精确相等
            if transmitted_data >= size_to_transmit - 1e-9:
                break

        if current_time >= max_simulation_time and transmitted_data < size_to_transmit:
             return float('inf'), float('inf') # 表示传输失败或时间过长

        completion_time = start_time + total_transmission_time

        return total_transmission_time, completion_time
    
    def step(self, action_index):
        k_i, target_buffer = self.action_map[action_index]
        x_i = 0 # 기본적으로 스킵 안함
        chunk_idx = len(self.u_i)

        if chunk_idx >= self.num_chunks:
            last_valid_idx = self.num_chunks - 1
            if last_valid_idx < 0:
                return self.reset(), 0.0, True, {}
            current_state = self.get_state_for_decision(self.num_chunks)
            return current_state, 0.0, True, {}
        

        # u_i (전송 시작 시간) 계산
        chunk_generation_time = self.generated_chunk_times.get(chunk_idx, (chunk_idx + 1) * self.t_i)
        prev_completion_time = 0.0
        if chunk_idx > 0:
            prev_action_info = self.chunk_actions.get(chunk_idx - 1)
            if prev_action_info:
                if prev_action_info.get('x', 0) == 1:
                    prev_completion_time = prev_action_info.get('u', 0.0)
                else:
                    prev_completion_time =self.actual_completion_times.get(chunk_idx - 1, 0.0)
                    if prev_completion_time == float('inf'):
                        prev_completion_time = 0.0
        else:
            prev_completion_time = 0.0

        u_i = max(chunk_generation_time, prev_completion_time)
        if self.u_i: u_i = max(u_i, self.u_i[-1] + 1e-9)
        self.u_i.append(u_i)

        # tau_at_decision = self._calculate_live_latency(u_i, chunk_idx)
        buffer_at_decision = self.calculate_remaining_play_time(u_i, chunk_idx)

        # 수정
        # 버퍼가 7초보다 크다면, 강제로 스킵하는 기능이 존재해야 함
        if buffer_at_decision > 7.0:
            x_i = 1

        slow_play_threshold, fast_play_threshold = target_buffer - self.speed_rule_margin, target_buffer + self.speed_rule_margin

        current_playback_speed = self.playback_speeds[1]
        if buffer_at_decision < slow_play_threshold: 
            current_playback_speed = self.playback_speeds[0]
        elif buffer_at_decision > fast_play_threshold: 
            current_playback_speed = self.playback_speeds[2]

        self.speed_j.append(current_playback_speed)
        self.k_j.append(k_i)
        
        self.x_j.append(x_i)

        # 실행 (전송 시뮬레이션 및 스킵)
        l_i = 0.0
        completion_time = u_i
        p_i = float('inf')
        d_i = 0.0

        if x_i == 0:
            # 전송 수행
            l_i, completion_time = self._simulate_chunk_transmission(chunk_idx, k_i, u_i)

            # 재생 시작 및 스톨 계산
            if l_i != float('inf'):
                ready_time = completion_time
                prev_play_end_time = self.t_buffer

                for j in range(chunk_idx - 1, -1, -1):
                    prev_info = self.chunk_actions.get(j)
                    if prev_info and prev_info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                        prev_play_end_time = self.p_i[j] + self.t_i / prev_info.get('speed', 1.0)
                        break
                
                p_i = max(ready_time, prev_play_end_time)
                d_i = max(0.0, ready_time - prev_play_end_time)
        else:
            l_i = 0.0
            completion_time = u_i
            p_i = float('inf')
            d_i = 0.0

        self.l_i.append(l_i)
        self.actual_completion_times[chunk_idx] = completion_time
        self.p_i.append(p_i)
        self.d_i.append(d_i)
        self.chunk_actions[chunk_idx] = {'k': k_i, 'u': u_i, 'speed': current_playback_speed, 'x': x_i,}

        # latency (tau) 계산
        current_tau = float('inf')
        if x_i == 0 and p_i != float('inf'):
            current_tau = p_i - chunk_idx * self.t_i

        prev_tau = self.t_buffer
        valid_taus = [t for t in self.tau_i if t != float('inf')]
        if valid_taus:
            prev_tau = valid_taus[-1]
        
        self.tau_i.append(current_tau)
        
        tau_for_reward = float('inf')

        if x_i == 0: # 재생 시
            if current_tau != float('inf'):
                time_saved = self.t_i - (self.t_i / current_playback_speed)
                tau_for_reward = current_tau + d_i - time_saved
        else:
            if prev_tau != float('inf'):
                tau_for_reward = prev_tau - self.t_i
        
        if tau_for_reward == float('inf'):
            tau_for_reward = self.t_buffer

        base_reward = self._get_reward(k_i, x_i, d_i, tau_for_reward)
        tau_normalized = tau_for_reward / self.t_buffer
        history_entry = np.array([
            l_i/self.t_i, 
            ((self.bitrate_map[k_i]*1e6*self.t_i)/8)/1e6, 
            d_i/self.t_i, 
            buffer_at_decision/self.t_buffer, 
            tau_normalized, 
            k_i/len(self.quality_levels)
        ], dtype=np.float32)

        self.chunk_history.append(history_entry)
        next_state = np.array(self.chunk_history, dtype=np.float32)
        done = chunk_idx >= self.num_chunks - 1

        return next_state, base_reward, done, {}
    
    def get_state_for_decision(self, target_chunk_idx):
        chunk_generation_time = self.generated_chunk_times.get(target_chunk_idx, (target_chunk_idx + 1) * self.t_i)
        prev_completion = 0.0
        if target_chunk_idx > 0:
            prev_completion = self.actual_completion_times.get(target_chunk_idx - 1, 0.0)
            if prev_completion == float('inf'):
                prev_completion = self.u_i[-1] if self.u_i else 0.0
        
        u_i = max(chunk_generation_time, prev_completion)
        if self.u_i:
            u_i = max(u_i, self.u_i[-1] + 1e-9)
        
        n_i = -1
        for j in range(target_chunk_idx - 1, -1, -1):
            action_info = self.chunk_actions.get(j)
            if action_info and action_info.get('x', 0) == 0:
                p_j = self.p_i[j]
                if p_j != float('inf') and p_j <= u_i:
                    n_i = j
                    break
        
        tau_i = self.t_buffer
        if n_i != -1:
            p_n_i = self.p_i[n_i]
            action_n = self.chunk_actions[n_i]
            v_n = action_n['speed']
            eta_n = self.t_i

            sum_duration = 0.0
            for j in range(n_i):
                a_j = self.chunk_actions.get(j)
                if a_j and a_j.get('x', 0) == 0:
                    sum_duration += self.t_i / a_j['speed']
            
            play_end_time_n = p_n_i + (eta_n / v_n)
            current_stall = max(0.0, u_i - play_end_time_n)

            tau_i = p_n_i - sum_duration + current_stall
        
        else:
            tau_i = self.t_buffer
        
        remaining_buffer_time = self.calculate_remaining_play_time(u_i, target_chunk_idx)

        safe_tau = tau_i if np.isfinite(tau_i) else 10.0
        safe_buffer = remaining_buffer_time if np.isfinite(remaining_buffer_time) else 0.0

        return np.array([safe_tau, safe_buffer], dtype=np.float32)
    
    
    def _get_reward(self, k_i, x_i, last_complete_d_i, tau_i_plus_1):
        """
        보상 함수 구현 함수
        Proposed 보상 함수와 동일한 형태로 구현
        """
        quality_reward = 0
        if x_i == 0:
            quality_reward = self.W1 * (k_i**2)
        
        stall_penalty = self.W2 * last_complete_d_i
        delay_penalty = self.W_tau * (tau_i_plus_1 - self.t_i*x_i) ** 2
        skip_penalty = self.W3 * x_i
        
        raw_reward = quality_reward - delay_penalty - skip_penalty - stall_penalty
        reward = max(min(raw_reward, 20), -20)

        return reward
    
    def get_skip_statistics(self):
        if not self.x_j: return 0, 0
        skipped = sum(self.x_j)
        num_processed = len(self.x_j)
        played = num_processed - skipped
        return skipped, played

    def get_average_speed(self):
        # speed_j에 이미 실제 속도 값(예: 1.0, 1.1 등)이 들어있으므로 바로 평균을 구하면 됩니다.
        if not self.speed_j: return 1.0
        return sum(self.speed_j) / len(self.speed_j)
    
    # get_average_quality needs to check x_j
    def get_average_quality(self):
        # Quality of chunks that were chosen *and not skipped* and successfully started playing
        played_qualities = [self.k_j[i] for i in range(len(self.k_j))
                            if i < len(self.x_j) and self.x_j[i] == 0 # Not skipped
                            and i < len(self.p_i) and self.p_i[i] != float('inf')] # Played successfully
        if not played_qualities: return 0
        return sum(played_qualities) / len(played_qualities)

    def get_average_capacity(self):
        if not self.interval_capacities: return 0
        valid_capacities = [c for c in self.interval_capacities.values() if c is not None and c != float('inf')]
        if not valid_capacities: return 0
        avg_bytes_per_sec = sum(valid_capacities) / len(valid_capacities)
        return (avg_bytes_per_sec * 8) / 1e6
        
    def get_average_tau_i(self):
        """返回当前episode所有记录的tau_i的平均值 (忽略inf)"""
        if not self.tau_i: return 0.0
        valid_tau = [t for t in self.tau_i if t is not None and t != float('inf')]
        if not valid_tau: return 0.0 # 如果全是 inf 或空
        avg = sum(valid_tau) / len(valid_tau)
        return avg

    def get_total_stall_time(self):
        """返回当前episode的总卡顿时间 (忽略inf)"""
        if not self.d_i: return 0.0
        actual_stalls = [d for d in self.d_i if d is not None and d != float('inf') and d > 1e-9]
        total = sum(actual_stalls)
        return total

    def get_average_d_i(self):
        if not self.d_i: return 0.0
        valid_d_values = [d for d in self.d_i if isinstance(d, (int, float)) and d != float('inf')]
        if not valid_d_values: return 0.0
        return sum(valid_d_values) / len(valid_d_values)


#################################
# PPO 알고리즘 부분 - 수정 불필요
#################################
LEARNING_RATE = 1e-4
GAMMA = 0.96
LAMBDA = 0.97
BATCH_SIZE = 128
EPOCHS = 10
CLIP_EPSILON = 0.2
NUM_STEPS = 1024
MAX_EPISODES = 50000
ENTROPY_COEF = 0.01
HISTORY_LEN, NUM_FEATURES, ACTION_DIM = 48, 6, 40

class ActorNetworkTCLiVi(nn.Module):
    def __init__(self):
        super(ActorNetworkTCLiVi, self).__init__(); self.conv1 = nn.Conv1d(NUM_FEATURES, 32, 3); self.flatten_size = 32 * (HISTORY_LEN - 3 + 1)
        self.fc1 = nn.Linear(self.flatten_size, 128); self.fc2 = nn.Linear(128, ACTION_DIM)
    def forward(self, state):
        x = torch.relu(self.conv1(state.permute(0, 2, 1))).view(-1, self.flatten_size)
        x = torch.relu(self.fc1(x)); return torch.softmax(self.fc2(x), dim=-1)
    
class CriticNetworkTCLiVi(nn.Module):
    def __init__(self):
        super(CriticNetworkTCLiVi, self).__init__(); self.conv1 = nn.Conv1d(NUM_FEATURES, 32, 3); self.flatten_size = 32 * (HISTORY_LEN - 3 + 1)
        self.fc1 = nn.Linear(self.flatten_size, 128); self.fc2 = nn.Linear(128, 1)
    def forward(self, state):
        x = torch.relu(self.conv1(state.permute(0, 2, 1))).view(-1, self.flatten_size)
        x = torch.relu(self.fc1(x)); return self.fc2(x)
    
class PPOAgent_TCLiVi:
    def __init__(self):
        self.actor = ActorNetworkTCLiVi().to(device); self.critic = CriticNetworkTCLiVi().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE); self.memory = []

    def store_experience(self, *args): self.memory.append(args)

    def clear_memory(self): self.memory = []

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = self.actor(state_tensor); dist = Categorical(probs); action = dist.sample()
            log_prob = dist.log_prob(action); value = self.critic(state_tensor)
        return action.item(), log_prob.item(), value.item()
    
    def update(self):
        states, actions, rewards, next_states, dones, old_log_probs, values = zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(device); actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device); dones = torch.FloatTensor(dones).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device); values = torch.FloatTensor(values).to(device)
        with torch.no_grad():
            last_value = self.critic(torch.FloatTensor(next_states[-1]).unsqueeze(0).to(device)).item()
        advantages = torch.zeros_like(rewards).to(device); last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t+1] if t < len(rewards) - 1 else last_value
            delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + GAMMA * LAMBDA * next_non_terminal * last_gae_lam; advantages[t] = last_gae_lam
        returns = advantages + values; advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(EPOCHS):
            indices = np.arange(len(self.memory)); np.random.shuffle(indices)
            for start in range(0, len(self.memory), BATCH_SIZE):
                batch_indices = indices[start : start + BATCH_SIZE]
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = \
                    states[batch_indices], actions[batch_indices], old_log_probs[batch_indices], advantages[batch_indices], returns[batch_indices]
                probs = self.actor(batch_states); dist = Categorical(probs); new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean(); value_preds = self.critic(batch_states).squeeze()
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages; surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy
                critic_loss = nn.MSELoss()(value_preds, batch_returns)
                self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
        self.clear_memory()
        
    def save_models(self, episode, prefix="TCLiVi_rule_based_speed"):
        torch.save(self.actor.state_dict(), f'baseline/train/B5/models/{prefix}_actor_ep{episode}.pth')
        torch.save(self.critic.state_dict(), f'baseline/train/B5/models/{prefix}_critic_ep{episode}.pth')
        print(f"TCLiVi 모델 (규칙 기반 속도)이 에피소드 {episode}에 저장되었습니다")

#################################
# 훈련 함수 - 이전 수정과 일치
#################################
def train_tclivi():
    env = VideoStreamingEnv_TCLiVi(num_chunks=300, history_len=HISTORY_LEN, W1=1, W2=30, W3=100.0, W_tau=0.5)
    agent = PPOAgent_TCLiVi()
    episode_rewards, episode_avg_qualities, episode_stall_times, episode_avg_tau_i, episode_avg_speeds = [], [], [], [], []
    results_prefix = "TCLiVi_rule_based_speed"
    for episode in range(MAX_EPISODES):
        state = env.reset(); episode_reward = 0; current_episode_taus = []
        for step in range(env.num_chunks):
            current_episode_taus.append(state[-1][4] * env.t_buffer)
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done, log_prob, value)
            state = next_state; episode_reward += reward
            if done: break
        if len(agent.memory) >= NUM_STEPS: agent.update()
        episode_rewards.append(episode_reward)
        episode_avg_qualities.append(env.get_average_quality())
        episode_stall_times.append(env.get_total_stall_time())
        episode_avg_tau_i.append(np.mean(current_episode_taus) if current_episode_taus else 0)
        episode_avg_speeds.append(env.get_average_speed())
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]); avg_quality = np.mean(episode_avg_qualities[-100:])
            avg_stall = np.mean(episode_stall_times[-100:]); avg_tau = np.mean(episode_avg_tau_i[-100:])
            avg_speed = np.mean(episode_avg_speeds[-100:])
            print(f"Ep {episode+1}/{MAX_EPISODES} | Avg R: {avg_reward:.2f} | Avg Qual: {avg_quality:.2f} | Avg Stall: {avg_stall:.2f}s | Avg τ: {avg_tau:.2f}s | Avg Speed: {avg_speed:.2f}x")
            np.save(f'results/{results_prefix}_rewards.npy', np.array(episode_rewards))
            np.save(f'results/{results_prefix}_qualities.npy', np.array(episode_avg_qualities))
            np.save(f'results/{results_prefix}_stalls.npy', np.array(episode_stall_times))
            np.save(f'results/{results_prefix}_taus.npy', np.array(episode_avg_tau_i))
            np.save(f'results/{results_prefix}_speeds.npy', np.array(episode_avg_speeds))
        if (episode + 1) % 5000 == 0: agent.save_models(episode + 1)
    agent.save_models("final")

#################################
# ### NEW ###: 완전히 새로운 테스트 함수
#################################
def test_tclivi(episode_to_load, test_episodes=100):
    """
    저장된 모델을 사용하여 결정론적 평가 수행
    :param episode_to_load: 로드할 모델 에피소드 번호, e.g., "final", 50000
    :param test_episodes: 평가에 사용할 테스트 에피소드 수
    """
    print(f"\n--- 저장된 TCLiVi 모델 테스트 시작 (Episode: {episode_to_load}) ---")
    
    # 환경 및 에이전트 초기화
    env = VideoStreamingEnv_TCLiVi(num_chunks=300, history_len=HISTORY_LEN)
    agent = PPOAgent_TCLiVi()
    
    # 훈련된 Actor 모델 로드
    model_prefix = "TCLiVi_rule_based_speed"
    try:
        actor_path = f'models/{model_prefix}_actor_ep{episode_to_load}.pth'
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        print(f"Actor 모델 로드 성공: {actor_path}")
    except Exception as e:
        print(f"Actor 모델 로드 실패: {e}"); return
    
    # Actor 네트워크를 평가 모드로 설정
    agent.actor.eval()

    # 모든 테스트 에피소드 결과를 저장할 리스트
    all_qualities, all_stalls, all_taus, all_speeds = [], [], [], []

    for ep in range(test_episodes):
        state = env.reset()
        done = False
        current_episode_taus = []
        
        while not done:
            current_episode_taus.append(state[-1][4] * env.t_buffer)

            # 테스트 시, 무작위 샘플링 대신 최적의 동작(가장 높은 확률) 선택
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                probs = agent.actor(state_tensor)
                action = torch.argmax(probs).item() # <--- argmax를 사용하여 결정론적 선택

            next_state, _, done, _ = env.step(action)
            state = next_state
        
        # 현재 테스트 에피소드 결과 기록
        all_qualities.append(env.get_average_quality())
        all_stalls.append(env.get_total_stall_time())
        all_taus.append(np.mean(current_episode_taus) if current_episode_taus else 0)
        all_speeds.append(env.get_average_speed())
        
        if (ep + 1) % 10 == 0:
            print(f"  테스트 에피소드 완료: {ep + 1}/{test_episodes}", end='\r')

    # 모든 테스트 에피소드의 평균 성능 계산 및 출력
    avg_quality = np.mean(all_qualities); avg_stall = np.mean(all_stalls)
    avg_tau = np.mean(all_taus); avg_speed = np.mean(all_speeds)
    
    print("\n\n--- TCLiVi 베이스라인 최종 테스트 결과 ---")
    print(f"  {test_episodes}개 테스트 에피소드에서의 평균 성능:")
    print(f"  -> 평균 화질 (Avg Qual): {avg_quality:.3f} Mbps")
    print(f"  -> 평균 끊김 (Avg Stall): {avg_stall:.3f} s")
    print(f"  -> 평균 지연 (Avg τ):    {avg_tau:.3f} s")
    print(f"  -> 평균 속도 (Avg Speed): {avg_speed:.3f}x")
    print("---------------------------------\n")

#################################
# ### MODIFIED ###: 메인 실행 함수
#################################
if __name__ == "__main__":
    # --- 1단계: 모델 훈련 ---
    # 이미 훈련된 모델이 있다면, 아래 세 줄을 주석 처리하여 훈련을 건너뛸 수 있습니다.
    print("--- 1단계: TCLiVi 베이스라인 모델 훈련 시작 ---")
    start_time = time.time()
    train_tclivi()
    print(f"총 훈련 시간: {(time.time() - start_time)/60:.2f} 분")

    # --- 2단계: 모델 자동 테스트 ---
    # 훈련 완료 후, 최종 저장된 "final" 모델을 자동으로 로드하여 테스트
    # 여기의 "final"을 50000과 같이 테스트하려는 임의의 에피소드 번호로 수정할 수 있습니다.
    # print("\n--- 2단계: 최종 훈련된 모델 평가 시작 ---")
    # test_tclivi(episode_to_load="final", test_episodes=10)

