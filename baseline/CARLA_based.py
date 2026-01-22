#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# 결과 저장을 위한 디렉토리 생성
if not os.path.exists('results'):
    os.makedirs('results')

# 모델 저장을 위한 디렉토리 생성
if not os.path.exists('models'):
    os.makedirs('models')

#################################
# 환경 부분 (재생 속도 & 스킵 결합)
#################################

class DynamicChannelModel:
    """동적 무선 채널을 시뮬레이션하는 클래스."""
    def __init__(self, W=1e6, gamma_dB=35, distance=20, beta=2, sigma_dB=4, mu_dB=0):
        self.W = W
        self.gamma = 10 ** (gamma_dB / 10)
        self.distance = distance
        self.beta = beta
        self.sigma_dB = sigma_dB
        self.mu_dB = mu_dB
        self.current_channel_gain = None
        self.current_capacity = None
        self.channel_components = None

    def generate_channel_state(self):
        """새로운 채널 상태(경로 손실, 섀도잉, 페이딩)를 생성합니다."""
        path_loss = self.distance ** self.beta
        shadowing_dB = np.random.normal(self.mu_dB, self.sigma_dB)
        shadowing = 10 ** (shadowing_dB / 10)
        fast_fading_real = np.random.normal(0, 1/np.sqrt(2))
        fast_fading_imag = np.random.normal(0, 1/np.sqrt(2))
        fast_fading_complex = fast_fading_real + 1j * fast_fading_imag
        fast_fading = np.abs(fast_fading_complex) ** 2
        channel_gain_squared = (1 / path_loss) * shadowing * fast_fading
        self.current_channel_gain = np.sqrt(channel_gain_squared)
        return

    def calculate_capacity(self, regenerate_channel=True):
        """섀넌 용량 공식을 사용하여 채널 용량을 계산합니다."""
        if regenerate_channel or self.current_channel_gain is None:
            self.generate_channel_state()
        channel_gain_squared = self.current_channel_gain ** 2
        capacity_bps = self.W * np.log2(1 + self.gamma * channel_gain_squared)
        capacity_bytes_per_sec = capacity_bps / 8
        self.current_capacity = capacity_bytes_per_sec
        return capacity_bytes_per_sec

class VideoStreamingEnvCombined:
    """스킵(Skip)과 재생 속도(Speed)를 결합한 비디오 스트리밍 환경을 초기화합니다."""
    def __init__(self, num_chunks=300, t_buffer=4, t_i=2,
                 W1=25.0, W2=300.0, W3=100.0, W_tau=1.0,
                 verbose=False):
        self.num_chunks = num_chunks
        self.t_buffer = t_buffer
        self.t_i = t_i
        self.quality_levels = [1, 2, 3, 4]
        self.bitrate_map = {1: 0.8, 2: 1.2, 3: 2.4, 4: 4.8}
        self.playback_speeds = [0.9, 1.0, 1.1]
        self.num_speed_levels = len(self.playback_speeds)
        self.verbose = verbose

        # 보상 가중치
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.W_tau = W_tau

        self.channel_model = DynamicChannelModel()
        self.reset()

    def reset(self):
        """환경을 초기 상태로 리셋합니다."""
        self.interval_capacities = {}
        self.actual_completion_times = {}
        self.chunk_actions = {}
        self.generated_chunk_times = {i: (i + 1) * self.t_i for i in range(self.num_chunks)}
        self.u_i, self.p_i, self.d_i, self.k_j, self.speed_idx_j, self.l_i, self.x_j = [], [], [], [], [], [], []
        self.cumulative_skips = 0
        self.state = self.get_state_for_decision(0)
        return self.state

    def _calculate_live_latency(self, calculation_time_ui, decision_chunk_idx):
        """
        논문 공식(5)에 따라 결정 시점 u_i에서의 라이브 지연 시간 τ_i를 정확하게 계산합니다.
        τ_i = [과거 누적 지연] + [현재 스톨 지연]
        """
        # --- n(i) 찾기: calculation_time_ui 시점에 재생 중인 청크의 인덱스 ---
        currently_playing_chunk_idx = -1
        last_played_chunk_info = None
        for j in range(decision_chunk_idx - 1, -1, -1):
            action_info = self.chunk_actions.get(j)
            if action_info and action_info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] <= calculation_time_ui:
                currently_playing_chunk_idx = j
                last_played_chunk_info = action_info
                break
        
        if currently_playing_chunk_idx == -1:
            return self.t_buffer

        # --- 1부: 과거 누적 지연 계산 ---
        p_n_i = self.p_i[currently_playing_chunk_idx]
        sum_of_prev_play_durations = sum(
            self.t_i / info.get('speed', 1.0)
            for j, info in self.chunk_actions.items()
            if j < currently_playing_chunk_idx and info.get('x', 0) == 0
        )
        historical_lag = p_n_i - sum_of_prev_play_durations

        # --- 2부: 현재 스톨 지연 계산 ---
        current_chunk_speed = last_played_chunk_info.get('speed', 1.0)
        current_chunk_play_duration = self.t_i / current_chunk_speed
        current_chunk_theoretical_end_time = p_n_i + current_chunk_play_duration
        current_stall_lag = max(0, calculation_time_ui - current_chunk_theoretical_end_time)

        total_latency = historical_lag + current_stall_lag
        return total_latency

    def calculate_remaining_play_time(self, calculation_time, decision_chunk_idx):
        """버퍼에 남은 재생 시간 B_i를 계산합니다."""
        remaining_current_chunk_time = 0.0
        future_buffered_time = 0.0
        currently_playing_chunk_idx = -1
        for j in range(decision_chunk_idx - 1, -1, -1):
            action_info = self.chunk_actions.get(j)
            if action_info and action_info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                play_start_time = self.p_i[j]
                play_duration = self.t_i / action_info.get('speed', 1.0)
                play_end_time = play_start_time + play_duration
                if play_start_time <= calculation_time < play_end_time:
                    remaining_current_chunk_time = max(0, play_end_time - calculation_time)
                    currently_playing_chunk_idx = j
                    break
        
        start_index_for_future = currently_playing_chunk_idx + 1
        for j in range(start_index_for_future, decision_chunk_idx):
            action_info = self.chunk_actions.get(j)
            if action_info and action_info.get('x', 0) == 0:
                completion_time = self.actual_completion_times.get(j, float('inf'))
                play_start = self.p_i[j] if j < len(self.p_i) else float('inf')
                if completion_time <= calculation_time and play_start > calculation_time:
                    future_buffered_time += self.t_i / action_info.get('speed', 1.0)
        
        return remaining_current_chunk_time + future_buffered_time

    def _get_capacity_for_interval(self, interval_index):
        if interval_index < 0: return 0
        if interval_index not in self.interval_capacities:
            self.interval_capacities[interval_index] = self.channel_model.calculate_capacity()
        return self.interval_capacities[interval_index]

    def _simulate_chunk_transmission(self, chunk_idx, k_i, start_time):
        size_to_transmit = (self.bitrate_map[k_i] * 1_000_000 * self.t_i) / 8
        if size_to_transmit <= 1e-9: return 0.0, start_time
        
        transmitted_data, current_time, total_transmission_time = 0.0, start_time, 0.0
        max_simulation_time = start_time + 10 * self.num_chunks * self.t_i

        while transmitted_data < size_to_transmit and current_time < max_simulation_time:
            interval_idx = int(current_time // self.t_i)
            capacity = self._get_capacity_for_interval(interval_idx)
            time_left_in_interval = (interval_idx + 1) * self.t_i - current_time
            
            if capacity <= 1e-9:
                current_time += time_left_in_interval + 1e-9
                total_transmission_time = current_time - start_time
                continue

            time_needed = (size_to_transmit - transmitted_data) / capacity
            time_spent = min(time_left_in_interval, time_needed)
            
            transmitted_data += time_spent * capacity
            total_transmission_time += time_spent
            current_time += time_spent

        if current_time >= max_simulation_time: return float('inf'), float('inf')
        return total_transmission_time, start_time + total_transmission_time

    def step(self, action):
        k_i, speed_idx, x_i = action
        chunk_idx = len(self.u_i)

        if chunk_idx >= self.num_chunks:
            return self.reset(), 0.0, True, {}

        # --- 결정 시점 u_i 계산 ---
        chunk_gen_time = self.generated_chunk_times.get(chunk_idx, (chunk_idx + 1) * self.t_i)
        prev_comp_time = 0.0
        if chunk_idx > 0:
            prev_action = self.chunk_actions.get(chunk_idx - 1)
            if prev_action:
                prev_comp_time = prev_action.get('u') if prev_action.get('x') == 1 else self.actual_completion_times.get(chunk_idx - 1, 0.0)
        u_i = max(chunk_gen_time, prev_comp_time)
        if self.u_i: u_i = max(u_i, self.u_i[-1] + 1e-9)
        self.u_i.append(u_i)

        # === 결정 시점의 상태(τ_i, B_i)를 가져와 보상 계산에 사용 ===
        tau_at_decision = self._calculate_live_latency(u_i, chunk_idx)
        buffer_at_decision = self.calculate_remaining_play_time(u_i, chunk_idx)

        # --- 행동 저장 및 시뮬레이션 ---
        self.chunk_actions[chunk_idx] = {'k': k_i, 'speed_idx': speed_idx, 'speed': self.playback_speeds[speed_idx], 'x': x_i, 'u': u_i}
        self.k_j.append(k_i); self.speed_idx_j.append(speed_idx); self.x_j.append(x_i)
        if x_i == 1: self.cumulative_skips += 1
        
        l_i, completion_time = (0.0, u_i) if x_i == 1 else self._simulate_chunk_transmission(chunk_idx, k_i, u_i)
        self.l_i.append(l_i)
        self.actual_completion_times[chunk_idx] = completion_time

        # === 논문 공식에 따라 보상 함수를 위한 τ_{i+1} 계산 ===
        # τ_{i+1} = τ_i + max{l_i - B_i, 0}
        tau_for_reward = tau_at_decision + (max(0, l_i - buffer_at_decision) if x_i == 0 else 0)

        # --- p_i 와 d_i 계산 ---
        p_i, d_i = float('inf'), 0.0
        if x_i == 0 and l_i != float('inf'):
            ready_time = completion_time
            prev_play_end_time = self.t_buffer
            for j in range(chunk_idx - 1, -1, -1):
                prev_action = self.chunk_actions.get(j)
                if prev_action and prev_action.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                    prev_play_end_time = self.p_i[j] + self.t_i / prev_action.get('speed', 1.0)
                    break
            p_i = max(ready_time, prev_play_end_time)
            d_i = max(0, ready_time - prev_play_end_time)
        
        while len(self.p_i) <= chunk_idx: self.p_i.append(float('inf'))
        self.p_i[chunk_idx] = p_i
        while len(self.d_i) <= chunk_idx: self.d_i.append(0.0)
        self.d_i[chunk_idx] = d_i

        # --- 보상 계산 ---
        reward = self._get_reward(k_i, x_i, d_i, tau_for_reward)

        # --- 다음 스텝 준비 ---
        next_state = self.get_state_for_decision(chunk_idx + 1)
        done = (chunk_idx >= self.num_chunks - 1)
        return next_state, reward, done, {}

    def get_state_for_decision(self, target_chunk_idx):
        """'target_chunk_idx'에 대한 행동을 결정하기 직전의 상태 [τ, B]를 가져옵니다."""
        chunk_gen_time = self.generated_chunk_times.get(target_chunk_idx, (target_chunk_idx + 1) * self.t_i)
        calc_time_ui = chunk_gen_time
        if target_chunk_idx > 0:
            prev_comp = self.actual_completion_times.get(target_chunk_idx - 1, 0.0)
            if prev_comp != float('inf'):
                calc_time_ui = max(chunk_gen_time, prev_comp)
        if self.u_i:
            calc_time_ui = max(calc_time_ui, self.u_i[-1] + 1e-9)
            
        current_tau = self._calculate_live_latency(calc_time_ui, target_chunk_idx)
        remaining_buffer = self.calculate_remaining_play_time(calc_time_ui, target_chunk_idx)
        
        safe_tau = current_tau if np.isfinite(current_tau) else 10 * self.t_buffer
        safe_buffer = remaining_buffer if np.isfinite(remaining_buffer) else 0.0
        return np.array([safe_tau, safe_buffer], dtype=np.float32)

    def _get_reward(self, k_i, x_i, d_i, tau_i_plus_1):
        """보상 계산 (논문 공식 7)"""
        # r_i = w1*q(k_i)^2*(1-xi) - w2*d_i - w3*xi - (τ_{i+1}-ti*xi)^2
        quality_score = self.bitrate_map.get(k_i, 0)
        quality_reward = self.W1 * (quality_score ** 2) * (1 - x_i)
        stall_penalty = self.W2 * d_i
        skip_penalty = self.W3 * x_i
        delay_penalty = self.W_tau * (tau_i_plus_1 - self.t_i * x_i)**2
        
        reward = quality_reward - stall_penalty - skip_penalty - delay_penalty
        return reward

    # --- 아래는 통계 함수이며, 수정이 필요 없습니다 ---
    def get_skip_statistics(self):
        if not self.x_j: return 0, 0
        skipped = sum(self.x_j)
        return skipped, len(self.x_j) - skipped

    def get_average_speed(self):
        played_speeds = [self.playback_speeds[s_idx] for i, s_idx in enumerate(self.speed_idx_j) if i < len(self.x_j) and self.x_j[i] == 0]
        return np.mean(played_speeds) if played_speeds else 1.0

    def get_average_quality(self):
        played_qualities = [self.bitrate_map[k] for i, k in enumerate(self.k_j) if i < len(self.x_j) and self.x_j[i] == 0 and i < len(self.p_i) and self.p_i[i] != float('inf')]
        return np.mean(played_qualities) if played_qualities else 0

    def get_average_capacity(self):
        if not self.interval_capacities: return 0
        return (np.mean(list(self.interval_capacities.values())) * 8) / 1e6

    def get_total_stall_time(self):
        return sum(d for d in self.d_i if d is not None and d != float('inf'))

# #################################
# 강화학습 부분 (PPO 알고리즘 - 결합 버전)
# #################################

# --- 하이퍼파라미터 및 상수 ---
LEARNING_RATE = 1e-4
GAMMA = 0.96
LAMBDA = 0.97
EPOCHS = 10
BATCH_SIZE = 128
CLIP_EPSILON = 0.2
NUM_STEPS = 1024
MAX_EPISODES = 50000
entropy_coef = 0.01
STATE_DIM = 2
QUALITY_LEVELS = 4
SPEED_LEVELS = 3
SKIP_OPTIONS = 2

class ActorNetworkCombined(nn.Module):
    """Actor 신경망: 상태를 받아 품질, 속도, 스킵에 대한 행동 확률을 출력합니다."""
    def __init__(self):
        super(ActorNetworkCombined, self).__init__()
        self.base = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.quality_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, QUALITY_LEVELS))
        self.speed_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, SPEED_LEVELS))
        self.skip_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, SKIP_OPTIONS))

    def forward(self, state):
        if not isinstance(state, torch.Tensor): state = torch.FloatTensor(state).to(device)
        if state.dim() == 1: state = state.unsqueeze(0)
        base_features = self.base(state)
        quality_probs = torch.softmax(self.quality_head(base_features), dim=-1)
        speed_probs = torch.softmax(self.speed_head(base_features), dim=-1)
        skip_probs = torch.softmax(self.skip_head(base_features), dim=-1)
        return quality_probs, speed_probs, skip_probs

class CriticNetwork(nn.Module):
    """Critic 신경망: 상태의 가치를 추정합니다."""
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(STATE_DIM, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, state):
        if not isinstance(state, torch.Tensor): state = torch.FloatTensor(state).to(device)
        if state.dim() == 1: state = state.unsqueeze(0)
        return self.network(state)

class PPOAgentCombined:
    """PPO 에이전트: Actor와 Critic을 관리하고 학습을 수행합니다."""
    def __init__(self):
        self.actor = ActorNetworkCombined().to(device)
        self.critic = CriticNetwork().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.clear_memory()

    def select_action(self, state):
        with torch.no_grad():
            quality_probs, speed_probs, skip_probs = self.actor(state)
            quality_dist = Categorical(quality_probs)
            speed_dist = Categorical(speed_probs)
            skip_dist = Categorical(skip_probs)
            quality_action, speed_action, skip_action = quality_dist.sample(), speed_dist.sample(), skip_dist.sample()
            log_prob_q, log_prob_s, log_prob_sk = quality_dist.log_prob(quality_action), speed_dist.log_prob(speed_action), skip_dist.log_prob(skip_action)
            value = self.critic(state)
        return (quality_action.item() + 1, speed_action.item(), skip_action.item()), log_prob_q.item(), log_prob_s.item(), log_prob_sk.item(), value.item()

    def store_experience(self, state, action, reward, next_state, done, log_prob_q, log_prob_s, log_prob_sk, value):
        self.states.append(state)
        self.actions_quality.append(action[0] - 1)
        self.actions_speed.append(action[1])
        self.actions_skip.append(action[2])
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs_quality.append(log_prob_q)
        self.log_probs_speed.append(log_prob_s)
        self.log_probs_skip.append(log_prob_sk)
        self.values.append(value)

    def clear_memory(self):
        self.states, self.actions_quality, self.actions_speed, self.actions_skip, self.rewards, self.next_states, self.dones, self.log_probs_quality, self.log_probs_speed, self.log_probs_skip, self.values = [],[],[],[],[],[],[],[],[],[],[]

    def compute_returns_and_advantages(self):
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        values = torch.tensor(self.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            last_value = self.critic(torch.FloatTensor(self.next_states[-1]).to(device)).item()
        
        returns, advantages = torch.zeros_like(rewards), torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t] if t == len(rewards) - 1 else 1.0 - dones[t+1]
            next_values = last_value if t == len(rewards) - 1 else values[t+1]
            delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + GAMMA * LAMBDA * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self):
        returns, advantages = self.compute_returns_and_advantages()
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions_quality = torch.LongTensor(self.actions_quality).to(device)
        actions_speed = torch.LongTensor(self.actions_speed).to(device)
        actions_skip = torch.LongTensor(self.actions_skip).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs_quality).to(device) + \
                        torch.FloatTensor(self.log_probs_speed).to(device) + \
                        torch.FloatTensor(self.log_probs_skip).to(device)
        
        for _ in range(EPOCHS):
            indices = np.random.permutation(len(self.states))
            for start in range(0, len(self.states), BATCH_SIZE):
                batch_indices = indices[start:start+BATCH_SIZE]
                batch_states, batch_actions_q, batch_actions_s, batch_actions_sk = states[batch_indices], actions_quality[batch_indices], actions_speed[batch_indices], actions_skip[batch_indices]
                batch_old_log_probs, batch_advantages, batch_returns = old_log_probs[batch_indices], advantages[batch_indices], returns[batch_indices]

                q_probs, s_probs, sk_probs = self.actor(batch_states)
                value_preds = self.critic(batch_states).squeeze()
                
                q_dist, s_dist, sk_dist = Categorical(q_probs), Categorical(s_probs), Categorical(sk_probs)
                new_log_probs = q_dist.log_prob(batch_actions_q) + s_dist.log_prob(batch_actions_s) + sk_dist.log_prob(batch_actions_sk)
                entropy = (q_dist.entropy() + s_dist.entropy() + sk_dist.entropy()).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                critic_loss = nn.MSELoss()(value_preds, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

    def save_models(self, episode):
        """모델을 저장합니다."""
        model_name_base = f'Combined_ppo_episode_{episode}'
        torch.save(self.actor.state_dict(), f'models/{model_name_base}_actor.pth')
        torch.save(self.critic.state_dict(), f'models/{model_name_base}_critic.pth')
        print(f"{episode} 에피소드 모델 저장 완료")

    def load_models(self, episode):
        """저장된 모델을 불러옵니다."""
        model_name_base = f'Combined_ppo_episode_{episode}'
        try:
            self.actor.load_state_dict(torch.load(f'models/{model_name_base}_actor.pth', map_location=device))
            self.critic.load_state_dict(torch.load(f'models/{model_name_base}_critic.pth', map_location=device))
            print(f"{episode} 에피소드 모델 불러오기 완료")
        except FileNotFoundError:
            print(f"오류: {episode} 에피소드의 모델 파일을 찾을 수 없습니다.")

# #################################
# 그래프 그리는 함수
# #################################
def plot_learning_curve(rewards, save_dir='models', window_size=100):
    """
    학습 곡선을 시각화하고 저장합니다. (Best Practice)
    
    Args:
        rewards (list): 에피소드별 총 보상 리스트
        save_dir (str): 그래프 저장 경로
        window_size (int): 이동 평균을 계산할 윈도우 크기
    """
    # 데이터가 없으면 리턴
    if not rewards:
        print("시각화할 보상 데이터가 없습니다.")
        return

    plt.figure(figsize=(12, 6))
    
    # 1. Raw Data (회색, 투명하게): 에피소드별 변동성을 보여줌
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw Rewards')
    
    # 2. Moving Average (빨간색, 진하게): 학습의 수렴 경향성을 보여줌
    # 이동 평균은 강화학습의 성능 지표로 가장 많이 사용됨
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        # 이동 평균은 데이터 길이가 줄어드므로 x축을 맞춰줌
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, 
                 color='red', linewidth=2, label=f'Moving Avg (Window={window_size})')
    else:
        # 데이터가 적을 경우 전체 평균선을 그림
        plt.axhline(y=np.mean(rewards), color='red', linestyle='--', label='Mean Reward')

    plt.title('Training Learning Curve (PPO)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 그래프 저장 (파일명에 시간 정보를 넣어 덮어쓰기 방지)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_dir, f'learning_curve_{timestamp}.png')
    plt.savefig(save_path)
    plt.close() # 메모리 누수 방지를 위해 닫기
    print(f"학습 그래프가 저장되었습니다: {save_path}")



# #################################
# 훈련 함수
# #################################

def train_combined():
    """메인 훈련 루프 함수."""
    env = VideoStreamingEnvCombined()
    agent = PPOAgentCombined()

    # --- 로깅을 위한 리스트 초기화 ---
    episode_rewards, episode_lengths, episode_avg_capacities, episode_stall_times = [], [], [], []
    episode_skipped, episode_played, episode_avg_speeds, episode_avg_tau_i, episode_avg_qualities = [], [], [], [], []

    results_dir = 'models'
    results_prefix, models_prefix = "Combined_ppo", "Combined_ppo"

    print("--- 훈련 시작 ---")
    start_time_train = time.time()

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward, episode_step, done = 0, 0, False
        current_episode_taus = []

        for step in range(NUM_STEPS):
            current_episode_taus.append(state[0])
            action, log_prob_q, log_prob_s, log_prob_sk, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done, log_prob_q, log_prob_s, log_prob_sk, value)
            
            state = next_state
            episode_reward += reward
            episode_step += 1
            if done: break

        if len(agent.states) > 0:
            agent.update()
            agent.clear_memory()

        # --- 에피소드 결과 로깅 ---
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        episode_avg_capacities.append(env.get_average_capacity())
        episode_stall_times.append(env.get_total_stall_time())
        skipped, played_count = env.get_skip_statistics()
        episode_skipped.append(skipped)
        episode_played.append(played_count)
        episode_avg_speeds.append(env.get_average_speed())
        episode_avg_qualities.append(env.get_average_quality())
        episode_avg_tau_i.append(np.mean(current_episode_taus) if current_episode_taus else 0)

        if (episode + 1) % 100 == 0:
            recent_tau = episode_avg_tau_i[-100:]
            avg_tau = np.mean(recent_tau) if recent_tau else 0
            print(f"에피소드 {episode+1}/{MAX_EPISODES} | 평균 보상: {np.mean(episode_rewards[-100:]):.2f} | "
                  f"평균 채널: {np.mean(episode_avg_capacities[-100:]):.2f}Mbps | 총 스톨: {np.mean(episode_stall_times[-100:]):.2f}s | "
                  f"스킵/재생: {np.mean(episode_skipped[-100:]):.1f}/{np.mean(episode_played[-100:]):.1f} | "
                  f"평균 속도: {np.mean(episode_avg_speeds[-100:]):.2f}x | 평균 지연(τ): {avg_tau:.2f}s | "
                  f"평균 품질: {np.mean(episode_avg_qualities[-100:]):.2f}Mbps")
            
            # 결과 저장
            np.save(f'results/{results_prefix}_rewards.npy', np.array(episode_rewards))
            # 다른 결과들도 필요하다면 여기에 저장

        if (episode + 1) % 5000 == 0:
            agent.save_models(episode + 1)

    agent.save_models('final')

    # [추가] 훈련 종료 후 최종 그래프 그리기
    plot_learning_curve(episode_rewards, save_dir=results_dir)

# #################################
# 메인 실행 부분
# #################################
if __name__ == "__main__":
    train_combined()
    print(f"--- 훈련 종료 ---")



