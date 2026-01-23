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

class DynamicChannelModel:
    """채널 모델 변동 없음"""
    def __init__(self, W=1e6, gamma_dB=35, distance=20, beta=2, sigma_dB=4, mu_dB=0):
        self.W, self.gamma, self.distance, self.beta, self.sigma_dB, self.mu_dB = W, 10**(gamma_dB/10), distance, beta, sigma_dB, mu_dB
        self.current_channel_gain, self.current_capacity, self.channel_components = None, None, None

    def generate_channel_state(self):
        path_loss = self.distance ** self.beta
        shadowing = 10 ** (np.random.normal(self.mu_dB, self.sigma_dB) / 10)
        fast_fading = np.abs(np.random.normal(0, 1/np.sqrt(2)) + 1j * np.random.normal(0, 1/np.sqrt(2))) ** 2
        self.current_channel_gain = np.sqrt((1 / path_loss) * shadowing * fast_fading)

    def calculate_capacity(self, regenerate_channel=True):
        if regenerate_channel or self.current_channel_gain is None: self.generate_channel_state()
        capacity_bps = self.W * np.log2(1 + self.gamma * (self.current_channel_gain ** 2))
        self.current_capacity = capacity_bps / 8
        return self.current_capacity
    
    def get_channel_details(self):
        if self.current_capacity is None: self.calculate_capacity()
        return {'capacity_Mbps': (self.current_capacity * 8) / 1e6, 'channel_gain': self.current_channel_gain, 'components': self.channel_components}


class VideoStreamingEnv_TCLiVi:
    """환경 클래스 코드는 이전 수정과 완전히 동일하며, 완전성을 위해 이곳에 유지함"""
    def __init__(self, num_chunks=300, t_buffer=4, t_i=2,
                 W1=25.0, W2=300.0, W3=100.0, W_tau=1.0,
                 history_len=48, verbose=False, snr_db=35.0):
        self.num_chunks, self.t_buffer, self.t_i = num_chunks, t_buffer, t_i
        self.quality_levels = [1, 2, 3, 4]
        self.bitrate_map = {1: 0.8, 2: 1.2, 3: 2.4, 4: 4.8}
        self.verbose = verbose
        self.W1, self.W2, self.W3, self.W_tau = W1, W2, W3, W_tau
        self.history_len = history_len
        self.num_history_features = 6
        self.chunk_history = deque(maxlen=self.history_len)
        self.action_map = {0: (1, 2), 1: (2, 2), 2: (3, 2), 3: (4, 2), 4: (1, 3), 5: (2, 3), 6: (3, 3), 7: (4, 3)}
        self.playback_speeds = {'slow': 0.9, 'normal': 1.0, 'fast': 1.1}
        self.speed_rule_margin = 1.0
        self.channel_model = DynamicChannelModel(gamma_dB=snr_db)

        # 수정 사항
        # 이전 비트레이트 추적을 위한 변수 초기화
        self.last_bitrate = None
        
        self.reset()

    def reset(self):
        self.interval_capacities, self.actual_completion_times, self.chunk_actions = {}, {}, {}
        self.generated_chunk_times = {i: (i + 1) * self.t_i for i in range(self.num_chunks)}
        self.u_i, self.p_i, self.d_i, self.k_j, self.l_i, self.speed_j = [], [], [], [], [], []
        self.chunk_history.clear()
        for _ in range(self.history_len):
            self.chunk_history.append(np.zeros(self.num_history_features, dtype=np.float32))
        self.state = np.array(self.chunk_history, dtype=np.float32)

        # 수정 사항
        self.last_bitrate = 0

        return self.state
    
    def _calculate_live_latency(self, calculation_time_ui, decision_chunk_idx):
        currently_playing_chunk_idx, last_played_chunk_info = -1, None
        for j in range(decision_chunk_idx - 1, -1, -1):
            action_info = self.chunk_actions.get(j)
            if action_info and j < len(self.p_i) and self.p_i[j] <= calculation_time_ui:
                currently_playing_chunk_idx, last_played_chunk_info = j, action_info; break
        if currently_playing_chunk_idx == -1: return self.t_buffer
        p_n_i = self.p_i[currently_playing_chunk_idx]
        sum_of_prev_play_durations = sum(self.t_i / info.get('speed', 1.0) for j, info in self.chunk_actions.items() if j < currently_playing_chunk_idx)
        historical_lag = p_n_i - sum_of_prev_play_durations
        current_chunk_speed, current_chunk_play_duration = last_played_chunk_info.get('speed', 1.0), self.t_i / last_played_chunk_info.get('speed', 1.0)
        current_chunk_theoretical_end_time = p_n_i + current_chunk_play_duration
        current_stall_lag = max(0, calculation_time_ui - current_chunk_theoretical_end_time)
        return historical_lag + current_stall_lag
    
    def calculate_remaining_play_time(self, calculation_time, decision_chunk_idx):
        remaining_current, future_buffered, playing_idx = 0.0, 0.0, -1
        for j in range(decision_chunk_idx):
            action_info = self.chunk_actions.get(j)
            if action_info and j < len(self.p_i) and self.p_i[j] != float('inf'):
                play_start_time, chunk_speed = self.p_i[j], action_info.get('speed', 1.0)
                play_end_time = play_start_time + self.t_i / chunk_speed
                if play_start_time <= calculation_time < play_end_time:
                    remaining_current, playing_idx = max(0, play_end_time - calculation_time), j; break
        for j in range(playing_idx + 1, decision_chunk_idx):
            action_info = self.chunk_actions.get(j)
            if self.actual_completion_times.get(j, float('inf')) <= calculation_time and (self.p_i[j] if j < len(self.p_i) else float('inf')) > calculation_time:
                 future_buffered += self.t_i / action_info.get('speed', 1.0)
        return remaining_current + future_buffered
    
    def _get_capacity_for_interval(self, i):
        if i < 0: return 0
        return self.interval_capacities.setdefault(i, self.channel_model.calculate_capacity(True))
    
    def _simulate_chunk_transmission(self, chunk_idx, k_i, start_time):
        size_to_transmit = (self.bitrate_map[k_i] * 1_000_000 * self.t_i) / 8
        if size_to_transmit <= 1e-9: return 0.0, start_time
        transmitted_data, current_time, total_tx_time = 0.0, start_time, 0.0
        max_sim_time = start_time + 10 * self.num_chunks * self.t_i
        while transmitted_data < size_to_transmit and current_time < max_sim_time:
            interval_idx = int(current_time // self.t_i)
            capacity, interval_end_time = self._get_capacity_for_interval(interval_idx), (interval_idx + 1) * self.t_i
            time_in_interval = interval_end_time - current_time
            if capacity <= 1e-9:
                total_tx_time += time_in_interval + 1e-9; current_time += time_in_interval + 1e-9; continue
            time_needed = (size_to_transmit - transmitted_data) / capacity
            time_this_interval = min(time_in_interval, time_needed)
            transmitted_data += capacity * time_this_interval; total_tx_time += time_this_interval; current_time += time_this_interval
        if current_time >= max_sim_time: return float('inf'), float('inf')
        return total_tx_time, start_time + total_tx_time
    
    def step(self, action_index):
        k_i, target_buffer = self.action_map[action_index]; x_i = 0; chunk_idx = len(self.u_i)
        if chunk_idx >= self.num_chunks: return self.reset(), 0.0, True, {}
        chunk_generation_time = self.generated_chunk_times.get(chunk_idx, (chunk_idx + 1) * self.t_i)
        prev_completion_time = self.actual_completion_times.get(chunk_idx - 1, 0.0) if chunk_idx > 0 else 0.0
        u_i = max(chunk_generation_time, prev_completion_time)
        if self.u_i: u_i = max(u_i, self.u_i[-1] + 1e-9)
        self.u_i.append(u_i)

        # 수정
        # Switching Penalty 계산
        current_bitrate = self.bitrate_map[k_i]

        if self.last_bitrate == 0:
            switching_diff = 0.0
        else:
            switching_diff = abs(current_bitrate - self.last_bitrate)
        
        self.last_bitrate = current_bitrate

        tau_at_decision = self._calculate_live_latency(u_i, chunk_idx)
        buffer_at_decision = self.calculate_remaining_play_time(u_i, chunk_idx)

        # 수정
        # 버퍼가 7초보다 크다면, 강제로 스킵하는 기능이 존재해야 함
        if buffer_at_decision > 7.0:
            skip_duration = buffer_at_decision - target_buffer

            buffer_at_decision = target_buffer
            tau_at_decision = max(0.0, tau_at_decision - skip_duration)

        slow_play_threshold, fast_play_threshold = target_buffer - self.speed_rule_margin, target_buffer + self.speed_rule_margin
        current_playback_speed = self.playback_speeds['normal']
        if buffer_at_decision < slow_play_threshold: current_playback_speed = self.playback_speeds['slow']
        elif buffer_at_decision > fast_play_threshold: current_playback_speed = self.playback_speeds['fast']
        self.speed_j.append(current_playback_speed)
        self.k_j.append(k_i); self.chunk_actions[chunk_idx] = {'k': k_i, 'u': u_i, 'speed': current_playback_speed}
        l_i, completion_time = self._simulate_chunk_transmission(chunk_idx, k_i, u_i)
        self.l_i.append(l_i); self.actual_completion_times[chunk_idx] = completion_time
        p_i, d_i = float('inf'), 0.0
        if l_i != float('inf'):
            ready_time = completion_time; prev_play_end_time = self.t_buffer
            for j in range(chunk_idx - 1, -1, -1):
                prev_action_info = self.chunk_actions.get(j)
                if prev_action_info and j < len(self.p_i) and self.p_i[j] != float('inf'):
                    prev_play_end_time = self.p_i[j] + self.t_i / prev_action_info.get('speed', 1.0); break
            p_i = max(ready_time, prev_play_end_time); d_i = max(0, ready_time - prev_play_end_time)
        self.p_i.append(p_i); self.d_i.append(d_i)

        tau_for_reward = tau_at_decision + max(0, l_i - buffer_at_decision)
        base_reward = self._get_reward(k_i, d_i, tau_for_reward, switching_diff)
        history_entry = np.array([l_i/self.t_i, ((self.bitrate_map[k_i]*1e6*self.t_i)/8)/1e6, d_i/self.t_i, buffer_at_decision/self.t_buffer, tau_at_decision/self.t_buffer, k_i/len(self.quality_levels)], dtype=np.float32)
        self.chunk_history.append(history_entry)
        next_state = np.array(self.chunk_history, dtype=np.float32)
        done = chunk_idx >= self.num_chunks - 1

        return next_state, base_reward, done, {}
    
    def _get_reward(self, bitrate, stall_time, latency, last_bitrate):
        """
        논문 [12] TCLiVi Eq. (6) 기반 수정
        r_n = alpha * q_n - beta * T_n - gamma * D_n - eta * |q_n - q_{n-1}|
        """
        # 1. 하이퍼파라미터 재설정 (Mbps 환경에 맞춰 튜닝된 비율)
        # 화질(q)이 0.8~4.8 수준이므로, 지연(latency) 페널티와 균형을 맞춤
        alpha = 1.0    # 화질 가중치 (기존 제곱일 때보다 낮게 잡아도 됨)
        beta = 25.0   # Stall 페널티 (초당 페널티)
        gamma = 0.1   # Latency 페널티 (선형이므로 기존보다 임팩트가 작아짐, 1.0 유지)
        eta = 0.02    # 스무딩 페널티 (화질 급변 방지)

        # 2. 보상 계산 (모두 선형 Linear 사용)
        # q_n: 비트레이트 그 자체 (Mbps)
        reward_quality = alpha * bitrate
        
        # T_n: Stall 시간 (초)
        reward_stall = beta * stall_time
        
        # D_n: Latency (초) -> 논문의 핵심! 제곱하지 않음!
        reward_latency = gamma * latency
        
        # S_n: 스무딩 (현재 비트레이트 - 이전 비트레이트 절대값)
        smoothness_penalty = eta * abs(bitrate - last_bitrate)

        # 최종 보상 합산
        reward = reward_quality - reward_stall - reward_latency - smoothness_penalty
        
        return reward
    
    def get_average_quality(self):
        played_qualities = [self.bitrate_map[k] for i, k in enumerate(self.k_j) if i < len(self.p_i) and self.p_i[i] != float('inf')]
        return np.mean(played_qualities) if played_qualities else 0
    
    def get_total_stall_time(self): return sum(d for d in self.d_i if d is not None and d != float('inf'))

    def get_average_capacity(self):
        caps = self.interval_capacities.values(); return (np.mean(list(caps)) * 8 / 1e6) if caps else 0
    
    def get_average_speed(self): return np.mean(self.speed_j) if self.speed_j else 1.0


#################################
# PPO 알고리즘 부분 - 수정 불필요
#################################
LEARNING_RATE, GAMMA, LAMBDA, EPOCHS, BATCH_SIZE, CLIP_EPSILON, NUM_STEPS, MAX_EPISODES, ENTROPY_COEF = 1e-4, 0.96, 0.97, 10, 128, 0.2, 1024, 50000, 0.01
HISTORY_LEN, NUM_FEATURES, ACTION_DIM = 48, 6, 8

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
        torch.save(self.actor.state_dict(), f'baseline_models/b5/{prefix}_actor_ep{episode}.pth')
        torch.save(self.critic.state_dict(), f'baseline_models/b5/{prefix}_critic_ep{episode}.pth')
        print(f"TCLiVi 모델 (규칙 기반 속도)이 에피소드 {episode}에 저장되었습니다")

#################################
# 훈련 함수 - 이전 수정과 일치
#################################
def train_tclivi():
    env = VideoStreamingEnv_TCLiVi(num_chunks=300, history_len=HISTORY_LEN)
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

