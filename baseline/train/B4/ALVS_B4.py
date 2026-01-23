#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import sys

# 전역 장치 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 환경 및 채널 모델 코드 (변경 없음) ---
class DynamicChannelModel:
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

from collections import deque

class VideoStreamingEnvCombined:
    # --- MODIFIED: 새로운 매개변수 및 기록 추가 ---
    def __init__(self, num_chunks=300, t_buffer=4, t_i=2, 
                 W1=2.5, W2=1.5, W3=0.1, W4=0.005, # ALVS reward weights: c0, c1, c3, c2
                 history_len=8, verbose=False, snr_db=35.0):
        
        # 핵심 매개변수
        self.num_chunks, self.t_buffer, self.t_i = num_chunks, t_buffer, t_i
        
        # 원본 논문의 비트레이트 및 화질 레벨 매핑
        self.quality_levels = [1, 2, 3, 4]
        self.bitrate_map = {1: 0.8, 2: 1.2, 3: 2.4, 4: 4.8} # Mbps
        
        # ALVS 동작 공간
        # ALVS 논문은 6개 화질과 3개 속도를 사용하지만, 여기서는 호환성을 위해 
        # 4개 화질을 유지하고 속도는 ALVS의 3개를 사용합니다.
        # 완전히 일치시키려면 self.bitrate_map을 수정하십시오.
        self.playback_speeds = [0.9, 1.0, 1.1]
        
        # 채널 모델
        self.channel_model = DynamicChannelModel(gamma_dB=snr_db)
        
        # 보상 가중치 (W1=c0, W2=c1, W3=c3, W4=c2)
        self.W1, self.W2, self.W3, self.W4 = W1, W2, W3, W4
        self.verbose = verbose
        
        # ALVS 상태 기록
        self.history_len = history_len
        self.reset()

    # --- MODIFIED: reset 메서드는 모든 기록을 초기화해야 함 ---
    def reset(self):
        # 시뮬레이션 상태 변수
        self.interval_capacities, self.actual_completion_times, self.chunk_actions = {}, {}, {}
        self.generated_chunk_times = {i: (i + 1) * self.t_i for i in range(self.num_chunks)}
        self.u_i, self.p_i, self.d_i, self.k_j, self.speed_idx_j, self.l_i, self.x_j = [], [], [], [], [], [], []
        self.current_chunk_idx = 0
        
        # ALVS 고유 기록 상태
        self.last_quality = 1 # 초기 화질
        self.throughput_history = deque([0] * self.history_len, maxlen=self.history_len)
        self.download_time_history = deque([0] * self.history_len, maxlen=self.history_len)
        
        # 초기 상태 가져오기
        return self.get_state_for_decision(0)

    # _calculate_live_latency 및 calculate_remaining_play_time은 수정 불필요
    def _calculate_live_latency(self, calculation_time_ui, decision_chunk_idx):
        currently_playing_chunk_idx, last_played_chunk_info = -1, None
        for j in range(decision_chunk_idx - 1, -1, -1):
            info = self.chunk_actions.get(j)
            if info and info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] <= calculation_time_ui:
                currently_playing_chunk_idx, last_played_chunk_info = j, info; break
        if currently_playing_chunk_idx == -1: return self.t_buffer
        p_n_i = self.p_i[currently_playing_chunk_idx]
        sum_of_prev_play_durations = sum(self.t_i / info.get('speed', 1.0) for j, info in self.chunk_actions.items() if j < currently_playing_chunk_idx and info.get('x', 0) == 0)
        historical_lag = p_n_i - sum_of_prev_play_durations
        current_chunk_speed, current_chunk_play_duration = last_played_chunk_info.get('speed', 1.0), self.t_i / last_played_chunk_info.get('speed', 1.0)
        current_chunk_theoretical_end_time = p_n_i + current_chunk_play_duration
        current_stall_lag = max(0, calculation_time_ui - current_chunk_theoretical_end_time)
        return historical_lag + current_stall_lag
        
    def calculate_remaining_play_time(self, calculation_time, decision_chunk_idx):
        remaining_current, future_buffered, playing_idx = 0.0, 0.0, -1
        for j in range(decision_chunk_idx):
            info = self.chunk_actions.get(j)
            if info and info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                start, speed = self.p_i[j], info.get('speed', 1.0)
                end = start + self.t_i / speed
                if start <= calculation_time < end:
                    remaining_current, playing_idx = max(0, end - calculation_time), j; break
        for j in range(playing_idx + 1, decision_chunk_idx):
            info = self.chunk_actions.get(j)
            if info and info.get('x', 0) == 0 and self.actual_completion_times.get(j, float('inf')) <= calculation_time:
                future_buffered += self.t_i / info.get('speed', 1.0)
        return remaining_current + future_buffered

    # _get_capacity_for_interval 및 _simulate_chunk_transmission은 수정 불필요
    def _get_capacity_for_interval(self, interval_index):
        if interval_index < 0: return 0
        return self.interval_capacities.setdefault(interval_index, self.channel_model.calculate_capacity(True))

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
            transmitted_data += capacity * time_this_interval
            total_tx_time += time_this_interval; current_time += time_this_interval
        if current_time >= max_sim_time: return float('inf'), float('inf')
        return total_tx_time, start_time + total_tx_time

    # --- MODIFIED: step 메서드는 기록을 업데이트해야 함 ---
    def step(self, action):
        k_i, speed_idx, x_i = action # x_i는 무시됨 (ALVS는 건너뛰기 수행 안 함)
        chunk_idx = self.current_chunk_idx
        
        if chunk_idx >= self.num_chunks: return self.reset(), 0.0, True, {}
        
        # --- ALVS does not skip ---
        x_i = 0
        
        chunk_gen_time = self.generated_chunk_times.get(chunk_idx, (chunk_idx + 1) * self.t_i)
        prev_comp_time = self.actual_completion_times.get(chunk_idx - 1, 0.0) if chunk_idx > 0 else 0.0
        u_i = max(chunk_gen_time, prev_comp_time)
        if self.u_i: u_i = max(u_i, self.u_i[-1] + 1e-9)
        self.u_i.append(u_i)
        
        self.k_j.append(k_i); self.speed_idx_j.append(speed_idx); self.x_j.append(x_i)
        self.chunk_actions[chunk_idx] = {'k': k_i, 'speed': self.playback_speeds[speed_idx], 'x': x_i, 'u': u_i}
        
        l_i, comp_time = self._simulate_chunk_transmission(chunk_idx, k_i, u_i)
        self.l_i.append(l_i); self.actual_completion_times[chunk_idx] = comp_time
        
        # --- 기록 업데이트 ---
        chunk_size_bits = self.bitrate_map[k_i] * 1_000_000 * self.t_i
        throughput = chunk_size_bits / (l_i + 1e-9) # 0으로 나누기 방지를 위한 작은 값 추가
        self.throughput_history.append(throughput)
        self.download_time_history.append(l_i)
        
        # 재생 시점 및 끊김(Stall) 계산
        prev_play_end_time = self.t_buffer
        for j in range(chunk_idx - 1, -1, -1):
            info = self.chunk_actions.get(j)
            if info and info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                prev_play_end_time = self.p_i[j] + self.t_i / info.get('speed', 1.0); break

        p_i = max(comp_time, prev_play_end_time)
        d_i = max(0, comp_time - prev_play_end_time)
        
        while len(self.p_i) <= chunk_idx: self.p_i.append(float('inf'))
        self.p_i[chunk_idx] = p_i
        while len(self.d_i) <= chunk_idx: self.d_i.append(0.0)
        self.d_i[chunk_idx] = d_i
        
        # 보상 계산
        reward = self._get_reward(k_i, d_i, p_i - chunk_gen_time)
        
        # 이전 화질 선택 업데이트
        self.last_quality = k_i
        
        self.current_chunk_idx += 1
        next_state = self.get_state_for_decision(self.current_chunk_idx)
        done = self.current_chunk_idx >= self.num_chunks
        return next_state, reward, done, {}

    # --- NEW: ALVS에 맞춰 상태 가져오기 함수 완전 재작성 ---
    def get_state_for_decision(self, target_chunk_idx):
        if target_chunk_idx >= self.num_chunks:
            # 종료되면 0으로 채운 상태 반환
            return {
                'throughput_history': np.zeros(self.history_len, dtype=np.float32),
                'download_time_history': np.zeros(self.history_len, dtype=np.float32),
                'next_chunk_sizes': np.zeros(len(self.quality_levels), dtype=np.float32),
                'other_features': np.zeros(4, dtype=np.float32) # buffer, latency, quality, remaining
            }

        # 1. & 2. 처리량 및 다운로드 시간 기록 (8개)
        throughput_hist_norm = np.array(self.throughput_history) / 10_000_000 # 정규화
        download_time_hist_norm = np.array(self.download_time_history) / 10.0 # 정규화

        # 3. 다음 청크 크기 (4개)
        next_chunk_sizes = [ (self.bitrate_map[q] * 1_000_000 * self.t_i) / 8 for q in self.quality_levels]
        next_chunk_sizes_norm = np.array(next_chunk_sizes) / 1_000_000 # 정규화

        # 현재 시각 계산
        chunk_gen_time = self.generated_chunk_times.get(target_chunk_idx, (target_chunk_idx + 1) * self.t_i)
        prev_comp = self.actual_completion_times.get(target_chunk_idx - 1, 0.0)
        calc_time_ui = max(chunk_gen_time, prev_comp) if target_chunk_idx > 0 else chunk_gen_time
        if self.u_i: calc_time_ui = max(calc_time_ui, self.u_i[-1] + 1e-9)

        # 4. 버퍼 크기
        remaining_buffer = self.calculate_remaining_play_time(calc_time_ui, target_chunk_idx)
        
        # 5. 라이브 지연 시간
        current_tau = self._calculate_live_latency(calc_time_ui, target_chunk_idx)

        # 6. 현재 재생 비트레이트 (이전 선택 비트레이트 사용)
        last_bitrate = self.bitrate_map[self.last_quality]

        # 7. 남은 청크 수
        chunks_remaining = self.num_chunks - target_chunk_idx
        
        other_features = np.array([
            remaining_buffer / 10.0, # 정규화
            current_tau / 15.0, # 정규화
            last_bitrate / 5.0, # 정규화
            chunks_remaining / self.num_chunks # 정규화
        ], dtype=np.float32)
        
        return {
            'throughput_history': throughput_hist_norm.astype(np.float32),
            'download_time_history': download_time_hist_norm.astype(np.float32),
            'next_chunk_sizes': next_chunk_sizes_norm.astype(np.float32),
            'other_features': other_features
        }

    # --- MODIFIED: ALVS Eq. (8)에 맞춰 보상 함수 재작성 ---
    def _get_reward(self, k_i, d_i, live_latency_i):
        # r_t = c0*VQ_t - c1*stall - c2*|VQ_t - VQ_{t-1}| - c3*latency
        current_bitrate = self.bitrate_map[k_i]
        last_bitrate = self.bitrate_map[self.last_quality]
        
        quality_reward = self.W1 * current_bitrate
        stall_penalty = self.W2 * d_i
        latency_penalty = self.W3 * live_latency_i
        smoothness_penalty = self.W4 * abs(current_bitrate - last_bitrate)
        
        return quality_reward - stall_penalty - latency_penalty - smoothness_penalty

    # get_... 보조 함수 수정 불필요
    def get_average_quality(self):
        played_q = [self.bitrate_map[k] for i,k in enumerate(self.k_j) if self.x_j[i]==0 and i < len(self.p_i) and self.p_i[i]!=float('inf')]
        return np.mean(played_q) if played_q else 0
    def get_total_stall_time(self): return sum(d for d in self.d_i if np.isfinite(d))
    def get_skip_statistics(self): return sum(self.x_j), len(self.x_j) - sum(self.x_j)
    def get_average_speed(self):
        speeds = [self.playback_speeds[s] for i,s in enumerate(self.speed_idx_j) if self.x_j[i]==0]
        
        return np.mean(speeds) if speeds else 1.0

# --- 강화 학습 부분 (A3C 알고리즘 - ALVS 버전) ---

# --- NEW: 새로운 상태 차원 및 초매개변수 정의 ---
HISTORY_LEN = 8
QUALITY_LEVELS = 4 # 환경 설정과 일치
SPEED_LEVELS = 3
# State is a dictionary now, so STATE_DIM is not a single number

MAX_GLOBAL_EPISODES = 15000
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99 # ALVS uses 0.99
LEARNING_RATE = 1e-4 # ALVS uses 1e-3 for actor, 1e-4 for critic
ENTROPY_BETA = 0.01 # ALVS uses an entropy weight from 4 down to 0.2

# --- NEW: ALVS의 CNN+MLP 네트워크 구조 정의 ---
class ActorNetwork_ALVS(nn.Module):
    def __init__(self):
        super(ActorNetwork_ALVS, self).__init__()
        # CNN for throughput, download time, and chunk sizes
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=128, kernel_size=4),
            nn.ReLU()
        )
        # Flattened CNN output will be concatenated with other features
        # CNN output shape: (batch, 128, 5) -> flatten -> (batch, 128*5=640)
        # Other features: 4 (buffer, latency, quality, remaining)
        self.mlp = nn.Sequential(
            nn.Linear(640 + 4, 128),
            nn.ReLU()
        )
        self.quality_head = nn.Linear(128, QUALITY_LEVELS)
        self.speed_head = nn.Linear(128, SPEED_LEVELS)

    def forward(self, state_dict):
        # CNN 입력 준비
        # shape: (batch, history_len) -> (batch, 1, history_len)
        throughput = state_dict['throughput_history'].unsqueeze(1)
        download_time = state_dict['download_time_history'].unsqueeze(1)
        
        # next_chunk_sizes를 기록 데이터 길이와 동일하게 확장
        # shape: (batch, num_qualities) -> (batch, 1, num_qualities)
        chunk_sizes_unsqueezed = state_dict['next_chunk_sizes'].unsqueeze(1)
        # Pad to history_len
        padding_size = HISTORY_LEN - chunk_sizes_unsqueezed.size(2)
        chunk_sizes = nn.functional.pad(chunk_sizes_unsqueezed, (0, padding_size))

        # CNN 입력으로 병합: (batch, 3, history_len)
        cnn_input = torch.cat([throughput, download_time, chunk_sizes], dim=1)
        
        cnn_output = self.cnn(cnn_input)
        cnn_output_flat = cnn_output.view(cnn_output.size(0), -1)
        
        # MLP 입력 준비
        other_features = state_dict['other_features']
        
        # CNN 출력과 기타 특징 병합
        combined_features = torch.cat([cnn_output_flat, other_features], dim=1)
        
        # MLP 및 출력 헤드 통과
        base = self.mlp(combined_features)
        return self.quality_head(base), self.speed_head(base)

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        # Actor와 동일한 특징 추출기 사용
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=128, kernel_size=4),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(640 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 가치 값 출력
        )

    def forward(self, state_dict):
        # Actor의 특징 추출 로직 재사용
        throughput = state_dict['throughput_history'].unsqueeze(1)
        download_time = state_dict['download_time_history'].unsqueeze(1)
        chunk_sizes_unsqueezed = state_dict['next_chunk_sizes'].unsqueeze(1)
        padding_size = HISTORY_LEN - chunk_sizes_unsqueezed.size(2)
        chunk_sizes = nn.functional.pad(chunk_sizes_unsqueezed, (0, padding_size))

        cnn_input = torch.cat([throughput, download_time, chunk_sizes], dim=1)
        cnn_output = self.cnn(cnn_input)
        cnn_output_flat = cnn_output.view(cnn_output.size(0), -1)
        other_features = state_dict['other_features']
        combined_features = torch.cat([cnn_output_flat, other_features], dim=1)
        
        return self.mlp(combined_features)

class Worker(mp.Process):
    def __init__(self, g_actor, g_critic, g_optimizer, g_ep, worker_id, logging_queue, optimizer_lock):
        super(Worker, self).__init__()
        self.g_actor, self.g_critic, self.g_optimizer = g_actor, g_critic, g_optimizer
        self.g_ep, self.worker_id, self.logging_queue = g_ep, worker_id, logging_queue
        self.optimizer_lock = optimizer_lock
        
        # --- MODIFIED: 환경 초기화 시 ALVS 보상 가중치 전달 ---
        self.env = VideoStreamingEnvCombined(W1=1.0, W2=1.5, W3=0.02, W4=0.005)
        self.l_actor, self.l_critic = ActorNetwork_ALVS(), CriticNetwork()

    def run(self):
        try:
            self.l_actor.to(device)
            self.l_critic.to(device)
            total_step = 1
            
            while self.g_ep.value < MAX_GLOBAL_EPISODES:
                state = self.env.reset()
                # 버퍼는 이제 상태 사전을 저장함
                buffer_s, buffer_a_q, buffer_a_s, buffer_r = [], [], [], []
                ep_r, ep_taus = 0., []
                
                # 전역 네트워크에서 로컬 네트워크로 매개변수 동기화
                self.l_actor.load_state_dict(self.g_actor.state_dict())
                self.l_critic.load_state_dict(self.g_critic.state_dict())

                for t in range(self.env.num_chunks):
                    # 기록을 위해 상태 사전에서 지연 시간 추출
                    ep_taus.append(state['other_features'][1] * 15.0) # 역정규화
                    
                    # 상태 사전을 Tensor 사전으로 변환
                    state_tensor_dict = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in state.items()}
                    
                    with torch.no_grad():
                        q_logits, s_logits = self.l_actor(state_tensor_dict)
                    
                    q_probs, s_probs = torch.softmax(q_logits, -1), torch.softmax(s_logits, -1)
                    q_action, s_action = Categorical(q_probs).sample().item(), Categorical(s_probs).sample().item()
                    action_for_env = (q_action + 1, s_action, 0) # ALVS는 건너뛰기 안 함
                    
                    next_state, reward, done, _ = self.env.step(action_for_env)
                    ep_r += reward
                    buffer_s.append(state); buffer_a_q.append(q_action); buffer_a_s.append(s_action); buffer_r.append(reward)
                    
                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                        self.push_and_pull(done, next_state, buffer_s, buffer_a_q, buffer_a_s, buffer_r)
                        buffer_s, buffer_a_q, buffer_a_s, buffer_r = [], [], [], []
                        
                        # 최신 전역 정책 사용을 위해 매개변수 재동기화
                        self.l_actor.load_state_dict(self.g_actor.state_dict())
                        self.l_critic.load_state_dict(self.g_critic.state_dict())
                    
                    state = next_state
                    total_step += 1
                    
                    if done:
                        # 전역 에피소드 카운터를 안전하게 증가시키기 위해 잠금(Lock) 사용
                        with self.g_ep.get_lock():
                            self.g_ep.value += 1
                        
                        # 로그 데이터 준비 및 전송
                        log_data = {
                            "ep": self.g_ep.value, 
                            "reward": ep_r, 
                            "steps": t + 1, 
                            "stall": self.env.get_total_stall_time(), 
                            "skipped": self.env.get_skip_statistics()[0], 
                            "played": self.env.get_skip_statistics()[1], 
                            "speed": self.env.get_average_speed(), 
                            "quality": self.env.get_average_quality(), 
                            "tau": np.mean(ep_taus) if ep_taus else 0
                        }
                        self.logging_queue.put(log_data)
                        break # 현재 에피소드 루프 종료

        except Exception as e:
            # Worker 프로세스 실행 중 포착되지 않은 예외 발생 시
            print(f"!!! Worker {self.worker_id} CRASHED with error: {e}")
            import traceback
            traceback.print_exc() # 디버깅을 위해 상세 오류 스택 정보 출력

        finally:
            # Worker가 정상 종료되든 비정상 중단되든 이 블록은 반드시 실행
            # 이를 통해 logger 프로세스가 Worker 종료를 인지함
            self.logging_queue.put(None)

    def push_and_pull(self, done, next_state, buffer_s, buffer_a_q, buffer_a_s, buffer_r):
        # 1단계: 목표 V 값 계산 (이 부분은 잠금 외부에서 수행, 정상)
        with torch.no_grad():
            if done:
                R = 0.0
            else:
                next_state_tensor_dict = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in next_state.items()}
                R = self.l_critic(next_state_tensor_dict).item()
    
        buffer_v_target = []
        for r in buffer_r[::-1]:
            R = r + GAMMA * R
            buffer_v_target.append(R)
        buffer_v_target.reverse()

        # 2단계: Tensor 준비 (이 부분도 잠금 외부에서 수행, 정상)
        states_dict = {key: torch.tensor(np.array([d[key] for d in buffer_s]), dtype=torch.float32).to(device) for key in buffer_s[0]}
        actions_q = torch.LongTensor(buffer_a_q).to(device)
        actions_s = torch.LongTensor(buffer_a_s).to(device)
        v_targets = torch.FloatTensor(buffer_v_target).to(device)

        # 3단계: 로컬 모델에서 손실 및 기울기 계산 (!!! 핵심 수정: 잠금 외부에서 실행 !!!)
        values = self.l_critic(states_dict).squeeze()
        advantage = (v_targets - values).detach()

        q_logits, s_logits = self.l_actor(states_dict)
    
        q_dist = Categorical(torch.softmax(q_logits, dim=-1))
        s_dist = Categorical(torch.softmax(s_logits, dim=-1))
    
        entropy = (q_dist.entropy() + s_dist.entropy()).mean()
        log_probs = q_dist.log_prob(actions_q) + s_dist.log_prob(actions_s)
    
        actor_loss = -(log_probs * advantage).mean() - ENTROPY_BETA * entropy
        critic_loss = nn.MSELoss()(values, v_targets)
        total_loss = actor_loss + critic_loss

        # 전역 모델에 적용하기 전, 로컬 모델의 기울기 초기화
        self.l_actor.zero_grad()
        self.l_critic.zero_grad()
        # 로컬 기울기 계산 (로컬 모델 업데이트 안 함)
        total_loss.backward()

        # 4단계: 잠금 획득 후 로컬 기울기를 전역 모델에 동기화하고 업데이트
        with self.optimizer_lock:
            # 로컬 기울기를 전역 모델로 복사
            for lp, gp in zip(self.l_actor.parameters(), self.g_actor.parameters()):
                gp._grad = lp.grad
            for lp, gp in zip(self.l_critic.parameters(), self.g_critic.parameters()):
                gp._grad = lp.grad
            
            # 전역 옵티마이저를 사용하여 전역 모델 업데이트
            self.g_optimizer.step()

def logger(logging_queue, g_actor, g_critic, total_workers):
    if not os.path.exists('results'): os.makedirs('results')
    if not os.path.exists('models'): os.makedirs('models')
    ep_rewards, ep_lengths, ep_stalls = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    ep_skipped, ep_played, ep_speeds = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    ep_qualities, ep_taus = deque(maxlen=100), deque(maxlen=100)
    start_time, terminated_workers = time.time(), 0
    while terminated_workers < total_workers:
        log_data = logging_queue.get()
        if log_data is None:
            terminated_workers += 1; continue
        ep_rewards.append(log_data["reward"]); ep_lengths.append(log_data["steps"])
        ep_stalls.append(log_data["stall"]); ep_skipped.append(log_data["skipped"])
        ep_played.append(log_data["played"]); ep_speeds.append(log_data["speed"])
        ep_qualities.append(log_data["quality"]); ep_taus.append(log_data["tau"])
        ep = log_data["ep"]
        if ep % 100 == 0 and ep > 0:
            print(f"Ep {ep}/{MAX_GLOBAL_EPISODES} | Avg R: {np.mean(ep_rewards):.2f} | Steps: {np.mean(ep_lengths):.1f} | "
                  f"Stall: {np.mean(ep_stalls):.2f} | Skip/Play: {np.mean(ep_skipped):.1f}/{np.mean(ep_played):.1f} | "
                  f"Speed: {np.mean(ep_speeds):.2f}x | Qual: {np.mean(ep_qualities):.2f} | "
                  f"Avg τ: {np.mean(ep_taus):.2f}s")
        if ep % 5000 == 0 and ep > 0:
            print(f"--- Saving models at ep {ep} ---")
            torch.save(g_actor.state_dict(), f'models/ALVS_A3C_actor_ep{ep}.pth')
            torch.save(g_critic.state_dict(), f'models/ALVS_A3C_critic_ep{ep}.pth')
    print(f"Logger finished. Time: {(time.time() - start_time)/60:.2f}m")

# =================================================================================
# === 테스트 함수 (최종 출력 형식 수정) ===
# =================================================================================
def test(actor_model_path):
    print(f"\n--- Starting Evaluation for model: {actor_model_path} ---")
    N_TEST_EPISODES = 100
    env = VideoStreamingEnvCombined(W1=1.0, W2=1.5, W3=0.02, W4=0.005)
    actor = ActorNetwork_ALVS().to(device)
    
    try:
        actor.load_state_dict(torch.load(actor_model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}"); return

    actor.eval()
    all_rewards, all_stalls, all_qualities, all_speeds, all_taus = [], [], [], [], []
    all_played, all_skipped, all_lengths = [], [], []

    for ep in range(N_TEST_EPISODES):
        state = env.reset()
        ep_reward, ep_taus, ep_steps = 0.0, [], 0
        done = False

        while not done:
            ep_taus.append(state['other_features'][1] * 15.0) # 역정규화
            # --- MODIFIED: 사전 입력 준비 ---
            state_tensor_dict = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in state.items()}

            with torch.no_grad():
                q_logits, s_logits = actor(state_tensor_dict)
                q_probs, s_probs = torch.softmax(q_logits, -1), torch.softmax(s_logits, -1)
                q_action, s_action = torch.argmax(q_probs).item(), torch.argmax(s_probs).item()
                action_for_env = (q_action + 1, s_action, 0)
            
            state, reward, done, _ = env.step(action_for_env)
            ep_reward += reward
            ep_steps += 1

        all_rewards.append(ep_reward)
        all_stalls.append(env.get_total_stall_time())
        all_qualities.append(env.get_average_quality())
        all_speeds.append(env.get_average_speed())
        all_taus.append(np.mean(ep_taus) if ep_taus else 0)
        skipped, played = env.get_skip_statistics()
        all_skipped.append(skipped)
        all_played.append(played)
        all_lengths.append(ep_steps)

        print(f"  Test Episode {ep+1}/{N_TEST_EPISODES} completed.", end='\r')

    print("\n\n--- Evaluation Summary ---")
    print(f"[TEST RESULT] Episodes: {N_TEST_EPISODES} | Avg R: {np.mean(all_rewards):.2f} | Steps: {np.mean(all_lengths):.1f} | "
          f"Stall: {np.mean(all_stalls):.2f} | Skip/Play: {np.mean(all_skipped):.1f}/{np.mean(all_played):.1f} | "
          f"Speed: {np.mean(all_speeds):.2f}x | Qual: {np.mean(all_qualities):.2f} | "
          f"Avg τ: {np.mean(all_taus):.2f}s")
    print("--- End of Evaluation ---")


def train(): 
    print(f"프로그램 실행 시작 (GPU-Enabled ALVS A3C) | 주 장치: {device}")
    start_time = time.time()
    g_actor, g_critic = ActorNetwork_ALVS().to(device), CriticNetwork().to(device)
    g_optimizer = optim.Adam(list(g_actor.parameters()) + list(g_critic.parameters()), lr=LEARNING_RATE)
    g_ep, log_queue = mp.Value('i', 0), mp.Queue()
    optimizer_lock = mp.Lock()
    num_workers = 3
    log_process = mp.Process(target=logger, args=(log_queue, g_actor, g_critic, num_workers)); log_process.start()
    print(f"Starting {num_workers} A3C workers for training...")
    workers = [Worker(g_actor, g_critic, g_optimizer, g_ep, i, log_queue, optimizer_lock) for i in range(num_workers)]
    [w.start() for w in workers]
    [w.join() for w in workers]
    log_process.join()
    print("--- 모든 Worker 훈련 완료 ---")
    torch.save(g_actor.state_dict(), 'models/ALVS_A3C_actor_final.pth')
    torch.save(g_critic.state_dict(), 'models/ALVS_A3C_critic_final.pth')
    print(f"총 실행 시간: {(time.time() - start_time)/60:.2f} 분")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- 수정된 테스트 프로세스 ---
    
    # 1. 시간 소모적인 재훈련을 방지하기 위해 훈련 부분 주석 처리 또는 삭제
    print("--- 훈련 시작 ---")
    train()
    print("--- 훈련 종료 ---")

    # 2. 테스트하려는 기존 모델 파일 경로 지정
    #    로그에 따르면 ep=5000 모델이 저장되었을 것임.
    # model_to_test_path = 'models/ALVS_A3C_actor_ep10000.pth'

    # 3. 모델 파일 존재 여부 확인 후 test 함수 직접 호출
    # print(f"\n--- 테스트 모드 준비 시작 ---")
    # if os.path.exists(model_to_test_path):
    #     print(f"모델 찾음: {model_to_test_path}")
    #     test(model_to_test_path)
    # else:
    #     print(f"오류: 지정된 모델 파일 '{model_to_test_path}'을 찾을 수 없습니다.")
    #     print("경로가 올바른지 확인하거나 먼저 훈련을 실행하여 모델을 생성하십시오.")