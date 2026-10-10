#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp

from collections import deque

# Proposed.py 로부터 요소들을 가져오기 위한 과정
current_dir = os.path.dirname(os.path.abspath(__file__))
proposed_dir = os.path.join(current_dir, '../proposed')
sys.path.append(proposed_dir)

try:
    from proposed import DynamicChannelModel
except ImportError as e:
    print("오류: proposed.py 파일을 찾거나 불러올 수 없습니다.")
    print("파일명을 확인하거나 구체적인 경로를 확인해주세요.")
    print(f"오류 메시지는 다음과 같습니다: {e}")
    exit()

# 전역 장치 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- ALVS 베이스라인 ---")
print(f"사용 장치: {device}")

class VideoStreamingEnv_ALVS:
    # --- MODIFIED: 새로운 매개변수 및 기록 추가 ---
    def __init__(self, num_chunks=300, t_buffer=4, t_i=2, 
                 W1=1, W2=300, W3=100, W_tau=1,
                 history_len=8, verbose=False, snr_db=35.0):
        
        # 핵심 매개변수
        self.num_chunks = num_chunks
        self.t_buffer = t_buffer
        self.t_i = t_i
        
        # DASH-Train 구조에 맞춰 시스템 구조 변경
        self.quality_levels = list(range(1, 21))
        self.max_quality = max(self.quality_levels)
        self.W1, self.W2, self.W3, self.W_tau = W1, W2, W3, W_tau
        
        self.playback_speeds = [0.9, 1.0, 1.1]
        self.num_speed_levels = len(self.playback_speeds)

        self.bitrate_map = {
            1: 0.045,   2: 0.089,   3: 0.131,   4: 0.178,   5: 0.221,
            6: 0.262,   7: 0.334,   8: 0.396,   9: 0.522,   10: 0.595,
            11: 0.791,  12: 1.032,  13: 1.244,  14: 1.546,  15: 2.133,
            16: 2.484,  17: 3.078,  18: 3.526,  19: 3.840,  20: 4.219
        }

        # 채널 모델
        self.channel_model = DynamicChannelModel()
        self.verbose = verbose
        
        # ALVS 상태 기록
        self.history_len = history_len
        self.reset()

    # --- MODIFIED: reset 메서드는 모든 기록을 초기화해야 함 ---
    def reset(self):
        # 시뮬레이션 상태 변수
        self.interval_capacities, self.actual_completion_times, self.chunk_actions = {}, {}, {}
        self.chunk_play_end_times = {}
        self.generated_chunk_times = {i: (i + 1) * self.t_i for i in range(self.num_chunks)}
        self.u_i, self.p_i, self.d_i, self.k_j, self.speed_idx_j, self.l_i, self.x_j, self.received_chunks = [], [], [], [], [], [], [], []
        self.tau_i = []
        self.current_chunk_idx = 0
        
        # ALVS 고유 기록 상태
        self.last_quality = 1 # 초기 화질
        self.throughput_history = deque([0] * self.history_len, maxlen=self.history_len)
        self.download_time_history = deque([0] * self.history_len, maxlen=self.history_len)
        
        # 초기 상태 가져오기
        return self.get_state_for_decision(0)

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
    

    # --- MODIFIED: step 메서드는 기록을 업데이트해야 함 ---
    def step(self, action):
        k_i, speed_idx, x_i = action # x_i는 무시됨 (ALVS는 건너뛰기 수행 안 함)
        current_playback_speed = self.playback_speeds[speed_idx]
        chunk_idx = len(self.u_i)
        
        if chunk_idx >= self.num_chunks: 
            last_valid_idx = self.num_chunks - 1
            if last_valid_idx < 0:
                return self.reset(), 0.0, True, {}
            current_state = self.get_state_for_decision(self.num_chunks)
            return current_state, 0.0, True, {}
        
        # --- ALVS does not skip ---
        x_i = 0

        self.k_j.append(k_i)
        self.speed_idx_j.append(speed_idx)
        self.x_j.append(x_i)

        # u_i (전송 시작 시간) 계산
        chunk_generation_time = self.generated_chunk_times.get(chunk_idx, (chunk_idx + 1) * self.t_i)
        prev_completion_time = 0.0
        if chunk_idx > 0:
            prev_idx = chunk_idx - 1
            prev_action_info = self.chunk_actions.get(prev_idx)
            if prev_action_info:
                # no skip
                prev_completion_time = self.actual_completion_times.get(prev_idx, 0.0)
                if prev_completion_time == float('inf'):
                    prev_completion_time = 0.0
        else:
            prev_completion_time = 0.0

        u_i = max(chunk_generation_time, prev_completion_time)
        if self.u_i: 
            u_i = max(u_i, self.u_i[-1] + 1e-9)
        self.u_i.append(u_i)
        
        self.chunk_actions[chunk_idx] = {'k': k_i, 'speed_idx': speed_idx, 'speed': current_playback_speed, 'x': x_i, 'u': u_i}
        
        # 전송 시뮬레이션
        l_i, completion_time = self._simulate_chunk_transmission(chunk_idx, k_i, u_i)
        if l_i != float('inf'):
            if chunk_idx not in self.received_chunks:
                self.received_chunks.append(chunk_idx)

        # 전송 결과 저장
        while len(self.l_i) <= chunk_idx:
            self.l_i.append(0.0)
        self.l_i[chunk_idx] = l_i
        self.actual_completion_times[chunk_idx] = completion_time
        
        # --- 기록 업데이트 ---
        chunk_size_bits = self.bitrate_map[k_i] * 1_000_000 * self.t_i
        throughput = chunk_size_bits / (l_i + 1e-9) # 0으로 나누기 방지를 위한 작은 값 추가
        self.throughput_history.append(throughput)
        self.download_time_history.append(l_i)
        
        # 재생 시작 시점 (p_i) 및 Stall (d_i) 계산
        p_i = float('inf') # default
        d_i = 0.0 # default

        if l_i != float('inf'):
            ready_time = completion_time
            prev_played_idx = -1
            for j in range(chunk_idx - 1, -1, -1):
                action_info = self.chunk_actions.get(j)
                if action_info and j < len(self.p_i) and self.p_i[j] != float('inf'):
                    prev_played_idx = j
                    break
            
            if prev_played_idx != -1:
                prev_action_info = self.chunk_actions[prev_played_idx]
                prev_playback_speed = prev_action_info.get('speed', 1.0)
                prev_play_duration = self.t_i / prev_playback_speed
                prev_play_end_time = self.p_i[prev_played_idx] + prev_play_duration
                p_i = max(ready_time, prev_play_end_time)
                d_i = max(0, ready_time - prev_play_end_time)
            else:
                p_i = max(ready_time, self.t_buffer)
                d_i = max(0, ready_time - self.t_buffer)
        
        else: # 전송 실패
            p_i = float('inf')
            d_i = float('inf')
        
        while len(self.p_i) <= chunk_idx: self.p_i.append(float('inf'))
        self.p_i[chunk_idx] = p_i
        while len(self.d_i) <= chunk_idx: self.d_i.append(0.0)
        self.d_i[chunk_idx] = d_i
        
        # 성공적으로 재생됐을 경우, 전송 마지막 시간 기록
        if p_i != float('inf'):
            play_duration = self.t_i / current_playback_speed
            self.chunk_play_end_times[chunk_idx] = p_i + play_duration

        # 라이브 지연 (tau) 계산
        tau_i = float('inf')
        if chunk_idx == 0:
            prev_tau = self.t_buffer
        elif (chunk_idx - 1) < len(self.tau_i):
            prev_tau = self.tau_i[chunk_idx - 1]
        else:
            prev_tau = self.t_buffer
        
        if p_i != float('inf'):
            tau_i = p_i - chunk_idx * self.t_i
        
        # 라이브 지연 시간 저장
        while len(self.tau_i) <= chunk_idx:
            self.tau_i.append(float('inf'))
        self.tau_i[chunk_idx] = tau_i

        # 보상 계산을 위해, 다음 시점의 지연 (tau_i+1) 계산
        tau_i_plus_1 = float('inf')
        current_tau_i = self.tau_i[chunk_idx]
        current_d_i = self.d_i[chunk_idx]

        if current_tau_i != float('inf') and current_d_i != float('inf'):
            time_saved_or_lost = self.t_i - (self.t_i / current_playback_speed)
            tau_i_plus_1 = current_tau_i + max(0, current_d_i) - time_saved_or_lost
        
        # 보상 계산
        last_complete_d_i = 0.0
        if chunk_idx > 0:
             last_played_idx = -1
             for j in range(chunk_idx - 1, -1, -1):
                  action_info = self.chunk_actions.get(j)
                  if action_info and action_info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                       last_played_idx = j
                       break
             if last_played_idx != -1 and last_played_idx < len(self.d_i):
                  last_complete_d_i = self.d_i[last_played_idx]
                  if last_complete_d_i == float('inf'): last_complete_d_i = 0.0

        reward = self._get_reward(k_i, x_i, last_complete_d_i, tau_i_plus_1)
        
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
        chunk_generation_time = self.generated_chunk_times.get(target_chunk_idx, (target_chunk_idx + 1) * self.t_i)
        prev_completion = 0.0
        if target_chunk_idx > 0:
            prev_completion= self.actual_completion_times.get(target_chunk_idx - 1, 0.0)
            if prev_completion == float('inf'):
                prev_completion = self.u_i[-1] if self.u_i else 0.0
        
        u_i = max(chunk_generation_time, prev_completion)
        if self.u_i:
            u_i = max(u_i, self.u_i[-1] + 1e-9)
        
        # 라이브 지연 시간
        n_i = -1
        for j in range(target_chunk_idx - 1, -1, -1):
            action_info = self.chunk_actions.get(j)
            if action_info:
                p_j = self.p_i[j]
                if p_j != float('inf') and p_j <= u_i:
                    n_i = j
                    break
        
        tau_i = self.t_buffer
        if n_i != -1:
            p_n_i = self.p_i[n_i]
            action_n = self.chunk_actions[n_i]
            v_n = action_n['speed']

            # n_i 이전까지의 실제 재생 지속 시간 합계 계산
            sum_duration = 0.0
            for j in range(n_i):
                a_j = self.chunk_actions.get(j)
                if a_j and a_j.get('x', 0) == 0:
                    sum_duration += self.t_i / a_j['speed']
            
            # 현재 스톨 계산
            play_end_time_n = p_n_i + (self.t_i / v_n)
            current_stall = max(0.0, u_i - play_end_time_n)

            tau_i = p_n_i - sum_duration + current_stall
        else:
            tau_i = self.t_buffer
        
        # 남은 버퍼 시간 계산
        remaining_buffer_time = self.calculate_remaining_play_time(u_i, target_chunk_idx)
    
        # 6. 현재 재생 비트레이트 (이전 선택 비트레이트 사용)
        last_bitrate = self.bitrate_map[self.last_quality]

        # 7. 남은 청크 수
        chunks_remaining = self.num_chunks - target_chunk_idx
        
        safe_tau = tau_i if np.isfinite(tau_i) else 10.0
        safe_buffer = remaining_buffer_time if np.isfinite(remaining_buffer_time) else 0.0
        
        other_features = np.array([
            safe_buffer / 10.0, # 정규화
            safe_tau / 10.0, # 정규화
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
    def _get_reward(self, k_i, x_i, last_complete_d_i, tau_i_plus_1):
        """计算奖励"""
        """Unified Reward Function"""
        quality_reward = 0
        if x_i == 0:
            quality_reward = self.W1 * (k_i**2)
        
        stall_penalty = self.W2 * last_complete_d_i
        # Safe tau calculation
        safe_tau = tau_i_plus_1 if (tau_i_plus_1 != float('inf') and tau_i_plus_1 > -100) else 10.0
        delay_penalty = self.W_tau * (safe_tau - self.t_i * x_i)**2
        skip_penalty = self.W3 * x_i
        
        raw_reward = quality_reward - stall_penalty - delay_penalty - skip_penalty
        return max(min(raw_reward, 20), -20) # Clip

    # --- Add back get_skip_statistics ---
    def get_skip_statistics(self):
        if not self.x_j: return 0, 0
        skipped = sum(self.x_j)
        num_processed = len(self.x_j)
        played = num_processed - skipped
        return skipped, played

    def get_average_speed(self):
        if not self.speed_idx_j: return 1.0 # Default if no actions taken
        actual_speeds = [self.playback_speeds[idx] for idx in self.speed_idx_j]
        return sum(actual_speeds) / len(actual_speeds)
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

# --- 강화 학습 부분 (A3C 알고리즘 - ALVS 버전) ---

# --- NEW: 새로운 상태 차원 및 초매개변수 정의 ---
HISTORY_LEN = 8
QUALITY_LEVELS = 20 # 환경 설정과 일치
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
            nn.Linear(2176 + 4, 128),
            nn.ReLU()
        )
        self.quality_head = nn.Linear(128, QUALITY_LEVELS)
        self.speed_head = nn.Linear(128, SPEED_LEVELS)

    def forward(self, state_dict):
        # 1. 히스토리 데이터 (길이 8) -> (Batch, 1, 8)
        throughput = state_dict['throughput_history'].unsqueeze(1)
        download_time = state_dict['download_time_history'].unsqueeze(1)
        
        # 2. 다음 청크 크기 데이터 (길이 20) -> (Batch, 1, 20)
        chunk_sizes = state_dict['next_chunk_sizes'].unsqueeze(1)
        
        # 3. 히스토리 데이터를 길이 20으로 패딩 (앞부분은 유지, 뒷부분 0 채움)
        # target_len(20) - current_len(8) = 12만큼 패딩
        target_len = chunk_sizes.size(2) # 20
        current_len = throughput.size(2) # 8
        padding = target_len - current_len # 12
        
        # (Batch, 1, 8) -> (Batch, 1, 20)
        throughput_padded = nn.functional.pad(throughput, (0, padding))
        download_time_padded = nn.functional.pad(download_time, (0, padding))
        
        # 4. CNN 입력 병합: (Batch, 3, 20)
        cnn_input = torch.cat([throughput_padded, download_time_padded, chunk_sizes], dim=1)
        
        # 5. CNN 통과
        cnn_output = self.cnn(cnn_input) # Output shape: (Batch, 128, 17)
        cnn_output_flat = cnn_output.view(cnn_output.size(0), -1) # Flatten
        
        # 6. MLP 입력 병합
        other_features = state_dict['other_features']
        combined_features = torch.cat([cnn_output_flat, other_features], dim=1)
        
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
            nn.Linear(2176 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 가치 값 출력
        )

    def forward(self, state_dict):
        # Actor와 동일한 전처리 과정
        throughput = state_dict['throughput_history'].unsqueeze(1)
        download_time = state_dict['download_time_history'].unsqueeze(1)
        chunk_sizes = state_dict['next_chunk_sizes'].unsqueeze(1)
        
        target_len = chunk_sizes.size(2) # 20
        current_len = throughput.size(2) # 8
        padding = target_len - current_len # 12
        
        throughput_padded = nn.functional.pad(throughput, (0, padding))
        download_time_padded = nn.functional.pad(download_time, (0, padding))
        
        cnn_input = torch.cat([throughput_padded, download_time_padded, chunk_sizes], dim=1)
        
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
        self.env = VideoStreamingEnv_ALVS(W1=1.0, W2=30, W3=100, W_tau=0.5)
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
                    ep_taus.append(state['other_features'][1] * 10.0) # 역정규화
                    
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
    env = VideoStreamingEnv_ALVS(W1=1.0, W2=1.5, W3=0.02, W_tau=0.005)
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
            ep_taus.append(state['other_features'][1] * 10.0) # 역정규화
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