#!/usr/bin/env python
# coding: utf-8

import os
import argparse
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
    # (Keep the DynamicChannelModel class exactly as it is in baseline5.py or Proposed_ppo.py)
    def __init__(self, W=1e6, gamma_dB=35, distance=20, beta=2, sigma_dB=4, mu_dB=0): # Example gamma
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
        path_loss = self.distance ** self.beta
        shadowing_dB = np.random.normal(self.mu_dB, self.sigma_dB)
        shadowing = 10 ** (shadowing_dB / 10)
        fast_fading_real = np.random.normal(0, 1/np.sqrt(2))
        fast_fading_imag = np.random.normal(0, 1/np.sqrt(2))
        fast_fading_complex = fast_fading_real + 1j * fast_fading_imag
        fast_fading = np.abs(fast_fading_complex) ** 2
        channel_gain_squared = (1 / path_loss) * shadowing * fast_fading
        components = {
            'path_loss': path_loss, 'shadowing_dB': shadowing_dB, 'shadowing': shadowing,
            'fast_fading_complex': fast_fading_complex, 'fast_fading': fast_fading,
            'channel_gain_squared': channel_gain_squared
        }
        self.channel_components = components
        self.current_channel_gain = np.sqrt(channel_gain_squared)
        return components

    def calculate_capacity(self, regenerate_channel=True):
        if regenerate_channel or self.current_channel_gain is None:
            self.generate_channel_state()
        channel_gain_squared = self.current_channel_gain ** 2
        capacity_bps = self.W * np.log2(1 + self.gamma * channel_gain_squared)
        capacity_bytes_per_sec = capacity_bps / 8
        self.current_capacity = capacity_bytes_per_sec
        return capacity_bytes_per_sec

    def get_channel_details(self):
        if self.current_capacity is None:
            self.calculate_capacity()
        return {
            'capacity_Mbps': (self.current_capacity * 8) / 1e6,
            'channel_gain': self.current_channel_gain,
            'components': self.channel_components
        }


class VideoStreamingEnvCombined: # Inherit from baseline5's Env for structure
    def __init__(self, num_chunks=300, t_buffer=4, t_i=2,
                 W1=1, W2=300, W3=100.0, W_tau=1, # Added W3 back
                 verbose=False):
        """初始化结合 Skip 和 Speed 的视频流环境"""
        # Call parent __init__ from baseline5's VideoStreamingEnv
      

        self.num_chunks = num_chunks
        self.t_buffer = t_buffer
        self.t_i = t_i
        
        #self.quality_levels = [1, 2, 3, 4]
        self.quality_levels = list(range(1, 21))
        self.max_quality = max(self.quality_levels)
        self.verbose = verbose
        self.W1 = W1  # 质量奖励权重
        self.W2 = W2  # 卡顿惩罚权重
        # self.W3 = W3 # 跳过惩罚权重 (REMOVED)
        self.W_tau = W_tau  # 延迟惩罚权重

        # --- Playback Speed Definition ---
        self.playback_speeds = [0.9, 1.0, 1.1]
        #self.playback_speeds = [0.8, 1.0, 1.2]
        self.num_speed_levels = len(self.playback_speeds)
        # --- End Playback Speed Definition ---
        self.bitrate_map = {
            1: 0.045,   2: 0.089,   3: 0.131,   4: 0.178,   5: 0.221,
            6: 0.262,   7: 0.334,   8: 0.396,   9: 0.522,   10: 0.595,
            11: 0.791,  12: 1.032,  13: 1.244,  14: 1.546,  15: 2.133,
            16: 2.484,  17: 3.078,  18: 3.526,  19: 3.840,  20: 4.219
        }
        #self.bitrate_map = {
         #   1: 0.8, 2: 1.2, 3: 2.4, 4: 4.8
        #}
        self.channel_model = DynamicChannelModel()

        # --- 状态变量 ---
        self.interval_capacities = {}
        self.received_chunks = []
        self.played_chunks = [] # 跟踪已播放的chunks (不再需要，但保留可能无害)
        self.actual_completion_times = {}
        self.chunk_actions = {} # Stores {k, speed_idx, speed, u}
        self.generated_chunk_times = {}
        self.chunk_play_end_times = {} # 记录每个块的播放结束时间 (考虑速度)

        self.u_i = []
        self.p_i = []
        self.d_i = []
        self.tau_i = []
        self.k_j = []
        # self.x_j = [] # 跳过决策 (REMOVED)
        self.speed_idx_j = [] # Store chosen speed index (0, 1, 2)
        self.l_i = []
        self.tau_history = [self.t_buffer] # Start with initial buffer as latency?
        # self.cumulative_skips = 0 # (REMOVED)
        self.state = None

        
        self.W3 = W3  # 跳过惩罚权重 (Re-introduced)
        self.x_j = [] # 跳过决策 (Re-introduced)
        self.cumulative_skips = 0 # (Re-introduced)
        # --- End Skip specific elements ---

        # Modify chunk_actions comment
        # Now stores {k, speed_idx, speed, x, u}

        # Reset needs to be called again to initialize new variables
        self.reset() # Call the modified reset

    def reset(self):
        """重置环境 (Combined Version)"""
        # Call parent reset first
        self.interval_capacities = {}
        self.received_chunks = []
        self.played_chunks = []
        self.actual_completion_times = {}
        self.chunk_actions = {} # Clear actions dict
        self.generated_chunk_times = {}
        self.chunk_play_end_times = {}

        self.u_i = []
        self.p_i = []
        self.d_i = []
        self.tau_i = []
        self.k_j = []
        # self.x_j = [] # (REMOVED)
        self.speed_idx_j = [] # Reset speed index list
        self.l_i = []
        self.tau_history = [self.t_buffer]
        # self.cumulative_skips = 0 # (REMOVED)

        for i in range(self.num_chunks):
            self.generated_chunk_times[i] = (i + 1) * self.t_i

        # Initial state calculation needs u_0, so we call get_state_for_decision(0)
        

        #super().reset() # This already resets most variables from baseline5

        # --- Reset Skip specific elements ---
        self.x_j = []
        self.cumulative_skips = 0
        # --- End Reset Skip specific elements ---

        # Recalculate initial state using the correct get_state_for_decision
        self.state = self.get_state_for_decision(0)
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

    def step(self, action):
        """处理包含质量、速度和跳过的动作 (Combined Version)"""
        k_i, speed_idx, x_i = action # Action now has 3 elements
        current_playback_speed = self.playback_speeds[speed_idx]
        chunk_idx = len(self.u_i)

        if chunk_idx >= self.num_chunks:
             # Episode finished logic remains the same as baseline5
             last_valid_idx = self.num_chunks - 1
             if last_valid_idx < 0: return self.reset(), 0.0, True, {}
             current_state = self.get_state_for_decision(self.num_chunks)
             return current_state, 0.0, True, {}

        self.k_j.append(k_i)
        self.speed_idx_j.append(speed_idx)
        self.x_j.append(x_i) # Store skip decision

        if x_i == 1:
            self.cumulative_skips += 1 # Increment cumulative skips

        # --- Calculate u_i (Transmission Start Time) ---
        chunk_generation_time = self.generated_chunk_times.get(chunk_idx, (chunk_idx + 1) * self.t_i)
        prev_completion_time = 0.0
        if chunk_idx > 0:
            prev_idx = chunk_idx - 1
            prev_action_info = self.chunk_actions.get(prev_idx)
            if prev_action_info:
                # If previous was skipped, base start time on its u_i
                if prev_action_info.get('x', 0) == 1:
                    prev_completion_time = prev_action_info.get('u', 0.0)
                # If previous was not skipped, use its completion time
                else:
                    prev_completion_time = self.actual_completion_times.get(prev_idx, 0.0)
                    if prev_completion_time == float('inf'): # Handle failure
                        prev_completion_time = 0.0
        else: # First chunk
             prev_completion_time = 0.0

        u_i = max(chunk_generation_time, prev_completion_time)
        if self.u_i: u_i = max(u_i, self.u_i[-1] + 1e-9)
        self.u_i.append(u_i)

        # Store action details including skip
        self.chunk_actions[chunk_idx] = {'k': k_i, 'speed_idx': speed_idx, 'speed': current_playback_speed, 'x': x_i, 'u': u_i}

        # --- Simulate Transmission (only if not skipped) ---
        l_i = 0.0
        completion_time = u_i # Default if skipped

        if x_i == 0: # Only transmit if not skipped
            l_i, completion_time = self._simulate_chunk_transmission(chunk_idx, k_i, u_i)
            if l_i != float('inf'):
                 if chunk_idx not in self.received_chunks:
                     self.received_chunks.append(chunk_idx)
        # else: l_i remains 0.0

        # Store transmission results
        while len(self.l_i) <= chunk_idx: self.l_i.append(0.0)
        self.l_i[chunk_idx] = l_i
        self.actual_completion_times[chunk_idx] = completion_time

        # --- Calculate Play Start (p_i) and Stall (d_i) ---
        p_i = float('inf') # Default to invalid
        d_i = 0.0          # Default stall is 0

        if x_i == 0 and l_i != float('inf'): # Not skipped and transmission succeeded
            ready_time = completion_time
            # Find the previous *played* chunk (x=0 and valid p_i)
            prev_played_idx = -1
            for j in range(chunk_idx - 1, -1, -1):
                 action_info = self.chunk_actions.get(j)
                 if action_info and action_info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                      prev_played_idx = j
                      break

            if prev_played_idx != -1:
                prev_action_info = self.chunk_actions[prev_played_idx]
                prev_playback_speed = prev_action_info.get('speed', 1.0)
                prev_play_duration = self.t_i / prev_playback_speed
                prev_play_end_time = self.p_i[prev_played_idx] + prev_play_duration
                p_i = max(ready_time, prev_play_end_time)
                d_i = max(0, ready_time - prev_play_end_time)
            else: # No previous played chunk (or first chunk)
                p_i = max(ready_time, self.t_buffer)
                d_i = max(0, ready_time - self.t_buffer)

        elif x_i == 1: # Skipped
            p_i = float('inf') # Indicate no playback
            d_i = 0.0          # No stall for skipped chunk
        else: # x_i == 0 and l_i == float('inf') -> Transmission Failed
             p_i = float('inf')
             d_i = float('inf') # Stall is effectively infinite

        # Store p_i and d_i
        while len(self.p_i) <= chunk_idx: self.p_i.append(float('inf'))
        self.p_i[chunk_idx] = p_i
        while len(self.d_i) <= chunk_idx: self.d_i.append(0.0) # Init with 0
        self.d_i[chunk_idx] = d_i

        # Record play end time (if played successfully)
        if x_i == 0 and p_i != float('inf'):
            play_duration = self.t_i / current_playback_speed
            self.chunk_play_end_times[chunk_idx] = p_i + play_duration

        # --- Calculate Live Latency (τ_i) ---
        tau_i = float('inf') # Default to invalid
        # Get previous tau (τ_{i-1})
        if chunk_idx == 0:
            prev_tau = self.t_buffer
        elif (chunk_idx - 1) < len(self.tau_i):
             prev_tau = self.tau_i[chunk_idx - 1]
        else:
             prev_tau = self.t_buffer # Fallback

        if x_i == 0 and p_i != float('inf'): # Played successfully
            tau_i = p_i - chunk_idx * self.t_i
        elif x_i == 1: # Skipped, inherit previous tau
            tau_i = prev_tau
        # else: Failed transmission, tau_i remains inf

        # Store tau_i
        while len(self.tau_i) <= chunk_idx: self.tau_i.append(float('inf'))
        self.tau_i[chunk_idx] = tau_i

        # --- Calculate τ_{i+1} (Predicted latency for next state) ---
        tau_i_plus_1 = float('inf')
        # Use tau_i just calculated
        current_tau_i = self.tau_i[chunk_idx]
        current_d_i = self.d_i[chunk_idx] # Use d_i of the current chunk

        if x_i == 0: # Not Skipped
            if current_tau_i != float('inf') and current_d_i != float('inf'):
                time_saved_or_lost = self.t_i - (self.t_i / current_playback_speed)
                # τ_{i+1} = τ_i + d_i - time_saved_by_speed
                tau_i_plus_1 = current_tau_i + max(0, current_d_i) - time_saved_or_lost
            # else: remains inf
        elif x_i == 1: # Skipped
            # Reduce latency by the duration of the skipped chunk
            if prev_tau != float('inf'): # Base it on previous valid tau
                 tau_i_plus_1 = prev_tau - self.t_i # Simpler definition: skipping moves reference point
            # else: remains inf

        # Store tau_i_plus_1 in history (optional, but maybe useful for reward)
        # self.tau_history.append(tau_i_plus_1)

        # --- Calculate Reward ---
        # Find stall before the *last played* chunk
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

        # Pass k_i, x_i, last_complete_d_i, and calculated tau_i_plus_1 to reward
        reward = self._get_reward(k_i, x_i, last_complete_d_i, tau_i_plus_1)

        # --- Prepare for Next Step ---
        next_chunk_idx = chunk_idx + 1
        next_state = self.get_state_for_decision(next_chunk_idx)
        done = chunk_idx >= self.num_chunks - 1

        return next_state, reward, done, {}

    def get_state_for_decision(self, target_chunk_idx):
       
        
        
        chunk_generation_time = self.generated_chunk_times.get(target_chunk_idx, (target_chunk_idx + 1) * self.t_i)
        prev_completion = 0.0
        if target_chunk_idx > 0:
             # 使用上一个块的实际完成时间，如果失败则估算
             prev_completion = self.actual_completion_times.get(target_chunk_idx - 1, 0.0)
             if prev_completion == float('inf'): 
                 prev_completion = self.u_i[-1] if self.u_i else 0.0
        
        # u_i 是生成时间和上一个传输完成时间的较大值
        u_i = max(chunk_generation_time, prev_completion)
        # 保持时间单调性
        if self.u_i:
            u_i = max(u_i, self.u_i[-1] + 1e-9)

        
        n_i = -1 
        
        
        for j in range(target_chunk_idx - 1, -1, -1):
            action_info = self.chunk_actions.get(j)
            if action_info and action_info.get('x', 0) == 0: # 没被跳过
                 p_j = self.p_i[j]
                 if p_j != float('inf') and p_j <= u_i:
                     n_i = j
                     break
        
        
        
        tau_i = self.t_buffer # 默认值 (比如初始状态)
        
        if n_i != -1:
            
            p_n_i = self.p_i[n_i]
            
            
            action_n = self.chunk_actions[n_i]
            v_n = action_n['speed']
            eta_n = self.t_i # 假设每个块时长固定为 2s
            
            
            sum_duration = 0.0
            for j in range(n_i):
                a_j = self.chunk_actions.get(j)
                if a_j and a_j.get('x', 0) == 0: # 只计算播放了的块
                    sum_duration += self.t_i / a_j['speed']
            
            # 计算 Max 项 (当前时刻 u_i 是否处于卡顿中)
            # 块 n(i) 的播放结束时间
            play_end_time_n = p_n_i + (eta_n / v_n)
            current_stall = max(0.0, u_i - play_end_time_n)
            
            # 最终公式
            tau_i = p_n_i - sum_duration + current_stall
            
        else:
            
            tau_i = self.t_buffer

        # 4. 计算 B_i (Buffer), 这部分逻辑你之前的代码是对的，保持即可
        remaining_buffer_time = self.calculate_remaining_play_time(u_i, target_chunk_idx)

        # 确保数值安全
        safe_tau = tau_i if np.isfinite(tau_i) else 10.0
        safe_buffer = remaining_buffer_time if np.isfinite(remaining_buffer_time) else 0.0

        return np.array([safe_tau, safe_buffer], dtype=np.float32)

    def _get_reward(self, k_i, x_i, last_complete_d_i, tau_i_plus_1):
        """计算奖励"""
        quality_reward = 0
        if x_i == 0:
            
            quality_reward = self.W1 * (k_i**2) 

        # 卡顿惩罚 (基于上一个播放块的卡顿)
        stall_penalty = self.W2 * last_complete_d_i

        
        delay_penalty = self.W_tau * (tau_i_plus_1 - self.t_i*x_i) ** 2

        # 跳过惩罚
        skip_penalty = self.W3 * x_i

        raw_reward = quality_reward  - delay_penalty - skip_penalty - stall_penalty
        reward = max(min(raw_reward, 20), -20)

        

        return reward

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
QUALITY_LEVELS = 20
SPEED_LEVELS = 3
SKIP_OPTIONS = 2

class ActorNetworkCombined(nn.Module):
    """Actor 신경망: 상태를 받아 품질, 속도, 스킵에 대한 행동 확률을 출력합니다."""
    def __init__(self):
        super(ActorNetworkCombined, self).__init__()
        self.base = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
        self.quality_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, QUALITY_LEVELS))
        self.speed_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, SPEED_LEVELS))
        self.skip_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, SKIP_OPTIONS))

    def forward(self, state):
        if not isinstance(state, torch.Tensor): 
            state = torch.FloatTensor(state).to(device)
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
        self.network = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state):
        if not isinstance(state, torch.Tensor): 
            state = torch.FloatTensor(state).to(device)
        if state.dim() == 1: state = state.unsqueeze(0)
        return self.network(state)

class PPOAgentCombined:
    """
    PPO 에이전트: Actor와 Critic을 관리하고 학습을 수행합니다.
    (추가) 베이스라인 학습에 대비합니다.
    """
    def __init__(self, learn_speed: bool = True, learn_skip: bool = True, fixed_speed_idx: int = 1, fixed_skip: int = 0):
        self.learn_speed = learn_speed
        self.learn_skip = learn_skip
        self.fixed_speed_idx = fixed_speed_idx
        self.fixed_skip = fixed_skip

        self.actor = ActorNetworkCombined().to(device)
        self.critic = CriticNetwork().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        
        self.states = []
        self.actions_quality = []
        self.actions_speed = [] # Keep speed index
        self.actions_skip = []  # Add skip action storage
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs_quality = []
        self.log_probs_speed = [] # Keep speed log_prob
        self.log_probs_skip = []  # Add skip log_prob storage
        self.values = []

    def select_action(self, state):
        with torch.no_grad():
            quality_probs, speed_probs, skip_probs = self.actor(state)

            quality_dist = Categorical(quality_probs)
            quality_action = quality_dist.sample()

            log_prob_quality = quality_dist.log_prob(quality_action)
            entropy = quality_dist.entropy()

            # speed
            if self.learn_speed:
                speed_dist = Categorical(speed_probs)
                speed_action = speed_dist.sample()
                log_prob_speed = speed_dist.log_prob(speed_action)
                entropy = entropy + speed_dist.entropy()
                speed_action_item = speed_action.item()
            else:
                speed_action_item = self.fixed_speed_idx
            
            # skip
            if self.learn_skip:
                skip_dist = Categorical(skip_probs)
                skip_action = skip_dist.sample()
                log_prob_skip = skip_dist.log_prob(skip_action)
                entropy = entropy + skip_dist.entropy()
                skip_action_item = skip_action.item()
            else:
                skip_action_item = self.fixed_skip

            value = self.critic(state)
        
        action = (quality_action.item() + 1, speed_action_item, skip_action_item)
        return action, log_prob_quality.item(), log_prob_speed.item(), log_prob_skip.item(), value.item()

    def store_experience(self, state, action, reward, next_state, done,
                         log_prob_quality, log_prob_speed, log_prob_skip, value):
        self.states.append(state)
        self.actions_quality.append(action[0] - 1)
        self.actions_speed.append(action[1])
        self.actions_skip.append(action[2])
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs_quality.append(log_prob_quality)
        self.log_probs_speed.append(log_prob_speed)
        self.log_probs_skip.append(log_prob_skip)
        self.values.append(value)

    def clear_memory(self):
        self.states = []
        self.actions_quality = []
        self.actions_speed = []
        self.actions_skip = []     
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs_quality = []
        self.log_probs_speed = []
        self.log_probs_skip = []      
        self.values = []

    def compute_returns_and_advantages(self):
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        values = torch.tensor(self.values, dtype=torch.float32).to(device)

        with torch.no_grad():
            last_next_state = self.critic(torch.FloatTensor(self.next_states[-1]).to(device))
            last_value = self.critic(last_next_state).item()
        
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
        old_log_probs_quality = torch.FloatTensor(self.log_probs_quality).to(device)
        old_log_probs_speed = torch.FloatTensor(self.log_probs_speed).to(device)
        old_log_probs_skip = torch.FloatTensor(self.log_probs_skip).to(device) # Skip log_probs tensor
        
        old_log_probs = old_log_probs_quality + old_log_probs_speed + old_log_probs_skip

        for _ in range(EPOCHS):
            num_samples = len(self.states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, len(self.states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions_quality = actions_quality[batch_indices]
                batch_actions_speed = actions_speed[batch_indices]
                batch_actions_skip = actions_skip[batch_indices] # Skip actions batch
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                q_probs, s_probs, sk_probs = self.actor(batch_states)
                value_predictions = self.critic(batch_states).squeeze()

                q_dist = Categorical(q_probs)
                new_log_probs_quality = q_dist.log_prob(batch_actions_quality)
                entropy = q_dist.entropy()

                if self.learn_speed:
                    s_dist = Categorical(s_probs)
                    new_log_probs_speed = s_dist.log_prob(batch_actions_speed)
                    entropy = entropy + s_dist.entropy()
                
                if self.learn_skip:
                    sk_dist = Categorical(sk_probs)
                    new_log_probs_skip = sk_dist.log_prob(batch_actions_skip)
                    entropy = entropy + sk_dist.entropy()

                new_log_probs = new_log_probs_quality + new_log_probs_speed + new_log_probs_skip
                entropy = entropy.mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                critic_loss = nn.MSELoss()(value_predictions, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

    def save_models(self, episode, mode):
        """모델을 저장합니다."""
        model_name_base = f'{mode}_Combined_ppo_episode_{episode}'

        if mode == "proposed":
            torch.save(self.actor.state_dict(), f'models/{model_name_base}_actor.pth')
            torch.save(self.critic.state_dict(), f'models/{model_name_base}_critic.pth')
            print(f"{mode}_{episode} 에피소드 모델 저장 완료")

        elif mode == "b2":
            torch.save(self.actor.state_dict(), f'baseline_models/b2/b2_{episode}_actor.pth')
            torch.save(self.critic.state_dict(), f'baseline_models/b2/b2_{episode}_critic.pth')
            print(f"{mode}_{episode} 에피소드 모델 저장 완료")
        
        elif mode == "b3":
            torch.save(self.actor.state_dict(), f'baseline_models/b3/b3_{episode}_actor.pth')
            torch.save(self.critic.state_dict(), f'baseline_models/b3/b3_{episode}_critic.pth')
            print(f"{mode}_{episode} 에피소드 모델 저장 완료")
        
        else:
            raise ValueError(f"존재하지 않는 모드입니다. 이름을 확인하세요: {mode}")


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

def train_combined(mode: str = "proposed", episode: int=50000):
    """메인 훈련 루프 함수."""
    env = VideoStreamingEnvCombined(num_chunks=300, verbose=False,
                                    W1=1, W2=30, W3=100.0, W_tau=0.5)
    
    fixed_speed_idx = 1
    fixed_skip = 0

    if mode == "proposed":
        agent = PPOAgentCombined(learn_speed=True, learn_skip=True,
                                 fixed_speed_idx=fixed_speed_idx, fixed_skip=fixed_skip)
        out_actor = "models/proposed_actor.pth"
        out_baseline = None

    elif mode == "b2":
        agent = PPOAgentCombined(learn_speed=False, learn_skip=False,
                                 fixed_speed_idx=fixed_speed_idx, fixed_skip=fixed_skip)
        out_baseline = f"baseline_models/{episode}_b2.pt"
    
    elif mode == "b3":
        agent = PPOAgentCombined(learn_speed=False, learn_skip=True,
                                 fixed_speed_idx=fixed_speed_idx, fixed_skip=fixed_skip)
        out_baseline = f"baseline_models/{episode}_b3.pt"
    
    else:
        raise ValueError(mode)

    # --- 로깅을 위한 리스트 초기화 ---
    episode_rewards = []
    episode_lengths = []
    episode_avg_capacities = []
    episode_stall_times = []
    episode_skipped = [] # Add back skipped tracking
    episode_played = []  # Add back played tracking
    episode_avg_speeds = [] # Keep average speed tracking
    episode_avg_tau_i = []
    episode_avg_qualities = []
    update_count = 0
    
    results_dir = 'models'
    results_prefix, models_prefix = "Combined_ppo", "Combined_ppo"

    print(f"--- {mode} ep: {episode} 훈련 시작 ---")
    start_time_train = time.time()

    for episode in range(MAX_EPISODES):
        state = env.reset()
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)

        episode_reward = 0
        episode_step = 0
        done = False

        for step in range(NUM_STEPS):
            if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)

            # --- Get Combined Action ---
            action, log_prob_quality, log_prob_speed, log_prob_skip, value = agent.select_action(state)
            # --- End Get ---

            next_state, reward, done, _ = env.step(action) # Pass combined action

            if not isinstance(next_state, np.ndarray): next_state = np.array(next_state, dtype=np.float32)
            if state.shape != (STATE_DIM,): state = state.reshape(STATE_DIM)
            if next_state.shape != (STATE_DIM,): next_state = next_state.reshape(STATE_DIM)

            # --- Store Combined Experience ---
            agent.store_experience(state, action, reward, next_state, done,
                                 log_prob_quality, log_prob_speed, log_prob_skip, value)
            # --- End Store ---

            state = next_state
            episode_reward += reward
            episode_step += 1
            if done: break

        if len(agent.states) > 0:
             agent.update()
             agent.clear_memory()
             update_count += 1

        # --- Update Logging ---
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        avg_capacity = env.get_average_capacity()
        episode_avg_capacities.append(avg_capacity)
        total_stall = env.get_total_stall_time()
        episode_stall_times.append(total_stall)
        skipped, played_count = env.get_skip_statistics() # Use the restored function
        episode_skipped.append(skipped)                   # Log skips
        episode_played.append(played_count)               # Log played
        avg_speed = env.get_average_speed()
        episode_avg_speeds.append(avg_speed)
        avg_tau_i = env.get_average_tau_i()
        episode_avg_tau_i.append(avg_tau_i)
        avg_quality = env.get_average_quality()
        episode_avg_qualities.append(avg_quality)
        # --- End Update ---

        if (episode + 1) % 100 == 0:
            # --- Update Averaging and Printing ---
            recent_rewards = episode_rewards[-100:]
            recent_avg_speeds = episode_avg_speeds[-100:]
            recent_avg_tau_i = episode_avg_tau_i[-100:]
            recent_avg_qualities = episode_avg_qualities[-100:]
            recent_stall_times = episode_stall_times[-100:]
            recent_avg_capacities = episode_avg_capacities[-100:]
            recent_lengths = episode_lengths[-100:]
            recent_skipped = episode_skipped[-100:] # Get recent skips
            recent_played_counts = episode_played[-100:] # Get recent played

            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            avg_length = np.mean(recent_lengths) if recent_lengths else 0
            avg_capacity_mbps = np.mean(recent_avg_capacities) if recent_avg_capacities else 0
            avg_stall_time = np.mean(recent_stall_times) if recent_stall_times else 0
            avg_skipped = np.mean(recent_skipped) if recent_skipped else 0 # Average skips
            avg_played_count = np.mean(recent_played_counts) if recent_played_counts else 0 # Average played
            avg_speed_val = np.mean(recent_avg_speeds) if recent_avg_speeds else 1.0
            avg_tau = np.mean(recent_avg_tau_i) if recent_avg_tau_i else 0
            avg_played_quality = np.mean(recent_avg_qualities) if recent_avg_qualities else 0

            # Modify print statement to include all metrics
            print(f"Ep {episode+1}/{MAX_EPISODES} | Avg R: {avg_reward:.2f} | Steps: {avg_length:.1f} | "
                  f"Cap: {avg_capacity_mbps:.2f} | Stall: {avg_stall_time:.2f} | Skip/Play: {avg_skipped:.1f}/{avg_played_count:.1f} | "
                  f"Speed: {avg_speed_val:.2f}x | τ: {avg_tau:.2f} | Qual: {avg_played_quality:.2f}")
            # --- End Update ---

            # --- Update Saving (use prefix) ---
            np.save(f'results/{results_prefix}_episode_rewards.npy', np.array(episode_rewards))
            np.save(f'results/{results_prefix}_episode_lengths.npy', np.array(episode_lengths))
            np.save(f'results/{results_prefix}_episode_avg_capacities.npy', np.array(episode_avg_capacities))
            np.save(f'results/{results_prefix}_episode_stall_times.npy', np.array(episode_stall_times))
            np.save(f'results/{results_prefix}_episode_skipped.npy', np.array(episode_skipped)) # Save skips
            np.save(f'results/{results_prefix}_episode_played.npy', np.array(episode_played))   # Save played
            np.save(f'results/{results_prefix}_episode_avg_speeds.npy', np.array(episode_avg_speeds))
            np.save(f'results/{results_prefix}_episode_avg_tau_i.npy', np.array(episode_avg_tau_i))
            np.save(f'results/{results_prefix}_episode_avg_qualities.npy', np.array(episode_avg_qualities))
            # --- End Update ---

        if (episode + 1) % 5000 == 0:
            agent.save_models(f'{models_prefix}_episode_{episode + 1}') # Use prefix

    agent.save_models(f'{models_prefix}_final') # Use prefix


# #################################
# 메인 실행 부분
# #################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["proposed", "b2", "b3"], default="proposed")
    args = parser.parse_args()

    train_combined(mode=args.mode, episode=50000)
    print(f"--- 훈련 종료 (mode={args.mode}) ---")