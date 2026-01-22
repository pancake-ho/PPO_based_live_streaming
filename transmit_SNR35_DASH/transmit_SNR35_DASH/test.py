#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import torch
import torch.nn as nn

# 检查GPU可用性
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Changed back to cuda:0 for consistency
print(f"使用设备: {device}")

# 创建保存结果的目录
if not os.path.exists('results'):
    os.makedirs('results')

# 创建保存模型的目录
if not os.path.exists('models'):
    os.makedirs('models')

#################################
# 环境部分 (Combined Speed & Skip)
#################################

class DynamicChannelModel:
    # (Keep the DynamicChannelModel class exactly as it is in baseline5.py or Proposed_ppo.py)
    def __init__(self, W=1e6, gamma_dB=36.2, distance=20, beta=2, sigma_dB=4, mu_dB=0): # Example gamma
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
    def __init__(self, num_chunks=300, t_buffer=2, t_i=2,
                 W1=100.0, W2=250.0, W3=100.0, W_tau=1, # Added W3 back
                 verbose=False):
        """初始化结合 Skip 和 Speed 的视频流环境"""
        # Call parent __init__ from baseline5's VideoStreamingEnv
      

        self.num_chunks = num_chunks
        self.t_buffer = t_buffer
        self.t_i = t_i
        self.quality_levels = [1, 2, 3, 4]
        self.max_quality = max(self.quality_levels)
        self.verbose = verbose
        self.W1 = W1  # 质量奖励权重
        self.W2 = W2  # 卡顿惩罚权重
        # self.W3 = W3 # 跳过惩罚权重 (REMOVED)
        self.W_tau = W_tau  # 延迟惩罚权重

        # --- Playback Speed Definition ---
        #self.playback_speeds = [0.9, 1.0, 1.1]
        self.playback_speeds = [0.8, 1.0, 1.2]
        self.num_speed_levels = len(self.playback_speeds)
        # --- End Playback Speed Definition ---

        self.bitrate_map = {
            1: 0.8, 2: 1.2, 3: 2.4, 4: 4.8
        }
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
        """
        获取决定块 'target_chunk_idx' 动作之前的状态。
        状态 = [上一个块的延迟 τ_{i-1}, 此刻缓冲区剩余播放时间 B_i]
        (结构不变，但依赖的 tau_i 和 buffer calculation 已改变)
        """
        # 1. Get latency from the previous step (τ_{target-1})
        last_tau = self.t_buffer # Default for the very first chunk (i=0)
        if target_chunk_idx > 0:
            prev_idx = target_chunk_idx - 1
            if prev_idx < len(self.tau_i):
                last_tau = self.tau_i[prev_idx]
                # Handle potential infinity from previous step failure
                if last_tau == float('inf'):
                    last_tau = 10 * self.t_buffer # Use a large finite value instead of inf
            # else: tau for prev_idx doesn't exist yet (shouldn't happen if called sequentially)

        # 2. Calculate remaining buffer time at the estimated start time u_i
        # Estimate u_i for the target_chunk_idx
        chunk_generation_time = self.generated_chunk_times.get(target_chunk_idx, (target_chunk_idx + 1) * self.t_i)
        # Estimate based on previous completion or generation time
        calc_time_ui = chunk_generation_time
        if target_chunk_idx > 0:
             prev_completion = self.actual_completion_times.get(target_chunk_idx - 1, 0.0)
             # Handle inf case for prev completion
             if prev_completion == float('inf'):
                  calc_time_ui = chunk_generation_time # Fallback if previous failed
             else:
                   calc_time_ui = max(chunk_generation_time, prev_completion)
        # Ensure time monotonicity based on last actual u_i
        if self.u_i:
            calc_time_ui = max(calc_time_ui, self.u_i[-1] + 1e-9)


        remaining_buffer_time = self.calculate_remaining_play_time(calc_time_ui, target_chunk_idx)

        # Normalize or scale state? For now, return raw values.
        # Ensure values are finite for the NN
        safe_last_tau = last_tau if np.isfinite(last_tau) else 10*self.t_buffer
        safe_remaining_buffer = remaining_buffer_time if np.isfinite(remaining_buffer_time) else 0.0


        # Return state vector
        # return np.array([last_tau, remaining_buffer_time], dtype=np.float32)
        return np.array([safe_last_tau, safe_remaining_buffer], dtype=np.float32)

    def _get_reward(self, k_i, x_i, last_complete_d_i, tau_i_plus_1):
        """计算奖励"""
        quality_reward = 0
        if x_i == 0:
            # 奖励与质量的平方成正比? 或者线性? 论文中可能是线性的 q(k_i)
            # quality_reward = self.W1 * k_i
            quality_reward = self.W1 * (k_i**2) # 假设 W1 包含了基础质量的价值

        # 卡顿惩罚 (基于上一个播放块的卡顿)
        stall_penalty = self.W2 * last_complete_d_i

        # 延迟惩罚 (基于当前块计算出的延迟)
        # 论文公式: W_tau * (τ_i)^2 ? 或者 W_tau * |τ_i - τ_target| ?
        # 原代码是 W_tau * (tau_{i+1} - x_i*t_i)^2，逻辑复杂
        # 简化为惩罚当前延迟的平方
        delay_penalty = self.W_tau * (tau_i_plus_1 - self.t_i*x_i) ** 2

        # 跳过惩罚
        skip_penalty = self.W3 * x_i

        reward = quality_reward  - delay_penalty - skip_penalty - stall_penalty

        

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
# 测试参数和常量
# #################################

NUM_TEST_EPISODES = 500  # 测试的回合数
MODEL_EPISODE_TO_LOAD = 'final' # 指定要加载的模型的回合数 (例如 '30000' 或 'final')
MODEL_DIR = 'models' # 模型所在的目录
STATE_DIM = 2        # 与 propoesd2nd 训练时使用的状态维度一致
QUALITY_LEVELS = 4   # 与 propoesd2nd 训练时使用的码率等级数一致
SPEED_LEVELS = 3     # 与 propoesd2nd 训练时使用的速度等级数一致
SKIP_OPTIONS = 2     # 与 propoesd2nd 训练时使用的跳过选项数一致

# #################################
# 测试智能体 (Test Agent for Combined Model - propoesd2nd)
# #################################

class ActorNetworkCombined(nn.Module): # Renamed for clarity
    def __init__(self):
        super(ActorNetworkCombined, self).__init__()
        # Base network remains the same
        self.base = nn.Sequential(
            nn.Linear(STATE_DIM, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU()
        )
        # Quality head remains the same
        self.quality_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, QUALITY_LEVELS)
        )
        # Speed head remains the same
        self.speed_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, SPEED_LEVELS)
        )
        # --- Add Skip Head ---
        self.skip_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, SKIP_OPTIONS) # Outputs logits for skip (0 or 1)
        )
        # --- End Add Skip Head ---

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        if state.dim() == 1: state = state.unsqueeze(0)

        base_features = self.base(state)
        # Calculate all three probability sets
        quality_logits = self.quality_head(base_features)
        quality_probs = torch.softmax(quality_logits, dim=-1)
        speed_logits = self.speed_head(base_features)
        speed_probs = torch.softmax(speed_logits, dim=-1)
        skip_logits = self.skip_head(base_features) # Calculate skip logits
        skip_probs = torch.softmax(skip_logits, dim=-1) # Calculate skip probs

        return quality_probs, speed_probs, skip_probs # Return all three

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 状态价值估计
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        value = self.network(state)
        return value

class TestAgent_Combined:
    def __init__(self, model_episode_to_load, model_dir='models'):
        # 使用 propoesd2nd.py 中定义的 ActorNetworkCombined 结构
        self.actor = ActorNetworkCombined().to(device)
        self.model_episode = model_episode_to_load
        self.model_dir = model_dir
        self.load_models()
        self.actor.eval()  # 设置 Actor 为评估模式

    def load_models(self):
        """加载模型 (Combined Version) - 已修正"""
        # 使用 self.model_episode (例如 'final' 或 '30000') 来构建基础名称
        model_name_base = f'Combined_ppo_episode_{self.model_episode}'
        # 构建正确的 actor 文件名
        actor_filename = f'{model_name_base}_actor.pth'
        # 构建完整路径
        actor_path = os.path.join(self.model_dir, actor_filename)

        print(f"C: {actor_path}") # 添加打印方便调试

        try:
            # 加载模型状态字典
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
            print(f"成功加载 Combined (propoesd2nd) Actor 模型: {actor_path}")
            # 如果你的训练代码也保存了Critic模型，并且TestAgent也初始化了self.critic，
            # 那么也应该在这里加载Critic模型：
            # critic_filename = f'{model_name_base}_critic.pth'
            # critic_path = os.path.join(self.model_dir, critic_filename)
            # self.critic.load_state_dict(torch.load(critic_path, map_location=device))
            # print(f"成功加载 Combined (propoesd2nd) Critic 模型: {critic_path}")

        except FileNotFoundError:
            print(f"错误: 无法在 '{self.model_dir}' 目录下找到 Combined 模型文件 '{actor_filename}'。")
            print(f"请检查 'models' 文件夹中是否存在此文件，并确认 MODEL_EPISODE_TO_LOAD ('{self.model_episode}') 设置正确。")
            # 打印 models 目录内容方便调试 (可选)
            try:
                print(f"'models' 目录内容: {os.listdir(self.model_dir)}")
            except FileNotFoundError:
                 print(f"'models' 目录不存在！")
            except Exception as list_e:
                 print(f"无法列出 'models' 目录内容: {list_e}")
            exit()
        except Exception as e:
            print(f"加载 Combined 模型时出错: {e}")
            exit()

    def select_action(self, state):
        """在测试时选择确定性动作 (选择概率最大的动作)"""
        with torch.no_grad(): # 测试时不需要计算梯度
            state_tensor = torch.FloatTensor(state).to(device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            # Actor 网络输出码率、速度和跳过的概率
            quality_probs, speed_probs, skip_probs = self.actor(state_tensor)

            # 选择概率最高的码率索引 (0-3)
            quality_action_idx = torch.argmax(quality_probs, dim=1).item()
            # 选择概率最高的速度索引 (0-2)
            speed_action_idx = torch.argmax(speed_probs, dim=1).item()
            # 选择概率最高的跳过动作 (0 or 1)
            skip_action_idx = torch.argmax(skip_probs, dim=1).item()

        # 将码率索引转换为环境期望的级别 (1-4)
        quality_level = quality_action_idx + 1

        # 返回动作元组 (码率级别, 速度索引, 跳过索引)
        return (quality_level, speed_action_idx, skip_action_idx)

# #################################
# 测试执行函数
# #################################

def run_test_combined():
    print(f"--- 开始 Combined PPO (propoesd2nd) 测试 ({NUM_TEST_EPISODES} 回合) ---")

    # --- 实例化 Combined 环境 ---
    # !! 注意：这里的权重 W1, W2, W3, W_tau 需要与你 propoesd2nd.py 训练时使用的权重严格一致 !!
    # 根据 propoesd2nd.py 中的 train_combined 函数设置
    env = VideoStreamingEnvCombined(num_chunks=300, verbose=False,
                                    W1=35.0, W2=300.0, W3=100.0, W_tau=1.0) # 使用 propoesd2nd 中的权重

    # 创建测试智能体并加载模型
    agent = TestAgent_Combined(model_episode_to_load=MODEL_EPISODE_TO_LOAD, model_dir=MODEL_DIR)

    # --- 初始化用于存储测试结果的列表 ---
    all_rewards = []
    all_steps = []
    all_stalls = []
    all_avg_taus = []
    all_avg_qualities = []
    all_avg_speeds = []
    all_skip_counts = [] # 重新加入 Skip 统计
    all_play_counts = [] # 重新加入 Play 统计
    all_avg_capacities = []
    # --- 结束初始化 ---

    start_time = time.time()

    for i_episode in range(NUM_TEST_EPISODES):
        state = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            # 确保状态格式正确
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            if state.shape != (STATE_DIM,):
                 try:
                     state = state.reshape(STATE_DIM)
                 except ValueError as e:
                     print(f"错误：无法将状态调整为所需形状 ({STATE_DIM},)。当前状态: {state}, 形状: {state.shape}")
                     print(f"Episode {i_episode+1}, Step {steps}")
                     raise e

            # 选择确定性动作 (quality, speed, skip)
            action = agent.select_action(state)
            # 执行动作
            next_state, reward, done, _ = env.step(action)

            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)

            episode_reward += reward
            state = next_state
            steps += 1
            if done:
                break

        # --- 回合结束后，收集统计指标 ---
        total_stall = env.get_total_stall_time()
        avg_tau = env.get_average_tau_i()
        avg_quality = env.get_average_quality()
        skipped, played = env.get_skip_statistics() # 获取跳过/播放统计
        avg_speed = env.get_average_speed()
        avg_capacity = env.get_average_capacity()

        # 打印当前回合的详细指标
        print(f"Episode {i_episode+1}/{NUM_TEST_EPISODES}: Reward: {episode_reward:.2f} | Stall: {total_stall:.2f}s | Tau: {avg_tau:.2f}s | Quality: {avg_quality:.2f} | Skip/Play: {skipped}/{played} | Avg Speed: {avg_speed:.2f}x")

        # 将指标添加到列表中
        all_rewards.append(episode_reward)
        all_steps.append(steps)
        all_stalls.append(total_stall)
        all_avg_taus.append(avg_tau)
        all_avg_qualities.append(avg_quality)
        all_skip_counts.append(skipped) # 存储跳过数
        all_play_counts.append(played)   # 存储播放数
        all_avg_speeds.append(avg_speed)
        all_avg_capacities.append(avg_capacity)
        # --- 结束指标收集 ---

    end_time = time.time()
    print(f"\n--- 测试运行在 {end_time - start_time:.2f} 秒内完成 ---")

    # --- 计算并打印所有测试回合的平均性能 ---
    avg_overall_reward = np.mean(all_rewards) if all_rewards else 0
    avg_overall_steps = np.mean(all_steps) if all_steps else 0
    avg_overall_stall = np.mean(all_stalls) if all_stalls else 0
    avg_overall_tau = np.mean(all_avg_taus) if all_avg_taus else 0
    avg_overall_quality = np.mean(all_avg_qualities) if all_avg_qualities else 0
    avg_overall_skipped = np.mean(all_skip_counts) if all_skip_counts else 0 # 计算平均跳过
    avg_overall_played = np.mean(all_play_counts) if all_play_counts else 0 # 计算平均播放
    avg_overall_speed = np.mean(all_avg_speeds) if all_avg_speeds else 0
    avg_overall_capacity = np.mean(all_avg_capacities) if all_avg_capacities else 0

    print("\n--- Baseline 5 (自适应速度) 在 {} 个测试回合中的平均性能 ---".format(NUM_TEST_EPISODES))
    print(f"Avg Steps: {avg_overall_steps:.1f}")
    print(f"Avg Reward: {avg_overall_reward:.2f}")
    print(f"Avg Channel Capacity: {avg_overall_capacity:.2f} Mbps")
    print(f"Avg Total Stall Time: {avg_overall_stall:.2f} s")
    print(f"Avg Live Delay (τ): {avg_overall_tau:.2f} s")
    print(f"Avg Played Quality: {avg_overall_quality:.2f}")
    print(f"Avg Played speed: {avg_overall_speed:.2f} x") # 报告平均速度
    print(f"Avg Skip/Play Counts: {avg_overall_skipped:.1f} / {avg_overall_played:.1f}")
    # --- 结束结果打印 ---

# #################################
# 主执行部分
# #################################
if __name__ == "__main__":
    # 检查模型目录是否存在
    if not os.path.exists(MODEL_DIR):
        print(f"错误: 模型目录 '{MODEL_DIR}' 未找到。")
        print("请确保该目录存在，并且包含训练好的 Combined PPO 模型文件。")
        exit()

    # 运行测试函数
    run_test_combined()





# class Test_PPOAgent:
#     """
#     Generic PPO actor policy wrapper:
#       - loads actor weights from:
#           * .pt containing {"actor": state_dict, "critic": ...}  OR
#           * .pth state_dict directly (actor-only)
#       - supports fixing speed/skip heads (for B2/B4) or enabling them (B5/B6/Proposed)
#       - optional rule skip (B4)
#     """
#     def __init__(
#         self,
#         env_module,
#         model_episode_to_load: str,
#         model_dir='models',
#         learn_speed: bool = True,
#         learn_skip: bool = True,
#         fixed_speed_idx: int = 1,
#         fixed_skip: int = 0,
#         greedy: bool = True,
#     ):
#         self.env_module = env_module
#         self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#         self.actor = env_module.ActorNetworkCombined().to(self.device)
#         self.model_episode = model_episode_to_load
#         self.model_dir = model_dir

#         self._load_model()
#         self.actor.eval()

#         self.learn_speed = learn_speed
#         self.learn_skip = learn_skip
#         self.fixed_speed_idx = int(fixed_speed_idx)
#         self.fixed_skip = int(fixed_skip)
#         self.greedy = greedy


#     def _load_model(self, path: str):
#         model_name_base = f'Combined_ppo_episode_{self.model_episode}'
#         actor_filename = f'{model_name_base}_actor.pth'
#         actor_path = os.path.join(self.model_dir, actor_filename)

#         print(f"C: {actor_path}")

#         try:
#             self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
#             print(f"Combined Actor {actor_path} 불러오기에 성공했습니다.")
        
#         except FileNotFoundError:
#             print(f"{self.model_dir} 디렉토리에서 Combined 모델 파일 {actor_filename} 을 찾을 수 없습니다.")
#             try:
#                 print(f"모델 디렉토리 내용: {os.listdir(self.model_dir)}")
#             except FileNotFoundError:
#                 print(f"models 디렉토리가 존재하지 않습니다.")
#             except Exception as e:
#                 print(f"모델을 불러오는 도중 다음과 같은 오류가 발생했습니다: {e}")

#         except Exception as e:
#             print(f"모델을 불러오는 도중 다음과 같은 오류가 발생했습니다: {e}")
#             exit()

#     def reset(self):
#         pass

#     @torch.no_grad()
#     def select_action(self, state: np.ndarray, env) -> Tuple[int, int, int]:
#         state_tensor = torch.FloatTensor(state).to(self.device)
#         if state_tensor.dim() == 1:
#             state_tensor = state_tensor.unsqueeze(0)

#         quality_probs, speed_probs, skip_probs = self.actor(state_tensor)  # each: [1,n]

#         # quality
#         if self.greedy:
#             quality_action_idx = int(torch.argmax(quality_probs, dim=-1).item()) # a_q
#         else:
#             quality_action_idx = int(torch.distributions.Categorical(quality_probs).sample().item()) # a_q

#         # speed
#         if self.learn_speed:
#             speed_action_idx = int(torch.argmax(speed_probs, dim=-1).item()) # a_v
#         else:
#             speed_action_idx = int(self.fixed_speed_idx) # a_v

#         if self.learn_skip:
#             skip_action_idx = int(torch.argmax(skip_probs, dim=-1).item()) # a_x
#         else:
#             skip_action_idx = int(self.fixed_skip) # a_x

#         quality_level = quality_action_idx + 1

#         return (quality_level, speed_action_idx, skip_action_idx)