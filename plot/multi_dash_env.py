import sys
import os
import torch
import numpy as np
import torch.nn as nn

from data.dash_adapter import BigBuckBunnyDataset, sintelDataset, Swiss_Dataset, Elephants_Dataset, Forest_men_Dataset
from train.proposed import DynamicChannelModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
class Multi_dash_VideoStreamingEnv:
    def __init__(self, num_chunks=300, t_buffer=4, t_i=2,
                 verbose=False, dash_dataset: Swiss_Dataset | None=None):
        
        self.W1 = 1.0
        self.W2 = 300
        self.W3 = 100
        self.W_tau = 1
        self.verbose = verbose

        self.dash_dataset = dash_dataset
        if self.dash_dataset is not None:
            self.num_chunks = self.dash_dataset.num_chunks
            self.t_i = self.dash_dataset.get_chunk_duration(1)
        else:
            self.num_chunks = num_chunks
            self.t_i = t_i
        
        self.t_buffer = t_buffer
        self.bitrate_map = {}

        if self.dash_dataset is not None:
            first_chunk_idx = 1
            # 첫 번째 청크에 대한 모든 rep 정보 가져오기
            reps_for_first = self.dash_dataset.chunk_table[first_chunk_idx]

            # bitrate 기준으로 정렬
            sorted_reps = sorted(
                reps_for_first.items(),
                key=lambda x: x[1].bitrate_kbps
            )

            rep_indices_sorted = []
            for rep_idx, info in sorted_reps:
                rep_indices_sorted.append(rep_idx)

                self.bitrate_map[rep_idx] = info.bitrate_kbps / 1000.0
            
            self.quality_levels = rep_indices_sorted
            
            # print(f"{type(self.dash_dataset).__name__} 데이터에 대한 Bitrate map 로드에 성공했습니다.")
            # print(f"\n 다음과 같습니다: {self.bitrate_map}")
        
        else:
            print("DASH 데이터셋을 로드하지 않았습니다. 코드를 확인해주세요.")
            sys.exit()
        
        self.playback_speeds = [0.9, 1.0, 1.1]
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

        self.x_j = [] # 跳过决策 (Re-introduced)
        self.cumulative_skips = 0 # (Re-introduced)
        # --- End Skip specific elements ---

        # Modify chunk_actions comment
        # Now stores {k, speed_idx, speed, x, u}

        # Reset needs to be called again to initialize new variables
        self.reset() # Call the modified reset

    def reset(self):
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
    
    def get_average_quality(self):
        """
        [수정됨] 재생된 청크들의 '인덱스'가 아닌 실제 '비트레이트(Mbps)'의 평균을 반환
        """
        # 1. 플레이된(스킵되지 않고, 재생 시작된) 청크의 인덱스들 추출
        played_indices = [self.k_j[i] for i in range(len(self.k_j))
                          if i < len(self.x_j) and self.x_j[i] == 0 # Not skipped
                          and i < len(self.p_i) and self.p_i[i] != float('inf')] # Played successfully

        if not played_indices: 
            return 0.0

        # 2. 인덱스(1~20)를 실제 비트레이트(Mbps)로 변환
        played_bitrates = []
        for k in played_indices:
            # bitrate_map에서 Mbps 값 조회 (없으면 0.0)
            # self.bitrate_map은 {1: 0.045, ..., 20: 4.219} 형태
            if hasattr(self, "bitrate_map") and k in self.bitrate_map:
                played_bitrates.append(self.bitrate_map[k])
            else:
                # 맵에 없는 경우 (예외 처리), 단순히 k를 넣거나 0 처리
                played_bitrates.append(0.0) 

        # 3. 실제 비트레이트의 평균 반환
        return sum(played_bitrates) / len(played_bitrates)

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
    
# --- Constants ---
LEARNING_RATE = 1e-4
GAMMA = 0.96
LAMBDA = 0.97
EPOCHS = 10
BATCH_SIZE = 128
CLIP_EPSILON = 0.2
NUM_STEPS = 1024 # Or maybe more needed for larger action space?
MAX_EPISODES = 50000 # Or more
entropy_coef = 0.01 # May need tuning
STATE_DIM = 2
#QUALITY_LEVELS = 4
QUALITY_LEVELS = 20
SPEED_LEVELS = 3
SKIP_OPTIONS = 2 # Added skip options

class ActorNetworkCombined(nn.Module):
    def __init__(self):
        super(ActorNetworkCombined, self).__init__()
        
        # 1. 基座网络：输出特征维度提升到 256
        self.base = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU()
        )
        
        # 2. Quality Head：输入 256 -> 128 -> 20 (QUALITY_LEVELS)
        self.quality_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),  # 修改：输入改为 256
            nn.Linear(128, QUALITY_LEVELS)
        )
        
        # 3. Speed Head：输入必须改为 256！
        self.speed_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),   # 修改：输入 64->256，隐藏层 32->64
            nn.Linear(64, SPEED_LEVELS)
        )
        
        # 4. Skip Head：输入必须改为 256！
        self.skip_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),   # 修改：输入 64->256，隐藏层 32->64
            nn.Linear(64, SKIP_OPTIONS)
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        if state.dim() == 1: state = state.unsqueeze(0)

        base_features = self.base(state)
        
        quality_logits = self.quality_head(base_features)
        quality_probs = torch.softmax(quality_logits, dim=-1)
        
        speed_logits = self.speed_head(base_features)
        speed_probs = torch.softmax(speed_logits, dim=-1)
        
        skip_logits = self.skip_head(base_features)
        skip_probs = torch.softmax(skip_logits, dim=-1)

        return quality_probs, speed_probs, skip_probs

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        # Critic 必须同步增强，否则无法准确评估复杂策略的价值
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 128),  # 64 -> 128
            nn.ReLU(),
            nn.Linear(128, 256),        # 128 -> 256
            nn.ReLU(),
            nn.Linear(256, 1)           # 输出 1 个价值
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        value = self.network(state)
        return value


env = Multi_dash_VideoStreamingEnv(dash_dataset=BigBuckBunnyDataset(csv_path="./data/bbb_2s_dataset_approx.csv"))
print(env.__init__)