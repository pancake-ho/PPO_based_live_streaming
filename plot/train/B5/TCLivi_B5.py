#!/usr/bin/env python
# coding: utf-8

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

# GPU check (Aligned with proposed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- TCLiVi 베이스라인 ---")
print(f"사용 장치: {device}")

if not os.path.exists('results'): os.makedirs('results')
if not os.path.exists('models'): os.makedirs('models')

#################################
# Environment (Modified to match Proposed Logic)
#################################

class DynamicChannelModel:
    """Channel Model (Same as Proposed)"""
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


class VideoStreamingEnv_TCLiVi_Unified:
    """
    TCLivi Environment modified to support Unified Actions (Quality, Speed, Skip)
    and Unified Reward.
    """
    def __init__(self, num_chunks=300, t_buffer=4, t_i=2,
                 W1=1.0, W2=30.0, W3=100.0, W_tau=0.5, # Weights matched to Proposed
                 history_len=10, verbose=False):
        self.num_chunks, self.t_buffer, self.t_i = num_chunks, t_buffer, t_i
        
        # --- Unified Definitions ---
        self.quality_levels = list(range(1, 21))
        self.bitrate_map = {
            1: 0.045,   2: 0.089,   3: 0.131,   4: 0.178,   5: 0.221,
            6: 0.262,   7: 0.334,   8: 0.396,   9: 0.522,   10: 0.595,
            11: 0.791,  12: 1.032,  13: 1.244,  14: 1.546,  15: 2.133,
            16: 2.484,  17: 3.078,  18: 3.526,  19: 3.840,  20: 4.219
        }
        self.playback_speeds = [0.9, 1.0, 1.1]
        self.num_speed_levels = len(self.playback_speeds)
        
        # Reward Weights
        self.W1, self.W2, self.W3, self.W_tau = W1, W2, W3, W_tau
        
        # TCLivi Specific: History State
        self.history_len = history_len
        self.num_history_features = 6 # [l_i, bit_rate, d_i, buffer, tau, k_i]
        self.chunk_history = deque(maxlen=self.history_len)
        
        self.channel_model = DynamicChannelModel()
        self.reset()

    def reset(self):
        self.interval_capacities, self.actual_completion_times, self.chunk_actions = {}, {}, {}
        self.generated_chunk_times = {i: (i + 1) * self.t_i for i in range(self.num_chunks)}
        
        self.u_i, self.p_i, self.d_i, self.l_i, self.tau_i = [], [], [], [], []
        self.k_j, self.speed_idx_j, self.x_j = [], [], [] # Track actions
        
        # Reset History State
        self.chunk_history.clear()
        for _ in range(self.history_len):
            self.chunk_history.append(np.zeros(self.num_history_features, dtype=np.float32))
        
        # Initial State
        self.state = np.array(self.chunk_history, dtype=np.float32)
        return self.state

    def _get_capacity_for_interval(self, i):
        if i < 0: return 0
        return self.interval_capacities.setdefault(i, self.channel_model.calculate_capacity(True))

    def _simulate_chunk_transmission(self, chunk_idx, k_i, start_time):
        # Using the Unified Bitrate Map
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

    def calculate_remaining_play_time(self, calculation_time, decision_chunk_idx):
        """Unified Buffer Calculation Logic"""
        remaining_current, future_buffered, playing_idx = 0.0, 0.0, -1
        # 1. Currently playing chunk
        for j in range(decision_chunk_idx):
            action_info = self.chunk_actions.get(j)
            if action_info and action_info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                play_start_time, chunk_speed = self.p_i[j], action_info.get('speed', 1.0)
                play_end_time = play_start_time + self.t_i / chunk_speed
                if play_start_time <= calculation_time < play_end_time:
                    remaining_current, playing_idx = max(0, play_end_time - calculation_time), j; break
        # 2. Future buffered
        for j in range(playing_idx + 1, decision_chunk_idx):
            action_info = self.chunk_actions.get(j)
            # Must be received, not skipped, and not started yet
            if action_info and action_info.get('x', 0) == 0:
                if self.actual_completion_times.get(j, float('inf')) <= calculation_time:
                    if (j < len(self.p_i) and self.p_i[j] > calculation_time) or j >= len(self.p_i):
                         future_buffered += self.t_i / action_info.get('speed', 1.0)
        return remaining_current + future_buffered

    def step(self, action_tuple):
        """
        Modified Step Function accepting (Quality, Speed, Skip)
        Aligned with Proposed Logic.
        """
        k_i, speed_idx, x_i = action_tuple # Unified Actions
        current_playback_speed = self.playback_speeds[speed_idx]
        chunk_idx = len(self.u_i)
        
        if chunk_idx >= self.num_chunks: return self.reset(), 0.0, True, {}

        # 1. Store Actions
        self.k_j.append(k_i); self.speed_idx_j.append(speed_idx); self.x_j.append(x_i)
        
        # 2. Calculate u_i (Start Time)
        chunk_generation_time = self.generated_chunk_times.get(chunk_idx, (chunk_idx + 1) * self.t_i)
        prev_completion_time = 0.0
        if chunk_idx > 0:
            prev_info = self.chunk_actions.get(chunk_idx - 1)
            if prev_info and prev_info.get('x', 0) == 1: # If prev skipped, use its decision time
                prev_completion_time = prev_info.get('u', 0.0)
            else:
                prev_completion_time = self.actual_completion_times.get(chunk_idx - 1, 0.0)
                if prev_completion_time == float('inf'): prev_completion_time = 0.0
                
        u_i = max(chunk_generation_time, prev_completion_time)
        if self.u_i: u_i = max(u_i, self.u_i[-1] + 1e-9)
        self.u_i.append(u_i)
        self.chunk_actions[chunk_idx] = {'k': k_i, 'speed_idx': speed_idx, 'speed': current_playback_speed, 'x': x_i, 'u': u_i}

        # 3. Simulate Transmission
        l_i, completion_time = 0.0, u_i
        if x_i == 0:
            l_i, completion_time = self._simulate_chunk_transmission(chunk_idx, k_i, u_i)
        
        self.l_i.append(l_i)
        self.actual_completion_times[chunk_idx] = completion_time

        # 4. Calculate Play Start (p_i) and Stall (d_i)
        p_i, d_i = float('inf'), 0.0
        if x_i == 0 and l_i != float('inf'):
            ready_time = completion_time
            prev_played_idx = -1
            for j in range(chunk_idx - 1, -1, -1):
                a_info = self.chunk_actions.get(j)
                if a_info and a_info.get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                    prev_played_idx = j; break
            
            if prev_played_idx != -1:
                prev_end = self.p_i[prev_played_idx] + self.t_i / self.chunk_actions[prev_played_idx]['speed']
                p_i = max(ready_time, prev_end)
                d_i = max(0, ready_time - prev_end)
            else:
                p_i = max(ready_time, self.t_buffer)
                d_i = max(0, ready_time - self.t_buffer)
        
        self.p_i.append(p_i); self.d_i.append(d_i)

        # 5. Calculate Latency (tau)
        tau_i = float('inf')
        prev_tau = self.tau_i[chunk_idx-1] if chunk_idx > 0 and chunk_idx-1 < len(self.tau_i) else self.t_buffer
        
        if x_i == 0 and p_i != float('inf'):
            tau_i = p_i - chunk_idx * self.t_i
        elif x_i == 1:
            tau_i = prev_tau # Inherit
            
        self.tau_i.append(tau_i)

        # 6. Calculate Reward Inputs (Unified)
        # Latency prediction for next step (tau_{i+1})
        tau_i_plus_1 = float('inf')
        if x_i == 0 and tau_i != float('inf'):
            time_saved = self.t_i - (self.t_i / current_playback_speed)
            tau_i_plus_1 = tau_i + d_i - time_saved
        elif x_i == 1 and prev_tau != float('inf'):
            tau_i_plus_1 = prev_tau - self.t_i

        # Last complete stall (for penalty)
        last_complete_d_i = 0.0
        if chunk_idx > 0:
            last_played = -1
            for j in range(chunk_idx - 1, -1, -1):
                if self.chunk_actions.get(j, {}).get('x', 0) == 0 and j < len(self.p_i) and self.p_i[j] != float('inf'):
                    last_played = j; break
            if last_played != -1 and last_played < len(self.d_i):
                last_complete_d_i = self.d_i[last_played]

        reward = self._get_reward(k_i, x_i, last_complete_d_i, tau_i_plus_1)

        # 7. Update State (TCLivi Specific History State)
        buffer_val = self.calculate_remaining_play_time(u_i, chunk_idx+1)
        # Normalize features
        norm_l = l_i / self.t_i if l_i != float('inf') else 2.0
        norm_bitrate = self.bitrate_map[k_i] / 4.219 # Max bitrate roughly
        norm_d = d_i / self.t_i if d_i != float('inf') else 1.0
        norm_buf = buffer_val / self.t_buffer
        norm_tau = tau_i / self.t_buffer if tau_i != float('inf') else 2.0
        norm_k = k_i / 20.0
        
        history_entry = np.array([norm_l, norm_bitrate, norm_d, norm_buf, norm_tau, norm_k], dtype=np.float32)
        self.chunk_history.append(history_entry)
        next_state = np.array(self.chunk_history, dtype=np.float32)
        
        done = chunk_idx >= self.num_chunks - 1
        return next_state, reward, done, {}

    def _get_reward(self, k_i, x_i, last_complete_d_i, tau_i_plus_1):
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

    # Stats getters
    def get_average_quality(self):
        played = [self.k_j[i] for i in range(len(self.k_j)) if self.x_j[i] == 0 and i < len(self.p_i) and self.p_i[i] != float('inf')]
        return np.mean(played) if played else 0
    def get_total_stall_time(self): return sum([d for d in self.d_i if d != float('inf')])
    def get_average_speed(self): 
        speeds = [self.playback_speeds[i] for i in self.speed_idx_j]
        return np.mean(speeds) if speeds else 1.0
    def get_average_tau(self): 
        taus = [t for t in self.tau_i if t != float('inf')]
        return np.mean(taus) if taus else 0.0
    def get_skip_count(self): return sum(self.x_j)


#################################
# PPO Algorithm (Modified for Unified Actions)
#################################
LEARNING_RATE, GAMMA, LAMBDA, EPOCHS, BATCH_SIZE, CLIP_EPSILON = 1e-4, 0.96, 0.97, 10, 128, 0.2
NUM_STEPS, MAX_EPISODES, ENTROPY_COEF = 1024, 50000, 0.01
HISTORY_LEN, NUM_FEATURES = 10, 6

# Constants for Unified Actions
QUALITY_LEVELS = 20
SPEED_LEVELS = 3
SKIP_OPTIONS = 2

class ActorNetworkTCLiVi_Unified(nn.Module):
    """
    TCLivi Actor modified to have Multi-Head Output (Quality, Speed, Skip)
    while keeping the Convolutional History Input structure.
    """
    def __init__(self):
        super(ActorNetworkTCLiVi_Unified, self).__init__()
        # Input processing (Same as original TCLivi: Conv1d for history)
        self.conv1 = nn.Conv1d(NUM_FEATURES, 32, 3) 
        self.flatten_size = 32 * 8 # (10 - 3 + 1) = 8 output width
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 256) # Increased size for multi-head
        
        # Unified Heads
        self.quality_head = nn.Linear(256, QUALITY_LEVELS)
        self.speed_head = nn.Linear(256, SPEED_LEVELS)
        self.skip_head = nn.Linear(256, SKIP_OPTIONS)

    def forward(self, state):
        # state: [Batch, History, Features] -> Permute for Conv1d: [Batch, Features, History]
        x = torch.relu(self.conv1(state.permute(0, 2, 1))).view(-1, self.flatten_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        q_probs = torch.softmax(self.quality_head(x), dim=-1)
        s_probs = torch.softmax(self.speed_head(x), dim=-1)
        x_probs = torch.softmax(self.skip_head(x), dim=-1)
        return q_probs, s_probs, x_probs

class CriticNetworkTCLiVi(nn.Module):
    """Critic remains largely similar, just updated hidden layers"""
    def __init__(self):
        super(CriticNetworkTCLiVi, self).__init__()
        self.conv1 = nn.Conv1d(NUM_FEATURES, 32, 3)
        self.flatten_size = 32 * 8
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 1)
    def forward(self, state):
        x = torch.relu(self.conv1(state.permute(0, 2, 1))).view(-1, self.flatten_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class PPOAgent_TCLiVi_Unified:
    def __init__(self):
        self.actor = ActorNetworkTCLiVi_Unified().to(device)
        self.critic = CriticNetworkTCLiVi().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.clear_memory()

    def clear_memory(self):
        self.states, self.next_states, self.dones, self.rewards, self.values = [], [], [], [], []
        self.actions_q, self.actions_s, self.actions_x = [], [], []
        self.log_probs_q, self.log_probs_s, self.log_probs_x = [], [], []

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_probs, s_probs, x_probs = self.actor(state_tensor)
            
            dist_q, dist_s, dist_x = Categorical(q_probs), Categorical(s_probs), Categorical(x_probs)
            action_q, action_s, action_x = dist_q.sample(), dist_s.sample(), dist_x.sample()
            
            value = self.critic(state_tensor)
        
        # Return tuple of values, tuple of log_probs, value
        actions = (action_q.item() + 1, action_s.item(), action_x.item()) # Q is 1-20
        log_probs = (dist_q.log_prob(action_q).item(), dist_s.log_prob(action_s).item(), dist_x.log_prob(action_x).item())
        return actions, log_probs, value.item()

    def store_experience(self, state, action_tuple, reward, next_state, done, log_prob_tuple, value):
        self.states.append(state)
        self.actions_q.append(action_tuple[0] - 1) # Store index 0-19
        self.actions_s.append(action_tuple[1])
        self.actions_x.append(action_tuple[2])
        self.rewards.append(reward); self.next_states.append(next_state); self.dones.append(done)
        self.log_probs_q.append(log_prob_tuple[0]); self.log_probs_s.append(log_prob_tuple[1]); self.log_probs_x.append(log_prob_tuple[2])
        self.values.append(value)

    def update(self):
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions_q = torch.LongTensor(self.actions_q).to(device)
        actions_s = torch.LongTensor(self.actions_s).to(device)
        actions_x = torch.LongTensor(self.actions_x).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        dones = torch.FloatTensor(self.dones).to(device)
        values = torch.FloatTensor(self.values).to(device)
        
        # Combine old log probs
        old_log_probs = torch.FloatTensor(self.log_probs_q).to(device) + \
                        torch.FloatTensor(self.log_probs_s).to(device) + \
                        torch.FloatTensor(self.log_probs_x).to(device)

        # GAE
        with torch.no_grad():
            last_val = self.critic(torch.FloatTensor(self.next_states[-1]).unsqueeze(0).to(device)).item()
        advantages = torch.zeros_like(rewards).to(device)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t+1] if t < len(rewards)-1 else last_val
            delta = rewards[t] + GAMMA * next_val * (1-dones[t]) - values[t]
            last_gae = delta + GAMMA * LAMBDA * (1-dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Epochs
        for _ in range(EPOCHS):
            indices = np.arange(len(self.states)); np.random.shuffle(indices)
            for start in range(0, len(self.states), BATCH_SIZE):
                idx = indices[start:start+BATCH_SIZE]
                
                # New Probs
                q_probs, s_probs, x_probs = self.actor(states[idx])
                dist_q, dist_s, dist_x = Categorical(q_probs), Categorical(s_probs), Categorical(x_probs)
                
                new_lp = dist_q.log_prob(actions_q[idx]) + dist_s.log_prob(actions_s[idx]) + dist_x.log_prob(actions_x[idx])
                entropy = (dist_q.entropy() + dist_s.entropy() + dist_x.entropy()).mean()
                
                ratio = torch.exp(new_lp - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages[idx]
                
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy
                critic_loss = nn.MSELoss()(self.critic(states[idx]).squeeze(), returns[idx])
                
                self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
        self.clear_memory()

    def save_models(self, episode, prefix="TCLiVi_Unified"):
        torch.save(self.actor.state_dict(), f'models/{prefix}_actor_ep{episode}.pth')
        print(f"Model Saved: Episode {episode}")

#################################
# Training Loop
#################################
def train_tclivi_unified():
    env = VideoStreamingEnv_TCLiVi_Unified(num_chunks=300, history_len=HISTORY_LEN)
    agent = PPOAgent_TCLiVi_Unified()
    
    # Metrics
    ep_rewards, ep_qualities, ep_stalls, ep_taus, ep_speeds, ep_skips = [], [], [], [], [], []
    
    for episode in range(MAX_EPISODES):
        state = env.reset()
        ep_r = 0
        for step in range(env.num_chunks):
            # Select Action (Tuple)
            action, log_probs, value = agent.select_action(state)
            # Step
            next_state, reward, done, _ = env.step(action)
            # Store
            agent.store_experience(state, action, reward, next_state, done, log_probs, value)
            
            state = next_state
            ep_r += reward
            if done: break
            
        if len(agent.states) >= NUM_STEPS: agent.update()
        
        # Logging
        ep_rewards.append(ep_r)
        ep_qualities.append(env.get_average_quality())
        ep_stalls.append(env.get_total_stall_time())
        ep_taus.append(env.get_average_tau())
        ep_speeds.append(env.get_average_speed())
        ep_skips.append(env.get_skip_count())
        
        if (episode + 1) % 100 == 0:
            print(f"Ep {episode+1}/{MAX_EPISODES} | R: {np.mean(ep_rewards[-100:]):.2f} | "
                  f"Q: {np.mean(ep_qualities[-100:]):.2f} | Stall: {np.mean(ep_stalls[-100:]):.2f}s | "
                  f"Tau: {np.mean(ep_taus[-100:]):.2f}s | Skip: {np.mean(ep_skips[-100:]):.1f}")
            
            # Save stats (Simplified saving for brevity)
            np.save('results/TCLivi_Unified_rewards.npy', ep_rewards)
            
        if (episode + 1) % 5000 == 0:
            agent.save_models(episode + 1)
            
    agent.save_models("final")

#################################
# Testing Loop (Unified)
#################################
def test_tclivi_unified(episode_to_load, test_episodes=10):
    print(f"\n--- Testing TCLiVi Unified (Ep: {episode_to_load}) ---")
    env = VideoStreamingEnv_TCLiVi_Unified(num_chunks=300, history_len=HISTORY_LEN)
    agent = PPOAgent_TCLiVi_Unified()
    
    try:
        path = f'models/TCLiVi_Unified_actor_ep{episode_to_load}.pth'
        agent.actor.load_state_dict(torch.load(path, map_location=device))
        agent.actor.eval()
    except Exception as e: print(f"Load failed: {e}"); return

    stats = {'q': [], 'stall': [], 'tau': [], 'speed': [], 'skip': []}
    
    for ep in range(test_episodes):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_p, s_p, x_p = agent.actor(st)
                # Greedy Action for Test
                act = (torch.argmax(q_p).item()+1, torch.argmax(s_p).item(), torch.argmax(x_p).item())
            state, _, done, _ = env.step(act)
            
        stats['q'].append(env.get_average_quality())
        stats['stall'].append(env.get_total_stall_time())
        stats['tau'].append(env.get_average_tau())
        stats['skip'].append(env.get_skip_count())
        print(f"Test Ep {ep+1} done", end='\r')

    print(f"\nResults over {test_episodes} eps:")
    print(f"Avg Q: {np.mean(stats['q']):.3f} | Stall: {np.mean(stats['stall']):.3f} | "
          f"Tau: {np.mean(stats['tau']):.3f} | Skips: {np.mean(stats['skip']):.1f}")

if __name__ == "__main__":
    train_tclivi_unified()
    test_tclivi_unified("final")

