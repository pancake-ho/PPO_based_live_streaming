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

try:
    from proposed import DynamicChannelModel, VideoStreamingEnvCombined
    from proposed import STATE_DIM, QUALITY_LEVELS, SPEED_LEVELS, SKIP_OPTIONS
except ImportError:
    print("오류: proposed.py 파일을 찾거나 불러올 수 없습니다.")
    print("파일명을 확인하거나 구체적인 경로를 확인해주세요.")
    exit()

# --- 하이퍼파라미터 ---
LEARNING_RATE = 1e-4
GAMMA = 0.96
LAMBDA = 0.97
EPOCHS = 10
BATCH_SIZE = 128
CLIP_EPSILON = 0.2
NUM_STEPS = 1024 # Or maybe more needed for larger action space?
# MAX_EPISODES = 50000 # Or more
entropy_coef = 0.01 # May need tuning


# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# 결과 및 모델 저장을 위한 디렉토리 생성
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('baseline_models/b2', exist_ok=True)
os.makedirs('baseline_models/b3', exist_ok=True)

# #################################
# 강화학습 부분 (PPO 알고리즘 - 결합 버전)
# #################################

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
    (추가) 베이스라인 B2, B3 학습에 대비합니다.
    """
    def __init__(self, mode: str = "proposed"):
        self.actor = ActorNetworkCombined().to(device)
        self.critic = CriticNetwork().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        
        self.mode = mode
        # mode 에 따른 학습 여부 결정
        # proposed: 모두 학습
        # baseline b2: Speed 고정(1.0x), Skip 고정(No skip), bitrate 만 학습
        # baseline b3: Speed 고정(1.0x), bitrate 와 skip 학습
        self.learn_speed = True if mode == "proposed" else False
        self.learn_skip = False if mode == "b2" else True

        print(f"Agent 가 [{mode}] mode 로 학습을 수행합니다. Speed 학습 여부: {self.learn_speed}, Skip 학습 여부: {self.learn_skip}")

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
            value = self.critic(state)

            quality_dist = Categorical(quality_probs)
            quality_action = quality_dist.sample()
            log_prob_quality = quality_dist.log_prob(quality_action)

            # speed
            if self.learn_speed:
                speed_dist = Categorical(speed_probs)
                speed_action = speed_dist.sample()
                log_prob_speed = speed_dist.log_prob(speed_action)
                speed_idx = speed_action.item()
            else:
                speed_idx = 1 # 1.0x
                log_prob_speed = torch.tensor(0.0).to(device)

            # skip
            if self.learn_skip:
                skip_dist = Categorical(skip_probs)
                skip_action = skip_dist.sample()
                log_prob_skip = skip_dist.log_prob(skip_action)
                skip_value = skip_action.item()
            else:
                skip_value = 0 # No skip
                log_prob_skip = torch.tensor(0.0).to(device)

        
        action = (quality_action.item() + 1, speed_idx, skip_value)
        return action, log_prob_quality.item(), log_prob_speed.item(), log_prob_skip.item() , value.item()

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
            last_state_tensor = torch.FloatTensor(self.next_states[-1]).to(device)
            last_value = self.critic(last_state_tensor).item()

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
        
        old_log_probs = old_log_probs_quality
        if self.learn_speed: old_log_probs += old_log_probs_speed
        if self.learn_skip: old_log_probs += old_log_probs_skip

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

                new_log_probs = new_log_probs_quality

                if self.learn_speed:
                    s_dist = Categorical(s_probs)
                    new_log_probs += s_dist.log_prob(batch_actions_speed)
                    entropy = entropy + s_dist.entropy()
                
                if self.learn_skip:
                    sk_dist = Categorical(sk_probs)
                    new_log_probs += sk_dist.log_prob(batch_actions_skip)
                    entropy = entropy + sk_dist.entropy()

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

def train(mode: str = "proposed", max_episodes: int=50000):
    """메인 훈련 루프 함수."""
    env = VideoStreamingEnvCombined(num_chunks=300, verbose=False,
                                    W1=1, W2=30, W3=100.0, W_tau=0.5)
    
    agent = PPOAgentCombined(mode=mode)

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
    
    results_prefix = f"{mode}_Combined_ppo"

    prefix = f"mode_{mode}"
    print(f"--- 훈련 시작: {prefix} (Episodes: {max_episodes}) ---")

    start_time = time.time()

    for episode in range(max_episodes):
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
            print(f"Ep {episode+1}/{max_episodes} | Avg R: {avg_reward:.2f} | Steps: {avg_length:.1f} | "
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
            agent.save_models(episode=episode + 1, mode=mode) # Use prefix
    
    agent.save_models(episode="final", mode=mode)
    print(f"--- 훈련 종료. 소요 시간: {(time.time() - start_time)/60:.1f} 분 ---")

# #################################
# 메인 실행 부분
# #################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["proposed", "b2", "b3"], default="proposed")
    parser.add_argument("--episodes", type=int, default=50000)
    args = parser.parse_args()

    train(mode=args.mode, max_episodes=args.episodes)