from __future__ import annotations
from dataclasses import dataclass
import threading
from typing import List, Tuple, Optional, Dict

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from Common.Config import VIDEO, PPO, NETWORK
from Common.Metrics import StateLOG, ActionLOG
from Playback import RX
from Live import TX
# from Env import GstVideoStreamingEnv

import json
import logging
import socket
import time
import subprocess

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GLib

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ControlSender:
    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send(self, k: int, v: float, x: int) -> None:
        payload = json.dumps({"k": int(k), "v": float(v), "x": int(x)})
        self.sock.sendto(payload.encode("utf-8"), self.addr)

class ActorNetworkCombined(nn.Module):
    def __init__(self):
        super(ActorNetworkCombined, self).__init__()
        self.base = nn.Sequential(nn.Linear(PPO.STATE_DIM, 32), nn.ReLU(), 
                                  nn.Linear(32, 64), nn.ReLU())
        self.quality_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, PPO.QUALITY_LEVELS))
        self.speed_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, PPO.SPEED_LEVELS))
        self.skip_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, PPO.SKIP_OPTIONS))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        base_features = self.base(state)
        quality_probs = torch.softmax(self.quality_head(base_features), dim=-1)
        speed_probs = torch.softmax(self.speed_head(base_features), dim=-1)
        skip_probs = torch.softmax(self.skip_head(base_features), dim=-1)
        return quality_probs, speed_probs, skip_probs

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(PPO.STATE_DIM, 64), nn.ReLU(), 
                                     nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.network(state)

class PPOAgent:
    def __init__(self) -> None:
        self.actor = ActorNetworkCombined().to(device)
        self.critic = CriticNetwork().to(device)

        lr = PPO.LEARNING_RATE
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma   = PPO.GAMMA
        self.lam     = PPO.LAMBDA
        self.clip_ep = PPO.CLIP_EPSILON
        self.ent_c   = PPO.entropy_coef
        self.batch_size   = PPO.BATCH_SIZE
        self.epochs  = PPO.EPOCHS

        # 환경 매핑
        self.rates  = list(int(r) for r in VIDEO.BIT_RATES_KBPS)
        self.speeds = list(float(s) for s in VIDEO.SPEEDS)

        self.clear_memory()
        self.update_interval = PPO.NUM_STEPS
        self.step_counter = 0

        self.last_state = None
        self.last_action = None
        self.last_logps = None
        self.last_value = None

    def select_action(self, state: np.ndarray) -> Tuple[Tuple[int, int, int], float, float, float, float]:
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            quality_probs, speed_probs, skip_probs = self.actor(state_tensor)
            quality_dist, speed_dist, skip_dist = Categorical(quality_probs), Categorical(speed_probs), Categorical(skip_probs)
            quality_action, speed_action, skip_action = quality_dist.sample(), speed_dist.sample(), skip_dist.sample()
            log_prob_q, log_prob_s, log_prob_sk = quality_dist.log_prob(quality_action).item(), speed_dist.log_prob(speed_action).item(), skip_dist.log_prob(skip_action).item()
            value = self.critic(state_tensor).item()
        return ((quality_action.item() + 1, speed_action.item(), skip_action.item()), log_prob_q, log_prob_s, log_prob_sk, value)
    

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
        self.states: List[np.ndarray] = []
        self.actions_quality: List[int] = []
        self.actions_speed: List[float] = []
        self.actions_skip: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.next_states: List[np.ndarray] = []
        self.log_probs_quality: List[float] = []
        self.log_probs_speed: List[float] = []
        self.log_probs_skip: List[float] = []

    def decide(self, tau: float, B: float) -> tuple[int, int, int]:
        state = np.array([tau, B], dtype=np.float32)
        (kvs, logp_q, logp_v, logp_x, value) = self.select_action(state)
        k_level, v_index, skip_flag = int(kvs[0]), int(kvs[1]), int(kvs[2])
        self.last_state = state
        self.last_action = (k_level, v_index, skip_flag)
        self.last_logps = (float(logp_q), float(logp_v), float(logp_x))
        self.last_value = float(value)
        return (k_level, v_index, skip_flag)

    def compute_returns_and_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages
    
    def update(self) -> None:
        returns, advantages = self.compute_returns_and_advantages()
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions_quality = torch.LongTensor(self.actions_quality).to(device)
        actions_speed = torch.LongTensor(self.actions_speed).to(device)
        actions_skip = torch.LongTensor(self.actions_skip).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs_quality).to(device) + \
                        torch.FloatTensor(self.log_probs_speed).to(device) + \
                        torch.FloatTensor(self.log_probs_skip).to(device)

        for _ in range(PPO.EPOCHS):
            indices = np.random.permutation(len(self.states))
            for start in range(0, len(self.states), self.batch_size):
                batch_indices = indices[start : start+self.batch_size]
                batch_states, batch_actions_q, batch_actions_s, batch_actions_sk = states[batch_indices], actions_quality[batch_indices], \
                                                                                actions_speed[batch_indices], actions_skip[batch_indices]
                batch_old_log_probs, batch_adv, batch_returns = old_log_probs[batch_indices], advantages[batch_indices], returns[batch_indices]

                q_probs, s_probs, sk_probs = self.actor(batch_states)
                value_probs = self.critic(batch_states).squeeze()

                q_dist, s_dist, sk_dist = Categorical(q_probs), Categorical(s_probs), Categorical(sk_probs)
                new_log_probs = q_dist.log_prob(batch_actions_q) + s_dist.log_prob(batch_actions_s) + sk_dist.log_prob(batch_actions_sk)
                entropy = (q_dist.entropy() + s_dist.entropy() + sk_dist.entropy()).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ep, 1.0 + self.clip_ep) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_c * entropy
                critic_loss = nn.MSELoss()(value_probs, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
    
    def observe(self, reward: float, tau_next: float, B_next: float, done: bool) -> bool:
        if self.last_state is None or self.last_action is None or self.last_logps is None or self.last_value is None:
            return False
        next_state = np.array([tau_next, B_next], dtype=np.float32)
        lq, lv, lx = self.last_logps
        self.store_experience(
            self.last_state, self.last_action, float(reward), next_state, bool(done),
            float(lq), float(lv), float(lx), float(self.last_value)
        )
        self.step_counter += 1
        updated = False
        if (self.step_counter % int(self.update_interval) == 0) or bool(done):
            self.compute_returns_and_advantages()
            self.update()
            self.clear_memory()
            updated = True
        if done:
            self.last_state = self.last_action = self.last_logps = self.last_value = None
        else:
            self.last_state = next_state
        return updated
    
    def save_models(self, episode: int) -> None:
        os.makedirs("models", exist_ok=True)
        model_name_base = f"GST_PPO_episode_{episode}"
        torch.save(self.actor.state_dict(), f"models/{model_name_base}_actor.pth")
        torch.save(self.critic.state_dict(), f"models/{model_name_base}_critic.pth")
        print(f"{episode} 에피소드 모델 저장 완료")
    
    def load_models(self, episode: int) -> None:
        model_name_base = f"GST_PPO_episode_{episode}"
        try:
            self.actor.load_state_dict(torch.load(f"models/{model_name_base}_actor.pth", map_location=device))
            self.critic.load_state_dict(torch.load(f"models/{model_name_base}_critic.pth", map_location=device))
            print(f"{episode} 에피소드 모델 로드 성공")
        except FileNotFoundError:
            print(f"오류: {episode} 에피소드의 모델 파일을 찾을 수 없습니다.")


def train_combined():
    agent = PPOAgent()

    episode_rewards, episode_lengths = [], []
    episode_avg_quality, episode_stall_time, episode_skip_rate = [], [], []

    logging.info("--- 훈련 시작 ---")
    start_time = time.time()
    
    for ep in range(PPO.MAX_EPISODES):
        # TX 구동
        tx_process = None
        try:
            tx_process = subprocess.Popen(
                ["python3", "TX.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logging.error(f"TX Process 불러오기 실패: {e}")
            continue
        
        # RX 프로세서
        rx = RX.RX(ctrl_host=NETWORK.TX_HOST, ctrl_port=NETWORK.CTRL_TX_PORT, show=False)
        rx.ctrl.start_listener()
        rx.ctrl.hello()
        rx.ctrl.wait_for_hello_ack(timeout=5.0)
        rx.pipeline.set_state(Gst.State.PLAYING)

        rx._bus_thread = threading.Thread(target=rx._bus_loop, daemon=True)
        rx._bus_thread.start()
                                         
        logging.info(f"Episode {ep+1}: TX/RX 파이프라인 시작")

        episode_reward = 0.0
        skip_count = 0
        quality_sum = 0.0
        played_chunks = 0
        total_stall = 0.0

        chunk_idx = 1
        rx.metrics.record_decision_time(chunk_idx, time.time())
        tau_i = float(rx.metrics.tau_state(chunk_idx))
        B_i = float(rx.metrics.buffer_state(chunk_idx))
        state = np.array([tau_i, B_i], dtype=np.float32)
        done = False

        skip_plan = {}

        while not done:
            action, logp_q, logp_v, logp_x, value = agent.select_action(state)
            k_level = int(action[0])
            v_index = int(action[1])
            skip_flag = bool(action[2])

            if float(rx.metrics.buffer_state(chunk_idx)) <= 0.0:
                skip_flag = False
            
            skip_plan[chunk_idx + 1] = skip_flag
            skip_count += int(skip_flag)

            k_level = max(1, min(k_level, len(agent.rates)))
            bitrate = int(agent.rates[k_level - 1])
            speed = float(agent.speeds[v_index])

            rx.ctrl.send_decision(chunk_idx, k_level, bitrate, speed, skip_flag)

            prev_skip = skip_plan.get(chunk_idx - 1, False)
            stall_time = 0.0 if prev_skip else float(rx.metrics.buffer_state(chunk_idx))
            rx.metrics.log_state(StateLOG(chunk_idx=chunk_idx, tau=tau_i, B=B_i, stall=stall_time))
            rx.metrics.log_action(ActionLOG(chunk_idx=chunk_idx, u_ts=time.time(), quality_level=k_level, speed=speed, skip=skip_flag))

            try:
                arrival_log = rx.metrics.arrivals.get(chunk_idx)
                arrived_ts = float(getattr(arrival_log, "arrived_ts", time.time()))
                rx.metrics.record_play_start(chunk_idx, arrived_ts, prev_skip, speed)
            except Exception as e:
                logging.warning(f"바보야 {e}")
            
            rx._tick.wait(timeout=5.0)
            rx._tick.clear()

            tau_next = float(rx.metrics.tau_state(chunk_idx + 1))
            B_next = float(rx.metrics.buffer_state(chunk_idx + 1))
            next_state = np.array([tau_next, B_next], dtype=np.float32)

            try:
                reward = float(rx.metrics.reward(chunk_idx))
            except Exception as e:
                logging.error(f"reward 계산 실패: {e}")
                reward = 0.0
            episode_reward += reward

            if not skip_flag:
                played_chunks += 1
                quality_val = bitrate if bitrate is not None else (agent.rates[k_level - 1])
                quality_sum += float(quality_val)
            if prev_skip is False:
                total_stall += float(stall_time)
            
            done = bool(getattr(rx, "_eos", False))
            action_tuple = (k_level, v_index, skip_flag)
            agent.store_experience(state, action_tuple, reward, next_state, done, 
                                    logp_q, logp_v, logp_x, value)
            
            state = next_state
            tau_i, B_i = tau_next, B_next
            chunk_idx += 1

            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(chunk_idx - 1)
        try:
            total_stall_time = float(rx.metrics.get_total_stall_time())
        except Exception:
            total_stall_time = total_stall

        episode_avg_quality.append((quality_sum / max(played_chunks, 1)))
        episode_stall_time.append(total_stall_time)
        episode_skip_rate.append(skip_count / max((chunk_idx - 1), 1))

        logging.info(f"Episode {ep+1} 완료 | Total Reward = {episode_reward:.3f}, "
                     f"Avg Quality = {episode_avg_quality[-1]:.3f} kbps, Total stall = {total_stall_time:.2f}s, Skip Rate = {episode_skip_rate[-1]:.3f}")
        
        agent.update()
        agent.clear_memory()

        rx.stop()
        if tx_process:
            tx_process.terminate()
            tx_process.wait()
        
        if (ep + 1) % 100 == 0:
            agent.save_models(ep + 1)
    
    agent.save_models('final')
    logging.info(f"훈련 완료 | {time.time() - start_time:.1f} secs")