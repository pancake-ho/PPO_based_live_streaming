"""
1) B1 및 B3 (학습 없이 평가만 이용하는 베이스라인) 실행 코드 예시

python paper_baselines_b1_b6.py --env_path /mnt/data/DASH_inference.py --mode eval --baseline B1 --episodes 200
python paper_baselines_b1_b6.py --env_path /mnt/data/DASH_inference.py --mode eval --baseline B3 --episodes 200


2) B2/B4/B5/B6 (학습 이후 평가하는 베이스라인) 실행 코드 예시

# B2 학습
python paper_baselines_b1_b6.py --env_path /mnt/data/DASH_inference.py --mode train --baseline B2 --episodes 5000 --model_path ./models/b2.pt
# B2 평가
python paper_baselines_b1_b6.py --env_path /mnt/data/DASH_inference.py --mode eval --baseline B2 --episodes 200 --model_path ./models/b2.pt
"""

import os
import argparse
import importlib.util
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# 유틸
def load_module_from_path(module_path: str, module_name: str = "env_module"):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 규칙 기반 (B1, B3)
@dataclass
class BufferABRConfig:
    t1: float
    t2: float
    t3: float

class B1_BufferBasedABR:
    """
    B1: Conventional buffer-based ABR (bitrate)
    """
    def __init__(self, cfg: BufferABRConfig):
        self.cfg = cfg
    
    def reset(self):
        pass

    def act(self, state: np.ndarray, env) -> Tuple[int, int, int]:
        _, B = float(state[0]), float(state[1])
        q_levels: List[int] = list(env.quality_levels) # DASH: [0, 1, 2, 3] / CARLA: [1, 2, 3, 4]
        q_levels_sorted = sorted(q_levels)

        if B < self.cfg.t1:
            k = q_levels_sorted[0]
        elif B < self.cfg.t2:
            k = q_levels_sorted[min(1, len(q_levels_sorted) - 1)]
        elif B < self.cfg.t3:
            k = q_levels_sorted[min(2, len(q_levels_sorted) - 1)]
        else:
            k = q_levels_sorted[-1]
        
        speed_idx = int(env.playback_speeds.index(1.0))
        x = 0
        return (k, speed_idx, x)

class B3_BufferABR_WithRuleSkip(B1_BufferBasedABR):
    """
    B3: B1 + rule_based skip (만약 tau > 15 이면 스킵)
    """
    def __init__(self, cfg: BufferABRConfig, tau_thr: float = 15.0):
        super().__init__(cfg)
        self.tau_thr = tau_thr
    
    def act(self, state: np.ndarray, env) -> Tuple[int, int, int]:
        tau, _ = float(state[0]), float(state[1])
        if tau > self.tau_thr:
            q_levels_sorted = sorted(list(env.quality_levels))
            k = q_levels_sorted[0] # 비트레이트는 제일 낮게[
            speed_idx = int(env.playback_speeds.index(1.0))
            return (k, speed_idx, 1)
        
        return super().act(state, env)

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

# PPO 기반 (B2, B4, B5, B6)
@dataclass
class PPOConfig:
    lr: float = 1e-4
    gamma: float = 0.96
    lamda: float = 0.97
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    epochs: int = 10
    minibatch_size: int = 128
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

@dataclass
class ActionMask:
    learn_quality: bool
    learn_speed: bool
    learn_skip: bool
    fixed_speed_idx: int = 1
    fixed_skip: int = 0

class PPOMaskedAgent:
    """
    env 모듈의 클래스를 그대로 사용하되, 학습할 액션만 logprob/sampling 계산하게 함
    """
    def __init__(self, env_module, mask: ActionMask, ppo_cfg: PPOConfig):
        self.mask = mask
        self.cfg = ppo_cfg
        
        self.actor: nn.Module = env_module.ActorNetworkCombined()
        self.critic: nn.Module = env_module.CriticNetwork()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        self.reset()
    
    def reset(self):
        self.states: List[torch.Tensor] = []
        self.actions_k: List[int] = []
        self.actions_v: List[int] = []
        self.actions_x: List[int] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def _tensor_state(self, state: np.ndarray) -> torch.Tensor:
        st = torch.tensor(state, dtype=torch.float32, device=self.device)
        if st.dim() == 1:
            st = st.unsqueeze(0)
        return st
    
    @torch.no_grad()
    def act(self, state: np.ndarray, env, rule_skip: bool = False, tau_thr: float = 15.0) -> Tuple[int, int, int]:
        """
        rule_skip = True 이면 (B4 용) B2 처럼, tau > 15 이면 skip 강제
        이 경우 skip 은 학습하지 않음
        """
        st = self._tensor_state(state)
        q_probs, v_probs, x_probs = self.actor(st) # [1, n]

        value = self.critic(st).squeeze(-1) # [1]

        # B4
        forced_skip = 0
        if rule_skip and float(state[0]) > tau_thr:
            forced_skip = 1

        logp_sum = torch.zeros(1, device=self.device)

        # quality
        if self.mask.learn_quality:
            q_dist = Categorical(q_probs)
            action_q = int(q_dist.sample().item())
            logp_sum += q_dist.log_prob(torch.tensor(action_q, device=self.device))
        else:
            action_q = 0
        
        # speed
        if self.mask.learn_speed:
            v_dist = Categorical(v_probs)
            action_v = int(v_dist.sample().item())
            logp_sum += v_dist.log_prob(torch.tensor(action_v, device=self.device))
        else:
            action_v = int(self.mask.fixed_speed_idx)

        # skip
        if forced_skip == 1:
            action_x = 1
        else:
            if self.mask.learn_skip:
                x_dist = Categorical(x_probs)
                action_x = int(x_dist.sample().item())
                logp_sum += x_dist.log_prob(torch.tensor(action_x, device=self.device))
            else:
                action_x = int(self.mask.fixed_skip)

        q_levels_sorted = sorted(list(env.quality_levels))
        k = q_levels_sorted[action_q]

        self.states.append(st.squeeze(0))
        self.actions_k.append(action_q)
        self.actions_v.append(action_v)
        self.actions_x.append(action_x)
        self.logprobs.append(logp_sum.squeeze(0))
        self.values.append(value.squeeze(0))

        return (k, action_v, action_x)
    
    def record_reward(self, reward: float, done: bool):
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def _compute_gae(self, last_value: float = 0.0):
        rewards = self.rewards
        dones = self.dones
        values = [v.item() for v in self.values] + [last_value]

        adv = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * values[t+1] * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lamda * mask * gae
            adv[t] = gae

        returns = adv + np.array(values[:-1], dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return torch.tensor(adv, device=self.device), torch.tensor(returns, device=self.device)
    
    def _recompute_logprob(self, states: torch.Tensor, a_q: torch.Tensor, a_v: torch.Tensor, a_x: torch.Tensor,
                           rule_skip_forced: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        저장된 action 기준으로, logprob 재계산
        이때 rule_skip_forced = 1 이면 skip 은 강제
        """
        q_probs, v_probs, x_probs = self.actor(states)

        logp = torch.zeros(states.size(0), device=self.device)

        if self.mask.learn_quality:
            q_dist = Categorical(q_probs)
            logp += q_dist.log_prob(a_q)
        
        if self.mask.learn_speed:
            v_dist = Categorical(v_probs)
            logp += v_dist.log_prob(a_v)
        
        if self.mask.learn_skip:
            x_dist = Categorical(x_probs)
            lp_x = x_dist.log_prob(a_x)
            if rule_skip_forced is not None:
                lp_x = lp_x * (1.0 - rule_skip_forced.float())  # 강제 스킵 구간 제외
            logp += lp_x
        return logp
    
    def update(self, rule_skip: bool = False, tau_thr: float = 15.0):
        if len(self.rewards) == 0:
            return 
        
        states = torch.stack(self.states, dim=0).to(self.device) # [T,2]
        a_q = torch.tensor(self.actions_k, device=self.device, dtype=torch.long)
        a_v = torch.tensor(self.actions_v, device=self.device, dtype=torch.long)
        a_x = torch.tensor(self.actions_x, device=self.device, dtype=torch.long)
        old_logp = torch.stack(self.logprobs, dim=0).detach()
        old_values = torch.stack(self.values, dim=0).detach()

        forced = None
        if rule_skip:
            forced = (states[:, 0] > tau_thr).long()

        adv, returns = self._compute_gae(last_value=0.0)

        n = states.size(0)
        idxs = np.arange(n)

        for _ in range(self.cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idxs[start:start + self.cfg.minibatch_size]
                mb_states = states[mb]
                mb_aq = a_q[mb]
                mb_av = a_v[mb]
                mb_ax = a_x[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = adv[mb]
                mb_ret = returns[mb]

                # policy loss
                new_logp = self._recompute_logprob(mb_states, mb_aq, mb_av, mb_ax,
                                                   rule_skip_forced=(forced[mb] if forced is not None else None))
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # entropy (학습하는 head들만)
                q_probs, v_probs, x_probs = self.actor(mb_states)
                ent = 0.0
                if self.mask.learn_quality:
                    ent += Categorical(q_probs).entropy().mean()
                if self.mask.learn_speed:
                    ent += Categorical(v_probs).entropy().mean()
                if self.mask.learn_skip:
                    # rule skip 강제면 해당 샘플은 entropy 반영해도 큰 문제 없지만,
                    # 논문 baseline 의도에 맞게 강제 샘플은 제외
                    if forced is None:
                        ent += Categorical(x_probs).entropy().mean()
                    else:
                        e = Categorical(x_probs).entropy()
                        e = e * (1.0 - forced[mb].float())
                        ent += e.mean()

                # value loss
                values = self.critic(mb_states).squeeze(-1)
                value_loss = (mb_ret - values).pow(2).mean()

                loss = policy_loss - self.cfg.entropy_coef * ent + self.cfg.value_coef * value_loss

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.cfg.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.reset()
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# 학습 및 평가 루프
def make_env(env_module, use_dash: bool = False, dash_name: str = "bbb"):
    """
    DASH 사용은 try/except 로 처리
    """
    dash_dataset = None
    if use_dash:
        try:
            if dash_name.lower() in ("swiss", "swiss_dataset"):
                dash_dataset = env_module.Swiss_Dataset()
            elif dash_name.lower() in ("bbb", "bigbuckbunny"):
                dash_dataset = env_module.BigBuckBunnyDataset()
            elif dash_name.lower() in ("sintel", ):
                dash_dataset = env_module.sintelDataset()
        except Exception as e:
            print(f"DASH 데이터 불러오기 실패: {e}")
            dash_dataset = None
    
    if dash_dataset is None:
        env = env_module.VideoStreamingEnvCombined(
            num_chunks=300,
            t_buffer=4,
            t_i=2,
            W1=35.0,
            W2=300.0,
            W3=100.0,
            W_tau=1.0,
            verbose=False
        )
    else:
        env = env_module.VideoStreamingEnvCombined(
            W1=35.0,
            W2=300.0,
            W3=100.0,
            W_tau=1.0,
            verbose=False,
            dash_dataset=dash_dataset
        )
    return env

def evaluate_policy(env_module, mode: str, policy, episodes: int = 50, seed: int = 0,
                    use_dash: bool = False, dash_name: str = "bbb") -> Dict[str, float]:
    set_seed(seed)
    agg = {
        "avg_quality": [],
        "avg_speed": [],
        "avg_capacity": [], 
        "total_stall": [],
        "skip_rate": [],
        "episode_return": [],
    }

    for ep in range(episodes):
        env = make_env(env_module, use_dash=use_dash, dash_name=dash_name)
        state = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        
        done = False
        ep_ret = 0.0
        while not done:
            action = policy.act(state, env)
            state, r, done, _ = env.step(action)
            ep_ret += float(r)
        
        skipped, played = env.get_skip_statistics()
        agg["avg_quality"].append(float(env.get_average_quality()))
        agg["avg_speed"].append(float(env.get_average_speed()))
        agg["avg_capacity"].append(float(env.get_average_capacity()))
        agg["total_stall"].append(float(env.get_total_stall_time()))
        agg["skip_rate"].append(float(skipped / max(1, skipped + played)))
        agg["episode_return"].append(ep_ret)

    return {k: float(np.mean(v)) for k, v in agg.items()}

def train_ppo_baseline(env_module, baseline: str, total_episodes: int, seed: int,
                        model_out: str, use_dash: bool = False, dash_name: str = "bbb"):
    """
    Baseline:
        - B2: PPO bitrate-only
        - B4: PPO bitrate-only + rule-skip(tau>15)
        - B5: PPO bitrate+speed (skip=0)
        - B6: PPO bitrate+skip (speed=1.0)
    """
    set_seed(seed)

    if baseline.upper() == "B2":
        mask = ActionMask(learn_quality=True, learn_speed=False, learn_skip=False,
                          fixed_speed_idx=1, fixed_skip=0)
        rule_skip = False
    
    elif baseline.upper() == "B4":
        mask = ActionMask(learn_quality=True, learn_speed=False, learn_skip=False,
                          fixed_speed_idx=1, fixed_skip=0)
        rule_skip = True   # tau>15 이면 skip 강제

    elif baseline.upper() == "B5":
        mask = ActionMask(learn_quality=True, learn_speed=True, learn_skip=False,
                          fixed_speed_idx=1, fixed_skip=0)
        rule_skip = False

    elif baseline.upper() == "B6":
        mask = ActionMask(learn_quality=True, learn_speed=False, learn_skip=True,
                          fixed_speed_idx=1, fixed_skip=0)
        rule_skip = False

    else:
        raise ValueError(f"잘못된 PPO 베이스라인: {baseline}")
    
    agent = PPOMaskedAgent(env_module, mask=mask, ppo_cfg=PPOConfig())

    # 학습 루프
    for ep in range(1, total_episodes+1):
        env = make_env(env_module, use_dash=use_dash, dash_name=dash_name)
        state = env.reset()

        done = False
        while not done:
            action = agent.act(state, env, rule_skip=rule_skip, tau_thr=15.0)
            state, r, done, _ = env.step(action)
            agent.record_reward(r, done)
        
        agent.update(rule_skip=rule_skip, tau_thr=15.0)

        if ep % 100 == 0:
            stats = evaluate_policy(
                env_module,
                mode=baseline,
                policy=PPOEvalWrapper(agent, rule_skip=rule_skip),
                episodes=10,
                seed=seed + 1000 + ep,
                use_dash=use_dash,
                dash_name=dash_name
            )
            print(f"[{baseline}] ep={ep:5d} | return={stats['episode_return']:.3f} | "
                f"Avg_Quality={stats['avg_quality']:.3f} | stall={stats['total_stall']:.3f} | "
                f"tau-skip={stats['skip_rate']:.3f}")
    
    agent.save(model_out)
    print(f"[DONE] {baseline} 모델 저장: {model_out}")

class PPOEvalWrapper:
    """
    평가용 wrapper 로, 학습된 agent 를 이용해 행동만 뽑음
    """
    def __init__(self, agent: PPOMaskedAgent, rule_skip: bool):
        self.agent = agent
        self.rule_skip = rule_skip
    
    def reset(self):
        pass

    def act(self, state: np.ndarray, env):
        with torch.no_grad():
            st = self.agent._tensor_state(state)
            q_probs, v_probs, x_probs = self.agent.actor(st)

            forced_skip = 1 if (self.rule_skip and float(state[0]) > 15.0) else 0

            # quality
            q_dist = Categorical(q_probs)
            a_q = int(torch.argmax(q_probs, dim=-1).item())  # greedy 평가(원하면 sample로 바꿔도 됨)

            # speed
            if self.agent.mask.learn_speed:
                a_v = int(torch.argmax(v_probs, dim=-1).item())
            else:
                a_v = int(self.agent.mask.fixed_speed_idx)

            # skip
            if forced_skip == 1:
                a_x = 1
            else:
                if self.agent.mask.learn_skip:
                    a_x = int(torch.argmax(x_probs, dim=-1).item())
                else:
                    a_x = int(self.agent.mask.fixed_skip)

            q_levels_sorted = sorted(list(env.quality_levels))
            k = q_levels_sorted[a_q]
            return (k, a_v, a_x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_path", type=str, required=True,
                   help="env 경로: 'CARLA_based.py' or 'DASH_inference.py'")
    p.add_argument("--use_dash", action="store_true",
                   help="DASH 데이터 사용 여부")
    p.add_argument("--dash_name", type=str, default="bbb", help="bbb | swiss | sintel")
    p.add_argument("--mode", type=str, required=True,
                   choices=["train", "eval"],
                   help="train: B2/B4/B5/B6, eval: B1 ~ B6")
    p.add_argument("--baseline", type=str, required=True,
                   choices=["B1", "B2", "B3", "B4", "B5", "B6"])
    p.add_argument("--episodes", type=int, default=5000,
                   help="PPO 베이스라인 훈련 및 추론 에피소드 지정")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str, default="./models/baseline.pt",
                   help="PPO 베이스라인 저장 모델")
    
    args = p.parse_args()
    env_module = load_module_from_path(args.env_path, module_name="env_module")

    # 규칙 베이스라인 B1/B3
    # threshold 기본 같은 경우, 논문의 tbuf=4 에 맞춰 설정
    cfg = BufferABRConfig(t1=2.0, t2=4.0, t3=8.0)

    if args.mode == "train":
        if args.baseline.upper() not in ("B2", "B4", "B5", "B6"):
            raise ValueError("Train 모드는 PPO 베이스라인(B2/B4/B5/B6) 만 지원합니다")
        train_ppo_baseline(
            env_module=env_module,
            baseline=args.baseline.upper(),
            total_episodes=args.episodes,
            seed=args.seed,
            model_out=args.model_path,
            use_dash=args.use_dash,
            dash_name=args.dash_name
        )
        return
    
    # eval mode
    baseline = args.baseline.upper()
    if baseline == "B1":
        policy = B1_BufferBasedABR(cfg)
    elif baseline == "B3":
        policy = B3_BufferABR_WithRuleSkip(cfg, tau_thr=15.0)
    elif baseline in ("B2", "B4", "B5", "B6"):
        if baseline == "B2":
            mask = ActionMask(learn_quality=True, learn_speed=False, learn_skip=False, fixed_speed_idx=1, fixed_skip=0)
            rule_skip = False
        elif baseline == "B4":
            mask = ActionMask(learn_quality=True, learn_speed=False, learn_skip=False, fixed_speed_idx=1, fixed_skip=0)
            rule_skip = True
        elif baseline == "B5":
            mask = ActionMask(learn_quality=True, learn_speed=True, learn_skip=False, fixed_speed_idx=1, fixed_skip=0)
            rule_skip = False
        else:  # B6
            mask = ActionMask(learn_quality=True, learn_speed=False, learn_skip=True, fixed_speed_idx=1, fixed_skip=0)
            rule_skip = False
        
        agent = PPOMaskedAgent(env_module, mask=mask, ppo_cfg=PPOConfig())
        agent.load(args.model_path)
        policy = PPOEvalWrapper(agent, rule_skip=rule_skip)
    else:
        raise ValueError(f"베이스라인 선택 실패, B1 ~ B6 만 지원합니다: {baseline}")
    
    stats = evaluate_policy(
        env_module=env_module,
        mode=baseline,
        policy=policy,
        episodes=args.episodes,
        seed=args.seed,
        use_dash=args.use_dash,
        dash_name=args.dash_name,
    )
    print(f"[EVAL {baseline}] {stats}")

if __name__ == "__main__":
    main()


"""
Baseline:
    - B1: (학습 필요x) Bitrate 만 버퍼 기반 선택
    - B2: (PPO 학습 필요) Proposed 알고리즘 사용 + Bitrate 만 학습
    - B3: (PPO 학습 필요) Proposed 알고리즘 사용 + Bitrate 및 Skip 만 학습
    - B4: (A3C 학습 필요) ALVS 알고리즘 사용 + Bitrate 및 Speed 만 학습
    - B5: (PPO 학습 필요) TCLiVi 알고리즘 사용 + Bitrate 학습, Speed 는 간접적으로 학습
""" 


