#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
논문에 첨부하기 위한 그래프를 그리는 코드:
  Proposed(DASH) + Baselines(B1~B5)

그래프에는 4가지의 정보를 포함 (4-panel like paper):
  (a) Average quality (Mbps)
  (b) Skip rate
  (c) Total stall time (s)
  (d) Live latency (avg tau at decision times, s)

Default: Buffer sweep (Fig.3 style)
Optional: SNR sweep (Fig.4 style) by modifying env.channel_model.gamma_dB (SNR)

0111: DASH Inference 수행 버전임
0121: DASH-Training Env 버전으로 수정 (proposed)
"""

from __future__ import annotations
from collections import deque
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import argparse
import importlib.util
import os
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from baseline.train.B4.ALVS_B4 import ActorNetwork_ALVS
from baseline.train.B5.TCLivi_B5 import ActorNetworkTCLiVi

# ----------------------------
# Utils: dynamic import
# ----------------------------

def get_bitrate_mbps(env, k: int, default: float=0.8) -> float:
    """
    non-DASH: env.bitrate_map[k]
    DASH: env.group_bitrate_mbps[k]
    """
    # 기존 env 가 bitrate_map 제공하는 경우 (non-DASH)
    if hasattr(env, "bitrate_map"):
        bit_map = getattr(env, "bitrate_map")

        if isinstance(bit_map, dict):
            if k in bit_map:
                return float(bit_map[k])
            str_k = str(k)
            if str_k in bit_map:
                return float(bit_map[str_k])
    
    return float(default)


@dataclass(frozen=True)
class DatasetSpec:
    name: str # 예) "bbb", "sintel"
    kind: str # 예) "bbb", "sintel"
    csv_path: str

def parse_datasets_arg(s: str) -> list[DatasetSpec]:
    """
    Dataset 을 csv 로 변환할 때 사용하는 함수
    """
    out: list[DatasetSpec] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        kind, path = token.split(":", 1)
        kind = kind.strip()
        path = path.strip()
        out.append(DatasetSpec(name=kind, kind=kind, csv_path=path))
    if not out:
        raise ValueError("파싱된 데이터셋이 없습니다.")
    return out

def make_env(env_module, ds: DatasetSpec, t_buffer: float, snr_db: float | None):
    """
    DASH 데이터에 따라, Env 를 다르게 설정함
    """
    dataset = None
    
    if ds.kind.lower() in ("non-dash", "none"):
        dataset = None
    elif ds.kind.lower() in ("bbb", "bigbuckbunny"): # 더 추가 가능
        dataset = env_module.BigBuckBunnyDataset(ds.csv_path)
    elif ds.kind.lower() in ("swiss", "swiss_account"):
        dataset = env_module.Swiss_Dataset(ds.csv_path)
    elif ds.kind.lower() in ("elephants"):
        dataset = env_module.Elephants_Dataset(ds.csv_path)

    # Forest_men 데이터는 SD 데이터로, 
    elif ds.kind.lower() in ("forest_men", "Forest_men"):
        dataset = env_module.Forest_men_Dataset(ds.csv_path)
    else:
        raise RuntimeError(f"존재하지 않는 데이터셋입니다. 이름을 확인하세요: {ds.kind}")
    
    # snr_db 가 None 이면 기본값 35 사용
    target_snr = float(snr_db) if snr_db is not None else 35.0
    
    env = env_module.VideoStreamingEnvCombined(
        t_buffer = float(t_buffer),
        verbose=False,
        dash_dataset=dataset,
        snr_db=target_snr
    )
    if snr_db is not None:
        env.channel_model.gamma = 10 ** (target_snr / 10.0)
        
    return env


def load_module_from_path(module_path: str, module_name: str = "env_module"):
    module_path = str(Path(module_path).resolve())
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)

    # ensure relative imports (e.g., dash_adapter.py) work
    module_dir = str(Path(module_path).parent.resolve())
    if module_dir not in os.sys.path:
        os.sys.path.insert(0, module_dir)

    spec.loader.exec_module(module)
    return module


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_csv_floats(s: str) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ----------------------------
# Policies: Baselines + PPO
# ----------------------------
@dataclass
class BufferABRConfig:
    t1: float
    t2: float
    t3: float


class B1_BufferBasedABR:
    """
    (Baseline B1) Bitrate 만 버퍼 임계값 기준으로 선택, Speed 와 Skip 은 각각 1.0, 없다고 가정
    """
    def __init__(self, cfg: BufferABRConfig):
        self.cfg = cfg

    def reset(self):
        pass

    def act(self, state: np.ndarray, env) -> Tuple[int, int, int]:
        _, B = float(state[0]), float(state[1])
        q_levels_sorted = sorted(list(env.quality_levels))

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


class Test_PPOAgent:
    """
    DASH Dataset 에 맞춰, Proposed & B2, B3 에 대해
    Inference 를 진행하는 PPO Agent
    """
    def __init__(
        self,
        env_module,
        model_episode_to_load: str,
        model_dir='models',
        learn_speed: bool = True,
        learn_skip: bool = True,
        fixed_speed_idx: int = 1,
        fixed_skip: int = 0,
        greedy: bool = True,
    ):
        self.env_module = env_module
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.actor = env_module.ActorNetworkCombined().to(self.device)
        self.model_episode = model_episode_to_load
        self.model_dir = model_dir

        self._load_model()
        self.actor.eval()

        self.learn_speed = learn_speed
        self.learn_skip = learn_skip
        self.fixed_speed_idx = int(fixed_speed_idx)
        self.fixed_skip = int(fixed_skip)
        self.greedy = greedy


    def _load_model(self):
        model_name_base = f'Combined_ppo_episode_{self.model_episode}'
        actor_filename = f'{model_name_base}_actor.pth'
        actor_path = os.path.join(self.model_dir, actor_filename)

        print(f"C: {actor_path}")

        try:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            print(f"Combined Actor {actor_path} 불러오기에 성공했습니다.")
        
        except FileNotFoundError:
            print(f"{self.model_dir} 디렉토리에서 Combined 모델 파일 {actor_filename} 을 찾을 수 없습니다.")
            try:
                print(f"모델 디렉토리 내용: {os.listdir(self.model_dir)}")
            except FileNotFoundError:
                print(f"models 디렉토리가 존재하지 않습니다.")
            except Exception as e:
                print(f"모델을 불러오는 도중 다음과 같은 오류가 발생했습니다: {e}")

        except Exception as e:
            print(f"모델을 불러오는 도중 다음과 같은 오류가 발생했습니다: {e}")
            exit()

    def reset(self):
        pass

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> Tuple[int, int, int]:
        state_tensor = torch.FloatTensor(state).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        quality_probs, speed_probs, skip_probs = self.actor(state_tensor)  # each: [1,n]

        # quality
        if self.greedy:
            quality_action_idx = int(torch.argmax(quality_probs, dim=-1).item()) # a_q
        else:
            quality_action_idx = int(torch.distributions.Categorical(quality_probs).sample().item()) # a_q

        # speed
        if self.learn_speed:
            speed_action_idx = int(torch.argmax(speed_probs, dim=-1).item()) # a_v
        else:
            speed_action_idx = int(self.fixed_speed_idx) # a_v

        if self.learn_skip:
            skip_action_idx = int(torch.argmax(skip_probs, dim=-1).item()) # a_x
        else:
            skip_action_idx = int(self.fixed_skip) # a_x

        quality_level = quality_action_idx + 1
        return (quality_level, speed_action_idx, skip_action_idx)

class ALVS_PolicyWrapper:
    """
    B4 의 ActorNetwork_ALVS 를 사용하되,
    env_module 의 실행 정보로 ALVS 입력을 구성해서 Action 반환하게 하는 클래스
    """
    def __init__(self, ckpt_path: str, device: Optional[str]=None, greedy: bool=True):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.greedy = greedy
        self.network = ActorNetwork_ALVS().to(self.device)
        self.network.eval()
        self.history_len = 8

        # history 버퍼
        self.throughput_hist = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.dl_time_hist = deque([0.0] * self.history_len, maxlen=self.history_len)

        self.last_quality = 1

        self._load_ckpt(ckpt_path)
    
    def _load_ckpt(self, path: str):
        ckpt = torch.load(str(Path(path).resolve()), map_location="cpu")
        
        if isinstance(ckpt, dict) and "actor" in ckpt:
            self.network.load_state_dict(ckpt["actor"])
        
        elif isinstance(ckpt, dict):
            self.network.load_state_dict(ckpt)
        
        else:
            raise RuntimeError(f"[ALVS-B4] Path 주소를 확인해주세요: {path}")
        self.network.to(self.device)
        self.network.eval()
    
    def reset(self):
        # history 버퍼
        self.throughput_hist = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.dl_time_hist = deque([0.0] * self.history_len, maxlen=self.history_len)
        
        self.last_quality = 1

    def _update_hist_from_env(self, env):
        l_i = float(env.l_i[-1]) if hasattr(env, "l_i") and len(env.l_i) > 0 else 0.0
        k = int(env.k_j[-1]) if hasattr(env, "k_j") and len(env.k_j) > 0 else self.last_quality

        self.last_quality = k

        if hasattr(env, "bitrate_map") and hasattr(env, "t_i"):
            chunk_bits = get_bitrate_mbps(env, k) * 1_000_000.0 * float(env.t_i)
        else:
            chunk_bits = 0.0
        
        # throughput = bits / secs
        thr = chunk_bits / max(l_i, 1e-9)
        self.throughput_hist.append(thr)
        self.dl_time_hist.append(l_i)

    def _build_state_dict(self, state: np.ndarray, env):
        """
        ALVS_B4 내 forward 함수가 기대하는 state 를 구성하는 함수
        """
        tau = float(state[0])
        B = float(state[1])

        # 다음 청크 크기
        q_levels_sorted = sorted(list(env.quality_levels))
        if len(q_levels_sorted) > 0 and self.last_quality not in q_levels_sorted:
            self.last_quality = q_levels_sorted[0]

        next_chunk_sizes = []
        for k in q_levels_sorted:
            size_bytes = (get_bitrate_mbps(env, k) * 1_000_000 * float(env.t_i)) / 8.0
            next_chunk_sizes.append(size_bytes)
        
        # 남은 청크 수
        if hasattr(env, "num_chunks") and hasattr(env, "k_j"):
            chunks_remaining = float(env.num_chunks - len(env.k_j))
        else:
            chunks_remaining = 0.0
        
        default_k = q_levels_sorted[0] if len(q_levels_sorted) > 0 else self.last_quality
        last_bitrate = get_bitrate_mbps(env, self.last_quality, default=get_bitrate_mbps(env, default_k, default=0.8))


        # 정규화
        throughput_hist_norm = np.array(self.throughput_hist, dtype=np.float32) / 10_000_000.0
        dl_time_hist_norm    = np.array(self.dl_time_hist, dtype=np.float32) / 10.0
        next_chunk_sizes_norm = np.array(next_chunk_sizes, dtype=np.float32) / 1_000_000.0

        other_features = np.array([
            B / 10.0,
            tau / 15.0,
            last_bitrate / 5.0,
            (chunks_remaining / float(env.num_chunks)) if hasattr(env, "num_chunks") else 0.0
        ], dtype=np.float32)

        state_dict = {
            "throughput_history": torch.tensor(throughput_hist_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            "download_time_history": torch.tensor(dl_time_hist_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            "next_chunk_sizes": torch.tensor(next_chunk_sizes_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            "other_features": torch.tensor(other_features, dtype=torch.float32, device=self.device).unsqueeze(0),
        }
        return state_dict

    @torch.no_grad()
    def act(self, state: np.ndarray, env) -> Tuple[int, int, int]:
        # 직전의 step 결과로 history 갱신
        self._update_hist_from_env(env)

        state_dict = self._build_state_dict(state, env)
        q_logits, s_logits = self.network(state_dict)

        q_probs = torch.softmax(q_logits, dim=-1)
        s_probs = torch.softmax(s_logits, dim=-1)

        q_action = int(torch.argmax(q_probs, dim=-1).item()) if self.greedy else int(torch.distributions.Categorical(q_probs).sample().item())
        s_action = int(torch.argmax(s_probs, dim=-1).item()) if self.greedy else int(torch.distributions.Categorical(s_probs).sample().item())

        q_levels_sorted = sorted(list(env.quality_levels))
        k = q_levels_sorted[q_action]
        return (k, s_action, 0)


class TCLivi_PolicyWrapper:
    """
    B5 의 ActorNetworkTCLivi 의 action index (0~7) 을 
    이용하여 Action 반환하게 하는 클래스
    """
    def __init__(self, ckpt_path: str, device: Optional[str]=None, greedy: bool=True, history_len: int=10, speed_rule_margin: float=1.0):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.greedy = greedy
        self.network = ActorNetworkTCLiVi().to(self.device)
        self.network.eval()

        self.history_len = history_len
        self.speed_rule_margin = speed_rule_margin

        # TCLivi feature
        self.num_features = 6
        self.hist = deque([np.zeros(self.num_features, dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)

        self.action_map = {
            0: (1, 0.5), 1: (2, 0.5), 2: (3, 0.5), 3: (4, 0.5),
            4: (1, 0.75), 5: (2, 0.75), 6: (3, 0.75), 7: (4, 0.75)
        } # 정규화 적용

        self.last_quality = 1
        self._load_ckpt(ckpt_path)
    
    def _load_ckpt(self, path: str):
        ckpt = torch.load(str(Path(path).resolve()), map_location="cpu")

        if isinstance(ckpt, dict) and "actor" in ckpt:
            self.network.load_state_dict(ckpt["actor"])
        
        elif isinstance(ckpt, dict):
            self.network.load_state_dict(ckpt)
        
        else:
            raise RuntimeError(f"[TCLivi-B5] Path 주소를 확인해주세요: {path}")
        self.network.to(self.device)
        self.network.eval()
    
    def reset(self):
        self.hist = deque([np.zeros(self.num_features, dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        self.last_quality = 1
    
    def _update_hist_from_env(self, state: np.ndarray, env):
        """
        env 의 내부 기록으로 TCLivi feature 구성하는 함수
        """
        tau = float(state[0])
        B = float(state[1])

        t_i = float(getattr(env, "t_i", 2.0))
        t_buffer = float(getattr(env, "t_buffer", 4.0))

        l_i = float(env.l_i[-1]) if hasattr(env, "l_i") and len(env.l_i) > 0 else 0.0
        d_i = float(env.d_i[-1]) if hasattr(env, "d_i") and len(env.d_i) > 0 else 0.0

        k_env = int(env.k_j[-1]) if hasattr(env, "k_j") and len(env.k_j) > 0 else None
        
        if hasattr(env, "dash_dataset") and env.dash_dataset is not None:
            k_model = (k_env + 1) if k_env is not None else self.last_quality
        else:
            k_model = int(env.k_j[-1]) if k_env is not None else self.last_quality

        self.last_quality = k_model

        # chunk size
        if hasattr(env, "dash_dataset") and env.dash_dataset is not None and k_env is not None:
            bitrate_mbps = get_bitrate_mbps(env, k_env, default=0.8)
        else:
            bitrate_mbps = get_bitrate_mbps(env, k_model, default=0.8)

        chunk_size = ((bitrate_mbps * 1_000_000.0 * t_i) / 8.0) / 1e6
        k_norm = k_model / 4.0

        
        features = np.array(
            [
                l_i / max(t_i, 1e-9),
                chunk_size,
                d_i / max(t_i, 1e-9),
                B / max(t_buffer, 1e-9),
                tau / max(t_buffer, 1e-9),
                k_norm
            ],
            dtype=np.float32
        )
        self.hist.append(features)
    
    @torch.no_grad()
    def act(self, state: np.ndarray, env) -> Tuple[int, int, int]:
        # 직전 결과를 history 에 반영
        self._update_hist_from_env(state, env)

        hist_np = np.stack(list(self.hist), axis=0)
        state_tensor = torch.tensor(hist_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.network(state_tensor)
        action = int(torch.argmax(probs, dim=-1).item()) if self.greedy else int(torch.distributions.Categorical(probs).sample().item())

        k, tb_ratio = self.action_map[action]
        target_buffer = tb_ratio * float(getattr(env, "t_buffer", 4.0))

        B = float(state[1])
        slow_thr = target_buffer - self.speed_rule_margin
        fast_thr = target_buffer + self.speed_rule_margin

        if B < slow_thr:
            speed = 0.9
        elif B > fast_thr:
            speed = 1.1
        else:
            speed = 1.0
        
        speed_idx = int(env.playback_speeds.index(speed)) if hasattr(env, "playback_speeds") else 1

        # DASH Env 호환 처리
        # DASH Env 호환 처리
        if hasattr(env, "dash_dataset") and env.dash_dataset is not None:
            k_env = int(k) - 1
            if hasattr(env, "rep_groups"):
                k_env = max(0, min(k_env, len(env.rep_groups) - 1))
            k_to_step = k_env
        else:
            # [수정] Non-DASH 모드여도, 환경이 0-based 인덱스(0~3)를 사용하는 경우 변환 필요
            # DASH_inference.py는 quality_levels가 [0, 1, 2, 3]임
            if hasattr(env, "quality_levels") and 0 in env.quality_levels and 4 not in env.quality_levels:
                k_to_step = int(k) - 1  # 1~4 -> 0~3 변환
            else:
                k_to_step = int(k)      # 기존 CARLA_based.py (1~4) 호환 유지

        return (k_to_step, speed_idx, 0)
# ----------------------------
# Env creation + evaluation
# ----------------------------
def write_rows_csv(out_csv: str, rows: list[dict]):
    os.makedirs(str(Path(out_csv).parent), exist_ok=True)
    cols = ["sweep", "x", "dataset", "method",
            "avg_quality", "skip_rate", "total_stall", "live_latency", "episode_return", "avg_speed"]
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)



def run_one_episode(env, policy) -> Dict[str, float]:
    state = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    taus: List[float] = []
    done = False
    ep_ret = 0.0

    while not done:
        taus.append(float(state[0]))  # tau at decision time u_i (matches paper state definition)
        action = policy.act(state, env)
        state, r, done, _ = env.step(action)
        ep_ret += float(r)

    skipped, played = env.get_skip_statistics()
    skip_rate = float(skipped / max(1, skipped + played))

    out = {
        "avg_quality": float(env.get_average_quality()),
        "skip_rate": skip_rate,
        "total_stall": float(env.get_total_stall_time()),
        "live_latency": float(np.mean(taus) if len(taus) > 0 else 0.0),
        "episode_return": float(ep_ret),
        "avg_speed": float(env.get_average_speed()),
    }
    return out


def evaluate(env_module, ds: DatasetSpec, t_buffer: float, snr_db: Optional[float], policy, episodes: int, seed: int) -> Dict[str, float]:
    agg: Dict[str, List[float]] = {
        "avg_quality": [],
        "skip_rate": [],
        "total_stall": [],
        "live_latency": [],
        "episode_return": [],
        "avg_speed": [],
    }

    for ep in range(episodes):
        set_seed(seed + ep)
        env = make_env(env_module, ds=ds, t_buffer=t_buffer, snr_db=snr_db)
        stats = run_one_episode(env, policy)
        for k in agg:
            agg[k].append(stats[k])

    return {k: float(np.mean(v)) for k, v in agg.items()}


# ----------------------------
# Auto-find proposed model
# ----------------------------
def auto_find_proposed_ckpt(episode: int = 15000) -> Optional[str]:
    """
    Proposed actor checkpoint auto-find (safe for actor/critic separated saves).
    Priority:
      1) combined/dict-style ckpt (name does NOT include actor/critic)
      2) actor-only ckpt (name includes actor)
      3) fallback: any .pt/.pth that is not critic (last resort)
    """
    dirs = [Path("DASH_TRAIN_models"), Path("models"), Path(".")]
    episode_str = str(episode)

    # gather candidates (include extensionless patterns too)
    candidates: list[Path] = []
    for d in dirs:
        if d.exists() and d.is_dir():
            candidates += list(d.glob(f"*{episode_str}*"))
            candidates += list(d.glob(f"*{episode_str}*.pt"))
            candidates += list(d.glob(f"*{episode_str}*.pth"))

    # keep files only
    candidates = [p for p in candidates if p.is_file()]

    # 1) prefer "combined" ckpt name without actor/critic
    combined = [
        p for p in candidates
        if ("combined" in p.name.lower() or "dash" in p.name.lower())
        and ("actor" not in p.name.lower())
        and ("critic" not in p.name.lower())
    ]
    if combined:
        return str(sorted(combined)[0])

    # 2) actor-only
    actors = [p for p in candidates if "actor" in p.name.lower()]
    if actors:
        return str(sorted(actors)[0])

    # 3) last resort: anything not critic
    not_critic = [p for p in candidates if "critic" not in p.name.lower()]
    if not_critic:
        return str(sorted(not_critic)[0])

    return None



# ----------------------------
# Plotting
# ----------------------------
def plot_4panel(
    x: List[float],
    curves: Dict[str, Dict[str, List[float]]],
    xlabel: str,
    out_path: str,
    title_prefix: str,
):
    """
    curves[method][metric] = y_list aligned to x
    metrics: avg_quality, skip_rate, total_stall, live_latency
    """

    # ---- method order (legend/plot order 고정) ----
    preferred_order = ["Proposed(DASH)", "B5", "B4", "B2", "B3", "B1"]
    methods = [m for m in preferred_order if m in curves] + [m for m in curves.keys() if m not in preferred_order]

    # ---- style map (선 구분) ----
    style_map = {
        "Proposed(DASH)": dict(marker="*", linestyle="-", linewidth=3.0, markersize=10, zorder=5),
        "B5":            dict(marker="s", linestyle="--", linewidth=2.0, markersize=6),
        "B4":            dict(marker="^", linestyle="-.", linewidth=2.0, markersize=6),
        "B2":            dict(marker="D", linestyle=":", linewidth=2.0, markersize=5),
        "B3":            dict(marker="v", linestyle="--", linewidth=2.0, markersize=6),
        "B1":            dict(marker="P", linestyle=":", linewidth=2.0, markersize=5),
    }

    fig, axs = plt.subplots(2, 2, figsize=(15.5, 8.5))
    axs = axs.reshape(2, 2)

    panels = [
        ("avg_quality",  "(a) Average quality (Mbps)"),
        ("skip_rate",    "(b) Skip rate"),
        ("total_stall",  "(c) Total stall time (s)"),
        ("live_latency", "(d) Live latency (s)"),
    ]

    x_arr = np.asarray(x, dtype=float)
    xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))

    def robust_ylim(y_all: np.ndarray, *, ratio_thr: float = 6.0, pct: float = 70.0):
        """
        outlier가 일부 방법 때문에 커져도(10% 넘겨도) '저값 구간'을 메인으로 줌.
        - 기준: max가 median 대비 ratio_thr 배 이상이면 outlier로 간주
        - 메인 상한: pct 퍼센타일(기본 70%) 근처로 설정
        """
        y = y_all[np.isfinite(y_all)]
        if y.size == 0:
            return None, None, None

        y_max = float(np.max(y))
        y_med = float(np.median(y))
        y_pct = float(np.percentile(y, pct))

        eps = 1e-6
        if y_max > ratio_thr * max(y_med, eps):
            main_hi = max(y_pct * 1.20, y_med * 1.50, eps)
            full_hi = y_max * 1.05
            return 0.0, main_hi, full_hi

        return 0.0, y_max * 1.05, None


    for (metric, title), ax in zip(panels, axs.flatten()):
        # plot lines
        for m in methods:
            y = curves[m][metric]
            st = style_map.get(m, dict(marker="o", linestyle="-", linewidth=2.0, markersize=6))
            ax.plot(x, y, label=m, **st)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x)

        # ---- axis formatting / scaling ----
        y_all = np.concatenate([np.asarray(curves[m][metric], dtype=float) for m in methods], axis=0)

        if metric == "skip_rate":
            ax.set_ylim(-0.02, 0.6)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        elif metric == "avg_quality":
            ax.set_ylim(bottom=0.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        elif metric in ("total_stall", "live_latency"):
            main_lo, main_hi, full_hi = robust_ylim(y_all, ratio_thr=6.0, pct=70.0)
            ax.set_ylim(main_lo, main_hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

            # outlier 심한 경우: inset으로 전체 범위(큰 값)도 같이 표시
            if full_hi is not None:
                axins = inset_axes(ax, width="42%", height="42%", loc="upper right", borderpad=1.0)
                for m in methods:
                    y = curves[m][metric]
                    st = style_map.get(m, dict(marker="o", linestyle="-", linewidth=1.5, markersize=4))
                    axins.plot(x, y, **{**st, "linewidth": 1.2, "markersize": 4})
                axins.set_xlim(xmin, xmax)
                axins.set_ylim(0.0, full_hi)
                axins.set_xticks([])
                axins.set_yticks([])
                axins.grid(True, alpha=0.15)

                ax.text(
                    0.02, 0.02,
                    "main: zoomed (outliers in inset)",
                    transform=ax.transAxes,
                    fontsize=9,
                    alpha=0.8,
                )

        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # ---- legend: 오른쪽 바깥으로 빼서 절대 안 겹치게 ----
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(right=0.80, top=0.90, wspace=0.25, hspace=0.30)

    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.50),
        frameon=False,
    )

    fig.suptitle(title_prefix, y=0.98, fontsize=14)

    os.makedirs(str(Path(out_path).parent), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_path", type=str, required=True, help="path to DASH_inference.py")
    p.add_argument("--datasets", type=str, default="", help="DASH Datasets such as BBB, Sintel")
    p.add_argument("--dataset_csv", type=str, default="", help="BBB csv path (e.g., ./data/bbb_2s_dataset_approx.csv)")
    p.add_argument("--baseline_dir", type=str, default="./baseline_models", help="dir containing b2.pt b4.pt b5.pt b6.pt")
    p.add_argument("--episodes", type=int, default=200, help="episodes per point")
    p.add_argument("--seed", type=int, default=42)

    # buffer sweep
    p.add_argument("--buffers", type=str, default="", help="comma-separated buffer times (seconds)")

    # optional SNR sweep (gamma_dB)
    p.add_argument("--snrs", type=str, default="", help="comma-separated SNR(dB) for sweep; if empty, skip")

    # proposed model override
    p.add_argument("--dash_ckpt", type=str, default="", help="proposed checkpoint (.pt dict with actor/critic OR actor-only)")
    p.add_argument("--dash_actor", type=str, default="", help="actor-only path (.pth) if you want to force actor loading")

    args = p.parse_args()

    env_module = load_module_from_path(args.env_path, module_name="env_module")

    baseline_dir = Path(args.baseline_dir)
    b2_path = baseline_dir / "b2/b2_25000_actor.pth"
    b3_path = baseline_dir / "b3/b3_50000_actor.pth"
    b4_path = baseline_dir / "b4/ALVS_A3C_actor_ep15000.pth"
    b5_path = baseline_dir / "b5/TCLiVi_rule_based_speed_actor_ep50000.pth"

    # Proposed checkpoint auto-find unless user specifies
    proposed_ckpt = args.dash_ckpt.strip() or args.dash_actor.strip() or auto_find_proposed_ckpt(episode=50000)
    if not proposed_ckpt:
        raise RuntimeError(
            "Proposed(DASH) 모델을 자동으로 못 찾았어.\n"
            "아래처럼 경로를 직접 지정해줘:\n"
            "  --dash_ckpt ./DASH_TRAIN_models/xxx.pt   (dict)\n"
            "  또는 --dash_actor ./DASH_TRAIN_models/xxx_actor.pth (actor-only)\n"
        )
    
    # Build policies
    buffers = [float(x) for x in parse_csv_ints(args.buffers)] if args.buffers.strip() else []

    def cfg_for_buffer(tb: float) -> BufferABRConfig:
        # buffer가 바뀌면 threshold도 논리적으로 같이 스케일 (buffer=4 => 2,4,8)
        return BufferABRConfig(t1=0.5 * tb, t2=1.0 * tb, t3=2.0 * tb)

    # Proposed (learn all heads)
    proposed_policy = Test_PPOAgent(
        env_module,
        model_episode_to_load=args.episodes,
        learn_speed=True,
        learn_skip=True,
        greedy=True,
    )

    # Baselines
    policies: Dict[str, object] = {
        "Proposed(DASH)": proposed_policy,
        "B1": None,  # buffer-dependent (created per point)
    }
    tmp_env = env_module.VideoStreamingEnvCombined()
    fixed_speed_idx = int(tmp_env.playback_speeds.index(1.0))

    # PPO baselines from baseline_models
    if b2_path.exists():
        policies["B2"] = Test_PPOAgent(env_module, actor_ckpt=str(b2_path), learn_speed=False, learn_skip=False, fixed_speed_idx=fixed_speed_idx, fixed_skip=0, greedy=True)
    else:
        print(f"[WARN] missing: {b2_path}")
    
    if b3_path.exists():
        policies["B3"] = Test_PPOAgent(env_module, actor_ckpt=str(b3_path), learn_speed=False, learn_skip=True, fixed_speed_idx=fixed_speed_idx, greedy=True)
    else:
        print(f"[WARN] missing: {b3_path}")

    if b4_path.exists(): # ALVS
        policies["B4"] = ALVS_PolicyWrapper(str(b4_path), greedy=True)
    else:
        print(f"[WARN] missing: {b4_path}")

    if b5_path.exists(): # TCLivi
        policies["B5"] = TCLivi_PolicyWrapper(str(b5_path), greedy=True, history_len=48)
    else:
        print(f"[WARN] missing: {b5_path}")

    if args.datasets.strip():
        datasets = parse_datasets_arg(args.datasets)
    elif args.dataset_csv.strip():
        datasets = [DatasetSpec(name="bbb", kind="bbb", csv_path=args.dataset_csv)]
    else:
        print("데이터셋 경로가 없습니다. Non-DASH 모드로 Inference 실행합니다.")
        datasets = [DatasetSpec(name="none", kind="none", csv_path="")]

    out_dir = Path("results")
    # -----------------------
    # Buffer sweep (Fig.3 style)
    # -----------------------
    if len(buffers) > 0:
        curves_buf: Dict[str, Dict[str, List[float]]] = {}
        method_names = list(policies.keys())

        # initialize
        for m in method_names:
            curves_buf[m] = {"avg_quality": [], "skip_rate": [], "total_stall": [], "live_latency": []}

        rows = [] # csv 용

        for tb in buffers:
            # buffer-based methods must be re-instantiated with scaled thresholds
            cfg = cfg_for_buffer(tb)
            policies["B1"] = B1_BufferBasedABR(cfg)

            print(f"\n[BUFFER={tb:.1f}s] evaluating...")
            for m in method_names:
                pol = policies[m]
                if pol is None:
                    continue

                # per-dataset 결과
                per_ds_stats = []
                for dsi, ds in enumerate(datasets):
                    stats = evaluate(
                        env_module=env_module,
                        ds=ds,
                        t_buffer=tb,
                        snr_db=None,
                        policy=pol,
                        episodes=args.episodes,
                        seed=args.seed + 1000 * dsi,
                    )
                    per_ds_stats.append(stats)

                    rows.append({
                        "sweep": "buffer",
                        "x": tb,
                        "dataset": ds.name,
                        "method": m,
                        **stats,
                    })
                
                # dataset 평균
                mean_stats = {k: float(np.mean([s[k] for s in per_ds_stats])) for k in per_ds_stats[0].keys()}
                
                curves_buf[m]["avg_quality"].append(mean_stats["avg_quality"])
                curves_buf[m]["skip_rate"].append(mean_stats["skip_rate"])
                curves_buf[m]["total_stall"].append(mean_stats["total_stall"])
                curves_buf[m]["live_latency"].append(mean_stats["live_latency"])

                print(f"  {m:14s} | Q={mean_stats['avg_quality']:.3f}Mbps | skip={mean_stats['skip_rate']:.3f} | stall={mean_stats['total_stall']:.3f}s | tau={mean_stats['live_latency']:.3f}s | speed={mean_stats['avg_speed']:.3f}x")


        plot_4panel(
            x=buffers,
            curves=curves_buf,
            xlabel="Buffer time (s)",
            out_path=str(out_dir / "fig_buffer_sweep_non_dash.png"),
            title_prefix=f"BBB (DASH) - Buffer sweep (episodes={args.episodes})",
        )
        print(f"\n[SAVED] {out_dir / 'fig_buffer_sweep_dash.png'}")
    
    else:
        method_names = list(policies.keys())
        rows = []

    # -----------------------
    # Optional SNR sweep (Fig.4 style)
    # -----------------------
    snrs = parse_csv_floats(args.snrs)
    snrs.sort()
    if len(snrs) > 0:
        if len(buffers) > 0:
            tb0 = 4.0 if 4.0 in buffers else float(buffers[0])
        else:
            tb0 = 10.0
        cfg = cfg_for_buffer(tb0)
        policies["B1"] = B1_BufferBasedABR(cfg)

        curves_snr: Dict[str, Dict[str, List[float]]] = {
            m: {"avg_quality": [], "skip_rate": [], "total_stall": [], "live_latency": []}
            for m in method_names
        }

        for snr_db in snrs:
            snr_db = float(snr_db)

            print(f"\n[SNR={snr_db:.1f}dB | buffer={tb0:.1f}s] evaluating...")
            for m in method_names:
                pol = policies[m]
                if pol is None:
                    continue

                per_ds_stats = []
                for dsi, ds in enumerate(datasets):
                    stats = evaluate(
                        env_module=env_module,
                        ds=ds,
                        t_buffer=tb0,
                        snr_db=snr_db,
                        policy=pol,
                        episodes=args.episodes,
                        seed=args.seed + 1000 * dsi,
                    )
                    per_ds_stats.append(stats)

                    rows.append({
                        "sweep": "snr",
                        "x": snr_db,
                        "dataset": ds.name,
                        "method": m,
                        **stats,
                    })

                mean_stats = {k: float(np.mean([s[k] for s in per_ds_stats])) for k in per_ds_stats[0].keys()}
                curves_snr[m]["avg_quality"].append(mean_stats["avg_quality"])
                curves_snr[m]["skip_rate"].append(mean_stats["skip_rate"])
                curves_snr[m]["total_stall"].append(mean_stats["total_stall"])
                curves_snr[m]["live_latency"].append(mean_stats["live_latency"])

                print(f"  {m:14s} | Q={mean_stats['avg_quality']:.3f}Mbps | skip={mean_stats['skip_rate']:.3f} | "
                    f"stall={mean_stats['total_stall']:.3f}s | tau={mean_stats['live_latency']:.3f}s")

        plot_4panel(
            x=snrs,
            curves=curves_snr,
            xlabel="Transmit SNR (dB)",
            out_path=str(out_dir / "fig_snr_sweep_bbb_non_dash.png"),
            title_prefix=f"BBB (DASH) - SNR sweep @ buffer={tb0:.1f}s (episodes={args.episodes})",
        )
        print(f"\n[SAVED] {out_dir / 'fig_snr_sweep_bbb_dash.png'}")

    write_rows_csv("results/multi_datasets_csv.csv", rows)


if __name__ == "__main__":
    main()
