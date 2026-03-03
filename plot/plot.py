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

import sys
import argparse
import importlib.util
import os
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from baseline.train.B4.ALVS_B4 import ActorNetwork_ALVS
from baseline.train.B5.TCLivi_B5 import ActorNetworkTCLiVi

# 유틸

def get_bitrate_mbps(env, k: int, default: float=0.8) -> float:
    """
    env.bitrate_mbps 를 우선적으로 사용하여 비트레이트(Mbps 단위) 반환하는 함수
    현재는 multi_dash_env.py 에서 bitrate 를 가져옴
    """
    # env.bitrate_map 이 존재하는 지 사전적으로 확인
    if hasattr(env, "bitrate_map") and isinstance(env.bitrate_map, dict):
        bit_map = getattr(env, "bitrate_map")

        if k in bit_map:
            return float(bit_map[k])
        
        # key 가 문자열일 경우 대비
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
    DASH 데이터셋을 로드하고, 올바른 환경(Multi_dash)을 구성하는 함수
    """
    # 1. 데이터셋 클래스 로딩 헬퍼
    def get_dataset_class(module, class_name):
        # proposed.py 확인
        if hasattr(module, class_name): return getattr(module, class_name)
        # multi_dash_env.py 확인
        try:
            import multi_dash_env
            if hasattr(multi_dash_env, class_name): return getattr(multi_dash_env, class_name)
        except ImportError: pass
        raise AttributeError(f"'{class_name}'을 찾을 수 없습니다.")

    # 2. 데이터셋 로드
    dataset = None
    if ds.kind.lower() not in ("non-dash", "none"):
        if ds.kind.lower() in ("bbb", "bigbuckbunny"):
            dataset = get_dataset_class(env_module, "BigBuckBunnyDataset")(ds.csv_path)
        elif ds.kind.lower() in ("swiss", "swiss_account"):
            dataset = get_dataset_class(env_module, "Swiss_Dataset")(ds.csv_path)
        elif ds.kind.lower() in ("elephants"):
            dataset = get_dataset_class(env_module, "Elephants_Dataset")(ds.csv_path)
        elif ds.kind.lower() in ("forest_men", "Forest_men"):
            dataset = get_dataset_class(env_module, "Forest_men_Dataset")(ds.csv_path)
        else:
            raise RuntimeError(f"존재하지 않는 데이터셋입니다: {ds.kind}")
        
    # 3. SNR 설정
    target_snr = float(snr_db) if snr_db is not None else 35.0
    
    # 4. [핵심 수정] Env Class 선택 로직 개선
    # 데이터셋이 존재하면 -> 무조건 Multi_dash_VideoStreamingEnv (inference용) 사용
    envclass = None
    
    if dataset is not None:
        # 우선 multi_dash_env 모듈에서 찾기 (가장 확실함)
        try:
            import multi_dash_env
            if hasattr(multi_dash_env, "Multi_dash_VideoStreamingEnv"):
                envclass = multi_dash_env.Multi_dash_VideoStreamingEnv
        except ImportError:
            pass
        
        # 없으면 env_module(proposed.py)에서 찾기
        if envclass is None and hasattr(env_module, "Multi_dash_VideoStreamingEnv"):
            envclass = env_module.Multi_dash_VideoStreamingEnv
            
    # 데이터셋이 없거나 위에서 못 찾은 경우 기존 로직 (Training용 Env)
    if envclass is None:
        if hasattr(env_module, "VideoStreamingEnvCombined"):
            envclass = env_module.VideoStreamingEnvCombined
        else:
            # Fallback
            import multi_dash_env
            envclass = multi_dash_env.VideoStreamingEnvCombined

    # 5. 환경 생성 (인자 구분)
    if dataset is not None:
        # Multi_dash 환경은 dash_dataset 인자를 받음
        env = envclass(
            t_buffer=float(t_buffer),
            verbose=False,
            dash_dataset=dataset
        )
    else:
        # Training 환경은 dash_dataset 인자를 받지 않음
        env = envclass(
            t_buffer=float(t_buffer),
            verbose=False
            # num_chunks 등 기본값 사용
        )

    # 6. Channel Model 업데이트
    if hasattr(env, "channel_model"):
        env.channel_model.gamma = 10 ** (target_snr / 10.0)

    # 7. Bitrate Map 설정 (데이터셋 기반)
    if dataset is not None and (not hasattr(env, "bitrate_map") or not env.bitrate_map):
        env.bitrate_map = {}
        first_chunk_idx = 1
        if hasattr(dataset, "chunk_table") and first_chunk_idx in dataset.chunk_table:
            reps = dataset.chunk_table[first_chunk_idx]
            sorted_reps = sorted(reps.items(), key=lambda x: x[1].bitrate_kbps)
            for rep_idx, info in sorted_reps:
                env.bitrate_map[rep_idx] = info.bitrate_kbps / 1000.0
                
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
        model_episode_to_load: str = None,
        model_dir='models',
        actor_ckpt: str = None,
        learn_speed: bool = True,
        learn_skip: bool = True,
        fixed_speed_idx: int = 1,
        fixed_skip: int = 0,
        greedy: bool = True,
    ):
        self.env_module = env_module
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        if hasattr(env_module, "ActorNetworkCombined"):
            self.actor = env_module.ActorNetworkCombined().to(self.device)
        else:
            raise RuntimeError("env_module 내에 ActorNetworkCombined 클래스가 정의되어 있지 않습니다.")
        
        self.model_episode = model_episode_to_load
        self.model_dir = model_dir
        self.actor_ckpt = actor_ckpt

        self._load_model()
        self.actor.eval()

        self.learn_speed = learn_speed
        self.learn_skip = learn_skip
        self.fixed_speed_idx = int(fixed_speed_idx)
        self.fixed_skip = int(fixed_skip)
        self.greedy = greedy


    def _load_model(self):
        # [수정] actor_ckpt가 직접 제공된 경우 (B2, B3 등)
        if self.actor_ckpt is not None:
            actor_path = self.actor_ckpt
        # 제공되지 않은 경우, 디렉토리와 에피소드 번호로 조합 (Proposed)
        else:
            model_name_base = f'Combined_ppo_episode_{self.model_episode}'
            actor_filename = f'{model_name_base}_actor.pth'
            actor_path = os.path.join(self.model_dir, actor_filename)

        print(f"Loading Model: {actor_path}")

        try:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            print(f"Combined Actor {actor_path} 불러오기에 성공했습니다.")
        
        except FileNotFoundError:
            print(f"오류: 파일 {actor_path} 을(를) 찾을 수 없습니다.")
            # 디버깅 힌트 제공
            if self.actor_ckpt:
                print(f"지정된 경로가 올바른지 확인하세요.")
            else:
                print(f"{self.model_dir} 디렉토리를 확인하세요.")

        except Exception as e:
            print(f"모델을 불러오는 도중 다음과 같은 오류가 발생했습니다: {e}")
            # 치명적인 오류가 아니면 넘어갈 수 있도록 exit() 제거 또는 유지 선택
            # 여기서는 파일 경로 문제일 확률이 높으므로 exit() 대신 경고만 출력하고 진행할 수도 있지만,
            # 모델 없이 진행하면 에러나니까 exit() 유지
            exit()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, state: np.ndarray, env=None) -> Tuple[int, int, int]:
        # [수정] select_action -> act 로 변경
        # [수정] env=None 인자 추가 (run_one_episode 함수와의 호환성 유지)
        
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

        # skip
        if self.learn_skip:
            skip_action_idx = int(torch.argmax(skip_probs, dim=-1).item()) # a_x
        else:
            skip_action_idx = int(self.fixed_skip) # a_x

        quality_level = quality_action_idx + 1
        return (quality_level, speed_action_idx, skip_action_idx)

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
        self.quality_head = nn.Linear(128, 20)
        self.speed_head = nn.Linear(128, 3)

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


class ALVS_PolicyWrapper:
    """
    [수정됨] B4 (ALVS) 의 로직을 완벽하게 반영한 Wrapper
    - 학습 코드(ALVS_B4.py)의 20-level Bitrate Map을 사용하여 입력 차원 고정
    - Latency 정규화 상수 수정 (15.0 -> 10.0)
    - Action Mapping: 20-level 출력을 현재 Env의 레벨로 근사 매핑
    """
    def __init__(self, ckpt_path: str, device: Optional[str]=None, greedy: bool=True):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.greedy = greedy
        
        # ALVS_B4.py에 정의된 네트워크 구조를 가져와야 함
        # (plot.py 상단에서 이미 import 되었다고 가정)
        self.actor = ActorNetwork_ALVS().to(self.device)
        self._load_ckpt(ckpt_path)

        self.actor.eval()
        self.history_len = 8

        # history 버퍼
        self.throughput_hist = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.dl_time_hist = deque([0.0] * self.history_len, maxlen=self.history_len)
        
        # 초기화: ALVS 학습 코드 기준 초기 화질 (Index 0 -> Level 1)
        self.last_quality_idx_alvs = 0 

        # --- [중요] ALVS_B4.py 학습에 사용된 20-level Bitrate Map ---
        self.alvs_bitrate_map = {
            1: 0.045,   2: 0.089,   3: 0.131,   4: 0.178,   5: 0.221,
            6: 0.262,   7: 0.334,   8: 0.396,   9: 0.522,   10: 0.595,
            11: 0.791,  12: 1.032,  13: 1.244,  14: 1.546,  15: 2.133,
            16: 2.484,  17: 3.078,  18: 3.526,  19: 3.840,  20: 4.219
        }
        self.alvs_quality_levels = sorted(list(self.alvs_bitrate_map.keys())) # 1~20

    def _load_ckpt(self, path: str):
        ckpt = torch.load(str(Path(path).resolve()), map_location="cpu")
        if isinstance(ckpt, dict) and "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])
        elif isinstance(ckpt, dict):
            self.actor.load_state_dict(ckpt)
        else:
            self.actor.load_state_dict(ckpt) # 직접 저장된 경우

    def reset(self):
        # history 버퍼 초기화
        self.throughput_hist = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.dl_time_hist = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.last_quality_idx_alvs = 0

    def _update_hist_from_env(self, env):
        """
        Env의 실행 결과를 바탕으로 History 업데이트
        """
        # Env에서 마지막 다운로드 시간 및 화질 가져오기
        l_i = float(env.l_i[-1]) if hasattr(env, "l_i") and len(env.l_i) > 0 else 0.0
        
        # 현재 Env에서 선택된 화질의 Bitrate 계산
        # (주의: Env의 화질 인덱스가 ALVS의 20단계와 다르더라도 실제 전송된 비트레이트를 기준으로 계산해야 함)
        k_env_idx = int(env.k_j[-1]) if hasattr(env, "k_j") and len(env.k_j) > 0 else 0
        
        # Env의 Bitrate 가져오기
        real_bitrate_mbps = get_bitrate_mbps(env, k_env_idx, default=0.5) # plot.py의 유틸 함수 사용
        t_i = float(getattr(env, "t_i", 2.0))
        
        chunk_bits = real_bitrate_mbps * 1_000_000.0 * t_i
        
        # Throughput (bps)
        thr = chunk_bits / max(l_i, 1e-9)
        
        self.throughput_hist.append(thr)
        self.dl_time_hist.append(l_i)

    def _build_state_dict(self, state: np.ndarray, env):
        """
        ALVS_B4 ActorNetwork에 맞는 State Dict 생성
        """
        tau = float(state[0])
        B = float(state[1])
        t_i = float(getattr(env, "t_i", 2.0))

        # --- [수정 1] Next Chunk Sizes는 항상 20개(ALVS 기준)로 고정 ---
        # 실제 Env의 화질 개수와 무관하게, 학습된 네트워크는 20개의 입력을 기대함
        next_chunk_sizes = []
        for q in self.alvs_quality_levels: # 1 ~ 20
            bitrate = self.alvs_bitrate_map[q]
            size_bytes = (bitrate * 1_000_000 * t_i) / 8.0
            next_chunk_sizes.append(size_bytes)
        
        # 남은 청크 비율
        if hasattr(env, "num_chunks") and hasattr(env, "k_j"):
            chunks_remaining_ratio = float(env.num_chunks - len(env.k_j)) / float(env.num_chunks)
        else:
            chunks_remaining_ratio = 0.0
        
        # 이전 화질의 Bitrate (ALVS 기준)
        last_bitrate = self.alvs_bitrate_map[self.last_quality_idx_alvs + 1]

        # --- [수정 2] Normalization 상수 맞춤 ---
        # ALVS_B4.py: throughput/10M, time/10.0, sizes/1M, B/10.0, tau/10.0, bitrate/5.0
        throughput_hist_norm = np.array(self.throughput_hist, dtype=np.float32) / 10_000_000.0
        dl_time_hist_norm    = np.array(self.dl_time_hist, dtype=np.float32) / 10.0
        next_chunk_sizes_norm = np.array(next_chunk_sizes, dtype=np.float32) / 1_000_000.0

        other_features = np.array([
            B / 10.0,
            tau / 10.0,      # [중요] 기존 15.0 -> 10.0 (ALVS_B4.py와 일치)
            last_bitrate / 5.0,
            chunks_remaining_ratio
        ], dtype=np.float32)

        state_dict = {
            "throughput_history": torch.tensor(throughput_hist_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            "download_time_history": torch.tensor(dl_time_hist_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            "next_chunk_sizes": torch.tensor(next_chunk_sizes_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            "other_features": torch.tensor(other_features, dtype=torch.float32, device=self.device).unsqueeze(0),
        }
        return state_dict

    def _map_action_to_env(self, alvs_action_idx: int, env) -> int:
        """
        ALVS(20-level)의 출력을 현재 Multi-DASH Env의 화질 인덱스로 매핑
        """
        # 1. ALVS가 의도한 목표 비트레이트 (Mbps)
        # self.alvs_bitrate_map은 1~20의 Key를 가짐 (이전 답변 코드 참조)
        target_bitrate = self.alvs_bitrate_map[alvs_action_idx + 1]

        # 2. 현재 Env의 비트레이트 맵 가져오기
        # multi_dash_env.py에 의해 env.bitrate_map은 {rep_idx: mbps} 형태임
        if not hasattr(env, "bitrate_map") or not env.bitrate_map:
             # Map이 없으면 안전하게 0번(혹은 1번) 인덱스 반환
            return 1 if hasattr(env, "quality_levels") and 1 in env.quality_levels else 0

        # 3. Nearest Neighbor 검색
        best_k = None
        min_diff = float('inf')

        # env.bitrate_map: {1: 0.45, 2: 0.89, ...}
        for k, br in env.bitrate_map.items():
            diff = abs(br - target_bitrate)
            if diff < min_diff:
                min_diff = diff
                best_k = k
        
        # 4. 반환 (Env에 전달할 Action Index)
        # Multi_dash_VideoStreamingEnv의 step 함수는 k_i를 받아서 self.bitrate_map[k_i]를 조회함.
        # 따라서 Key값 자체를 반환해야 함.
        return best_k

    @torch.no_grad()
    def act(self, state: np.ndarray, env) -> Tuple[int, int, int]:
        # 1. 직전의 step 결과로 history 갱신
        self._update_hist_from_env(env)

        # 2. ALVS 기준의 State Dict 생성 (20-level chunk sizes 포함)
        state_dict = self._build_state_dict(state, env)
        
        # 3. Inference
        q_logits, s_logits = self.actor(state_dict)

        q_probs = torch.softmax(q_logits, dim=-1)
        s_probs = torch.softmax(s_logits, dim=-1)

        # 4. ALVS Action (0 ~ 19) 선택
        if self.greedy:
            alvs_q_action = int(torch.argmax(q_probs, dim=-1).item())
            s_action = int(torch.argmax(s_probs, dim=-1).item())
        else:
            alvs_q_action = int(torch.distributions.Categorical(q_probs).sample().item())
            s_action = int(torch.distributions.Categorical(s_probs).sample().item())

        # 내부 추적용 업데이트 (다음 스텝의 last_bitrate 계산용)
        self.last_quality_idx_alvs = alvs_q_action

        # 5. [중요] ALVS Action -> Current Env Action 매핑
        env_k = self._map_action_to_env(alvs_q_action, env)
        
        return (env_k, s_action, 0)

class ActorNetworkTCLiVi(nn.Module):
    def __init__(self):
        super(ActorNetworkTCLiVi, self).__init__() 
        self.conv1 = nn.Conv1d(6, 32, 3)
        self.flatten_size = 32 * (48 - 3 + 1)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 40)
        
    def forward(self, state):
        x = torch.relu(self.conv1(state.permute(0, 2, 1))).view(-1, self.flatten_size)
        x = torch.relu(self.fc1(x)); return torch.softmax(self.fc2(x), dim=-1)



class TCLivi_PolicyWrapper:
    """
    [수정됨] B5 (TCLiVi) 의 학습 로직(20 Qualities, 2 Target Buffers)을 
    완벽하게 반영하고, 이종 데이터셋(BBB 등)에 적응하는 Wrapper
    """
    def __init__(self, ckpt_path: str, device: Optional[str]=None, greedy: bool=True, history_len: int=48, speed_rule_margin: float=1.0):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.greedy = greedy
        
        # [중요] TCLivi_B5.py에 정의된 ActorNetworkTCLiVi 클래스 사용
        self.network = ActorNetworkTCLiVi().to(self.device)
        self._load_ckpt(ckpt_path)
        self.network.eval()

        self.history_len = history_len
        self.speed_rule_margin = speed_rule_margin # 기본값 1.0

        # TCLivi Feature (6 dim)
        self.num_features = 6
        self.hist = deque([np.zeros(self.num_features, dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)

        # --- [1. Action Map 동기화] (Source: TCLivi_B5.py) ---
        # B5는 20개의 화질과 2개의 타겟 버퍼(2.0, 3.0) 조합으로 총 40개의 액션을 가짐
        self.b5_quality_levels = list(range(1, 21)) # 1 ~ 20
        self.b5_target_buffers = [2.0, 3.0]
        self.action_map = {}
        
        action_id = 0
        for quality in self.b5_quality_levels:
            for buf in self.b5_target_buffers:
                self.action_map[action_id] = (quality, buf)
                action_id += 1
        
        # --- [2. B5 학습용 Bitrate Map 정의] ---
        # 네트워크 입력(Chunk Size 정규화) 및 출력 매핑에 사용
        self.b5_bitrate_map = {
            1: 0.045,   2: 0.089,   3: 0.131,   4: 0.178,   5: 0.221,
            6: 0.262,   7: 0.334,   8: 0.396,   9: 0.522,   10: 0.595,
            11: 0.791,  12: 1.032,  13: 1.244,  14: 1.546,  15: 2.133,
            16: 2.484,  17: 3.078,  18: 3.526,  19: 3.840,  20: 4.219
        }

        # 초기 상태 추적용
        self.last_quality_b5_idx = 0 # 0-based index (0 -> level 1)

    def _load_ckpt(self, path: str):
        ckpt = torch.load(str(Path(path).resolve()), map_location="cpu")
        if isinstance(ckpt, dict) and "actor" in ckpt:
            self.network.load_state_dict(ckpt["actor"])
        elif isinstance(ckpt, dict):
            self.network.load_state_dict(ckpt)
        else:
            self.network.load_state_dict(ckpt)
        self.network.to(self.device)

    def reset(self):
        self.hist = deque([np.zeros(self.num_features, dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        self.last_quality_b5_idx = 0

    def _update_hist_from_env(self, state: np.ndarray, env):
        """
        Env의 상태를 B5 네트워크가 이해하는 형태(20-level 정규화 등)로 변환하여 히스토리 저장
        """
        tau = float(state[0])
        B = float(state[1])

        t_i = float(getattr(env, "t_i", 2.0))
        t_buffer = float(getattr(env, "t_buffer", 4.0))

        # 마지막 Env 결과 가져오기
        l_i = float(env.l_i[-1]) if hasattr(env, "l_i") and len(env.l_i) > 0 else 0.0
        d_i = float(env.d_i[-1]) if hasattr(env, "d_i") and len(env.d_i) > 0 else 0.0
        
        # --- [3. Chunk Size 계산 수정] ---
        # 학습 코드는 "이전 step에서 선택한 화질의 bitrate"를 기준으로 chunk size를 계산함.
        # 따라서 현재 Env의 화질이 아니라, "B5 관점에서의 화질(last_quality_b5_idx)"을 사용해야
        # 네트워크가 학습된 데이터 분포와 일치하는 입력을 받음.
        
        b5_quality_level = self.last_quality_b5_idx + 1 # 1 ~ 20
        b5_bitrate_mbps = self.b5_bitrate_map[b5_quality_level]
        
        # MB 단위 계산 (TCLivi_B5.py와 동일 로직)
        # ((Mbps * 1e6 * t_i) / 8) / 1e6 = MB
        chunk_size_MB = ((b5_bitrate_mbps * 1_000_000.0 * t_i) / 8.0) / 1e6

        # --- [4. Quality Normalization 수정] ---
        # 학습 코드는 k_i / 20.0 으로 정규화함
        k_norm = b5_quality_level / 20.0 

        features = np.array(
            [
                l_i / max(t_i, 1e-9),       # download time norm
                chunk_size_MB,              # chunk size (MB) - No norm in source? (Check source: Yes, raw MB value)
                d_i / max(t_i, 1e-9),       # stall norm
                B / max(t_buffer, 1e-9),    # buffer norm
                tau / max(t_buffer, 1e-9),  # latency norm
                k_norm                      # quality norm (0~1)
            ],
            dtype=np.float32
        )
        self.hist.append(features)

    def _map_b5_action_to_env(self, b5_k: int, env) -> int:
        """
        B5가 선택한 화질(1~20)을 현재 Env의 화질 Index로 매핑 (Nearest Neighbor)
        """
        target_bitrate = self.b5_bitrate_map[b5_k]
        
        # Env의 Bitrate Map이 없으면 안전하게 0 리턴
        if not hasattr(env, "bitrate_map") or not env.bitrate_map:
            return 0
            
        best_k = None
        min_diff = float('inf')

        for k, br in env.bitrate_map.items():
            diff = abs(br - target_bitrate)
            if diff < min_diff:
                min_diff = diff
                best_k = k
        return best_k

    @torch.no_grad()
    def act(self, state: np.ndarray, env) -> Tuple[int, int, int]:
        # 1. 히스토리 업데이트
        self._update_hist_from_env(state, env)

        # 2. Inference
        # Input Shape: (Batch, History, Features) -> Network 내부에서 Permute(0,2,1)로 (Batch, Feat, Hist) 변환됨
        hist_np = np.stack(list(self.hist), axis=0) # (48, 6)
        state_tensor = torch.tensor(hist_np, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, 48, 6)
        
        probs = self.network(state_tensor) # Output: (1, 40)
        
        if self.greedy:
            action_idx = int(torch.argmax(probs, dim=-1).item())
        else:
            action_idx = int(torch.distributions.Categorical(probs).sample().item())

        # 3. Action Decoding (0~39 -> Quality(1~20), TargetBuffer)
        b5_k, target_buffer = self.action_map[action_idx]
        
        # 내부 상태 업데이트 (다음 step의 입력 feature 생성을 위해)
        self.last_quality_b5_idx = b5_k - 1

        # 4. Speed Control (Rule-based from TCLivi_B5.py)
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

        # 5. Quality Mapping (B5 20-level -> Env N-level)
        env_k = self._map_b5_action_to_env(b5_k, env)

        return (env_k, speed_idx, 0)


# ----------------------------
# Env creation + evaluation
# ----------------------------
def write_rows_csv(out_csv: str, rows: list[dict]):
    """
    각 데이터 당 그래프를 그리기 위한 csv 파일을 만드는 함수
    """
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

    b2_path = Path("./train/proposed/baseline_models/b2/b2_25000_actor.pth")
    b3_path = Path("./train/proposed/baseline_models/b3/b3_50000_actor.pth")
    b4_path = Path("./train/B4/models/ALVS_A3C_actor_ep15000.pth")
    b5_path = Path("./train/B5/models/TCLiVi_rule_based_speed_actor_ep50000.pth")

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
        actor_ckpt=proposed_ckpt,
        learn_speed=True,
        learn_skip=True,
        greedy=True,
    )

    # Baselines
    policies: Dict[str, object] = {
        "Proposed(DASH)": proposed_policy,
        "B1": None,  # buffer-dependent (created per point)
    }

    # PPO baselines from baseline_models
    if b2_path.exists():
        policies["B2"] = Test_PPOAgent(env_module, actor_ckpt=str(b2_path), learn_speed=False, learn_skip=False, fixed_speed_idx=1, fixed_skip=0, greedy=True)
    else:
        print(f"[WARN] missing: {b2_path}")
    
    if b3_path.exists():
        policies["B3"] = Test_PPOAgent(env_module, actor_ckpt=str(b3_path), learn_speed=False, learn_skip=True, fixed_speed_idx=1, greedy=True)
    else:
        print(f"[WARN] missing: {b3_path}")

    if b4_path.exists(): # ALVS
        policies["B4"] = ALVS_PolicyWrapper(str(b4_path), greedy=True)
    else:
        print(f"[WARN] missing: {b4_path}")

    if b5_path.exists(): # TCLivi
        policies["B5"] = TCLivi_PolicyWrapper(str(b5_path), greedy=True)
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
