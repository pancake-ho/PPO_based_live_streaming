from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict
import os

@dataclass(frozen=True)
class NetworkConfig:
    # 로컬 2대로 송수신 흐름 체크할 때 수정해야 할 사항들 !!!
    RX_HOST: str = os.getenv("RX_HOST", "127.0.0.1") # RX 측 IP 주소 입력하기
    RX_RTP_PORT: int = int(os.getenv("RX_RTP_PORT", "5000")) # RX 측 PORT 입력하기
    TX_HOST: str = os.getenv("TX_HOST", "127.0.0.1") # TX 측 IP 주소 입력하기
    CTRL_TX_PORT: int = int(os.getenv("CTRL_TX_PORT", "5001")) # TX 측 PORT 입력하기

    # 컨트롤 채널 (ZMQ 용 주소. UDP 쓸 경우 CTRL_TX_PORT 사용)
    CTRL_BIND: str = os.getenv("CTRL_BIND", "tcp://0.0.0.0:5555")      # Playback -> 수정할 필요 없음 !!
    CTRL_CONNECT: str = os.getenv("CTRL_CONNECT", "tcp://127.0.0.1:5555") # Live가 연결 -> 수정할 필요 없음 !!

@dataclass(frozen=True)
class VideoConfig:
    WIDTH: int = 640
    HEIGHT: int = 360
    FPS: int = 30
    CHUNK_SEC: float = 2.0
    INIT_BUFFERING_TIME: int = 4

    QUALITY_SCORES: Tuple[float, float, float, float] = (0.8, 1.2, 2.4, 4.8)
    QUALITY_LEVELS: Dict[int, float] = field(default_factory=lambda : {1: 0.8, 2: 1.2, 3: 2.4, 4: 4.8})
    SPEEDS: Tuple[float, float, float] = (0.9, 1.0, 1.1)

    @property
    def BIT_RATES_KBPS(self) -> Tuple[int, int, int, int]:
        return tuple(int(x * 1000) for x in self.QUALITY_SCORES)

@dataclass(frozen=True)
class PPOConfig:
    LEARNING_RATE = 1e-4
    GAMMA = 0.96
    LAMBDA = 0.97
    EPOCHS = 10
    BATCH_SIZE = 128
    CLIP_EPSILON = 0.2
    NUM_STEPS = 3 # 원 논문은 1024
    MAX_EPISODES = 50000
    entropy_coef = 0.01
    STATE_DIM = 2
    QUALITY_LEVELS = 4
    SPEED_LEVELS = 3
    SKIP_OPTIONS = 2

@dataclass(frozen=True)
class RewardConfig:
    W1: float = 35.0 # quality
    W2: float = 300.0 # stall 기존 300
    W3: float = 100.0 # skip
    W_TAU: float = 1.0 # 기존 1.0

def mbps_to_kbps(mbps: float) -> int:
    return int(mbps * 1000)

NETWORK = NetworkConfig()
VIDEO = VideoConfig()
PPO = PPOConfig()
REWARD = RewardConfig()