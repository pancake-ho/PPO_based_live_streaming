# Common/Metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import math

# 외부 설정(상수) 의존
from Common.Config import VIDEO

# -----------------------------
# Dataclasses (RX에서 import 사용)
# -----------------------------
@dataclass
class StateLOG:
    chunk_idx: int
    tau: float
    B: float
    stall: float = 0.0  # 관측 스톨(선택)

@dataclass
class ActionLOG:
    chunk_idx: int
    u_ts: float          # 절대시각(초)
    quality_level: int
    speed: float
    skip: bool

@dataclass
class ArrivalLOG:
    chunk_idx: int
    sent_ts: Optional[float]  # TX가 넣어줄 수도/없을 수도
    arrived_ts: float         # RX 수신 절대시각
    size_bytes: int = 0

# -----------------------------
# Metrics 본체
# -----------------------------
class Metrics:
    """
    - t0(절대시각)를 기준으로 모든 내부 저장은 '상대시각' [초] 로 통일
      * u_decide[i]  : 의사결정 시각 u_i  (상대)
      * arrivals[i]  : ArrivalLOG (arrived_ts 절대 보관하되 필요시 상대 변환)
      * p_start/end[i]: 재생 스케줄의 시작/종료 시각 (상대)
      * actions[i]   : ActionLOG (skip, speed, quality_level)
      * d_play[i]    : 실제 재생 소요시간(η/v) (스킵이면 0)
      * d_stall[i]   : i 청크로 인해 추가된 스톨(상대)
      * trans_delay[i]: 전송/디코드 지연(도착–결정, 스킵이면 0)
    - 상태 캐시:
      * tau_rec[i], B_rec[i]
    """
    def __init__(self) -> None:
        # 시간 기준
        self.t0: Optional[float] = None           # 절대시각
        self.chunk_secs: float = float(VIDEO.CHUNK_SEC)
        self.init_buffer: float = float(getattr(VIDEO, "INIT_BUFFERING_TIME", 4.0))

        # 로그/스케줄
        self.states: Dict[int, StateLOG] = {}
        self.actions: Dict[int, ActionLOG] = {}
        self.arrivals: Dict[int, ArrivalLOG] = {}
        self.u_decide: Dict[int, float] = {}      # 상대시각 u_i
        self.p_start: Dict[int, float] = {}       # 상대시각
        self.p_end: Dict[int, float] = {}         # 상대시각

        # 유도량
        self.d_play: Dict[int, float] = {}
        self.d_stall: Dict[int, float] = {}
        self.trans_delay: Dict[int, float] = {}   # (도착 - u_i) 양수만 저장

        # 캐시
        self.tau_rec: Dict[int, float] = {}
        self.B_rec: Dict[int, float] = {}

        # 누계
        self.total_stall: float = 0.0

    # -----------------------------
    # 내부 유틸
    # -----------------------------
    def _ensure_t0(self, u_abs: float) -> None:
        if self.t0 is None:
            self.t0 = float(u_abs)

    def _to_rel(self, abs_ts: Optional[float]) -> Optional[float]:
        if abs_ts is None:
            return None
        if self.t0 is None:
            return 0.0
        return float(abs_ts) - float(self.t0)

    def _invalidate(self, i: int) -> None:
        self.tau_rec.pop(i, None)
        self.B_rec.pop(i, None)
        # i 이후도 영향 받을 수 있으므로 보수적으로 이전 캐시도 일부 제거
        self.tau_rec.pop(i+1, None)
        self.B_rec.pop(i+1, None)

    def _last_non_skip_end_before(self, i: int) -> float:
        """
        i 이전에 '스킵이 아닌' 마지막 청크의 p_end(상대)를 반환. 없으면 초기버퍼 길이 반환.
        """
        if i <= 1:
            return float(self.init_buffer)
        for j in range(i-1, 0, -1):
            act = self.actions.get(j, None)
            if act is None or getattr(act, "skip", False):
                continue
            pe = self.p_end.get(j, None)
            if pe is not None:
                return float(pe)
        return float(self.init_buffer)

    # -----------------------------
    # 로깅/기록 API (RX가 호출)
    # -----------------------------
    def log_state(self, s: StateLOG) -> None:
        self.states[int(s.chunk_idx)] = s

    def log_action(self, a: ActionLOG) -> None:
        """
        Action 전송 직후 보관(절대 u_ts를 상대로 추가 저장)
        """
        i = int(a.chunk_idx)
        self.actions[i] = a
        if a.u_ts is not None:
            self._ensure_t0(float(a.u_ts))
            self.u_decide[i] = float(self._to_rel(float(a.u_ts)))
        self._invalidate(i)

    # RX 최신본이 `log_decision(...)`을 직접 호출하므로, 동일 기능 별칭 제공
    def log_decision(self, chunk_idx: int, u_ts: float, quality_level: int, speed: float, skip: bool) -> None:
        self.log_action(ActionLOG(chunk_idx=chunk_idx, u_ts=float(u_ts),
                                  quality_level=int(quality_level), speed=float(speed), skip=bool(skip)))

    def record_decision_time(self, i: int, u_i_abs: float) -> None:
        """
        의사결정 시각 u_i 절대→상대 기록(없으면 생성/있으면 최신화)
        """
        self._ensure_t0(float(u_i_abs))
        self.u_decide[int(i)] = float(self._to_rel(float(u_i_abs)))
        self._invalidate(int(i))

    def log_arrival(self, a: ArrivalLOG) -> None:
        """
        도착 절대시각 기록. (논문: l_i = max(0, arrived_i - u_i_abs))
        """
        i = int(a.chunk_idx)
        self.arrivals[i] = a
        # trans_delay: 스킵이 아닌 경우에만 양수로 누적
        u_rel = self.u_decide.get(i, None)
        if u_rel is not None and a.arrived_ts is not None and self.t0 is not None:
            u_abs = float(self.t0) + float(u_rel)
            td = float(a.arrived_ts) - float(u_abs)
            if td > 0:
                self.trans_delay[i] = float(td)
        self._invalidate(i)

    def record_play_start(
        self,
        i: int,
        u_i_abs: float,
        arrived_ts: Optional[float],
        skip_current: bool,
        speed: float,
    ) -> None:
        """
        논문 정합 스케줄링:
          p_start(i) = max(last_end, u_i, arrived_i)
          p_end(i)   = p_start(i) + η/v_i             (스킵이면 0-duration)
          d_stall(i) = max(0, p_start(i) - max(last_end, u_i))
        """
        i = int(i)
        self._ensure_t0(float(u_i_abs))

        u_rel = float(self._to_rel(float(u_i_abs)))
        last_end = float(self._last_non_skip_end_before(i))
        v = float(max(speed, 1e-6))
        eta = float(self.chunk_secs)

        if skip_current:
            # 즉시-완료(0 duration) 마킹: p_start==p_end==last_end
            ps = float(last_end)
            pe = float(last_end)
            d_play = 0.0
            d_stall = 0.0
            self.trans_delay[i] = 0.0
        else:
            # 도착(절대) → 상대 변환
            a_rel = self._to_rel(arrived_ts) if (arrived_ts is not None) else None
            # 시작점 후보: last_end, u_i, arrived_i(있다면)
            cand_start = max(last_end, u_rel, (float(a_rel) if a_rel is not None else -math.inf))
            ps = float(cand_start)
            d_play = float(eta / v)
            pe = float(ps + d_play)
            # u_i 이후로 추가된 stall(디코딩 지연/late arrival 포함)
            base_gate = max(last_end, u_rel)
            d_stall = max(0.0, ps - base_gate)

            # trans_delay는 도착-결정(양수만)
            if arrived_ts is not None:
                u_abs = float(self.t0) + float(u_rel)
                td = float(arrived_ts) - float(u_abs)
                self.trans_delay[i] = float(td if td > 0 else 0.0)

        # 스케줄 반영
        self.p_start[i] = float(ps)
        self.p_end[i] = float(pe)
        self.d_play[i] = float(d_play)
        self.d_stall[i] = float(d_stall)
        self.total_stall += float(d_stall)

        # 상태 캐시 무효화
        self._invalidate(i)

    # -----------------------------
    # 상태(τ, B) 계산
    # -----------------------------
    def _current_playing_index(self, u_rel: float) -> Optional[int]:
        """
        u_rel에서 재생 중인(스킵 제외) 청크 인덱스 n, 없으면 None
        """
        u = float(u_rel)
        for j, ps in self.p_start.items():
            pe = self.p_end.get(j, None)
            if ps is None or pe is None:
                continue
            act = self.actions.get(j, None)
            if (act is None) or getattr(act, "skip", False):
                continue
            if float(ps) <= u < float(pe):
                return int(j)
        return None

    def _buffer_ahead(self, u_rel: float) -> float:
        """
        u_rel 기준 '앞으로 재생이 예약된 구간' 총량(초). 스킵 구간 제외.
        """
        u = float(u_rel)
        rem = 0.0
        for j, ps in self.p_start.items():
            pe = self.p_end.get(j, None)
            if pe is None:
                continue
            act = self.actions.get(j, None)
            if (act is None) or getattr(act, "skip", False):
                continue
            psf, pef = float(ps), float(pe)
            if pef <= u:
                continue
            # 남은 부분만 누적
            rem += max(0.0, pef - max(psf, u))
        return rem

    def tau_state(self, i: int) -> float:
        """
        τ_i = (재생 헤드가 가리키는 '컨텐츠 타임라인'과 라이브 생산 타임라인의 간격)
        근사식:
            content_head(u_i) = u_i + B_i
            live_head(i)      = i * η
            τ_i               = max(0, content_head - live_head)
        - 직관: 지금 시각(u_i)에서 앞으로 재생 예약(B_i)만큼 '앞서 플레이'가 가능하므로
                컨텐츠 헤드는 (u_i + B_i) (상대축) 로 접근 가능.
        - 실측 로그와 논문식의 절충 근사로, 스킵/감속/가속 모두 보정.
        """
        i = int(i)
        u_rel = float(self.u_decide.get(i, 0.0))  # 없으면 0
        # 캐시 재사용
        if i in self.tau_rec:
            return float(self.tau_rec[i])

        B_i = float(self.buffer_state(i))
        live_head = float(i) * float(self.chunk_secs)
        content_head = float(u_rel) + float(B_i)
        tau = max(0.0, content_head - live_head)
        self.tau_rec[i] = float(tau)
        return float(tau)

    def B_state(self, i: int) -> float:
        """
        B_i = u_i 시점 이후로 예약된 '유효 재생 구간' 총량(초).
        """
        i = int(i)
        if i in self.B_rec:
            return float(self.B_rec[i])
        u_rel = float(self.u_decide.get(i, 0.0))
        B = float(self._buffer_ahead(u_rel))
        self.B_rec[i] = float(B)
        return float(B)

    # RX측 일부 호출 호환(별칭)
    def buffer_state(self, i: int) -> float:
        return self.B_state(i)

    # -----------------------------
    # 보상
    # -----------------------------
    # def reward(self, i: int) -> float:


    # -----------------------------
    # 리포트
    # -----------------------------
    def get_total_stall_time(self) -> float:
        return float(self.total_stall)
