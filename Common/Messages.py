from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union

SCHEMA_VERSION: Tuple[int, int] = (2, 0) 

class Role(str, Enum):
    LIVE = "live" # 송신
    PLAYBACK = "playback" # 수신

class MessageType(str, Enum):
    """
    HELLO -> STATE -> DECISION -> ARRIVAL -> REWARD -> BYE
    1) HELLO 로 세션 시작
    2) STATE 및 DECISION 으로 상태, 행동 전송
    3) ARRIVAL 으로 계산값 전송
    4) REWARD 전송 후 반복 (BYE 는 종료 시 경우)
    """
    HELLO = "HELLO" 
    HELLO_ACK = "HELLO_ACK"
    STATE = "STATE" 
    DECISION = "DECISION" 
    ARRIVAL = "ARRIVAL" 
    REWARD = "REWARD" 

@dataclass
class Envelope:
    """
    메시지의 공통 헤더
    """
    schema: Tuple[int, int]
    msg_type: MessageType
    sender: Role
    session_id: str
    ts: float
    seq: Optional[int] = None

    @staticmethod
    def make(msg_type: MessageType, sender: Role, session_id: Optional[str] = None, seq: Optional[int] = None) -> "Envelope":
        return Envelope(
            schema=SCHEMA_VERSION,
            msg_type=msg_type,
            sender=sender,
            session_id=(session_id or str(uuid.uuid4())),
            ts=time.time(),
            seq=seq
        )

@dataclass
class Hello:
    who: Role
    node_id: str
    capabilities: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateAtDecision:
    chunk_idx: int
    tau: float   # 라이브 지연(초)
    B: float     # 버퍼 잔여 재생시간(초)

@dataclass
class Decision:
    chunk_idx: int
    k_level: Optional[int] = None
    bitrate: Optional[int] = None
    speed: Optional[float] = None
    skip: bool = False

@dataclass
class Arrival:
    chunk_idx: int
    size_bytes: int
    sent_ts: float # 보낸 시각
    arrived_ts: float # RX 기준 도착 시각
    tx_done_ts: float | None = None # TX 창 경계 / 완료 시각

@dataclass
class RewardReport:
    chunk_idx: int
    reward: float
    reward_inst: Optional[float] = None
    reward_cum: Optional[float] = None

PayloadT = Union[Hello, StateAtDecision, Decision, Arrival, RewardReport, Dict[str, Any]]

@dataclass
class WireMsg:
    envelope: Envelope
    payload: PayloadT

    def to_bytes(self) -> bytes:
        def _strip_none(obj: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in obj.items() if v is not None}
        
        payload_dict: Dict[str, Any]
        if hasattr(self.payload, "__dataclass_fields__"):
            payload_dict = _strip_none(asdict(self.payload))
        else:
            payload_dict = _strip_none(dict(self.payload))

        obj = {
            "envelope": asdict(self.envelope),
            "payload": payload_dict
        }
        return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    
    @staticmethod
    def from_bytes(raw: bytes) -> "WireMsg":
        d = json.loads(raw.decode("utf-8"))
        env = d["envelope"]
        envelope = Envelope(
            schema=tuple(env["schema"]),
            msg_type=MessageType(env["msg_type"]),
            sender=Role(env["sender"]),
            session_id=str(env["session_id"]),
            ts=float(env["ts"]),
            seq=env.get("seq")
        )
        # payload type dispatch
        mt = envelope.msg_type
        payload_map: Dict[MessageType, Type] = {
            MessageType.HELLO: Hello,
            MessageType.HELLO_ACK: Hello,
            MessageType.STATE: StateAtDecision,
            MessageType.DECISION: Decision,
            MessageType.ARRIVAL: Arrival,
            MessageType.REWARD: RewardReport
        }
        cls = payload_map.get(mt)
        if cls:
            payload = cls(**d["payload"])
        else:
            payload = d["payload"]
        return WireMsg(envelope, payload)

def make_wire(msg_type: MessageType, sender: Role, session_id: str, payload: PayloadT, seq: Optional[int] = None) -> WireMsg:
    env = Envelope.make(msg_type, sender, session_id, seq=seq)
    return WireMsg(env, payload)

def encode_decision_udp(
        k_level: Optional[int],
        speed: Optional[float],
        skip: bool,
        chunk_idx: Optional[int] = None,
        session_id: Optional[str] = None,
        include_meta: bool = True
) -> bytes:
    obj: Dict[str, Any] = {
        "k": k_level,
        "v": speed,
        "x": int(skip)
    }
    if include_meta:
        if chunk_idx is not None:
            obj["i"] = int(chunk_idx)
        if session_id:
            obj["sid"] = str(session_id)
        obj["mt"] = "DECISION"
        obj["t"] = time.time()
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")

@dataclass
class ParsedDecision:
    session_id: Optional[str]
    chunk_idx: Optional[int]
    k_level: Optional[int]
    bitrate_kbps: Optional[int]
    speed: Optional[float]
    skip: bool

def _nearest_level_from_kbps(kbps: int, known_rates_kbps: Optional[Iterable[int]]) -> Optional[int]:
    if not known_rates_kbps:
        return None
    try:
        pairs = list(enumerate(list(map(int, known_rates_kbps)), start=1))
        idx, _ = min(pairs, key= lambda it: abs(int(kbps) - it[1]))
        return int(idx)
    except Exception:
        return None

def parse_any_decision(raw: bytes, known_rates_kbps: Optional[Iterable[int]] = None) -> ParsedDecision:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return ParsedDecision(None, None, None, None, None, False)
    
    # Case 2: compact/legacy format
    if isinstance(obj, dict) and ("k" in obj or "v" in obj or "x" in obj):
        k = obj.get("k")
        v = obj.get("v")
        x = obj.get("x")
        return ParsedDecision(
            session_id=obj.get("sid"),
            chunk_idx=obj.get("i"),
            k_level=(int(k) if k is not None else None),
            bitrate_kbps=None,
            speed=(float(v) if v is not None else None),
            skip=(bool(int(x)) if isinstance(x, (int, float, str)) else bool(x)),
        )

    # Case 1: enveloped ND‑JSON
    if "envelope" in obj and "payload" in obj:
        env = obj["envelope"]
        p = obj["payload"]
        sid = env.get("session_id")
        seq = env.get("seq")
        k_level = p.get("k_level")
        br = p.get("bitrate_kbps")
        if k_level is None and br is not None:
            k_level = _nearest_level_from_kbps(int(br), known_rates_kbps)
        return ParsedDecision(
            session_id=sid,
            chunk_idx=(seq if seq is not None else p.get("chunk_idx")),
            k_level=(int(k_level) if k_level is not None else None),
            bitrate_kbps=(int(br) if br is not None else None),
            speed=(float(p.get("speed")) if p.get("speed") is not None else None),
            skip=bool(p.get("skip", False)),
        )

    # Case 3: payload‑only message (e.g. {"payload": {...}})
    if "payload" in obj and isinstance(obj["payload"], dict):
        p = obj["payload"]
        br = p.get("bitrate_kbps")
        k = p.get("k_level") or p.get("k")
        v = p.get("speed")
        x = p.get("skip", False)
        if k is None and br is not None:
            k = _nearest_level_from_kbps(int(br), known_rates_kbps)
        return ParsedDecision(
            session_id=None,
            chunk_idx=p.get("chunk_idx"),
            k_level=(int(k) if k is not None else None),
            bitrate_kbps=(int(br) if br is not None else None),
            speed=(float(v) if v is not None else None),
            skip=bool(x),
        )

    # Unrecognized format; return defaults
    return ParsedDecision(None, None, None, None, None, False)


if __name__ == "__main__":
    # Roundtrip test for ND‑JSON
    sess = "sess-001"
    dec = Decision(chunk_idx=5, k_level=2, speed=1.0, skip=False)
    wire = make_wire(MessageType.DECISION, Role.PLAYBACK, sess, dec, seq=5)
    data = wire.to_bytes()
    roundtripped = WireMsg.from_bytes(data)
    assert roundtripped.envelope.msg_type == MessageType.DECISION
    assert roundtripped.envelope.session_id == sess
    assert isinstance(roundtripped.payload, Decision)

    # Legacy roundtrip
    legacy = encode_decision_udp(3, 1.0, False, chunk_idx=6, session_id=sess)
    parsed = parse_any_decision(legacy, known_rates_kbps=(800, 1200, 2400, 4800))
    assert parsed.k_level == 3 and parsed.chunk_idx == 6

    print("Messages.py v2.0 OK")