from __future__ import annotations
# --- project root path injection ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # /home/.../Ver.5
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
# -----------------------------------

import argparse
import json
import logging
import socket
import threading
import time
import uuid
from typing import NamedTuple, Optional, Tuple, Dict

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

from Live.Gst_Common import build_tx_pipeline, set_encoder_bitrate, set_skip_gate, force_key_unit
from Common.Config import VIDEO, NETWORK
from Common.Messages import WireMsg, MessageType, Role, Decision, StateAtDecision, Hello, RewardReport,  make_wire, Arrival

class Action(NamedTuple):
    k_level: Optional[int] # 비트레이트 인덱스
    speed: Optional[float] # 속도
    skip: bool # 스킵

def _nearest_level_from_kbps(kbps: int) -> int:
    rates = getattr(VIDEO, "BIT_RATES_KBPS", (800, 1200, 2400, 4800))
    idx, _ = min(enumerate(rates, start=1), key=lambda it : abs(int(kbps) - int(it[1])))
    return int(idx)

def _loads_action(dat: bytes) -> Action:
    """
    평문 JSON 포맷을 읽기 위한 함수
    새 프로토콜 기반 파싱 실패 시 호환용으로 설정
    """
    try:
        obj = json.loads(dat.decode("utf-8"))
    except Exception:
        return Action(None, None, False)

    if "payload" in obj and isinstance(obj["payload"], dict):
        p = obj["payload"]
        k = p.get("k")
        if k is None and "bitrate_kbps" in p:
            k = _nearest_level_from_kbps(int(p["bitrate_kbps"]))
        v = p.get("speed")
        x = bool(p.get("skip", False))
        return Action(int(k) if k is not None else None, float(v) if v is not None else None, x)

    # flat
    k = obj.get("k")
    v = obj.get("v")
    x = bool(obj.get("x", False))
    return Action(int(k) if k is not None else None, float(v) if v is not None else None, x)

class LiveTX:
    """
    송신(TX, Live) 용 클래스

    1) 수신(RX, Playback) 측에서 오는 Action 을 UDP 로 수신
    2) 품질 k : 인코더 bitrate 적용 | 스킵 x 및 재생 속도 v : 토글
    """
    def __init__(self, dest_host: str, rtp_port: int, ctrl_port: int) -> None:
        self.dest_host = dest_host
        self.rtp_port = int(rtp_port)
        self.ctrl_port = int(ctrl_port)

        self.pipeline = build_tx_pipeline(self.dest_host, self.rtp_port)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", ctrl_port))
        self._sock.settimeout(0.5)

        self._stop = threading.Event()
        self._ctrl_th: Optional[threading.Thread] = None
        self._bus_th: Optional[threading.Thread] = None

        self._dropping = False
        self._drop_lock = threading.Lock()
        self._gate_lock = threading.Lock()
        self._current_k: Optional[int] = None
        self._session_id = f"tx-{uuid.uuid4().hex[:8]}"

        self._peer_addr: Optional[Tuple[str, int]] = None
        self._chunk_sec = getattr(VIDEO, 'CHUNK_SEC', 2.0)
        self._chunk_idx = 0
        self._win_start = time.time()
        self._bytes_in_win = 0
        self._probe_lock = threading.Lock()
    
    def _install_pay_probe(self) -> None:
        pay = self.pipeline.get_by_name("pay")
        if not pay:
            logging.warning("ARRIVAL: 'pay' not found; disable ARRIVAL sending")
            return
        pad = pay.get_static_pad("src")
        if not pad:
            logging.warning("ARRIVAL: 'pay.src' not found; disable ARRIVAL sending")
            return
        
        def _on_probe(_pad, info):
            buf = info.get_buffer()
            if not buf:
                return Gst.PadProbeReturn.OK
            try:
                sz = buf.get_size()
            except Exception:
                sz = 0
            now = time.time()
            with self._probe_lock:
                self._bytes_in_win += int(sz)
                if now - self._win_start >= self._chunk_sec:
                    self._win_start = now
                    self._bytes_in_win = 0
                    self._chunk_idx += 1
                    sent_ts = now  # 청크 종료 시각을 전송 기준으로 사용

                    # 창 경계에서 ARRIVAL 송신
                    if self._peer_addr and self._chunk_idx > 0:
                            arr = Arrival(
                                chunk_idx=int(self._chunk_idx),
                                size_bytes=0,
                                sent_ts=float(sent_ts),
                                arrived_ts=0.0,  # RX가 수신 즉시 로컬 시간으로 채움
                                tx_done_ts=float(now)
                            )
                            wire = make_wire(MessageType.ARRIVAL, Role.LIVE, self._session_id, arr, seq=self._chunk_idx)
                            self._sock.sendto(wire.to_bytes(), self._peer_addr)
            return Gst.PadProbeReturn.OK

        pad.add_probe(Gst.PadProbeType.BUFFER, _on_probe)
        logging.info("ARRIVAL: pay.src probe installed (chunk=%.3fs)", self._chunk_sec)

    def start(self) -> None:
        logging.info("| TX 파이프라인 가동: RTP -> %s:%d | CTRL <- %d", self.dest_host, self.rtp_port, self.ctrl_port)
        self.pipeline.set_state(Gst.State.PLAYING)

        self._bus_th = threading.Thread(target=self._bus_loop, daemon=True)
        self._bus_th.start()

        self._ctrl_th = threading.Thread(target=self._ctrl_loop, daemon=True)
        self._ctrl_th.start()
        self._install_pay_probe()
     
    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass
        
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        if self._ctrl_th and self._ctrl_th.is_alive():
            self._ctrl_th.join(timeout=1.0)
        if self._bus_th and self._bus_th.is_alive():
            self._bus_th.join(timeout=1.0)
        
        logging.info("TX stopped")

    def _bus_loop(self) -> None:
        bus = self.pipeline.get_bus()
        while not self._stop.is_set():
            msg = bus.timed_pop(500 * Gst.MSECOND)

            if not msg:
                continue
            t = msg.type

            if t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                logging.error("GstError: %s | %s", err, dbg)

            elif t == Gst.MessageType.WARNING:
                err, dbg = msg.parse_warning()
                logging.warning("GstWarning: %s | %s", err, dbg)

            elif t == Gst.MessageType.STATE_CHANGED and msg.src == self.pipeline:
                old, new, pend = msg.parse_state_changed()
                logging.info("| TX 파이프라인 상태 : %s -> %s", old.value_nick, new.value_nick)
    
    def _ctrl_loop(self) -> None:
        """
        Playback 측에서 오는 제어 메시지 수신 및 처리

        - 우선 파싱: WireMsg (based on "Messages.py")
        - 실패: JSON 포맷
        """
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            
            try:
                msg = WireMsg.from_bytes(data)
                mt = msg.envelope.msg_type
                self._peer_addr = addr
            except Exception:
                act = _loads_action(data)
                logging.debug("Control (legacy) %s -> %s", addr, act)
                self._apply_action(act)
                continue

            # HELLO / 세션 시작 응답
            if mt == MessageType.HELLO:
                try:
                    p = msg.payload
                    RX_node_id = getattr(p, "node_id", "")
                except Exception:
                    RX_node_id = ""
                logging.info("| HELLO 메시지 수신 성공 | 송신부: %s (%s)", addr, RX_node_id)
                ack_payload = Hello(who=Role.LIVE, node_id=self._session_id, capabilities={})
                ack_msg = make_wire(MessageType.HELLO_ACK, Role.LIVE, self._session_id, ack_payload)
                
                try:
                    time.sleep(0.1)
                    self._sock.sendto(ack_msg.to_bytes(), addr)
                    logging.info("| HELLO 응답 메시지 송신부로 전달\n")
                except Exception as e:
                    logging.warning("| Hello 응답 메시지 전송 실패: %s", e)
                continue
            
            # DECISION / 비트레이트, 스킵, 속도 적용
            if mt == MessageType.DECISION:
                dec: Decision = msg.payload
                if dec.k_level is not None:
                    k = int(dec.k_level)
                else:
                    k = None
                act = Action(k_level=k, speed=dec.speed, skip=dec.skip)
                logging.info("| [청크 %d] Action 수신 성공 | %s", dec.chunk_idx, act)
                self._apply_action(act, dec.chunk_idx)
                continue

            # REWARD / 보상 값 로그
            if mt == MessageType.REWARD:
                reward: RewardReport = msg.payload
                # 확장 필드가 있으면 함께 출력
                inst = getattr(reward, "reward_inst", None)
                cum  = getattr(reward, "reward_cum", None)
                if inst is not None or cum is not None:
                    logging.info("| [청크 %d] 보상 수신 성공 | 보상 = %.4f%s",
                                reward.chunk_idx, float(inst if inst is not None else reward.reward),
                                (f", 누적 보상 = {float(cum):.4f}\n" if cum is not None else ""))
                else:
                    # 하위호환(구형 RX)
                    logging.info("Received 'REWARD' for chunk %d: %.4f", reward.chunk_idx, reward.reward)
            continue

    def _apply_action(self, act: Action, chunk_idx: int) -> None:
        """
        DECISION 메시지에 따른 동작 (청크 번호 추가)
        """
        # Bitrate
        if act.k_level is not None and act.k_level != self._current_k:
            rates = VIDEO.BIT_RATES_KBPS
            n = len(rates)
            k_raw = int(act.k_level)
            k = max(1, min(k_raw, n))
            kbps = int(rates[k - 1])
            set_encoder_bitrate(self.pipeline, kbps)
            force_key_unit(self.pipeline)
            self._current_k = k
            logging.info(f"[Chunk {chunk_idx}] Bitrate set to {kbps} kbps")
        
        # Skip: 현재 chunk_idx에 대한 skip이므로, 
        # 다음 창(chunk_idx+1)부터 dropping 시작해야 함
        # 하지만 구현 복잡도를 고려하여 즉시 반영
        if act.skip:
            now = time.time()
            with self._probe_lock:
                elapsed = now - self._win_start
                remain  = max(0.0, self._chunk_sec - elapsed)

            # 경계 시각에 도착한 스킵은 '다음 청크' 드롭 금지 (논문 정합)
            if remain <= 1e-6:
                logging.info(f"[Chunk {chunk_idx}] Skip at boundary -> NO drop of next chunk")
            else:
                logging.info(f"[Chunk {chunk_idx}] Skip NOW: drop for remaining {remain:.3f}s (this chunk only)")
                # ★ 남은 시간만 드롭 → 현재 청크의 나머지만 생략
                self._trigger_skip(remain)
        
    
    def _trigger_skip(self, sec: float) -> None:
        """
        지정한 시간만큼 청크 스킵
        """
        with self._drop_lock:
            if self._dropping:
                logging.debug("Skip already active: ignoring")
                return
            self._dropping = True
            set_skip_gate(self.pipeline, True)
            force_key_unit(self.pipeline)

        def _end():
            with self._drop_lock:
                set_skip_gate(self.pipeline, False)
                self._dropping = False
                force_key_unit(self.pipeline)
        
        t = threading.Timer(float(sec), _end)
        t.daemon = True
        t.start()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Live TX (RTP/H264 over UDP) with UDP control")
    ap.add_argument("--dest-host", default=NETWORK.RX_HOST)
    ap.add_argument("--rtp-port", type=int, default=NETWORK.RX_RTP_PORT)
    ap.add_argument("--ctrl-port", type=int, default=5001, help="제어용 UDP PORT (default: 5001)")
    ap.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.loglevel), format="%(asctime)s %(levelname)s %(message)s")

    tx = LiveTX(args.dest_host, args.rtp_port, args.ctrl_port)
    try:
        tx.start()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        tx.stop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())