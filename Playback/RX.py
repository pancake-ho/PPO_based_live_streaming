from __future__ import annotations
# --- project root path injection ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # /home/.../Ver.5
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
# -----------------------------------

import argparse
import logging
import socket
import threading
import time 
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GLib

from Common.Config import VIDEO, NETWORK
from Common.Metrics import Metrics, StateLOG, ActionLOG, ArrivalLOG
from Common.Messages import make_wire, WireMsg, MessageType, Role, StateAtDecision, Decision, RewardReport, Hello
from dataclasses import replace

try:
    from PPO_Agent import PPOAgent
except:
    pass

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
Gst.init(None)

def build_rx_pipeline(port: int, show: bool) -> Gst.Pipeline:
    disp = ("ximagesink name=disp sync=true" if show
            else "fakesink sync=true")
    desc = (
    f'udpsrc port={port} '
    'caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" '
    '! rtpjitterbuffer name=jb latency=300 '
    '! rtph264depay ! h264parse ! avdec_h264 ! videoconvert '
    '! tee name=vidtee '

    # isel 먼저 '정의'하고, 이후에 pad로 입력 연결
    'input-selector name=isel '

    # 일반/가속 입력(0)
    'vidtee. ! queue name=q_norm_1 ! identity name=pace drop-probability=0.0 ! queue name=q_norm_2 ! isel.sink_0 '

    # retime 입력(1)
    'appsrc name=retime_src is-live=true format=time do-timestamp=false emit-signals=false '
    '! queue name=q_retime_to_isel ! isel.sink_1 '

    # isel 소스 → 싱크
    'isel. ! queue name=q_isel_out '
    '! identity name=disp_tap signal-handoffs=true silent=false ! '
    f'{disp} '

    # retime 브리지 입력 탭
    'vidtee. ! queue name=q_retime_tap leaky=2 max-size-buffers=1 '
    '! appsink name=retime_in emit-signals=true sync=false max-buffers=1 drop=true '

    # grab 탭
    'vidtee. ! queue name=q_grab_tap leaky=2 max-size-buffers=1 '
    '! appsink name=grab emit-signals=true sync=false max-buffers=1 drop=true '
)

    pipeline = Gst.parse_launch(desc)

    # jitterbuffer 등 속성 추가 조정
    jb = pipeline.get_by_name('jb')
    if jb:
        try:
            jb.set_property('do-lost', True)
            jb.set_property('latency', 500)
        except Exception:
            pass

    # isel 속성은 None 가드 + 예외 가드
    isel = pipeline.get_by_name('isel')
    if isel:
        try:
            isel.set_property('sync-mode', 1)       # ACTIVE_SEGMENT
            isel.set_property('cache-buffers', True)
        except Exception:
            pass

    # [NEW] 모든 queue 요소에 공통 속성 일괄 적용 (이름 상관없이)
    it = pipeline.iterate_elements()
    while True:
        res = it.next()  # (Gst.IteratorResult, element)
        if res[0] != Gst.IteratorResult.OK:
            break
        elem = res[1]
        try:
            fac = elem.get_factory()
            if fac and fac.get_name() == 'queue':
                elem.set_property('leaky', 2)           # downstream
                elem.set_property('max-size-buffers', 0)
                elem.set_property('max-size-bytes', 0)
                elem.set_property('max-size-time', 0)
        except Exception:
            pass

    return pipeline

class CtrlLink:
    """
    송신 측(TX, Live) 과 제어 메시지를 교환하는 부분.

    - 전송 요소: HELLO / STATE / DECISION / REWARD
    """
    def __init__(self, host: str, port: int, timeout: float=0.5, on_arrival=None) -> None:
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.bind(("0.0.0.0", 0))
        except Exception as e:
            logging.debug("CTRL bind skipped: %s", e)
        self.session_id = f"rx-{uuid.uuid4().hex[:8]}"
        self._stop = threading.Event()
        self._recv_th: Optional[threading.Thread] = None
        self._on_arrival = on_arrival
        self._hello_ack_event = threading.Event()

    def start_listener(self) -> None:
        if self._recv_th and self._recv_th.is_alive():
            return
        self._recv_th = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_th.start()

    def stop_listener(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except Exception:
            pass
        if self._recv_th and self._recv_th.is_alive():
            self._recv_th.join(timeout=1.0)
    
    def hello(self) -> None:
        """
        HELLO 메시지를 TX 측으로 전송하여 세션 시작
        """
        payload = Hello(who=Role.PLAYBACK, node_id=self.session_id, capabilities={"dash": True})
        w = make_wire(MessageType.HELLO, Role.PLAYBACK, self.session_id, payload)
        try:
            self.sock.sendto(w.to_bytes(), self.addr)
            logging.info("| Hello 메시지 전송 성공 | 송신부: %s", self.addr)
        except Exception as e:
            logging.warning("| Hello 메시지 전송 실패 | 송신부: %s", e)

    def send_state(self, chunk_idx: int, tau: float, B: float) -> None:
        """
        현재 의사결정 시점의 상태(τ, B)를 TX로 전송 (선택)
        """
        st = StateAtDecision(chunk_idx=int(chunk_idx), tau=float(tau), B=float(B))
        wire = make_wire(MessageType.STATE, Role.PLAYBACK, self.session_id, st, seq=chunk_idx)
        try:
            self.sock.sendto(wire.to_bytes(), self.addr)
            logging.debug("| [청크 %d] [STATE 전송] tau=%.3f, B=%.3f", chunk_idx, tau, B)
        except Exception as e:
            logging.warning("STATE 전송 실패: %s", e)


    def send_decision(self, chunk_idx: int, k_level: int | None, bitrate: int | None, speed: float, skip: bool) -> None:
        """
        PPO Agent 가 선택한 행동을 TX 로 전송
        """
        dec = Decision(chunk_idx=int(chunk_idx), k_level=(int(k_level) if k_level is not None else None), 
                       bitrate=(int(bitrate) if bitrate is not None else None), speed=float(speed), skip=bool(skip))
        wire = make_wire(MessageType.DECISION, Role.PLAYBACK, self.session_id, dec, seq=chunk_idx)
        try:
            self.sock.sendto(wire.to_bytes(), self.addr)
            logging.info("| [청크 %d] [Action 전송 성공] k = %s (bitrate = %s), speed = %.3f, skip = %s", 
                          chunk_idx, str(k_level), str(bitrate), speed, str(skip))
        except Exception as e:
            logging.warning("Action 전송 실패: %s", e)
    
    def send_reward(self, chunk_idx: int, reward: float) -> None:
        """
        현재 시점의 청크에 대한 보상 값을 TX 로 전송
        """
        rr = RewardReport(chunk_idx=int(chunk_idx), reward=float(reward))
        wire = make_wire(MessageType.REWARD, Role.PLAYBACK, self.session_id, rr, seq=chunk_idx)
        try:
            self.sock.sendto(wire.to_bytes(), self.addr)
            logging.debug("| [chunk %d] Reward 전송 성공 | Reward = %.4f", chunk_idx, reward)
        except Exception as e:
            logging.warning("Reward 전송 실패: %s", e)
    
    def wait_for_hello_ack(self, timeout: float = 2.0) -> bool:
        return self._hello_ack_event.wait(timeout)
    
    def _recv_loop(self) -> None:
        """
        HELLO_ACK 및 ARRIVAL 수신 처리
        """
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
        
            try:
                msg = WireMsg.from_bytes(data)
            except Exception:
                continue

            mt = msg.envelope.msg_type
            if mt.name == "HELLO_ACK":
                logging.info("| HELLO 응답 메시지 수신 성공")
                self._hello_ack_event.set()
            elif mt.name == "ARRIVAL":
                arr = msg.payload
                now = time.time()
                if not getattr(arr, "arrived_ts", None):
                    arr.arrived_ts = float(now)
                if not hasattr(arr, "sent_ts"):
                    arr.sent_ts = None
                if callable(self._on_arrival):
                    try:
                        self._on_arrival(arr)
                    except Exception as e:
                        logging.warning("ARRIVAL callback failed: %s", e)

class RetimerBridge:
    """
    파이프라인의 appsink(retime_in) -> 재타임스태프 -> appsrc(retime_src) 전달 흐름 구현
    v < 1.0 구간(감속) 에서만 활성화
    """
    def __init__(self, pipeline: Gst.Pipeline, get_speed: Callable[[], float]):
        self.pipe = pipeline
        self.get_speed = get_speed
        self._in = self.pipe.get_by_name('retime_in')
        self._out = self.pipe.get_by_name('retime_src')
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_pts_ns: Optional[int] = None
        self._base_interval_ns: int = int(1e9 / 30.0)

        self._out.set_property('block', True)
        self._out.set_property('min-latency', 0)
        self._out.set_property('max-bytes', 0)
    
    def _detect_interval(self, sample: Gst.Sample) -> None:
        caps = sample.get_caps()
        s = caps.get_structure(0) if caps else None
        if s and s.has_field('framerate'):
            frac = s.get_value('framerate')
            try:
                num = frac.numerator
                den = frac.denominator
                if num > 0 and den > 0:
                    self._base_interval_ns = int(1e9 * den / num)
            except Exception:
                pass
    
    def _loop(self):
        gi.require_version('GstApp', '1.0')
        from gi.repository import GstApp
        self._running.set()
        self._last_pts_ns = None
        while self._running.is_set():
            sample = self._in.emit('try-pull-sample', 10_000_000)
            if not sample:
                continue
            if self._last_pts_ns is None:
                self._detect_interval(sample)
                try:
                    caps = sample.get_caps()
                    if caps:
                        self._out.set_property('caps', caps)
                except Exception:
                    pass
                self._last_pts_ns = int(time.time() * 1e9)
            
            buf = sample.get_buffer()
            dur_ns = buf.duration if buf and buf.duration > 0 else self._base_interval_ns

            v = max(self.get_speed(), 1e-6)
            target_interval_ns = int(dur_ns / v)

            now_ns = int(time.time() * 1e9)
            next_pts_ns = self._last_pts_ns + target_interval_ns
            sleep_ns = max(0, next_pts_ns - now_ns)
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            
            success, mapinfo = buf.map(Gst.MapFlags.READ) if buf else (False, None)
            try:
                outbuf = Gst.Buffer.new_allocate(None, mapinfo.size if success else 0, None)
                if success and mapinfo.size > 0:
                    outbuf.fill(0, mapinfo.data)
                outbuf.pts = next_pts_ns
                outbuf.dts = Gst.CLOCK_TIME_NONE
                outbuf.duration = target_interval_ns
                self._out.emit('push-buffer', outbuf)
            finally:
                if success:
                    buf.unmap(mapinfo)
            
            self._last_pts_ns = next_pts_ns
        self._out.emit('end-of-stream')
    
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.clear()
        self._thread = threading.Thread(target=self._loop, name='RetimerBridge', daemon=True)
        self._thread.start()
    
    def stop(self):
        if not self._thread:
            return
        self._running.clear()
        self._thread.join(timeout=1.0)
        self._thread = None


@dataclass
class RXWindowStats:
    frames: int = 0
    start_ts: float = 0.0
    end_ts: float = 0.0

class RX:
    """
    수신(RX, Playback) 용 클래스
    1) RL Agent 전담
    2) State 및 Action (Action 의 경우 Pull-Back 방식으로 전달됨) 전달
    3) Reward 전달
    """
    def __init__(self, ctrl_host: str, ctrl_port: int, show: bool=True) -> None:
        Gst.init(None)
        self.pipeline = build_rx_pipeline(NETWORK.RX_RTP_PORT, show=show)
        self.appsink = self.pipeline.get_by_name('grab')
        
        self._disp_frames = 0 # 표시 프레임 카운터
        disp_tap = self.pipeline.get_by_name('disp_tap')
        if disp_tap is not None:
            def _on_disp_handoff(elem, buf):
                with self._lock:
                    self._disp_frames += 1
                    self._window.end_ts = time.time()
            
            try:
                disp_tap.connect('handoff', _on_disp_handoff)
            except TypeError:
                disp_tap.connect('handoff', lambda elem, buf, pad : _on_disp_handoff(elem, buf))

        if self.appsink is None:
            raise RuntimeError(
                "appsink(name='grab')를 찾지 못했습니다. build_rx_pipeline() desc에 "
                "'appsink name=grab ...'이 포함되어 있는지 확인하세요."
            )
        self.appsink.connect("new-sample", self._on_new_sample)

        isel = self.pipeline.get_by_name('isel')
        srcpad = isel.get_static_pad('src')

        def _on_isel_buf(pad, info, user_data=None):
            with self._lock:
                self._window.frames += 1
                self._window.end_ts = time.time()
            return Gst.PadProbeReturn.OK

        srcpad.add_probe(Gst.PadProbeType.BUFFER, _on_isel_buf, None)
        
        self._speed = float(1.0)
        
        self._isel = isel
        if self._isel:
            try:
                self._isel.set_property('sync-mode', 1)
                self._isel.set_property('cache-buffers', True)
            except Exception:
                pass

        self._retimer = RetimerBridge(self.pipeline, get_speed= lambda : self._speed)
        self._sel_pad_normal = None
        self._sel_pad_retime = None
        self._R_cum = 0.0
        self._t0 = time.time()
        self._period = float(VIDEO.CHUNK_SEC)
        
        for pad in self._isel.iterate_sink_pads():
            name = pad.get_name()
            if name.endswith('sink_0'):
                self._sel_pad_normal = pad
            elif name.endswith('sink_1'):
                self._sel_pad_retime = pad
        
        if self._sel_pad_normal:
            self._isel.set_property('active-pad', self._sel_pad_normal)
    
        self._eos = False
        self.metrics = Metrics()
        # 청크 경계 신호
        self._tick = threading.Event()

        self._skip_plan: dict[int, bool] = {}

        def _on_arrival_msg(arr) -> None:
            now = time.time()
            idx = int(getattr(arr, "chunk_idx"))
            size_bytes = int(getattr(arr, "size_bytes", 0) or 0)
            is_drop = (size_bytes == 0)
            is_start = bool(getattr(arr, "is_start", False))  # TX가 넣어줄 수도, 아닐 수도

            # 1) 도착만 로깅
            self.metrics.log_arrival(ArrivalLOG(
                chunk_idx=idx,
                sent_ts=float(getattr(arr, "sent_ts", now)),
                arrived_ts=float(now),
                size_bytes=size_bytes,
            ))

            # 2) u_i가 '이미' 기록된 경우에만 p_start/p_end를 만든다
            u_ts_rel = float(self.metrics.u_decide.get(idx)) if idx in self.metrics.u_decide else None
            if (idx not in self.metrics.p_start) and (u_ts_rel is not None):
                # 시작 ARRIVAL이든 0B든 상관 없이, u_i가 존재하면 스케줄 생성
                u_abs = float(self.metrics.t0) + float(u_ts_rel)
                act = self.metrics.actions.get(idx)
                skip_planned = getattr(act, "skip", bool(self._skip_plan.get(idx, False)))
                speed = getattr(act, "speed", getattr(self, "_speed", 1.0))
                self.metrics.record_play_start(
                    i=idx,
                    u_i_abs=u_abs,
                    arrived_ts=float(now),
                    skip_current=bool(skip_planned),
                    speed=float(speed),
                )
            try:
                self._tick.set()
            except Exception:
                pass

        
        self.ctrl = CtrlLink(ctrl_host, ctrl_port, on_arrival=_on_arrival_msg)

        if PPOAgent is not None:
            try:
                self.agent = PPOAgent()
                logging.info("PPO 에이전트 불러오기에 성공했습니다.")
            except Exception:
                logging.warning("PPO 에이전트 불러오기에 실패했습니다.")
                self.agent = None
        else:
            self.agent = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frames_total = 0
        self._window = RXWindowStats(start_ts=time.time())

        self._decision_thread: Optional[threading.Thread] = None
        self._bus_thread: Optional[threading.Thread] = None

        self._i = 1
        self._tau = float(VIDEO.INIT_BUFFERING_TIME)

        self._B = VIDEO.INIT_BUFFERING_TIME
        self._recv_sec = 0.0
        self._played_sec = 0.0
        self._p_prev_end = 0.0
        
    
    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        """
        파이프라인의 appsink 로부터 새 프레임을 받을 때 호출
        """
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        with self._lock:
            self._frames_total += 1
            self._window.end_ts = time.time()
        return Gst.FlowReturn.OK    
    
    def _bus_loop(self) -> None:
        bus = self.pipeline.get_bus()
        while not self._stop.is_set():
            msg = bus.timed_pop_filtered(100 * Gst.MSECOND,
                                        Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.WARNING)
            if msg is None:
                continue

            t = msg.type
            if t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                logging.error("GstError: %s | %s", err, dbg)
                try:
                    q = self.pipeline.get_by_name("q_isel_out") # 현재 에러 발생 부분 보강
                    if q: 
                        static_pad = q.get_static_pad("src")
                        dp = static_pad.get_peer() if static_pad else None
                        logging.error("q_isel_out linked ? %s -> %s",
                                      bool(static_pad and static_pad.is_linked()), # 현재 False
                                      dp.get_parent_element().name if dp else None) # 현재 None
                    self.pipeline.get_state(Gst.State.PAUSED)
                    self.pipeline.get_state(Gst.State.PLAYING)

                except Exception:
                    pass

            
            elif t == Gst.MessageType.EOS:
                logging.info("RX EOS")
                self._eos = True
                break

            elif t == Gst.MessageType.WARNING:
                err, dbg = msg.parse_warning()
                logging.error("GstWarning: %s | %s", err, dbg)
                try:
                    disp = self.pipeline.get_by_name("disp")
                    if disp and disp.get_property("sync"):
                        disp.set_property("sync", False)  # 표시 경로만 비동기
                except Exception:
                    pass

            # [PATCH-RX-1] ARRIVAL 처리부 교체/보강
            elif t == MessageType.ARRIVAL:
                arr = msg.payload  # Messages.Arrival
                now_abs = time.time()

                # arrived_ts를 RX 기준 '수신 즉시'로 채움
                try:
                    arr.arrived_ts = float(now_abs)
                except Exception:
                    pass

                # Metrics 반영: (i) 도착 기록 + (ii) 스케줄 생성
                i = int(getattr(arr, "chunk_idx", 0) or 0)
                self.metrics.log_arrival(
                    ArrivalLOG(
                        chunk_idx=i,
                        sent_ts=float(getattr(arr, "sent_ts", now_abs) or now_abs),
                        arrived_ts=float(now_abs),
                        size_bytes=int(getattr(arr, "size_bytes", 0) or 0),
                    )
                )

                # 스킵이 아닌 경우에만 '정상' 스케줄. (스킵 청크는 아래 PATCH-RX-2에서 즉시-완료 마킹)
                act = self.metrics.actions.get(i)
                if act is not None and not bool(getattr(act, "skip", False)):
                    v = float(getattr(act, "speed", 1.0) or 1.0)
                    self.metrics.record_play_start(
                        i=i, 
                        u_i_abs=float(now_abs),         # 결정시각 근사, 이후 record_decision_time으로 최신화됨
                        arrived_ts=float(now_abs), 
                        skip_current=False, 
                        speed=v
                    )
                continue


    def _decision_loop(self) -> None:
        """
        RL Agent로부터 행동을 받아 STATE, DECISION, REWARD 메시지를 TX로 전송
        (CARLA 정합: τ_i/B_i는 상태식, τ_{i+1}는 전이식에서 계산)
        """
        fps = float(VIDEO.FPS)
        t_i = float(VIDEO.CHUNK_SEC)

        while not self._stop.is_set():
            # GOOD: 틱 경계 도달 후 "실제 지금 시각"으로 기록 → 그때까지 도착한 청크들이 B에 반영
            self._tick.wait(timeout=t_i * 1.2)
            self._tick.clear()

            u_abs = time.time()
            # t0 초기화(최초 1회)
            if self.metrics.t0 is None:
                self.metrics.t0 = u_abs
            self.metrics.record_decision_time(self._i, u_abs)


            # --- (참고) 창 프레임 스냅샷 (디스플레이/리워드 참고용) ---
            with self._lock:
                win_frames = self._window.frames
                disp_frames = self._disp_frames
                self._disp_frames = 0
                self._window = RXWindowStats(start_ts=time.time())

            # 디코드된 콘텐츠 시간(최대 t_i)
            eff_frames = win_frames if win_frames > 0 else disp_frames
            decoded_sec_for_state = min(eff_frames / max(fps, 1e-6), t_i)
            stall_actual = max(0.0, t_i - decoded_sec_for_state)  # 관찰용

            self._no_frame_streak = getattr(self, "_no_frame_streak", 0)
            if decoded_sec_for_state <= 1e-6:
                self._no_frame_streak += 1
            else:
                self._no_frame_streak = 0

            if self._no_frame_streak >= 2:
                try:
                    # retimer 경로 정지(있으면)
                    if hasattr(self, "_retimer") and self._retimer:
                        self._retimer.stop()
                    # 속도 일반화(1.0)
                    self._speed = 1.0
                    # input-selector를 정상 패드로 복귀
                    if getattr(self, "_isel", None) and getattr(self, "_sel_pad_normal", None):
                        self._isel.set_property('active-pad', self._sel_pad_normal)
                    # 디스플레이 싱크를 잠시 풀어 GST 클럭 대기로 막히는 것 방지
                    if getattr(self, "pipeline", None):
                        disp = self.pipeline.get_by_name("disp")
                        if disp and hasattr(disp, "set_property"):
                            disp.set_property("sync", False)
                except Exception:
                    # 복구 중 예외는 삼키되, 다음 윈도에서 다시 시도
                    pass
                finally:
                    self._no_frame_streak = 0
            
            if decoded_sec_for_state > 0.0 and getattr(self, "pipeline", None):
                disp = self.pipeline.get_by_name("disp")
                if disp and hasattr(disp, "get_property"):
                    try:
                        # 너무 자주 토글되는 것 방지: False로 풀려있는 경우에만 True로 원복
                        if hasattr(disp, "set_property") and disp.get_property("sync") is False:
                            disp.set_property("sync", True)
                    except Exception:
                        pass

            try:
                for j in range(0, self._i):
                    # 이미 완성된 건 패스
                    if (j in self.metrics.p_start) and (j in self.metrics.p_end):
                        continue
                    act_j = self.metrics.actions.get(j, None)
                    # 스킵은 스케줄 미생성 (d_j=0으로만 유지)
                    if act_j is not None and getattr(act_j, "skip", False):
                        self.metrics.d_stall[j] = 0.0
                        continue
                    arr_j = self.metrics.arrivals.get(j, None)
                    if arr_j is None or getattr(arr_j, "arrived_ts", None) is None:
                        continue
                    # u_j 상대시각이 없으면 현재 u_abs를 근사 사용
                    u_rel_j = self.metrics.u_decide.get(j)
                    if u_rel_j is None:
                        self.metrics.record_decision_time(j, u_abs)
                    spd_j = float(getattr(act_j, "speed", 1.0) if act_j else 1.0)
                    self.metrics.record_play_start(
                        i=j,
                        u_i_abs=float(self.metrics.t0 + self.metrics.u_decide.get(j, u_abs - self.metrics.t0)),
                        arrived_ts=float(arr_j.arrived_ts),
                        skip_current=False,
                        speed=spd_j,
                    )
            except Exception as e:
                logging.debug("hydrate schedules failed: %s", e)

            # 5) [NEW: Soft-ARRIVAL] 현재 청크 i에 ARRIVAL이 없는데 디코드가 있었다면, 도착을 u_abs로 백필
            try:
                has_arr = (self.metrics.arrivals.get(self._i, None) is not None)
                if not has_arr:
                    self.metrics.log_arrival(ArrivalLOG(
                        chunk_idx=int(self._i),
                        sent_ts=None,              # 지연 왜곡 방지 (None 유지)
                        arrived_ts=float(u_abs),   # 지금 시각으로 보강
                        size_bytes=0
                    ))
            except Exception as e:
                logging.debug("soft-arrival failed: %s", e)

            # --- 상태 계산(정합: Metrics만 사용) ---
            tau_i = float(self.metrics.tau_state(self._i))
            B_i   = float(self.metrics.buffer_state(self._i))

            self.metrics.log_state(StateLOG(chunk_idx=self._i, tau=tau_i, B=B_i, stall=stall_actual))

            # --- (선택) STATE 전송 ---
            try:
                if hasattr(self.ctrl, "send_state"):
                    self.ctrl.send_state(self._i, tau_i, B_i)
            except Exception as e:
                logging.debug("STATE 전송 생략/실패: %s", e)

            # --- RL Action 결정 ---
            k_level: int = 1
            bitrate: int | None = None
            speed: float = 1.0
            skip_current: bool = False

            if self.agent is not None:
                try:
                    dec_k, dec_speed_idx, dec_skip = self.agent.decide(tau_i, B_i)
                    # 품질 단계/비트레이트
                    if hasattr(self.agent, "rates"):
                        k_level = max(1, min(int(dec_k), len(self.agent.rates)))
                        bitrate = int(self.agent.rates[k_level - 1])
                    else:
                        k_level = max(1, int(dec_k))
                        if bitrate is None and hasattr(VIDEO, "BIT_RATES_KBPS"):
                            bitrate = int(VIDEO.BIT_RATES_KBPS[k_level - 1])
                    # 속도/스킵
                    speeds = list(getattr(VIDEO, "SPEEDS", (0.9, 1.0, 1.1)))
                    raw = dec_speed_idx

                    if isinstance(raw, int):
                        v_idx = max(0, min(raw, len(speeds) - 1))
                        speed = float(speeds[v_idx])
                    else:
                        # 실속도(float)로 온 경우
                        try:
                            val = float(raw)
                            # 선택: speeds 범위 밖이면 clamp (retimer/파이프라인 안전)
                            v_min, v_max = (min(speeds), max(speeds))
                            speed = float(min(max(val, v_min), v_max))
                        except Exception:
                            speed = 1.0  # 폴백

                    skip_current = bool(dec_skip)
                    if skip_current:
                        stall_actual = 0.0
                except Exception as e:
                    logging.warning("Agent decide 실패: %s (기본값 사용)", e)

            
            try:
                self.ctrl.send_decision(self._i, k_level, bitrate, speed, bool(skip_current))
            except Exception as e:
                logging.warning("Action 전송 실패: %s", e)

            try:
                self._speed = float(speed)
                if self._speed < 1.0:
                    self._retimer.start()
                    if self._isel and self._sel_pad_retime:
                        self._isel.set_property('active-pad', self._sel_pad_retime)
                else:
                    if self._retimer:
                        self._retimer.stop()
                    if self._isel and self._sel_pad_normal:
                        self._isel.set_property('active-pad', self._sel_pad_normal)

                # 행동 로그 (이미 있음)
                self.metrics.log_decision(
                    chunk_idx=self._i,
                    u_ts=float(self.metrics.u_decide[self._i]),
                    quality_level=int(k_level),
                    speed=float(speed),
                    skip=bool(skip_current),
                )
                
                if bool(skip_current):
                    try:
                        self.metrics.record_play_start(
                            i=self._i,
                            u_i_abs=time.time(),
                            arrived_ts=None,
                            skip_current=True,
                            speed=float(speed if speed else 1.0),
                        )
                    except Exception as e:
                        logging.debug("스킵 즉시-완료 마킹 실패: %s", e)

                # [여기서 스킵 즉시-완료 마킹: PATCH-RX-2]  ⬅️⬅️⬅️
                # 스킵이면 스케줄 잔여를 지우고(p_start/p_end), 0-duration 완료로 마킹
                if skip_current:
                    self.metrics.p_start.pop(self._i, None)
                    self.metrics.p_end.pop(self._i, None)

                arr_i = self.metrics.arrivals.get(self._i, None)
                arrived_ts = (float(arr_i.arrived_ts) if (arr_i and getattr(arr_i, "arrived_ts", None)) else None)

                self.metrics.record_play_start(
                    i=self._i,
                    u_i_abs=u_abs,          # 위에서 기록한 이번 결정 시각
                    arrived_ts=arrived_ts,  # 스킵이면 None이어도 됨(0-duration 처리)
                    skip_current=skip_current,
                    speed=float(speed),
                )

            except Exception as e:
                logging.debug("Metrics 스케줄 업데이트 실패: %s", e)


            try:
                # Eq.(1) 기반 전이: u_{i+1} = max(t_{i+1}, u_i + l_i*(1-x_i))
                t_i = float(VIDEO.CHUNK_SEC)
                # live 생성시각: t1=η1, ti=∑η ⇒ t_{i+1} = (i+1)*t_i (초기 t0=0 기준)
                t_next_abs = float(self.metrics.t0) + (self._i * t_i) + t_i

                u_i_abs = float(self.metrics.t0) + float(self.metrics.u_decide[self._i])
                l_i = 0.0 if skip_current else float(self.metrics.trans_delay.get(self._i, 0.0))
                u_next_abs = max(t_next_abs, u_i_abs + l_i)

                self.metrics.record_decision_time(self._i + 1, u_next_abs)

                # 상태 계산은 항상 Metrics를 통해
                tau_next = float(self.metrics.tau_state(self._i + 1))
                B_next   = float(self.metrics.buffer_state(self._i + 1))
            except Exception as e:
                logging.debug("다음 상태 계산 실패: %s", e)
                tau_next = float('nan')
                B_next = float('nan')

            # --- 리워드 계산/전송 ---
            try:
                reward_value = float(self.metrics.reward(self._i))
            except Exception as e:
                logging.warning("Reward 계산 실패: %s", e)
                reward_value = 0.0
            self._R_cum += reward_value
            try:
                self.ctrl.send_reward(self._i, reward_value)
            except Exception as e:
                logging.debug("Reward 전송 실패: %s", e)

           # --- 로그 ---
            logging.info(
                "| [청크 %d] [STATE] τ=%.3f, B=%.3f | [ACTION] k=%d (bitrate=%s), v=%.2f, skip=%s | "
                "[REWARD]=%.3f | [NEXT] τ'=%.3f, B'=%.3f | decoded_time=%.2fs, stall_time=%.2fs\n",
                self._i, tau_i, B_i, k_level, str(bitrate), speed, str(skip_current),
                reward_value, tau_next, B_next,
                decoded_sec_for_state, stall_actual
            )

            # --- 인덱스 증가 ---
            self._i += 1

                
    def start(self) -> None:
        self.ctrl.start_listener()
        time.sleep(0.1)
        self.ctrl.hello()
        if not self.ctrl.wait_for_hello_ack(timeout=10.0):
            logging.warning("| HELLO 미수신 상태에서 파이프라인을 시작합니다")
        self.pipeline.set_state(Gst.State.PLAYING)
        logging.info("| RX 파이프라인 가동 (포트 = %d)\n", NETWORK.RX_RTP_PORT)
        self._bus_thread = threading.Thread(target=self._bus_loop, daemon=True)
        self._bus_thread.start()
        self._decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self._decision_thread.start()
    
    # RetimerBridge.stop 보강(클래스 내부)
    def stop(self):
        if not self._thread:
            return
        self._running.clear()
        self._thread.join(timeout=1.0)
        # 보강: appsrc에 EOS를 한번 보내고, block=False로 풀었다가 다시 True
        try:
            self._out.emit('end-of-stream')
            self._out.set_property('block', False)
            time.sleep(0.01)
            self._out.set_property('block', True)
        except Exception:
            pass
        self._thread = None

    
                 
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Playback RX (UDP/RTP/H264) with optional agent")
    ap.add_argument("--ctrl-host", default=NETWORK.TX_HOST)
    ap.add_argument("--ctrl-port", type=int, default=NETWORK.CTRL_TX_PORT)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--train-runner", action="store_true",
                help="PPO_Agent.train_combined() 러너를 실행")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.train_runner:
        from PPO_Agent import train_combined
        train_combined()
        return 0
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s')
    logging.getLogger().setLevel(getattr(logging, args.loglevel))

    rx = RX(ctrl_host=args.ctrl_host, ctrl_port=args.ctrl_port, show=args.show)
    try:
        rx.start()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
    return 0


if __name__ == "__main__":
    main()