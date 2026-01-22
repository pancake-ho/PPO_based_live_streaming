from __future__ import annotations
import os
import gi
import logging
from typing import Optional

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo, GLib

from Common.Config import VIDEO, NETWORK
Gst.init(None)

def _source_desc() -> str:
    """
    환경변수 LIVE_SOURCE 에 따라 소스 선택
    """
    mode = (os.getenv("LIVE_SOURCE") or "test").lower()
    if mode == "file":
        path = os.getenv("LIVE_FILE_PATH")
        if path and os.path.exists(path):
            # filesrc는 일시 정지 없이 지속 재생을 위해 loop=true 로 queue 사용 가능
            return f'filesrc location="{path}" ! qtdemux ! h264parse ! avdec_h264 ! videoconvert'
    elif mode == "v4l2":
        dev = os.getenv("LIVE_V4L2_DEV") or "/dev/video0"
        if os.path.exists(dev):
            return f'v4l2src device={dev} ! videoconvert'
    # mode == "dash" 추가해야함
    
    # fallback
    return 'videotestsrc is-live=true pattern=ball ! videoconvert'

def build_tx_pipeline(dest_host: str, dest_port: int, vconf = VIDEO, nconf = NETWORK) -> Gst.Pipeline:
    w, h, fps = vconf.WIDTH, vconf.HEIGHT, vconf.FPS
    init_kbps = int(getattr(VIDEO, 'BIT_RATES_KBPS', (800,1200,2400,4800))[1] if hasattr(VIDEO, 'BIT_RATES_KBPS') else 1200)

    # key-int-max=FPS 설정하여 매초 IDR 설정
    desc = f"""
        {_source_desc()} !
        videorate ! video/x-raw,width={w},height={h},framerate={fps}/1 !
        x264enc name=enc tune=zerolatency speed-preset=ultrafast bframes=0 key-int-max={fps} bitrate={init_kbps} !
        h264parse config-interval=1 !
        rtph264pay pt=96 name=pay config-interval=1 mtu=1200 !
        valve name=gate drop=false !
        udpsink host={dest_host} port={dest_port} sync=false async=false    
        """
    pipeline = Gst.parse_launch(desc)
    assert isinstance(pipeline, Gst.Pipeline) # 검증

    return pipeline

def set_encoder_bitrate(pipeline: Gst.Pipeline, kbps: int) -> None:
    enc = pipeline.get_by_name("enc")
    if enc is not None:
        enc.set_property("bitrate", int(kbps))

def set_skip_gate(pipeline: Gst.Pipeline, drop: bool) -> None:
    gate = pipeline.get_by_name("gate")
    if gate is not None:
        gate.set_property("drop", bool(drop))

def force_key_unit(pipeline: Gst.Pipeline) -> bool:
    enc = pipeline.get_by_name("enc")
    if enc is None:
        logging.warning("force_key_unit: 'enc' not found !")
        return False

    src_pad = enc.get_static_pad("src")
    if src_pad is None:
        logging.warning("force_key_unit: enc src pad not found !")
        return False

    # Upstream FKU: 요청은 upstream로! (src 패드에서 보내는 게 정석)
    ev = GstVideo.video_event_new_upstream_force_key_unit(
        Gst.CLOCK_TIME_NONE,  
        True,                 
        0                       
    )
    ok = bool(src_pad.send_event(ev))
    return ok