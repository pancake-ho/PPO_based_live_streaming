import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# 설정: 처리할 MPD 파일들과 출력될 CSV 파일명 매핑
DATASETS = [
    # {
    #     "mpd_path": Path("SNE_DASH_SD_CASE1A_REVISED.mpd"),
    #     "output_csv": Path("tears_of_steel_dataset.csv"),
    #     "segment_duration": 2.0  # MPD 내부 정보를 확인하거나, 알고 있는 값 사용
    # },
    # {
    #     "mpd_path": Path("sintel.mpd"),
    #     "output_csv": Path("sintel_dataset.csv"),
    #     "segment_duration": 2.0
    # },
    # {
    #     "mpd_path": Path("TheSwissAccount_2s_onDemand_2014_05_09.mpd"),
    #     "output_csv": Path("swiss_account_dataset.csv"),
    #     "segment_duration": 2.0
    # }
    # {
    #     "mpd_path": Path("./data/ElephantsDream_2s_onDemand_2014_05_09.mpd"),
    #     "output_csv": Path("./data/elephants.csv"),
    #     "segment_duration": 2.0
    # },
    {
        "mpd_path": Path("./data/OfForestAndMen_2s_onDemand_2014_05_09.mpd"),
        "output_csv": Path("./data/forest_men.csv"),
        "segment_duration": 2.0
    }
]

# 네임스페이스 처리를 유연하게 하기 위한 함수
def find_ns(root):
    # MPD 파일의 root 태그에서 {urn:mpeg:dash...} 형태의 네임스페이스 추출
    m = re.match(r'\{(.*)\}', root.tag)
    return {"mpd": m.group(1)} if m else {}

def parse_duration_iso8601(duration_str: str) -> float:
    # 예: "PT0H9M56.46S" -> 초 단위 변환
    if not duration_str: return 0.0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(\d+(?:\.\d+)?)S", duration_str)
    if not m:
        # 가끔 숫자만 있는 경우도 처리 (드물지만 예외처리)
        try:
            return float(duration_str)
        except:
            return 0.0
            
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    s = float(m.group(3))
    return h * 3600 + m_ * 60 + s

def process_mpd(info: dict):
    mpd_path = info["mpd_path"]
    output_csv = info["output_csv"]
    seg_duration = info["segment_duration"]

    if not mpd_path.exists():
        print(f"[Skip] {mpd_path} not found.")
        return

    print(f"Processing {mpd_path}...")
    tree = ET.parse(mpd_path)
    root = tree.getroot()
    ns = find_ns(root)

    # 1. 전체 길이 파싱
    media_duration_str = root.attrib.get("mediaPresentationDuration")
    if not media_duration_str:
        # 일부 MPD는 Period에 duration이 있을 수 있음
        period = root.find("mpd:Period", ns)
        media_duration_str = period.attrib.get("duration")
    
    total_duration = parse_duration_iso8601(media_duration_str)
    num_chunks = int(total_duration // seg_duration)

    print(f"  - Total Duration: {total_duration:.2f}s")
    print(f"  - Chunks: {num_chunks} (Seg Duration: {seg_duration}s)")

    # 2. Representation(Bitrate) 정보 파싱
    period = root.find("mpd:Period", ns)
    # AdaptationSet이 여러 개일 수 있음 (Video, Audio 등). mimeType="video" 인 것 찾기
    video_adap = None
    for adap in period.findall("mpd:AdaptationSet", ns):
        mime = adap.attrib.get("mimeType", "")
        # contentType이 video이거나 mimeType에 video가 포함된 경우
        if "video" in mime or adap.attrib.get("contentType") == "video":
            video_adap = adap
            break
    
    # 만약 명시적 태그가 없으면 첫 번째 것을 가정 (단순화)
    if video_adap is None:
        video_adap = period.find("mpd:AdaptationSet", ns)

    reps = video_adap.findall("mpd:Representation", ns)
    
    rep_infos = []
    for rep_idx, rep in enumerate(reps):
        bw_bps = int(rep.attrib.get("bandwidth", 0))
        rep_id = rep.attrib.get("id", f"rep_{rep_idx}")
        # bandwidth가 0이면 skip
        if bw_bps > 0:
            rep_infos.append((rep_idx, rep_id, bw_bps))

    # 비트레이트 오름차순 정렬 (학습 시 일관성 유지)
    rep_infos.sort(key=lambda x: x[2])
    
    # 인덱스 재할당 (0부터 순서대로)
    final_rep_infos = []
    for new_idx, (_, r_id, bw) in enumerate(rep_infos):
        final_rep_infos.append((new_idx, r_id, bw))
        print(f"  - [Rep {new_idx}] ID={r_id}, Bandwidth={bw/1000:.0f} kbps")

    # 3. CSV 저장
    with output_csv.open("w", newline="") as f:
        fieldnames = ["chunk_idx", "rep_idx", "size_bytes", "duration_sec", "bitrate_kbps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for chunk_idx in range(1, num_chunks+1):
            for r_idx, r_id, bw_bps in final_rep_infos:
                # 근사치 계산: size = bitrate * duration
                size_bytes = int(bw_bps * seg_duration / 8.0)
                writer.writerow({
                    "chunk_idx": chunk_idx,
                    "rep_idx": r_idx,
                    "size_bytes": size_bytes,
                    "duration_sec": seg_duration,
                    "bitrate_kbps": int(bw_bps / 1000)
                })
    
    print(f"  -> Saved to {output_csv}\n")

if __name__ == "__main__":
    for info in DATASETS:
        process_mpd(info)