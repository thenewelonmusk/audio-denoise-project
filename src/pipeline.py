from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.audio_io import inspect_audio
from src.benchmark import summarize_timing
from src.config import OUTPUT_DIR
from src.enhancer import enhance_audio


def run_pipeline(input_path: str) -> dict:
    """
    Điều phối một lần xử lý audio từ đầu vào đến đầu ra.
    """

    # Bước 1: Kiểm tra đầu vào
    audio_info = inspect_audio(input_path)

    # Bước 2: Chuẩn bị thư mục đầu ra
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Tạo tên file duy nhất, tránh ghi đè file trước
    input_name = Path(input_path).stem
    unique_id = uuid4().hex[:8]

    output_path = OUTPUT_DIR / (
        f"{input_name}_enhanced_{unique_id}.wav"
    )

    # Bước 3: Bắt đầu đo thời gian
    start_time = perf_counter()

    try:
        # Bước 4: Gọi module AI
        saved_path = enhance_audio(
            input_path=audio_info["path"],
            output_path=str(output_path)
        )

    except Exception:
        # Xóa file dở dang nếu có lỗi
        output_path.unlink(missing_ok=True)
        raise

    # Bước 5: Kết thúc đo thời gian
    elapsed = perf_counter() - start_time

    # Bước 6: Tính các chỉ số thời gian
    stats = summarize_timing(
        elapsed_seconds=elapsed,
        audio_duration_seconds=audio_info["duration_seconds"]
    )

    # Bước 7: Trả kết quả cho giao diện
    return {
        "output_path": saved_path,
        "sample_rate": audio_info["sample_rate"],
        "channels": audio_info["channels"],
        **stats
    }