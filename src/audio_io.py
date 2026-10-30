from pathlib import Path

import soundfile as sf

from src.config import MAX_AUDIO_SECONDS


class AudioValidationError(ValueError):
    """Lỗi khi file đầu vào không đáp ứng yêu cầu."""


def inspect_audio(input_path: str) -> dict:
    """
    Kiểm tra file và trả về metadata cơ bản.
    Chưa thay đổi dữ liệu âm thanh.
    """

    path = Path(input_path)

    # 1. Kiểm tra đường dẫn
    if not path.exists() or not path.is_file():
        raise AudioValidationError(
            "Không tìm thấy file âm thanh."
        )

    # 2. MVP chỉ nhận WAV
    if path.suffix.lower() != ".wav":
        raise AudioValidationError(
            "Vui lòng sử dụng file WAV."
        )

    # 3. Đọc thông tin file mà không cần tải toàn bộ audio vào RAM
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise AudioValidationError(
            "File WAV không hợp lệ hoặc không đọc được."
        ) from exc

    if info.samplerate <= 0 or info.frames <= 0:
        raise AudioValidationError(
            "File âm thanh không có dữ liệu hợp lệ."
        )

    duration = info.frames / info.samplerate

    if duration > MAX_AUDIO_SECONDS:
        raise AudioValidationError(
            f"File dài {duration:.1f} giây. "
            f"Giới hạn hiện tại là {MAX_AUDIO_SECONDS} giây."
        )

    if info.channels > 2:
        raise AudioValidationError(
            "Bản demo chỉ hỗ trợ audio mono hoặc stereo."
        )

    # 4. Trả về thông tin để pipeline sử dụng
    return {
        "path": str(path.resolve()),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_seconds": duration,
        "frames": info.frames
    }