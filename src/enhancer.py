import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.config import MODEL_NAME, MODEL_TIMEOUT_SECONDS


def enhance_audio(
    input_path: str,
    output_path: str
) -> str:
    """
    Gọi DeepFilterNet2 bằng CLI.

    Parameters
    ----------
    input_path:
        Đường dẫn đến file WAV đầu vào.
    output_path:
        Đường dẫn mong muốn của file WAV đầu ra.

    Returns
    -------
    str:
        Đường dẫn của file đã xử lý.
    """

    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Tạo thư mục tạm riêng cho lần xử lý này
    with tempfile.TemporaryDirectory(
        prefix="deepfilter_"
    ) as temp_dir:

        # Gọi module CLI trong chính môi trường Python hiện tại
        command = [
            sys.executable,
            "-m",
            "df.enhance",
            "-m",
            MODEL_NAME,
            "--output-dir",
            temp_dir,
            str(source)
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=MODEL_TIMEOUT_SECONDS,
                check=False
            )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Mô hình xử lý quá thời gian cho phép."
            ) from exc

        # Nếu mô hình trả mã lỗi khác 0, coi như xử lý thất bại
        if completed.returncode != 0:
            error_details = (
                completed.stderr[-2000:]
                if completed.stderr
                else "Không có thông tin lỗi."
            )

            raise RuntimeError(
                "DeepFilterNet xử lý thất bại:\n"
                + error_details
            )

        # DeepFilterNet tự đặt tên file trong thư mục output.
        # Tìm file WAV được tạo bởi đúng lần chạy này.
        generated_files = list(
            Path(temp_dir).glob("*.wav")
        )

        if not generated_files:
            raise RuntimeError(
                "Mô hình kết thúc nhưng không tạo ra file WAV."
            )

        # Một input nên sinh một file output
        generated_file = generated_files[0]

        # Đưa kết quả về đường dẫn do pipeline quản lý
        shutil.copy2(
            generated_file,
            destination
        )

    return str(destination)