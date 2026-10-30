from pathlib import Path

# Thư mục gốc của dự án: audio-denoise/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Thư mục lưu file đã xử lý
OUTPUT_DIR = PROJECT_ROOT / "output"

# Giới hạn cho bản demo MVP
MAX_AUDIO_SECONDS = 30

# Tên mô hình pretrained cần sử dụng
MODEL_NAME = "DeepFilterNet2"

# Timeout tối đa cho một lần xử lý
MODEL_TIMEOUT_SECONDS = 300

