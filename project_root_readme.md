# ACOUSTIC STUDIO // Hệ thống Khử nhiễu Giọng nói & Âm thanh AI

**Đồ án môn học:** Trí tuệ Nhân tạo / Xử lý Tín hiệu Số (`UET.CS2046` — Nhóm 13)  
**Mô hình cốt lõi:** DeepFilterNet3 tinh chỉnh (Fine-tuned) trên dữ liệu tiếng Việt **VIVOS** & tạp âm **UrbanSound8K**  
**Giao diện:** Acoustic Studio — Digital Audio Workstation (DAW) xây dựng trên **Gradio**

---

## 1. Giới thiệu & Tính năng chính

Hệ thống cung cấp giải pháp khử nhiễu giọng nói toàn dải ($48\text{ kHz}$) chạy trực tiếp trên CPU phổ thông, hỗ trợ cả xử lý tệp âm thanh dài bất kỳ lẫn khử nhiễu luồng trực tiếp từ Microphone:

* **Đa định dạng đầu vào & Chuẩn hóa tự động (`CT-01`):** Hỗ trợ `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`. Tự động giải mã qua `ffmpeg`, trộn đa kênh về đơn kênh (`Mono`) và nội suy đa pha (*polyphase resampling*) về chuẩn $48\text{ kHz}$ `float32`.
* **Xử lý tệp dài không giới hạn thời lượng (`CT-04`):** Tự động chia nhỏ các tệp âm thanh $> 60\text{s}$ thành các phân đoạn $\le 60\text{s}$ với dải đệm gối đầu $0.5\text{s}$, sau đó tái cấu trúc bằng thuật toán **Overlap-Add (OLA)** sử dụng cửa sổ làm mượt Hanning ($w_{\text{out}} + w_{\text{in}} = 1.0$) giúp loại bỏ hoàn toàn tiếng lách cách (*click/pop*) tại biên nối.
* **Kiến trúc Plug-and-Play Engine (`CT-02`):** Cho phép chuyển đổi nóng giữa các mô hình ngay trên giao diện:
  * `dfn3_ft_vivos_v1`: DeepFilterNet3 tinh chỉnh chuyên biệt cho giọng nói tiếng Việt (Mặc định).
  * `dfnet2`: DeepFilterNet2 Pretrained gốc.
  * `mock`: Bộ lọc dải thông Butterworth ($100\text{ Hz} - 7.5\text{ kHz}$) siêu nhẹ dùng cho kiểm thử đường ống không cần GPU.
* **Giao diện DAW & Live Streaming (`CT-05`):**
  * **Chế độ Xử lý Tệp:** Đối chiếu âm thanh Before/After, hiển thị biểu đồ phổ tần số STFT Spectrogram và đo hiệu năng thời gian thực (`RTF`, `Peak RAM`).
  * **Chế độ Live Streaming:** Khử nhiễu luồng trực tiếp từ WebRTC Microphone theo cửa sổ trượt $200\text{ ms}$ có duy trì trạng thái phiên.

---

## 2. Cấu trúc thư mục dự án

```text
audio-denoise-project-app_dfn_dev/
├── README.md                          # Tài liệu hướng dẫn cài đặt & vận hành tổng thể
├── requirements.txt                   # Danh sách thư viện phụ thuộc của hệ thống
├── data/
│   ├── noisy/test.wav                 # Mẫu âm thanh kiểm thử cơ bản
│   └── samples/                       # Kho mẫu kiểm thử đa định dạng & tệp dài (15s - 150s)
├── docs/
│   └── final_report.md                # Báo cáo tổng kết đồ án cuối kỳ
├── ml/                                # Không gian huấn luyện & đánh giá ngoại tuyến (Offline Eval)
│   ├── final_models/
│   │   └── dfn3_ft_vivos_v1/          # Trọng số mô hình fine-tune (config.ini + checkpoints/)
│   ├── eval/                          # Bộ đo lường định lượng (SI-SDR, STOI, PESQ proxy, RTF)
│   └── training/                      # Mã nguồn huấn luyện & xuất mô hình
├── results/
│   ├── benchmark.csv                  # Số liệu đo hiệu năng CPU thực tế (RTF & Peak RAM)
│   └── evaluation.csv                 # Bảng đối chứng chất lượng âm thanh trên tập kiểm thử
└── VDenoise1.1/                       # Mã nguồn ứng dụng chính (Production App)
    ├── app.py                         # Giao diện Web DAW (Gradio)
    ├── models/                        # Các engine suy luận & registry cấu hình
    │   ├── dfn3_vivos.py              # Engine cho mô hình DeepFilterNet3 VIVOS
    │   ├── dfnet_engine.py            # Engine cho DeepFilterNet2
    │   ├── mock_engine.py             # Engine giả lập DSP cho kiểm thử tự động
    │   ├── _libdf_compat.py           # Lớp tương thích thuần Python cho Python 3.12+
    │   └── models_registry.json       # Danh mục đăng ký động các mô hình
    ├── src/                           # Lõi xử lý tín hiệu & điều phối hệ thống
    │   ├── config.py                  # Tham số cấu hình tập trung (Chunk 60s, OLA 0.5s, 48kHz)
    │   ├── audio_io.py                # Chuẩn hóa âm thanh, Chunking, OLA, vẽ Spectrogram
    │   ├── engine_interface.py        # Giao diện trừu tượng AudioDenoiseEngine & Factory
    │   ├── benchmark.py               # Đo thời gian thực (RTF) & đỉnh bộ nhớ RAM (psutil)
    │   ├── pipeline.py                # Luồng điều phối tổng thể run_pipeline (CT-04)
    │   └── theme.py                   # Giao diện chủ đề DAW tối màu
    └── tests/                         # Bộ kiểm thử tự động (pytest)
        ├── samples/                   # Tệp mẫu nạp nhanh 1-click cho UI
        ├── test_pipeline.py           # Kiểm thử tích hợp luồng cắt/ghép OLA & ngoại lệ
        └── test_qa_15_cases.py        # Bộ 15 kịch bản kiểm thử chất lượng phần mềm (QA)
```

---

## 3. Hướng dẫn cài đặt từ đầu (Clean Environment)

### Bước 1: Yêu cầu hệ thống
* **Hệ điều hành:** Windows 10/11, Linux hoặc macOS.
* **Python:** Khuyến nghị **Python 3.10 – 3.12**.
* **FFmpeg:** Hệ thống tích hợp sẵn qua gói `imageio-ffmpeg`, tuy nhiên khuyến nghị cài đặt `ffmpeg` vào `PATH` hệ thống để đạt hiệu năng giải mã tối ưu.

### Bước 2: Tạo môi trường ảo và cài đặt thư viện
Mở terminal tại thư mục gốc `audio-denoise-project-app_dfn_dev/` và chạy:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:
* **Windows (CMD / PowerShell):**
  ```powershell
  .venv\Scripts\activate
  ```
* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

Cài đặt toàn bộ gói phụ thuộc:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install psutil pytest imageio-ffmpeg
```

### Bước 3: Kiểm tra tệp trọng số mô hình Fine-tune
Đảm bảo thư mục `ml/final_models/dfn3_ft_vivos_v1/` chứa đủ 2 thành phần sau:
```text
ml/final_models/dfn3_ft_vivos_v1/
├── config.ini
└── checkpoints/
    └── model_7.ckpt.best
```

---

## 4. Hướng dẫn vận hành & Kiểm thử

### 4.1. Khởi chạy ứng dụng Web (Acoustic Studio DAW)
Di chuyển vào thư mục `VDenoise1.1/` và khởi động máy chủ Gradio:

```bash
cd VDenoise1.1
python app.py
```

Sau khi terminal hiển thị địa chỉ cục bộ (mặc định `http://127.0.0.1:7860`), mở trình duyệt để sử dụng:
1. **Tab Xử lý Tệp:** Tải tệp âm thanh lên (hoặc nhấn chip **Mẫu thử nhanh**), chọn mô hình tại mục **Động cơ khử nhiễu**, sau đó nhấn **Khử Nhiễu Âm Thanh**.
2. **Tab Live Streaming:** Cấp quyền Microphone cho trình duyệt và nói trực tiếp để nghe luồng âm thanh đã lọc tạp âm theo thời gian thực.

### 4.2. Chạy bộ kiểm thử tự động (`pytest`)
Tại thư mục `VDenoise1.1/`, chạy lệnh sau để thực thi toàn bộ 18 bài kiểm thử (kiểm tra cắt khối 60s, ghép nối Overlap-Add, đo Peak RAM và 15 kịch bản QA):

```bash
python -m pytest -v
```

### 4.3. Chạy đo hiệu năng hệ thống (Benchmark CLI)
Để đo tốc độ `RTF` và đỉnh bộ nhớ `Peak RAM` và ghi kết quả vào `results/benchmark.csv`:

```bash
python -c "
from src.pipeline import run_pipeline
res = run_pipeline('../data/samples/sample_long_150s.wav', engine_name='dfn3_ft_vivos_v1', log_benchmark=True)
print('Chunks:', res['num_chunks'], '| RTF:', res['rtf'], '| Peak RAM (MB):', res['peak_ram_mb'])
"
```