# BÁO CÁO TỔNG KẾT BÀI TẬP LỚN

**Môn học:** Trí tuệ Nhân tạo 

**Đề tài:** Hệ thống Khử Nhiễu Giọng Nói Tiếng Việt Thời Gian Thực Tích Hợp Học Sâu

**Nhóm thực hiện:** Nhóm 13 (Lớp CS2046_6)

## Tóm tắt (Abstract)

Ghi âm giọng nói tiếng Việt trong môi trường thực tế thường chịu tác động nghiêm trọng từ tạp âm nền như tiếng xe cộ, tiếng quạt gió, tiếng ồn đường phố và nhiễu thiết bị. Báo cáo này trình bày quy trình thiết kế, tinh chỉnh mô hình học sâu và xây dựng hệ thống khử nhiễu âm thanh đầu cuối **Acoustic Studio (VDenoise 1.1)**.

Về mặt mô hình, nhóm thực hiện tinh chỉnh (*fine-tuning*) kiến trúc **DeepFilterNet3** trên tập ngữ liệu tiếng Việt **VIVOS** kết hợp tạp âm môi trường **UrbanSound8K**, giúp cải thiện chỉ số `SI-SDR` trên tập kiểm định từ `17.51 dB` lên `23.37 dB` (`+5.86 dB`). Về mặt kỹ thuật hệ thống, ứng dụng được xây dựng theo kiến trúc phân tầng 3 lớp tách biệt, hỗ trợ đa định dạng đầu vào (`WAV`, `MP3`, `M4A`, `AAC`, `FLAC`, `OGG`), xử lý tệp thời lượng dài không giới hạn thông qua cơ chế phân đoạn $\le 60\text{s}$ kết hợp ghép nối **Overlap-Add (OLA)** dùng cửa sổ **Hanning cross-fade**, đồng thời hỗ trợ khử nhiễu luồng trực tiếp (**Live Streaming**) qua Microphone. Thực nghiệm trên CPU phổ thông cho thấy hệ thống đạt hệ số thời gian thực $\text{RTF} \in [0.081, 0.123]$ (nhanh gấp $8\text{–}12$ lần thời gian thực) và vượt qua $18/18$ kịch bản kiểm thử chất lượng phần mềm tự động.

## Chương 1. Tổng quan Bài toán và Mục tiêu Kỹ thuật

### 1.1. Đặt vấn đề

Trong các ứng dụng thực tế như ghi âm bài giảng, hội họp trực tuyến, phỏng vấn hiện trường và tiền xử lý cho hệ thống nhận dạng giọng nói tự động (ASR), chất lượng tín hiệu tiếng Việt thường bị suy giảm mạnh bởi tạp âm môi trường không dừng (*non-stationary noise*). Khác với các ngôn ngữ không thanh điệu, tiếng Việt sở hữu hệ thống 6 thanh điệu phụ thuộc chặt chẽ vào đường bao tần số cơ bản ($F_0$) và các formant ở vùng tần số thấp và trung. Các giải thuật lọc tín hiệu cổ điển (như trừ phổ *Spectral Subtraction* hay bộ lọc thông dải *Bandpass*) thường gây méo dạng sóng, làm mờ phụ âm cuối hoặc phát sinh nhiễu nhạc (*musical noise*), khiến giọng nói mất đi độ tự nhiên.

### 1.2. Mục tiêu đề tài

Nhóm đặt mục tiêu xây dựng một hệ thống hoàn chỉnh từ lõi mô hình học sâu đến giao diện người dùng cuối với 4 tiêu chí kỹ thuật cốt lõi:

1. **Nâng cao chất lượng khử nhiễu cho tiếng Việt:** Tinh chỉnh mô hình **DeepFilterNet3** trên dữ liệu tiếng Việt **VIVOS** + **UrbanSound8K**, cải thiện đồng thời cả độ rõ tín hiệu (`SI-SDR`), độ dễ hiểu lời nói (`STOI`) và chất lượng cảm nhận (`PESQ`) so với mô hình tiền huấn luyện gốc.

2. **Tương thích đa định dạng thực tế:** Tự động tiếp nhận và chuẩn hóa các định dạng tệp phổ biến (`WAV`, `MP3`, `M4A`, `AAC`, `FLAC`, `OGG`) về chuẩn **48 kHz, 1 kênh (Mono), kiểu dữ liệu `float32`**.

3. **Xử lý tệp dài an toàn bộ nhớ:** Hỗ trợ các bản ghi âm dài nhiều phút thông qua cơ chế chia khối tối đa $60\text{s}$ và tái cấu trúc mượt mà bằng kỹ thuật **Overlap-Add (OLA)**, triệt tiêu hiện tượng tiếng lách cách (*click/pop*) tại ranh giới các phân đoạn.

4. **Hiệu năng thời gian thực trên CPU & Trải nghiệm trực quan:** Đạt hệ số thời gian thực $\text{RTF} < 0.30$ ngay trên CPU laptop; cung cấp giao diện trạm âm thanh số (DAW) cho phép nghe đối chiếu trực tiếp, quan sát biểu đồ phổ **STFT Spectrogram** và khử nhiễu **Live Streaming** từ Microphone.

## Chương 2. Cơ sở Lý thuyết và Kiến trúc Hệ thống

### 2.1. Kiến trúc Mô hình DeepFilterNet3

**DeepFilterNet3** là mô hình khử nhiễu giọng nói toàn dải (*Full-band 48 kHz*) hoạt động trên miền tần số phức (STFT với khung cửa sổ $20\text{ ms}$, bước nhảy $10\text{ ms}$), tận dụng đặc trưng tâm lý âm học của thính giác người để tối ưu hóa tốc độ suy luận trên CPU:

```
Tín hiệu nhiễu x[n] (48 kHz)
       │
       ▼
   [ STFT ] ──► Phổ biên độ & pha phức X(k, m)
       │
       ├──► Nhánh 1: Trích xuất đặc trưng ERB (32 dải) ──► Bộ mã hóa/giải mã GRU ──► Độ lợi G_ERB ──┐
       │                                                                                          ├──► Kết hợp phổ ──► [ iSTFT ] ──► Tín hiệu sạch y[n]
       └──► Nhánh 2: Phổ phức dải thấp (< 5 kHz) ─────────► Bộ lọc sâu đa khung DF (L = 5 khung) ───┘

```

* **Giai đoạn 1 — Ước lượng độ lợi theo thang ERB (*ERB Gain Estimation*):** Toàn bộ dải tần $0\text{–}24\text{ kHz}$ được nén thành $32$ dải tần tương đương hình chữ nhật (*Equivalent Rectangular Bandwidth*). Mạng nơ-ron hồi quy dự đoán hệ số suy giảm nhiễu cho từng dải ERB, giúp loại bỏ nhanh tạp âm nền trên toàn dải phổ với số lượng tham số rất nhỏ.

* **Giai đoạn 2 — Lọc sâu đa khung trên dải tần thấp (*Deep Filtering*** $< 5\text{ kHz}$**):** Trên vùng tần số dưới $5\text{ kHz}$ (nơi tập trung năng lượng giọng nói và thông tin thanh điệu tiếng Việt), mô hình áp dụng bộ lọc tuyến tính phức trên $5$ khung thời gian liên tiếp để khôi phục cấu trúc hài âm (*harmonics*) và loại bỏ nhiễu cộng hưởng giữa các đỉnh formant.

### 2.2. Kiến trúc Phần mềm 3 Tầng (Modular Layered Architecture)

Hệ thống được thiết kế theo nguyên tắc tách biệt trách nhiệm (*Separation of Concerns*), kết nối qua bộ hợp đồng giao diện chuẩn hóa (`CT-01` đến `CT-05`):

```
+-----------------------------------------------------------------------------+
| TẦNG 3: GIAO DIỆN & TRẢI NGHIỆM NGƯỜI DÙNG (app.py, src/theme.py)           |
|  • File Mode: Nạp tệp đa định dạng, đối chiếu Audio & STFT Spectrogram      |
|  • Live Streaming Mode: Khử nhiễu WebRTC Mic (200ms/khối), giữ trạng thái   |
+--------------------------------------+--------------------------------------+
                                       │ Hợp đồng CT-04 / CT-05
+--------------------------------------v--------------------------------------+
| TẦNG 2: ĐIỀU PHỐI & XỬ LÝ TÍN HIỆU (src/pipeline.py, audio_io.py, config.py)|
|  • Giải mã đa định dạng qua ffmpeg (MP3, M4A, AAC, FLAC, OGG -> WAV)        |
|  • Chuẩn hóa 48 kHz Mono float32 & vẽ Spectrogram đầu vào (CT-01)           |
|  • Phân đoạn <= 60s & Tái cấu trúc Overlap-Add (Hanning Cross-fade 0.5s)    |
|  • Đo lường RTF & Peak RSS RAM đa luồng qua psutil (CT-03 - benchmark.py)   |
+--------------------------------------+--------------------------------------+
                                       │ Hợp đồng CT-02
+--------------------------------------v--------------------------------------+
| TẦNG 1: ĐỘNG CƠ SUY LUẬN AI PLUG-AND-PLAY (src/engine_interface.py)        |
|  • EngineRegistry tự động nạp và quản lý động các mô hình trong models/:    |
|     - dfn3_vivos.py   : DeepFilterNet3 Fine-tuned (VIVOS + UrbanSound8K)    |
|     - dfnet_engine.py : DeepFilterNet2 / DeepFilterNet3 Pretrained          |
|     - mock_engine.py  : Bộ lọc thông dải Butterworth (CPU Unit Testing)     |
|  • Lớp tương thích _libdf_compat.py hỗ trợ chạy thuần Python trên Py 3.12+  |
+-----------------------------------------------------------------------------+

```

### 2.3. Giải thuật Cắt đoạn và Ghép nối Overlap-Add (OLA)

Để xử lý các bản ghi âm thời lượng dài mà không gây quá tải bộ nhớ, `src/audio_io.py` và `src/pipeline.py` thực hiện quy trình chia khối và tái cấu trúc tín hiệu theo 3 bước:

1. **Phân đoạn gối đầu:** Nếu thời lượng âm thanh $T > 60\text{s}$, chuỗi mẫu được chia thành các khối có độ dài tối đa $L_{\text{chunk}} = 60\text{s}$ ($2.880.000$ mẫu tại $48\text{ kHz}$), trong đó hai khối liền kề chồng lấn nhau một khoảng $L_{\text{ov}} = 0.5\text{ s}$ ($24.000$ mẫu).

2. **Suy luận độc lập trong RAM:** Từng khối được đưa qua hàm `engine.process_array()` trực tiếp trên mảng NumPy mà không cần ghi tệp tạm ra ổ cứng.

3. **Tổng hợp bằng cửa sổ Hanning bù trừ năng lượng:** Tại vùng chồng lấn $n \in [0, L_{\text{ov}} - 1]$ giữa khối thứ $k$ và khối thứ $k+1$, tín hiệu đầu ra được nội suy tuyến tính theo dạng sóng cửa sổ Hanning:

$$
w_{\text{in}}[n] = \frac{1}{2}\left(1 - \cos\left(\frac{\pi n}{L_{\text{ov}} - 1}\right)\right), \qquad w_{\text{out}}[n] = 1 - w_{\text{in}}[n]
$$

$$
y_{\text{splice}}[n] = y_k[L_{\text{chunk}} - L_{\text{ov}} + n] \cdot w_{\text{out}}[n] + y_{k+1}[n] \cdot w_{\text{in}}[n]
$$

Vì $w_{\text{out}}[n] + w_{\text{in}}[n] = 1$ tại mọi chỉ số mẫu $n$, biên độ tín hiệu được giữ liên tục cả về giá trị lẫn đạo hàm bậc nhất, loại bỏ hoàn toàn hiện tượng gián đoạn pha gây tiếng lách cách (*click/pop*) tại biên ghép nối.

## Chương 3. Huấn luyện Mô hình và Kết quả Thực nghiệm

### 3.1. Thiết lập Dữ liệu và Quá trình Fine-tuning

* **Ngữ liệu giọng nói sạch:** Tập dữ liệu tiếng Việt **VIVOS** (chuẩn hóa về tần số lấy mẫu $48\text{ kHz}$, đơn kênh Mono).

* **Ngữ liệu tạp âm nền:** Tập âm thanh môi trường **UrbanSound8K** gồm đa dạng nguồn nhiễu đời sống (động cơ, còi xe, tiếng ồn giao thông, tiếng nói chuyện nền), được trộn động với giọng nói sạch theo nhiều mức tỷ số tín hiệu trên nhiễu (SNR).

* **Diễn biến hội tụ:** Trong quá trình huấn luyện $8$ epoch (đánh chỉ số từ `0` đến `7`), hàm mất mát huấn luyện (`Training Loss`) giảm đều từ **`1.023`** xuống đáy **`0.899`** tại Epoch 6. Chỉ số `Validation SI-SDR` cải thiện mạnh từ mức nền nhiễu **`17.51 dB`** lên đỉnh **`23.37 dB`** (tăng **`+5.86 dB`**), được lưu lại tại checkpoint `model_7.ckpt.best`.

### 3.2. Đánh giá Chất lượng Khử nhiễu Ngoại tuyến (`results/evaluation.csv`)

Nhóm thực hiện đánh giá định lượng lặp lại $n = 3$ lần trên tập kiểm thử tiếng Việt có nhiễu nền, đối chiếu giữa mô hình gốc `DeepFilterNet3` và mô hình tinh chỉnh `DFN3 Fine-tuned (VIVOS v1)`:

| Phiên bản / Mô hình | SI-SDR (dB) $\uparrow$ | STOI $\uparrow$ | PESQ torch loss $\downarrow$ | RTF (Eval) $\downarrow$ | 
| ----- | ----- | ----- | ----- | ----- | 
| Tín hiệu gốc chưa xử lý (*Noisy*) | `[Cập nhật final]` | `[Cập nhật final]` | `[Cập nhật final]` | — | 
| DeepFilterNet3 (*Pretrained gốc*) | `16.2266 ± 0.0000` | `0.9121 ± 0.0000` | `1.8847 ± 0.0000` | `0.5507 ± 0.0301` | 
| **DFN3 Fine-tuned (`VIVOS v1`)** | **`17.0006 ± 0.0000`** | **`0.9192 ± 0.0000`** | **`1.5990 ± 0.0000`** | **`0.0218 ± 0.0021`** | 

**Phân tích kết quả:**

* **Cải thiện toàn diện 3 chỉ số âm học:** Mô hình tinh chỉnh đạt `SI-SDR = 17.0006 dB` (tăng $+0.774\text{ dB}$ so với `DFN3` gốc), `STOI = 0.9192` (vượt ngưỡng mục tiêu $\ge 0.90$) và giảm hàm mất mát `PESQ torch loss` từ `1.8847` xuống **`1.5990`** (giúp giọng tiếng Việt trong hơn và giảm hiện tượng nuốt âm cuối).

* **Tính ổn định tất định:** Độ lệch chuẩn của các chỉ số chất lượng bằng `0.0000` qua $n=3$ lần chạy do mô hình vận hành ở chế độ suy luận tất định (`eval()`), trong khi chỉ số `RTF` của bản fine-tuned đạt **`0.0218`**.

### 3.3. Đánh giá Hiệu năng Ứng dụng Thực tế trên CPU (`results/benchmark.csv`)

Để kiểm chứng khả năng vận hành thực tế của toàn bộ chuỗi xử lý (`ffmpeg transcode` $\rightarrow$ `chuẩn hóa 48 kHz` $\rightarrow$ `chunking` $\rightarrow$ `suy luận AI` $\rightarrow$ `ghép OLA` $\rightarrow$ `xuất WAV & Spectrogram`), nhóm thực hiện đo lường trực tiếp trên CPU laptop với 4 tệp đại diện:

| Tên tệp kiểm thử | Định dạng | Thời lượng | Số khối (Chunks) | Thời gian xử lý | Hệ số RTF | Peak RAM (MB) | Đạt RTF (< 0.3) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| `sample_vn_15s.wav` | WAV Mono | 15.0s | 1 | 1.850s | **0.123** | 1128.59 MB | **ĐẠT (TRUE)** | 
| `sample_stereo_44k.mp3` | MP3 Stereo | 15.0s | 1 | 1.799s | **0.120** | 1203.67 MB | **ĐẠT (TRUE)** | 
| `sample_phone_record.m4a` | M4A (AAC) | 15.0s | 1 | 1.562s | **0.104** | 1242.50 MB | **ĐẠT (TRUE)** | 
| `sample_long_150s.wav` | WAV Mono | 150.0s | 3 | 12.181s | **0.081** | 2713.45 MB | **ĐẠT (TRUE)** | 

**Nhận xét kỹ thuật:**

* **Tốc độ xử lý nhanh hơn thời gian thực 8–12 lần:** Tất cả các định dạng (bao gồm cả tệp nén `.mp3` và `.m4a` cần giải mã qua `ffmpeg`) đều đạt $\text{RTF} \in [0.081, 0.123]$, vượt xa ngưỡng mục tiêu $\text{RTF} < 0.30$. Đặc biệt, với tệp dài $2.5\text{ phút}$ (`150s`), hệ thống chỉ mất **`12.181s`** để hoàn tất toàn bộ 3 phân đoạn.

* **Hành vi tiêu thụ bộ nhớ RAM:** Với tệp $15\text{s}$, mức RAM đỉnh duy trì quanh **`1.13 – 1.24 GB`**. Với tệp dài $150\text{s}$ chia làm 3 khối $60\text{s}$, đỉnh RAM đạt `2.71 GB` do kích thước tensor biến đổi phổ STFT của khung $60\text{s}$ trên PyTorch CPU — hoàn toàn vận hành mượt mà trên các máy tính cá nhân tiêu chuẩn (8–16 GB RAM).

## Chương 4. Kiểm thử Chất lượng Phần mềm (QA)

Hệ thống được kiểm thử tự động thông qua framework `pytest` (`tests/test_pipeline.py` và `tests/test_qa_15_cases.py`), đạt tỷ lệ **`18/18 PASSED` (100%)**, bao phủ đầy đủ 15 kịch bản chức năng, biên dữ liệu và xử lý ngoại lệ:

| Mã TC | Nhóm kiểm thử | Kịch bản & Dữ liệu đầu vào | Kết quả kỳ vọng | Trạng thái | 
| ----- | ----- | ----- | ----- | ----- | 
| **TC-01** | Luồng cơ bản | Tệp ngắn $\le 60\text{s}$ (`sample_vn_15s.wav`) | Xử lý 1 chunk duy nhất, xuất 1 tệp WAV 48 kHz | **PASS** | 
| **TC-02** | Đa định dạng | Tệp nén MP3 (`sample_stereo_44k.mp3`) | Tự động giải mã qua `ffmpeg` về WAV 48 kHz | **PASS** | 
| **TC-03** | Đa định dạng | Ghi âm điện thoại M4A (`sample_phone_record.m4a`) | Giải mã luồng AAC thành công, giữ đúng thời lượng | **PASS** | 
| **TC-04** | Đa định dạng | Tệp không nén FLAC (`48 kHz`) | Đọc và chuẩn hóa thành tệp WAV đầu ra hợp lệ | **PASS** | 
| **TC-05** | Chuẩn hóa kênh | Âm thanh 2 kênh (Stereo) | Tự động trộn kênh về 1 kênh Mono (`channels == 1`) | **PASS** | 
| **TC-06** | Chuẩn hóa tần số | Âm thanh băng hẹp `16 kHz` | Nội suy đa pha (*polyphase resample*) lên `48 kHz` | **PASS** | 
| **TC-07** | Chia khối dài | Tệp trung bình `90s` ($> 60\text{s}$) | Tự động cắt thành 2 chunks và ghép thành 1 file | **PASS** | 
| **TC-08** | Chia khối dài | Tệp dài `150s` ($2.5\text{ phút}$) | Chia 3 chunks, bảo toàn chính xác $7.200.000$ mẫu | **PASS** | 
| **TC-09** | Chất lượng nối OLA | Tệp `70s` đi qua biên nối tại mốc `60s` | Biên độ liên tục, không xuất hiện bước nhảy gây tiếng lách cách | **PASS** | 
| **TC-10** | Ổn định nhiễu nặng | Tín hiệu lẫn nhiễu biên độ lớn | Tín hiệu đầu ra hữu hạn, không chứa `NaN/Inf`, thuộc `[-1, 1]` | **PASS** | 
| **TC-11** | Xử lý ngoại lệ | Tệp rỗng dung lượng `0 byte` | Ném ngoại lệ `AudioValidationError`, không sập ứng dụng | **PASS** | 
| **TC-12** | Xử lý ngoại lệ | Tệp văn bản đổi tên giả mạo đuôi `.mp3` | Phát hiện lỗi giải mã và báo `AudioValidationError` | **PASS** | 
| **TC-13** | Xử lý ngoại lệ | Đường dẫn tệp không tồn tại trên ổ đĩa | Báo lỗi không tìm thấy tệp theo đúng hợp đồng | **PASS** | 
| **TC-14** | Giao diện UI | Bấm xử lý khi chưa chọn file & khi có file hợp lệ | Hiện cảnh báo `gr.Error` khi rỗng; trả đủ 4 thành phần CT-05 | **PASS** | 
| **TC-15** | Live Streaming | Luồng âm thanh `200ms`/khối & nút Reset Stream | Xử lý liên tục giữ trạng thái phiên và đặt lại bộ đệm | **PASS** | 

## Chương 5. Tổng kết, Hạn chế và Hướng Phát triển

### 5.1. Kết quả đạt được

* Xây dựng hoàn chỉnh ứng dụng **Acoustic Studio (VDenoise 1.1)** với giao diện DAW trực quan, hỗ trợ đối chiếu âm thanh và biểu đồ phổ STFT trước/sau khi khử nhiễu, cùng chế độ **Live Streaming** thời gian thực.

* Tinh chỉnh thành công mô hình **DeepFilterNet3** cho tiếng Việt, cải thiện rõ rệt các chỉ số `SI-SDR`, `STOI`, `PESQ loss` và đạt tốc độ nhanh gấp $8\text{–}12$ lần thời gian thực trên CPU ($\text{RTF} \in [0.081, 0.123]$).

* Kiến trúc hệ thống tuân thủ nghiêm ngặt phân tách tầng (`CT-01` đến `CT-05`), có bộ kiểm thử tự động `pytest` bao phủ 18 kịch bản kiểm thử đạt tỷ lệ thành công 100%.

### 5.2. Hạn chế và Hướng phát triển

1. **Tối ưu bộ nhớ cho thiết bị cấu hình thấp:** Với tệp dài trên 2 phút, kích thước phân đoạn $60\text{s}$ làm đỉnh cấp phát tensor của PyTorch đạt mức `2.71 GB`. Trong phiên bản tiếp theo, nhóm sẽ giảm kích thước khối suy luận xuống $15\text{–}20\text{s}$ kết hợp giải phóng bộ nhớ chủ động để giữ `Peak RAM` luôn dưới `1.2 GB` trên mọi độ dài tệp.

2. **Mở rộng dữ liệu huấn luyện thực tế:** Bổ sung thêm các kịch bản phòng có độ vang lớn (Room Impulse Response - RIR) và nhiễu tiếng người nói xen lẫn (*babble noise*) để nâng cao khả năng tách giọng trong môi trường hội thảo đông người.

3. **Đóng gói triển khai độc lập:** Chuyển đổi mô hình sang định dạng **ONNX Runtime** lượng tử hóa `INT8` để giảm dung lượng gói cài đặt và loại bỏ sự phụ thuộc vào môi trường PyTorch đầy đủ khi phân phối cho người dùng cuối.

## Chương 6. Phân công Công việc Nhóm

| Thành viên | Vai trò chính | Các hạng mục đã hoàn thành | 
| ----- | ----- | ----- | 
| **Thành viên A** | **Nhóm trưởng & Tích hợp Hệ thống** | Thiết kế kiến trúc 3 tầng và bộ hợp đồng `CT-01` $\rightarrow$ `CT-05`; xây dựng `src/config.py`, `src/benchmark.py` (đo RTF & Peak RAM bằng `psutil`), `src/pipeline.py` (điều phối Chunking & Overlap-Add); viết bộ kiểm thử tự động `tests/` (18 tests) và tổng hợp báo cáo. | 
| **Thành viên B** | **Nghiên cứu & Huấn luyện Mô hình AI** | Xây dựng pipeline tiền xử lý dữ liệu VIVOS + UrbanSound8K; thực hiện fine-tuning mô hình DeepFilterNet3 (`dfn3_ft_vivos_v1`); chạy đánh giá định lượng (`SI-SDR`, `STOI`, `PESQ torch loss`) và xuất trọng số `model_7.ckpt.best`. | 
| **Thành viên C** | **Phát triển Giao diện & Lõi Âm thanh** | Xây dựng giao diện Gradio DAW (`app.py`, `src/theme.py`); cài đặt hàm chuẩn hóa 48 kHz Mono, phân đoạn & tái cấu trúc OLA (`src/audio_io.py`), lớp giao diện `EngineRegistry` (`src/engine_interface.py`) và chế độ Live Streaming. | 
| **Thành viên D** | **Kiểm thử Dữ liệu & Đảm bảo Chất lượng** | Thu thập mẫu âm thanh kiểm thử đa định dạng, thực hiện kiểm thử thính giác biên nối OLA, đối soát kết quả thực nghiệm và hỗ trợ hoàn thiện tài liệu hướng dẫn sử dụng. | 
