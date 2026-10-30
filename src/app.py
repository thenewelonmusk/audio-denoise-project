import traceback

import gradio as gr

from src.audio_io import AudioValidationError
from src.pipeline import run_pipeline


def process_audio(input_path):
    """
    Hàm được gọi khi người dùng nhấn nút Khử nhiễu.
    input_path là đường dẫn file do Gradio cung cấp.
    """

    if not input_path:
        raise gr.Error("Vui lòng tải lên file âm thanh.")

    try:
        # Chuyển việc xử lý thực tế cho pipeline
        result = run_pipeline(input_path)

    except AudioValidationError as exc:
        # Lỗi do file đầu vào không hợp lệ
        raise gr.Error(str(exc)) from exc

    except Exception as exc:
        # Ghi đầy đủ lỗi cho nhóm debug tại terminal
        print("Lỗi khi xử lý âm thanh:")
        traceback.print_exc()

        # Chỉ hiện thông báo dễ hiểu cho người dùng
        raise gr.Error(
            "Xử lý thất bại. Hãy kiểm tra log trong terminal."
        ) from exc

    # Chuẩn bị nội dung hiển thị sau khi xử lý xong
    stats_text = (
        f"Thời lượng đầu vào: "
        f"{result['audio_duration_seconds']:.2f} giây\n"
        f"Thời gian xử lý: "
        f"{result['elapsed_seconds']:.2f} giây\n"
        f"Real-time factor: {result['rtf']:.3f}"
    )

    # Giá trị trả về phải tương ứng với các output của Gradio
    return result["output_path"], stats_text


with gr.Blocks(title="AI Speech Enhancement") as demo:
    gr.Markdown(
        "# AI Speech Enhancement\n"
        "Khử nhiễu giọng nói bằng mô hình AI chạy local."
    )

    input_audio = gr.Audio(
        sources=["upload"],
        type="filepath",
        format="wav",
        label="1. Âm thanh đầu vào"
    )

    process_button = gr.Button(
        "Khử nhiễu",
        variant="primary"
    )

    output_audio = gr.Audio(
        label="2. Âm thanh sau khử nhiễu",
        format="wav"
    )

    stats_output = gr.Textbox(
        label="Thông tin xử lý",
        interactive=False
    )

    process_button.click(
        fn=process_audio,
        inputs=[input_audio],
        outputs=[output_audio, stats_output]
    )


if __name__ == "__main__":
    demo.launch()