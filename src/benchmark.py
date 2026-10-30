def summarize_timing(
    elapsed_seconds: float,
    audio_duration_seconds: float
) -> dict:
    """
    Tính các chỉ số thời gian cho một lần xử lý.
    """

    if audio_duration_seconds <= 0:
        raise ValueError(
            "Thời lượng audio phải lớn hơn 0."
        )

    rtf = elapsed_seconds / audio_duration_seconds

    return {
        "elapsed_seconds": round(
            elapsed_seconds, 3
        ),
        "audio_duration_seconds": round(
            audio_duration_seconds, 3
        ),
        "rtf": round(rtf, 3)
    }