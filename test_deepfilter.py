
from pathlib import Path

import torch
from df.enhance import (
    init_df,
    load_audio,
    enhance,
    save_audio,
)


def main():
    input_path = Path("data/noisy/test.wav")
    output_path = Path("output/clean.wav")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load pretrained DeepFilterNet model
    model, df_state, _ = init_df()

    # Load audio and resample to the model's sample rate
    audio, _ = load_audio(
        str(input_path),
        sr=df_state.sr(),
    )

    # Move audio to the same device as the model
    device = next(model.parameters()).device
    audio = audio.to(device)

    # Noise suppression inference
    with torch.no_grad():
        enhanced = enhance(
            model,
            df_state,
            audio,
        )

    # Save the enhanced audio
    save_audio(
        str(output_path),
        enhanced.cpu(),
        df_state.sr(),
    )

    print(f"Finished: {output_path}")


if __name__ == "__main__":
    main()
