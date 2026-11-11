"""
src/theme.py - Custom Gradio Digital Audio Workstation (DAW) Theme.

Implements an Anti-AI-Slop studio rackmount aesthetic:
- Deep zinc/slate dark surfaces (#09090b, #18181b, #27272a)
- Emerald/cyan audio meter accents (#10b981, #06b6d4)
- High-contrast technical borders (#27272a, #3f3f46)
- Monospaced typography for telemetry metrics (JetBrains Mono / Space Grotesk)
- Zero generic purple/indigo gradients or glassmorphism blur
"""

from __future__ import annotations

from typing import Tuple

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class AudioWorkstationTheme(Base):
    """
    Studio Hardware Rackmount Theme for Audio Processing Application.
    Subclasses gr.themes.Base with specialized dark zinc/emerald tokens.
    """

    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.cyan,
        neutral_hue: colors.Color | str = colors.zinc,
        spacing_size: sizes.Size | str = sizes.spacing_sm,
        radius_size: sizes.Size | str = sizes.radius_sm,
        text_size: sizes.Size | str = sizes.text_md,
        font: tuple[fonts.Font | str, ...] = (
            fonts.GoogleFont("Space Grotesk", weights=(400, 500, 600, 700)),
            "ui-sans-serif",
            "system-ui",
            "-apple-system",
            "sans-serif",
        ),
        font_mono: tuple[fonts.Font | str, ...] = (
            fonts.GoogleFont("JetBrains Mono", weights=(400, 500, 700)),
            "Consolas",
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        self.name = "audio_workstation_theme"

        super().set(
            # Page body & container surfaces (Matte Zinc Dark)
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
            body_text_color="*neutral_200",
            body_text_color_dark="*neutral_200",
            background_fill_primary="*neutral_900",
            background_fill_primary_dark="*neutral_900",
            background_fill_secondary="*neutral_800",
            background_fill_secondary_dark="*neutral_800",
            border_color_primary="*neutral_800",
            border_color_primary_dark="*neutral_800",
            border_color_accent="*primary_500",
            border_color_accent_dark="*primary_500",

            # Blocks & Panels (Hardware Module Look)
            panel_background_fill="*neutral_900",
            panel_background_fill_dark="*neutral_900",
            panel_border_color="*neutral_800",
            panel_border_color_dark="*neutral_800",
            block_background_fill="*neutral_900",
            block_background_fill_dark="*neutral_900",
            block_border_color="*neutral_800",
            block_border_color_dark="*neutral_800",
            block_border_width="1px",
            block_radius="*radius_sm",
            block_shadow="none",
            block_label_background_fill="*neutral_800",
            block_label_background_fill_dark="*neutral_800",
            block_label_border_color="*neutral_700",
            block_label_border_color_dark="*neutral_700",
            block_label_text_color="*neutral_300",
            block_label_text_color_dark="*neutral_300",
            block_title_text_color="*neutral_100",
            block_title_text_color_dark="*neutral_100",
            block_title_text_weight="600",

            # Primary Action Button (Studio Emerald Signal)
            button_primary_background_fill="*primary_600",
            button_primary_background_fill_dark="*primary_600",
            button_primary_background_fill_hover="*primary_500",
            button_primary_background_fill_hover_dark="*primary_500",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_primary_border_color="*primary_500",
            button_primary_border_color_dark="*primary_500",

            # Secondary Action Button
            button_secondary_background_fill="*neutral_800",
            button_secondary_background_fill_dark="*neutral_800",
            button_secondary_background_fill_hover="*neutral_700",
            button_secondary_background_fill_hover_dark="*neutral_700",
            button_secondary_text_color="*neutral_200",
            button_secondary_text_color_dark="*neutral_200",
            button_secondary_border_color="*neutral_700",
            button_secondary_border_color_dark="*neutral_700",

            # Form Inputs & Audio Player Widgets
            input_background_fill="*neutral_900",
            input_background_fill_dark="*neutral_900",
            input_border_color="*neutral_700",
            input_border_color_dark="*neutral_700",
            input_border_color_focus="*primary_500",
            input_border_color_focus_dark="*primary_500",

            # Tabs styling
            color_accent_soft="*neutral_800",
            color_accent_soft_dark="*neutral_800",
        )


DAW_CUSTOM_CSS = """
/* Anti-AI-Slop DAW Workstation Styling */
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif !important;
    background-color: #09090b !important;
}

/* DAW Header Rack */
.daw-header {
    background: #121316;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.daw-header h1, .daw-header-title h1 {
    font-size: 20px;
    font-weight: 700;
    color: #f4f4f5;
    margin: 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.daw-header p, .daw-header-title p {
    font-size: 13px;
    color: #a1a1aa;
    margin: 4px 0 0 0;
}
.daw-header-badges {
    display: flex !important;
    gap: 10px !important;
    align-items: center !important;
    justify-content: flex-end !important;
    flex-wrap: nowrap !important;
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: fit-content !important;
}
.daw-header-badges > * {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: fit-content !important;
}
.daw-badge {
    background: #064e3b;
    color: #34d399;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 4px;
    border: 1px solid #059669;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
    height: 32px !important;
    box-sizing: border-box !important;
    line-height: 1 !important;
}
.daw-badge-mono {
    border-color: #06b6d4 !important;
    color: #22d3ee !important;
    background: #083344 !important;
}
.daw-badge-wrapper {
    display: inline-flex !important;
    align-items: center !important;
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: fit-content !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* Interactive Engine Ready Button */
button.daw-engine-btn {
    background: #064e3b !important;
    color: #34d399 !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 5px 12px !important;
    border-radius: 4px !important;
    border: 1px solid #059669 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    min-height: unset !important;
    height: 32px !important;
    box-sizing: border-box !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.15s ease-in-out !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}
button.daw-engine-btn:hover {
    background: #065f46 !important;
    border-color: #10b981 !important;
    color: #a7f3d0 !important;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.35) !important;
    transform: translateY(-1px) !important;
}
button.daw-engine-btn:active {
    background: #047857 !important;
    transform: translateY(0) !important;
}
button.daw-engine-btn:focus {
    outline: 2px solid #10b981 !important;
    outline-offset: 1px !important;
}

/* Engine Selector Rack Panel */
#daw-engine-rack,
.daw-engine-rack {
    background: #121316 !important;
    background-color: #121316 !important;
    border: 1px solid #059669 !important;
    border-radius: 6px !important;
    padding: 16px 18px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5) !important;
}
.daw-engine-rack-header {
    background: transparent !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #27272a !important;
    flex-wrap: nowrap !important;
    gap: 12px !important;
}
.daw-engine-rack-title {
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
    color: #34d399;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    flex: 1 1 auto;
}
button.daw-close-rack-btn {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 80px !important;
    height: 30px !important;
    padding: 4px 12px !important;
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #e4e4e7 !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}
button.daw-close-rack-btn:hover {
    background: #3f3f46 !important;
    color: #ffffff !important;
    border-color: #52525b !important;
}
.daw-engine-rack .block,
.daw-engine-rack .form,
.daw-engine-rack fieldset {
    background: #18181b !important;
    border-color: #27272a !important;
}
.daw-engine-rack label span {
    color: #d4d4d8 !important;
}
.daw-engine-rack select,
.daw-engine-rack input {
    background: #141518 !important;
    color: #f4f4f5 !important;
    border-color: #27272a !important;
}
.daw-engine-meta-card {
    background: #090a0c;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 12px;
    color: #d4d4d8;
    line-height: 1.6;
}
.daw-engine-meta-card strong {
    color: #22d3ee;
}
.daw-engine-status-tag {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.daw-engine-status-tag.active {
    background: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}
.daw-engine-status-tag.standby {
    background: #1e293b;
    color: #38bdf8;
    border: 1px solid #0284c7;
}

/* Telemetry Grid & Hardware Cards */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 12px 0 16px 0;
}
@media (max-width: 768px) {
    .telemetry-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.telemetry-card {
    background: #141518;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
}
.telemetry-label {
    font-size: 11px;
    font-weight: 600;
    color: #71717a;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.telemetry-value {
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 18px;
    font-weight: 700;
    color: #f4f4f5;
}
.telemetry-unit {
    font-size: 12px;
    color: #a1a1aa;
    font-weight: 400;
    margin-left: 2px;
}
.status-good {
    color: #10b981 !important;
}
.status-warn {
    color: #f59e0b !important;
}
.status-alert {
    color: #ef4444 !important;
}

/* Studio Audio Meters & Fader Bars */
.audio-meter-bar {
    height: 6px;
    background: #27272a;
    border-radius: 3px;
    overflow: hidden;
    position: relative;
    margin-top: 6px;
}
.audio-meter-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #06b6d4);
    border-radius: 3px;
    width: 100%;
}

/* Component Overrides */
.gr-audio {
    border-radius: 4px !important;
    border: 1px solid #27272a !important;
    background-color: #141518 !important;
}
.tab-nav button[aria-selected="true"] {
    border-bottom: 2px solid #10b981 !important;
    color: #f4f4f5 !important;
    font-weight: 600 !important;
}

/* Audio Panels Symmetry & Height Balance */
.daw-audio-comparison-row {
    align-items: stretch !important;
    margin-bottom: 12px !important;
}
.daw-audio-comparison-row > div {
    display: flex !important;
    flex-direction: column !important;
}
.daw-audio-panel {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: 275px !important;
    height: 100% !important;
}
.daw-audio-panel > .block,
.daw-audio-panel.block {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: 275px !important;
    height: 100% !important;
}
.daw-audio-panel .audio-container {
    min-height: 215px !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}
.daw-audio-panel .empty,
.daw-audio-panel .empty-wrap,
.daw-audio-panel .empty.small {
    min-height: 215px !important;
    height: 100% !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}
.daw-audio-panel .upload-container,
.daw-audio-panel .wrap {
    min-height: 195px !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}
.daw-process-btn {
    margin-top: 10px !important;
    margin-bottom: 14px !important;
    width: 100% !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
}
"""


def get_daw_theme() -> Tuple[gr.themes.Base, str]:
    """
    Return the initialized AudioWorkstationTheme instance and CSS overrides.

    Returns:
        Tuple containing:
            - AudioWorkstationTheme instance (subclassing gr.themes.Base)
            - Custom CSS string for studio rackmount styling
    """
    return AudioWorkstationTheme(), DAW_CUSTOM_CSS
