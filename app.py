import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from neuro_pipeline import (
    load_model,
    run_inference,
    full_analysis,
    generate_dashboard_html,
)

# ── Configuration ────────────────────────────────────────────────────────────

CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")
ASSETS_DIR = Path(CACHE_DIR) / "results" / "assets"
USE_TEXT = os.environ.get("USE_TEXT", "true").lower() == "true"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None

print(f"[startup] Device: {DEVICE}, CUDA available: {torch.cuda.is_available()}")
print(f"[startup] Text features: {USE_TEXT}")


def get_model():
    global MODEL
    if MODEL is None:
        print("[model] Loading TRIBE v2...")
        MODEL = load_model(cache_folder=CACHE_DIR, device=DEVICE, use_text=USE_TEXT)
        print("[model] TRIBE v2 loaded!")
    return MODEL


def analyze_video(video_path: str, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video first.")

    progress(0.05, desc="Loading TRIBE v2 model...")
    model = get_model()

    progress(0.15, desc="Extracting audio and video features...")
    preds = run_inference(video_path, model, use_text=USE_TEXT)

    progress(0.6, desc="Computing neural scores...")
    data = full_analysis(
        video_path=video_path,
        preds=preds,
        assets_dir=ASSETS_DIR if ASSETS_DIR.exists() else None,
    )

    scores = data["scores"]
    for k, v in scores.items():
        scores[k] = float(v)

    diag_text = "\n".join(
        f"{'✅' if d['type']=='success' else '⚠️' if d['type']=='warning' else '❌' if d['type']=='error' else 'ℹ️'} "
        f"**{d['title']}** — {d['text']}"
        for d in data["diagnostics"]
    )

    progress(1.0, desc="Analysis complete!")

    # Return ONLY lightweight data (no dashboard HTML file — it's 8MB and kills the Gradio Client)
    return (
        float(scores["hook"]),
        float(scores["semantic"]),
        float(scores["synergy"]),
        float(scores["coherence"]),
        diag_text,
    )


def _get_grade(score):
    if score >= 85: return "A+"
    if score >= 75: return "A"
    if score >= 65: return "B+"
    if score >= 55: return "B"
    if score >= 45: return "C+"
    if score >= 35: return "C"
    return "F"


# ── Gradio Interface ─────────────────────────────────────────────────────────

with gr.Blocks(title="Neuro Ads") as app:
    gr.Markdown(
        """
        # 🧠 Neuro Ads — Creative Brain Analysis

        Upload an ad video (up to 30s) and receive a complete neural analysis
        based on brain activity prediction (TRIBE v2 / Meta Research).
        """
    )

    video_input = gr.File(label="Upload Video", file_types=["video"])
    analyze_btn = gr.Button("🔬 Analyze Creative", variant="primary", size="lg")

    with gr.Row():
        hook_score = gr.Number(label="🎯 Hook (35%)", precision=1, interactive=False)
        semantic_score = gr.Number(label="💬 Semantic (20%)", precision=1, interactive=False)
        synergy_score = gr.Number(label="🔗 Synergy (25%)", precision=1, interactive=False)
        coherence_score = gr.Number(label="📊 Coherence (20%)", precision=1, interactive=False)

    diagnostics_display = gr.Markdown(label="Diagnostics")

    gr.Markdown("*Powered by TRIBE v2 (Meta Research)*")

    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[
            hook_score, semantic_score,
            synergy_score, coherence_score,
            diagnostics_display,
        ],
        api_name="analyze",
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
