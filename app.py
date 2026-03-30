"""Neuro Ads — Análise Cerebral de Criativos.

App Gradio para HuggingFace Spaces (Docker + GPU).
Faz upload de vídeo → roda TRIBE v2 → retorna análise neural completa.
"""

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
USE_TEXT = os.environ.get("USE_TEXT", "false").lower() == "true"
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
        raise gr.Error("Faça upload de um vídeo primeiro.")

    progress(0.05, desc="Carregando modelo TRIBE v2...")
    model = get_model()

    progress(0.15, desc="Extraindo features de áudio e vídeo...")
    preds = run_inference(video_path, model, use_text=USE_TEXT)

    progress(0.6, desc="Computando scores neurais...")
    data = full_analysis(
        video_path=video_path,
        preds=preds,
        assets_dir=ASSETS_DIR if ASSETS_DIR.exists() else None,
    )

    progress(0.85, desc="Gerando dashboard...")
    html = generate_dashboard_html(data)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", prefix=f"neuro_ads_{data['videoName']}_",
        delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()

    scores = data["scores"]
    grade = _get_grade(scores["neuroRank"])

    diag_text = "\n".join(
        f"{'✅' if d['type']=='success' else '⚠️' if d['type']=='warning' else '❌' if d['type']=='error' else 'ℹ️'} "
        f"**{d['title']}** — {d['text']}"
        for d in data["diagnostics"]
    )

    progress(1.0, desc="Análise completa!")

    return (
        f"## {scores['neuroRank']:.1f} / 100  —  Nota {grade}",
        scores["hook"],
        scores["semantic"],
        scores["synergy"],
        scores["coherence"],
        diag_text,
        tmp.name,
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
        # 🧠 Neuro Ads — Análise Cerebral de Criativos

        Faça upload de um vídeo de anúncio (até 30s) e receba uma análise neural completa
        baseada em predição de atividade cerebral (TRIBE v2 / Meta Research).
        """
    )

    video_input = gr.File(label="Upload do Vídeo", file_types=["video"])
    analyze_btn = gr.Button("🔬 Analisar Criativo", variant="primary", size="lg")
    neuro_rank_display = gr.Markdown(value="*Faça upload de um vídeo e clique em Analisar*")

    with gr.Row():
        hook_score = gr.Number(label="🎯 Hook (35%)", precision=1, interactive=False)
        semantic_score = gr.Number(label="💬 Semântica (20%)", precision=1, interactive=False)
        synergy_score = gr.Number(label="🔗 Sinergia (25%)", precision=1, interactive=False)
        coherence_score = gr.Number(label="📊 Coerência (20%)", precision=1, interactive=False)

    diagnostics_display = gr.Markdown(label="Diagnóstico")
    dashboard_file = gr.File(label="📥 Dashboard Completo (HTML)")

    gr.Markdown("*Powered by TRIBE v2 (Meta Research)*")

    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[
            neuro_rank_display, hook_score, semantic_score,
            synergy_score, coherence_score,
            diagnostics_display, dashboard_file,
        ],
        api_name="analyze",
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
