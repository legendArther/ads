"""Neuro Ads — Análise Cerebral de Criativos.

App Gradio para HuggingFace Spaces com ZeroGPU.
Faz upload de vídeo → roda TRIBE v2 → retorna análise neural completa.
"""

import subprocess
import sys

# Install neuralset and tribev2 with --no-deps to avoid exca>=0.5.20 conflict on Python 3.10
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "neuralset==0.0.2", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps",
                       "tribev2@git+https://github.com/facebookresearch/tribev2.git", "-q"])

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import spaces

from neuro_pipeline import (
    load_model,
    run_inference,
    full_analysis,
    generate_dashboard_html,
)

# ── Configuração ─────────────────────────────────────────────────────────────

CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")
ASSETS_DIR = Path(CACHE_DIR) / "results" / "assets"
USE_TEXT = os.environ.get("USE_TEXT", "true").lower() == "true"
MODEL = None


def get_model():
    """Carrega modelo na primeira chamada (lazy loading)."""
    global MODEL
    if MODEL is None:
        MODEL = load_model(cache_folder=CACHE_DIR, device="cpu", use_text=USE_TEXT)
    return MODEL


# ── Inferência com GPU ───────────────────────────────────────────────────────

@spaces.GPU(duration=180)
def inference_gpu(video_path: str, progress=gr.Progress()):
    """Roda inferência TRIBE v2 na GPU (ZeroGPU)."""
    model = get_model()
    progress(0.1, desc="Extraindo features de áudio, vídeo e texto...")
    preds = run_inference(video_path, model, use_text=USE_TEXT)
    progress(0.6, desc="Inferência concluída!")
    return preds


# ── Pipeline completo ────────────────────────────────────────────────────────

def analyze_video(video_path: str, progress=gr.Progress()):
    """Pipeline completo: upload → inferência → scoring → dashboard."""
    if not video_path:
        raise gr.Error("Faça upload de um vídeo primeiro.")

    progress(0.05, desc="Iniciando análise neural...")

    # 1. Inferência na GPU
    preds = inference_gpu(video_path, progress)
    progress(0.6, desc="Computando scores neurais...")

    # 2. Análise completa (CPU)
    data = full_analysis(
        video_path=video_path,
        preds=preds,
        assets_dir=ASSETS_DIR if ASSETS_DIR.exists() else None,
    )
    progress(0.85, desc="Gerando dashboard...")

    # 3. Gerar HTML
    html = generate_dashboard_html(data)

    # 4. Salvar HTML para download
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", prefix=f"neuro_ads_{data['videoName']}_",
        delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()

    # 5. Extrair dados para display no Gradio
    scores = data["scores"]
    grade = _get_grade(scores["neuroRank"])

    # Diagnostics text
    diag_text = "\n".join(
        f"{'✅' if d['type']=='success' else '⚠️' if d['type']=='warning' else '❌' if d['type']=='error' else 'ℹ️'} "
        f"**{d['title']}** — {d['text']}"
        for d in data["diagnostics"]
    )

    # Metrics prediction text
    analysis_text = data.get("creativeAnalysis", "")

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


# ── Interface Gradio ─────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#eef2ff", c100="#dde4ff", c200="#c2cfff", c300="#94aaff",
        c400="#6c8cff", c500="#3367ff", c600="#004eea", c700="#002376",
        c800="#001b61", c900="#000d3b", c950="#000724",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f1f3fc", c100="#e2e5f0", c200="#c5c9d6", c300="#a8abb3",
        c400="#72757d", c500="#44484f", c600="#20262f", c700="#1b2028",
        c800="#151a21", c900="#0f141a", c950="#0a0e14",
    ),
    font=["Inter", "sans-serif"],
    font_mono=["Space Grotesk", "monospace"],
)

css = """
.gradio-container { max-width: 900px !important; }
.score-box { text-align: center; }
.score-box .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
"""

with gr.Blocks(theme=THEME, css=css, title="Neuro Ads") as app:
    gr.Markdown(
        """
        # 🧠 Neuro Ads
        ### Análise Cerebral de Criativos

        Faça upload de um vídeo de anúncio e receba uma análise neural completa
        baseada em predição de atividade cerebral (TRIBE v2 / Meta Research).

        **Como funciona:** O modelo prediz como o cérebro humano reagiria ao seu vídeo,
        mapeando ativação em 20.484 pontos do córtex cerebral a cada segundo.
        """
    )

    with gr.Row():
        video_input = gr.Video(label="Upload do Vídeo", height=300)

    analyze_btn = gr.Button(
        "🔬 Analisar Criativo",
        variant="primary",
        size="lg",
    )

    # Results
    neuro_rank_display = gr.Markdown(
        value="*Faça upload de um vídeo e clique em Analisar*",
        label="Neuro Rank",
    )

    with gr.Row():
        hook_score = gr.Number(label="🎯 Impacto do Hook (35%)", precision=1, interactive=False)
        semantic_score = gr.Number(label="💬 Clareza Semântica (20%)", precision=1, interactive=False)
        synergy_score = gr.Number(label="🔗 Sinergia Multimodal (25%)", precision=1, interactive=False)
        coherence_score = gr.Number(label="📊 Coerência (20%)", precision=1, interactive=False)

    diagnostics_display = gr.Markdown(label="Diagnóstico")

    dashboard_file = gr.File(label="📥 Dashboard Completo (HTML)")

    gr.Markdown(
        """
        ---
        **Nota:** A análise usa apenas vídeo + áudio (sem transcrição de texto).
        Para resultados completos com análise de texto/legendas, configure o modelo LLaMA 3.2.

        Scores: **Hook** (atenção inicial) · **Semântica** (clareza da mensagem) ·
        **Sinergia** (integração multimodal) · **Coerência** (sustentação do engajamento)

        *Powered by TRIBE v2 (Meta Research) · Régua de scoring proprietária*
        """
    )

    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[
            neuro_rank_display,
            hook_score,
            semantic_score,
            synergy_score,
            coherence_score,
            diagnostics_display,
            dashboard_file,
        ],
        api_name="analyze",
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
