"""Neuro Ads — Análise Cerebral de Criativos.

App Gradio para HuggingFace Spaces com ZeroGPU.
Faz upload de vídeo → roda TRIBE v2 → retorna análise neural completa.
"""

import subprocess
import sys
import os
import pathlib
import site
import tempfile

# ── Step 1: Install neuralset --no-deps ──────────────────────────────────────
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "neuralset==0.0.2", "-q"])

# ── Step 2: Patch neuralset exca version check BEFORE any import ─────────────
for sp in site.getsitepackages() + ([site.getusersitepackages()] if hasattr(site, 'getusersitepackages') else []):
    ns_init = pathlib.Path(sp) / "neuralset" / "__init__.py"
    if ns_init.exists():
        src = ns_init.read_text()
        if 'raise RuntimeError' in src and 'exca' in src:
            # Replace the raise with pass, keeping indentation
            lines = src.split('\n')
            new_lines = []
            for line in lines:
                if 'raise RuntimeError' in line and 'exca' in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + 'pass  # patched for py3.10')
                else:
                    new_lines.append(line)
            ns_init.write_text('\n'.join(new_lines))
            print(f"[startup] Patched neuralset exca check at {ns_init}")
            break

# ── Step 3: Install tribev2 (patch requires-python) ─────────────────────────
_tmpdir = tempfile.mkdtemp()
subprocess.check_call(
    ["git", "clone", "--depth=1", "https://github.com/facebookresearch/tribev2.git", _tmpdir],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
_pyproj = os.path.join(_tmpdir, "pyproject.toml")
with open(_pyproj) as f:
    content = f.read()
content = content.replace('requires-python = ">=3.11"', 'requires-python = ">=3.10"')
with open(_pyproj, "w") as f:
    f.write(content)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", _tmpdir, "-q"])
print("[startup] Installed tribev2 (patched for Python 3.10)")

# ── Step 4: Patch neuralset base.py _get_discriminated_subclasses ────────────
for sp in site.getsitepackages() + ([site.getusersitepackages()] if hasattr(site, 'getusersitepackages') else []):
    base_py = pathlib.Path(sp) / "neuralset" / "events" / "transforms" / "base.py"
    if base_py.exists():
        src = base_py.read_text()
        if '_get_discriminated_subclasses' in src:
            # Replace the problematic method call with a try/except
            src = src.replace(
                'if name and name not in cls._get_discriminated_subclasses():',
                'if name and hasattr(cls, "_get_discriminated_subclasses") and name not in cls._get_discriminated_subclasses():'
            )
            base_py.write_text(src)
            print(f"[startup] Patched neuralset base.py at {base_py}")
            break

print("[startup] All patches applied, starting Gradio app...")

# ── Now import everything ────────────────────────────────────────────────────

import tempfile as _tempfile
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

# ── Configuration ────────────────────────────────────────────────────────────

CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")
ASSETS_DIR = Path(CACHE_DIR) / "results" / "assets"
USE_TEXT = os.environ.get("USE_TEXT", "true").lower() == "true"
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model(cache_folder=CACHE_DIR, device="cpu", use_text=USE_TEXT)
    return MODEL


@spaces.GPU(duration=180)
def inference_gpu(video_path: str, progress=gr.Progress()):
    model = get_model()
    progress(0.1, desc="Extraindo features de áudio, vídeo e texto...")
    preds = run_inference(video_path, model, use_text=USE_TEXT)
    progress(0.6, desc="Inferência concluída!")
    return preds


def analyze_video(video_path: str, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Faça upload de um vídeo primeiro.")

    progress(0.05, desc="Iniciando análise neural...")
    preds = inference_gpu(video_path, progress)
    progress(0.6, desc="Computando scores neurais...")

    data = full_analysis(
        video_path=video_path,
        preds=preds,
        assets_dir=ASSETS_DIR if ASSETS_DIR.exists() else None,
    )
    progress(0.85, desc="Gerando dashboard...")

    html = generate_dashboard_html(data)

    tmp = _tempfile.NamedTemporaryFile(
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

        Faça upload de um vídeo de anúncio e receba uma análise neural completa
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
