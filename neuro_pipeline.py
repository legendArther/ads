"""Neuro Ads Pipeline — Complete neural analysis of creatives.

Combines TRIBE v2 inference + scoring + HTML dashboard generation.
Designed to run on HuggingFace Spaces with ZeroGPU.
"""

import base64
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ── Functional systems (Destrieux atlas) ────────────────────────────────────

SYSTEMS = {
    "visual": {
        "regions": ["Pole_occipital", "G_occipital_middle", "S_oc_middle_and_Lunatus",
                     "G_and_S_occipital_inf", "S_oc_sup_and_transversal", "S_calcarine",
                     "G_occipital_sup", "G_cuneus"],
        "label": "Visual System (V1-V3)", "color": "#FF6B35",
    },
    "motion": {
        "regions": ["S_oc-temp_lat", "G_temporal_middle"],
        "label": "Áreas de Movimento (MT/V5)", "color": "#FF9F1C",
    },
    "auditory": {
        "regions": ["S_temporal_transverse", "G_temp_sup-G_T_transv",
                     "G_temp_sup-Plan_tempo", "G_temp_sup-Lateral"],
        "label": "Córtex Auditivo (A1/A5)", "color": "#2EC4B6",
    },
    "language": {
        "regions": ["G_front_inf-Opercular", "G_front_inf-Triangul", "S_temporal_sup"],
        "label": "Linguagem (Broca/STS)", "color": "#E71D36",
    },
    "vwfa": {
        "regions": ["G_oc-temp_lat-fusifor"], "hemisphere": "left",
        "label": "VWFA (Texto na Tela)", "color": "#7B2D8E",
    },
    "integration": {
        "regions": ["G_pariet_inf-Angular", "G_pariet_inf-Supramar",
                     "G_parietal_sup", "S_intrapariet_and_P_trans"],
        "label": "Integração (TPJ/TPO)", "color": "#4361EE",
    },
    "prefrontal": {
        "regions": ["G_front_middle", "G_front_sup", "S_front_sup",
                     "G_front_inf-Orbital", "G_orbital"],
        "label": "Pré-frontal (Decisão)", "color": "#3F37C9",
    },
    "ffa": {
        "regions": ["G_oc-temp_lat-fusifor"], "hemisphere": "right",
        "label": "FFA (Rostos)", "color": "#F72585",
    },
    "ppa": {
        "regions": ["G_oc-temp_med-Parahip"],
        "label": "PPA (Cenas/Contexto)", "color": "#3A0CA3",
    },
    "insula": {
        "regions": ["S_circular_insula_inf", "S_circular_insula_sup",
                     "G_insular_short", "G_Ins_lg_and_S_cent_ins"],
        "label": "Ínsula (Emoção)", "color": "#560BAD",
    },
}

WEIGHTS = {"hook": 0.35, "synergy": 0.25, "semantic": 0.20, "coherence": 0.20}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Inferência ───────────────────────────────────────────────────────────────

def load_model(cache_folder="./cache", device="auto", use_text=True):
    """Carrega o modelo TRIBE v2.

    Args:
        use_text: Se True, usa LLaMA para features de texto (mirror ungated).
                  Se False, usa apenas áudio + vídeo.
    """
    from tribev2.demo_utils import TribeModel

    if use_text:
        config_update = {
            "data.features_to_use": ["audio", "video", "text"],
            "data.text_feature.model_name": "meta-llama/Llama-3.2-3B",
        }
    else:
        config_update = {
            "data.features_to_use": ["audio", "video"],
        }

    if device == "cpu":
        config_update.update({
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
            "data.text_feature.device": "cpu",
        })

    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_folder,
        device=device,
        config_update=config_update,
    )
    return model


def run_inference(video_path: str, model, use_text=True, progress_callback=None) -> np.ndarray:
    """Roda inferência no vídeo e retorna predições cerebrais."""
    from tribev2.demo_utils import get_audio_and_text_events

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {path}")

    if progress_callback:
        progress_callback(0.1, "Extraindo features de áudio, vídeo e texto...")

    event = {
        "type": "Video",
        "filepath": str(path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=not use_text)

    if progress_callback:
        progress_callback(0.4, "Rodando modelo de encoding cerebral...")

    preds, segments = model.predict(events)

    if progress_callback:
        progress_callback(0.6, "Inferência concluída!")

    return preds


# ── Scoring ──────────────────────────────────────────────────────────────────

def _get_video_duration(video_path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        return round(float(out.stdout.strip()), 2)
    except Exception:
        return 30.0


def _load_atlas():
    from nilearn import datasets as ds
    destrieux = ds.fetch_atlas_surf_destrieux()
    labels_lh = np.array(destrieux.map_left)
    labels_rh = np.array(destrieux.map_right)
    label_names = [l if isinstance(l, str) else l.decode() for l in destrieux.labels]
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    return labels_lh, labels_rh, label_to_idx


def _region_activation(preds_slice, regions, labels_lh, labels_rh, label_to_idx, hemisphere="both"):
    n_hemi = preds_slice.shape[1] // 2
    all_vals = []
    for region in regions:
        idx = label_to_idx.get(region)
        if idx is None:
            continue
        lh_mask = labels_lh == idx
        rh_mask = labels_rh == idx
        if hemisphere == "left":
            rh_mask = np.zeros_like(rh_mask)
        elif hemisphere == "right":
            lh_mask = np.zeros_like(lh_mask)
        if lh_mask.any():
            all_vals.append(preds_slice[:, :n_hemi][:, lh_mask].mean(axis=1))
        if rh_mask.any():
            all_vals.append(preds_slice[:, n_hemi:][:, rh_mask].mean(axis=1))
    if not all_vals:
        return np.zeros(preds_slice.shape[0])
    return np.mean(all_vals, axis=0)


def sigmoid_score(value, midpoint, steepness=20):
    return float(100 / (1 + np.exp(-steepness * (value - midpoint))))


def compute_scores(preds: np.ndarray, video_duration: float):
    """Calcula os 4 scores + Neuro Rank."""
    n_segments, n_vertices = preds.shape
    seg_duration = video_duration / n_segments
    labels_lh, labels_rh, label_to_idx = _load_atlas()

    def ra(regions, hemi="both"):
        return _region_activation(preds, regions, labels_lh, labels_rh, label_to_idx, hemi)

    # Séries temporais por sistema
    system_ts = {}
    for sys_name, sys_info in SYSTEMS.items():
        hemi = sys_info.get("hemisphere", "both")
        system_ts[sys_name] = ra(sys_info["regions"], hemi)

    # Janelas temporais
    def win_seg(start, end):
        s = max(0, int(start / seg_duration))
        e = min(n_segments, int(np.ceil(end / seg_duration)))
        return s, e

    WINDOWS = {
        "hook": {"start": 0, "end": 1.5, "label": "Hook (0–1,5s)", "color": "#FF6B35"},
        "transition": {"start": 1.5, "end": 3.0, "label": "Transição (1,5–3s)", "color": "#FF9F1C"},
        "message": {"start": 3.0, "end": 8.0, "label": "Mensagem (3–8s)", "color": "#2EC4B6"},
        "sustain": {"start": 8.0, "end": video_duration, "label": "Sustentação (8s+)", "color": "#4361EE"},
    }

    window_activations = {}
    for win_name, win_info in WINDOWS.items():
        s, e = win_seg(win_info["start"], win_info["end"])
        if s >= e:
            s, e = 0, 1
        window_activations[win_name] = {}
        for sys_name in SYSTEMS:
            ts = system_ts[sys_name]
            window_activations[win_name][sys_name] = float(ts[s:e].mean()) if s < len(ts) else 0.0

    global_mean = float(preds.mean())

    # Hook Score
    hook_s, hook_e = win_seg(0, 1.5)
    hook_visual = system_ts["visual"][hook_s:hook_e].mean()
    hook_motion = system_ts["motion"][hook_s:hook_e].mean()
    hook_auditory = system_ts["auditory"][hook_s:hook_e].mean()
    hook_raw = hook_visual * 0.4 + hook_motion * 0.35 + hook_auditory * 0.25
    early_ts = preds[:min(4, n_segments)].mean(axis=1)
    hook_slope = float(np.polyfit(range(len(early_ts)), early_ts, 1)[0]) if len(early_ts) > 1 else 0.0
    hook_slope_bonus = max(0, min(20, hook_slope * 500))
    hook_score = min(100, sigmoid_score(hook_raw, global_mean, steepness=15) + hook_slope_bonus)

    # Semantic Score
    lang_ts = system_ts["language"]
    vwfa_ts = system_ts["vwfa"]
    msg_s, msg_e = win_seg(1.5, video_duration)
    semantic_raw = lang_ts[msg_s:msg_e].mean() * 0.6 + vwfa_ts[msg_s:msg_e].mean() * 0.4
    semantic_score = sigmoid_score(semantic_raw, global_mean * 0.5, steepness=18)
    msg3_s, msg3_e = win_seg(3, 8)
    lang_in_message = lang_ts[msg3_s:msg3_e].mean() if msg3_s < msg3_e else 0
    if lang_in_message < global_mean * 0.3:
        semantic_score *= 0.7

    # Synergy Score
    integration_ts = system_ts["integration"]
    prefrontal_ts = system_ts["prefrontal"]
    insula_ts = system_ts["insula"]
    synergy_raw = integration_ts.mean() * 0.5 + prefrontal_ts.mean() * 0.3 + insula_ts.mean() * 0.2
    synergy_score = sigmoid_score(synergy_raw, global_mean * 0.5, steepness=18)
    syn_in_msg = integration_ts[msg3_s:msg3_e].mean() if msg3_s < msg3_e else 0
    if syn_in_msg > integration_ts[:max(1, hook_e)].mean():
        synergy_score = min(100, synergy_score * 1.15)

    # Coherence Score
    overall_ts = preds.mean(axis=1)
    cv = float(overall_ts.std() / max(overall_ts.mean(), 1e-8))
    coherence_cv = max(0, 1 - cv) * 40
    half = n_segments // 2
    first_half = overall_ts[:half].mean()
    second_half = overall_ts[half:].mean()
    sustain_ratio = second_half / max(first_half, 1e-8)
    coherence_sustain = min(30, max(0, sustain_ratio * 20))
    drops = np.diff(overall_ts) if len(overall_ts) > 1 else np.array([0])
    max_relative_drop = float(abs(drops.min())) / max(overall_ts.mean(), 1e-8)
    coherence_nodrop = max(0, 30 - max_relative_drop * 100)
    coherence_score = min(100, coherence_cv + coherence_sustain + coherence_nodrop)

    # Neuro Rank
    neuro_rank = (WEIGHTS["hook"] * hook_score + WEIGHTS["synergy"] * synergy_score
                  + WEIGHTS["semantic"] * semantic_score + WEIGHTS["coherence"] * coherence_score)

    # Diagnostics
    diagnostics = _compute_diagnostics(
        hook_score, semantic_score, synergy_score, coherence_score,
        system_ts, overall_ts, global_mean, sustain_ratio, hook_e, msg3_s, msg3_e
    )

    # Creative analysis
    creative_analysis = _generate_creative_analysis(
        hook_score, semantic_score, synergy_score, coherence_score, neuro_rank,
        system_ts, overall_ts, global_mean, sustain_ratio, seg_duration,
        video_duration, integration_ts, prefrontal_ts, vwfa_ts, lang_ts
    )

    # Ensure all values are native Python types (not numpy) for JSON serialization
    def _f(v):
        return round(float(v), 6) if hasattr(v, 'item') else round(v, 6)

    # Convert window activations to native floats
    clean_wa = {}
    for wn, wv in window_activations.items():
        clean_wa[wn] = {sk: float(sv) for sk, sv in wv.items()}

    return {
        "scores": {
            "hook": round(float(hook_score), 1),
            "semantic": round(float(semantic_score), 1),
            "synergy": round(float(synergy_score), 1),
            "coherence": round(float(coherence_score), 1),
            "neuroRank": round(float(neuro_rank), 1),
        },
        "weights": WEIGHTS,
        "timeseries": {
            "overall": overall_ts.tolist(),
            "systems": {k: v.tolist() for k, v in system_ts.items()},
        },
        "systems": {k: {"label": v["label"], "color": v["color"]} for k, v in SYSTEMS.items()},
        "windows": {k: {"label": v["label"], "color": v["color"], "start": float(v["start"]), "end": float(v["end"])}
                    for k, v in WINDOWS.items()},
        "windowActivations": clean_wa,
        "diagnostics": diagnostics,
        "creativeAnalysis": creative_analysis,
        "details": {
            "hookSlope": round(float(hook_slope), 6),
            "sustainRatio": round(float(sustain_ratio), 3),
            "maxRelativeDrop": round(float(max_relative_drop), 3),
            "cv": round(float(cv), 3),
        },
    }


def _compute_diagnostics(hook_score, semantic_score, synergy_score, coherence_score,
                         system_ts, overall_ts, global_mean, sustain_ratio, hook_e, msg3_s, msg3_e):
    diagnostics = []

    if hook_score >= 70:
        diagnostics.append({"type": "success", "title": "Strong Hook",
                             "text": "Visual and auditory activation rises quickly in the first 1.5s. Good thumb-stop."})
    elif hook_score >= 40:
        diagnostics.append({"type": "warning", "title": "Moderate Hook",
                             "text": "Visual system activates but there is no strong peak. Consider more impactful elements in the first frames."})
    else:
        diagnostics.append({"type": "error", "title": "Weak Hook",
                             "text": "Low initial activation. The thumb-stop is weak. Needs more visual contrast, movement, or sound in the first 1.5s."})

    visual_mean = system_ts["visual"].mean()
    integration_mean = system_ts["integration"].mean()
    if visual_mean > global_mean * 1.5 and integration_mean < global_mean * 0.8:
        diagnostics.append({"type": "warning", "title": "High Visual, Low Integration",
                             "text": "Eye-catching hook but weak message. The ad grabs visual attention but does not 'close the loop'."})

    lang_mean = system_ts["language"].mean()
    if lang_mean > global_mean and visual_mean < global_mean * 0.8:
        diagnostics.append({"type": "info", "title": "Clear Promise, Low Stop Power",
                             "text": "Language active but visual weak. Good message but might not stop the scroll."})

    vwfa_mean = system_ts["vwfa"].mean()
    if vwfa_mean > visual_mean * 0.8:
        diagnostics.append({"type": "warning", "title": "Reading Dependent",
                             "text": "VWFA very active. Creative may depend too much on on-screen text."})

    integration_ts = system_ts["integration"]
    syn_early = integration_ts[:max(1, msg3_s)].mean()
    syn_late = integration_ts[msg3_s:].mean()
    if syn_late > syn_early * 1.5 and syn_early < global_mean * 0.5:
        diagnostics.append({"type": "warning", "title": "Late Integration",
                             "text": "Claim/proof/demo is entering late. Multimodal integration only appears after 8s."})

    if len(overall_ts) > 5:
        hook_peak = overall_ts[:3].max()
        post_hook = overall_ts[3:8].mean()
        if post_hook < hook_peak * 0.5:
            diagnostics.append({"type": "error", "title": "Sharp Drop After Hook",
                                 "text": "The middle of the video loses engagement. Needs a new visual reward or social proof after the hook."})

    if sustain_ratio > 1.1:
        diagnostics.append({"type": "success", "title": "Good Sustain",
                             "text": "The second half of the video maintains or increases in activation. Good for retention."})
    elif sustain_ratio < 0.6:
        diagnostics.append({"type": "error", "title": "Loss of Engagement",
                             "text": "The second half loses more than 40% of activation. Review the pacing after the hook."})

    ffa_mean = system_ts["ffa"].mean()
    if ffa_mean > global_mean * 1.2:
        diagnostics.append({"type": "info", "title": "Faces Detected",
                             "text": "FFA (face area) active. Presence of faces generates emotional connection."})

    if not diagnostics:
        diagnostics.append({"type": "info", "title": "Neutral Profile",
                             "text": "No strong pattern detected. The ad may need more emotional contrast."})

    return diagnostics


def _generate_creative_analysis(hook_score, semantic_score, synergy_score, coherence_score,
                                neuro_rank, system_ts, overall_ts, global_mean,
                                sustain_ratio, seg_duration, video_duration,
                                integration_ts, prefrontal_ts, vwfa_ts, lang_ts):
    lines = []

    lines.append("<h4>Creative Performance Diagnosis</h4>")

    if hook_score >= 70:
        lines.append(f"<p>The <strong>hook is the strong point</strong> of this creative (score {hook_score:.0f}/100). "
                     "The visual and auditory systems respond intensely in the first 1.5 seconds, "
                     "indicating a high potential for thumb-stopping in the feed. "
                     "This should translate to a strong 3s view rate.</p>")
    else:
        lines.append(f"<p>The <strong>hook is weak</strong> (score {hook_score:.0f}/100). "
                     "The visual system does not reach a relevant peak in the first 1.5s. "
                     "This likely results in a low 3s view rate and poor thumb-stopping.</p>")

    if coherence_score < 50:
        lines.append(f"<p>The <strong>main issue is coherence</strong> (score {coherence_score:.0f}/100). "
                     f"The second half / first half ratio is only {sustain_ratio:.2f}x, "
                     "indicating the brain 'shuts off' after the initial impact. "
                     "This directly affects retention and video completion rates.</p>")
    elif coherence_score >= 70:
        lines.append(f"<p>The <strong>coherence is solid</strong> (score {coherence_score:.0f}/100). "
                     "Neural engagement is maintained throughout the video without sharp drops. "
                     "This favors good retention rates.</p>")

    if synergy_score < 50:
        lines.append(f"<p>The <strong>multimodal synergy is low</strong> (score {synergy_score:.0f}/100). "
                     "Integration areas (TPJ, Prefrontal) are not activating as much as primary sensory areas. "
                     "This suggests the creative impacts the senses but does not 'connect the dots'— "
                     "the viewer sees and hears, but does not integrate the message.</p>")
    elif synergy_score >= 70:
        lines.append(f"<p>The <strong>multimodal synergy is good</strong> (score {synergy_score:.0f}/100). "
                     "Brain integration areas (TPJ/Prefrontal) activate well, indicating that video+audio "
                     "complement each other and the message 'closes the loop'.</p>")

    if semantic_score >= 60:
        lines.append(f"<p>The <strong>semantic clarity is adequate</strong> (score {semantic_score:.0f}/100). "
                     "Language areas respond to the verbal/text content of the ad.</p>")
    else:
        lines.append(f"<p>The <strong>semantic clarity is weak</strong> (score {semantic_score:.0f}/100). "
                     "Language areas (Broca/STS) are not processing the message effectively. "
                     "The claim may be unclear or entering too late.</p>")

    # Suggestions
    lines.append("<h4>Optimization Roadmap</h4><ol>")

    if coherence_score < 60:
        lines.append("<li><strong>Add 'rewards' in the middle of the video:</strong> "
                     "After the hook (3-8s), insert social proof, product demo, or a new visual cut. "
                     "The brain needs new stimuli every 3-5s to maintain activation.</li>")

    if synergy_score < 60:
        lines.append("<li><strong>Better synchronize video + audio + text:</strong> "
                     "When the verbal claim enters, ensure the visual reinforces the same message. "
                     "Multimodal integration (TPJ) triggers when senses converge on the same idea.</li>")

    integration_peak_idx = int(np.argmax(integration_ts))
    integration_peak_time = integration_peak_idx * seg_duration
    if integration_peak_time < video_duration * 0.7:
        lines.append(f"<li><strong>Reposition CTA/branding:</strong> "
                     f"The peak of brain integration happens around {integration_peak_time:.0f}s. "
                     "This is the best moment for the CTA—the brain is in 'closing loop' mode. "
                     "Placing the CTA alone at the end misses this window.</li>")

    if hook_score >= 70 and coherence_score < 50:
        lines.append("<li><strong>Hook works, body doesn't:</strong> "
                     "Keep the first 1.5s intact and rework the 3-15s segment. "
                     "Test with product demos, testimonials, or a quick sequence of benefits.</li>")

    visual_mean = system_ts["visual"].mean()
    vwfa_mean = vwfa_ts.mean()
    if vwfa_mean > visual_mean * 0.6:
        lines.append("<li><strong>Reduce dependency on on-screen text:</strong> "
                     "The VWFA is activating heavily. Replace part of the text with visual demonstration. "
                     "Remember 80% of feed videos are watched without sound—but reading does not replace visual impact.</li>")

    sensory_max = max(visual_mean, system_ts["auditory"].mean())
    if integration_ts.mean() < sensory_max * 0.5:
        lines.append("<li><strong>Build a clearer narrative:</strong> "
                     "Sensory areas activate well but integration areas don't follow. "
                     "This indicates 'visual/auditory noise' without a coherent story. "
                     "Simplify the message: 1 problem → 1 solution → 1 CTA.</li>")

    lines.append("</ol>")

    # Metrics prediction
    lines.append("<h4>Metric Projections</h4>")
    lines.append("<table class='metrics-table'><thead><tr>"
                 "<th>Metric</th><th>Projection</th><th>Neural Signal</th></tr></thead><tbody>")

    if hook_score >= 70:
        lines.append("<tr><td>3s view rate</td><td style='color:#22c55e'>High</td>"
                     "<td>Strong visual/auditory activation at hook</td></tr>")
    else:
        lines.append("<tr><td>3s view rate</td><td style='color:#ef4444'>Low</td>"
                     "<td>Hook lacking sufficient sensory impact</td></tr>")

    if coherence_score >= 60:
        lines.append("<tr><td>Retention</td><td style='color:#22c55e'>Good</td>"
                     "<td>Sustained engagement throughout the video</td></tr>")
    else:
        lines.append("<tr><td>Retention</td><td style='color:#ef4444'>Weak</td>"
                     "<td>Drop in activation after hook—viewer tends to drop off</td></tr>")

    if synergy_score >= 60 and semantic_score >= 50:
        lines.append("<tr><td>CTR</td><td style='color:#22c55e'>Good</td>"
                     "<td>Multimodal integration + message clarity</td></tr>")
    elif synergy_score >= 50 or semantic_score >= 50:
        lines.append("<tr><td>CTR</td><td style='color:#f59e0b'>Moderate</td>"
                     "<td>Message partially integrated</td></tr>")
    else:
        lines.append("<tr><td>CTR</td><td style='color:#ef4444'>Low</td>"
                     "<td>Lack of integration and clarity in proposal</td></tr>")

    if neuro_rank >= 70 and prefrontal_ts.mean() > global_mean:
        lines.append("<tr><td>Conversion</td><td style='color:#22c55e'>Promising</td>"
                     "<td>Active prefrontal + high neuro rank</td></tr>")
    elif neuro_rank >= 50:
        lines.append("<tr><td>Conversion</td><td style='color:#f59e0b'>Uncertain</td>"
                     "<td>Mixed brain signals—depends on audience and offer</td></tr>")
    else:
        lines.append("<tr><td>Conversion</td><td style='color:#ef4444'>Weak</td>"
                     "<td>Insufficient activation in decision areas</td></tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)
    return "\n".join(lines)


# ── Geração de imagens cerebrais ─────────────────────────────────────────────

# ── Análise profunda com Claude API ──────────────────────────────────────────

def generate_ai_analysis(scores: dict, diagnostics: list, details: dict,
                         system_timeseries: dict, window_activations: dict,
                         video_name: str, video_duration: float) -> str | None:
    """Generates deep creative analysis using Claude API.

    Returns HTML formatted with personalized insights, or None if the API is not configured.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        return None

    # Prepare neural context for the prompt
    scores_info = scores
    top_systems_per_window = {}
    for win_name, sys_acts in window_activations.items():
        sorted_sys = sorted(sys_acts.items(), key=lambda x: x[1], reverse=True)[:4]
        top_systems_per_window[win_name] = sorted_sys

    prompt = f"""You are an expert in neuromarketing and advertising creative analysis.
Analyze the brain activation data below for the video "{video_name}" ({video_duration:.0f}s).

NEURAL SCORES:
- Hook Impact: {scores_info['hook']}/100 (weight 35%)
- Semantic Clarity: {scores_info['semantic']}/100 (weight 20%)
- Multimodal Synergy: {scores_info['synergy']}/100 (weight 25%)
- Ad Coherence: {scores_info['coherence']}/100 (weight 20%)
- Final Neuro Rank: {scores_info['neuroRank']}/100

TECHNICAL DETAILS:
- Hook slope: {details['hookSlope']:.6f} (positive = increasing activation)
- Sustain ratio (2nd half / 1st half): {details['sustainRatio']:.2f}x
- Largest relative drop in activation: {details['maxRelativeDrop']:.2f}
- Coefficient of variation: {details['cv']:.3f}

AUTOMATIC DIAGNOSTICS:
{chr(10).join(f"- [{d['type'].upper()}] {d['title']}: {d['text']}" for d in diagnostics)}

ACTIVATION BY TIME WINDOW:
{chr(10).join(f"- {win}: " + ", ".join(f"{s}={v:.5f}" for s, v in acts) for win, acts in top_systems_per_window.items())}

INSTRUCTIONS:
Generate an analysis in HTML (no html/body/head tags, just content) with these sections:

1. <h4>Neural Creative Diagnosis</h4>
   Analyze the overall brain activation pattern. What do the data reveal about how the brain processes this ad? Connect the scores with what this means in terms of paid media performance. Be specific to THIS creative, not generic.

2. <h4>What Works and What Doesn't</h4>
   List strengths and weaknesses based on neural data. For each point, explain WHICH brain region indicates this and WHY it matters to the advertiser.

3. <h4>Optimization Roadmap</h4>
   Give 3-5 practical recommendations ordered by priority. For each one:
   - What to change in the creative
   - Which Meta metric this should impact (3s view, retention, CTR, conversion)
   - Which neural signal justifies this change

4. <h4>Performance Projection</h4>
   Assemble an HTML table with projections for: 3s view rate, Retention, CTR, relative CPA.
   Use colors: green (#22c55e) for good, orange (#f59e0b) for moderate, red (#ef4444) for weak.

Use <p>, <ul>, <li>, <strong>, <table> tags for formatting.
Write in English, direct and technical but accessible tone.
Do not use emojis. Do not repeat the scores — the user already sees them on the dashboard."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error in Claude API: {e}")
        return None


def generate_brain_images(preds: np.ndarray, video_name: str):
    """Generates static brain maps as base64 PNG."""
    from nilearn import datasets as ds, plotting

    n_hemi = preds.shape[1] // 2
    fsaverage = ds.fetch_surf_fsaverage("fsaverage5")
    avg_activation = preds.mean(axis=0)
    peak_idx = int(preds.mean(axis=1).argmax())
    peak_activation = preds[peak_idx]

    brain_images = {}
    for img_name, data, title in [
        ("avg", avg_activation, f"Average Brain Activation — {video_name}"),
        ("peak", peak_activation, f"Peak Activation (Segment {peak_idx}) — {video_name}"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={"projection": "3d"})
        for i, (hemi, hdata, mesh, bg) in enumerate([
            ("left", data[:n_hemi], fsaverage.pial_left, fsaverage.sulc_left),
            ("right", data[n_hemi:], fsaverage.pial_right, fsaverage.sulc_right),
        ]):
            for j, view in enumerate(["lateral", "medial"]):
                plotting.plot_surf_stat_map(
                    mesh, hdata, bg_map=bg, hemi=hemi, view=view,
                    colorbar=j == 1, cmap="hot", axes=axes[i][j],
                    title=f"{hemi.title()} — {view.title()}"
                )
        fig.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#131829")
        plt.close(fig)
        buf.seek(0)
        brain_images[img_name] = base64.b64encode(buf.read()).decode("ascii")

    # Temporal chart
    fig_t, ax_t = plt.subplots(figsize=(12, 4), facecolor="#131829")
    ax_t.set_facecolor("#131829")
    ts_data = preds.mean(axis=1)
    ax_t.plot(range(len(ts_data)), ts_data, "o-", color="#4f7cff", linewidth=2, markersize=5)
    ax_t.fill_between(range(len(ts_data)), ts_data, alpha=0.15, color="#4f7cff")
    ax_t.set_xlabel("Segment (time)", fontsize=11, color="#8892a8")
    ax_t.set_ylabel("Average brain activation", fontsize=11, color="#8892a8")
    ax_t.tick_params(colors="#8892a8")
    for spine in ax_t.spines.values():
        spine.set_color("#252d45")
    ax_t.grid(True, alpha=0.15, color="#8892a8")
    peak_s = ts_data.argmax()
    ax_t.annotate(f"Peak (seg. {peak_s})", xy=(peak_s, ts_data[peak_s]),
                  xytext=(peak_s - 5, ts_data[peak_s] + 0.01),
                  arrowprops=dict(arrowstyle="->", color="#FF6B35"), fontsize=10, color="#FF6B35")
    plt.tight_layout()
    buf2 = io.BytesIO()
    fig_t.savefig(buf2, format="png", dpi=150, bbox_inches="tight", facecolor="#131829")
    plt.close(fig_t)
    buf2.seek(0)
    brain_images["temporal"] = base64.b64encode(buf2.read()).decode("ascii")

    return brain_images


# ── 3D Data Generation ─────────────────────────────────────────────────────

def generate_3d_data(preds: np.ndarray, assets_dir: Path):
    """Generates face color data for the 3D viewer."""
    import trimesh

    n_segments = preds.shape[0]
    n_hemi = preds.shape[1] // 2

    glb_files = {
        "left": "brain-left-hemisphere-1b9f386f.glb",
        "right": "brain-right-hemisphere-f0dea562.glb",
        "leftInflated": "brain-left-hemisphere-inflated-23f77205.glb",
        "rightInflated": "brain-right-hemisphere-inflated-1ded8aca.glb",
        "head": "head-9ddb57ac.glb",
    }

    # Check if assets exist
    available_glbs = {}
    glb_meshes = {}
    for gname, gfname in glb_files.items():
        fpath = assets_dir / gfname
        if fpath.exists():
            available_glbs[gname] = fpath
            if gname in ("left", "right"):
                sc = trimesh.load(fpath)
                glb_meshes[gname] = list(sc.geometry.values())[0].faces

    if not glb_meshes:
        return None, None

    def value_to_rgb(val):
        if val < 0.25:
            t = val / 0.25
            return (int(t * 180), 0, 0)
        elif val < 0.5:
            t = (val - 0.25) / 0.25
            return (180 + int(t * 75), int(t * 80), 0)
        elif val < 0.75:
            t = (val - 0.5) / 0.25
            return (255, 80 + int(t * 175), int(t * 30))
        else:
            t = (val - 0.75) / 0.25
            return (255, 255, 30 + int(t * 225))

    vmin = float(np.percentile(preds, 5))
    vmax = float(np.percentile(preds, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    frame_data = []
    for seg_idx in range(n_segments):
        fd = {}
        for hname, hoff in [("left", 0), ("right", n_hemi)]:
            if hname not in glb_meshes:
                continue
            vvals = preds[seg_idx, hoff:hoff + n_hemi]
            normed = np.clip((vvals - vmin) / (vmax - vmin), 0, 1)
            faces = glb_meshes[hname]
            fvals = normed[faces].mean(axis=1)
            rgb = np.zeros((len(faces), 3), dtype=np.uint8)
            for i, v in enumerate(fvals):
                rgb[i] = value_to_rgb(float(v))
            fd[hname] = base64.b64encode(rgb.tobytes()).decode("ascii")
        frame_data.append(fd)

    glb_b64 = {}
    for gname, gpath in available_glbs.items():
        glb_b64[gname] = base64.b64encode(gpath.read_bytes()).decode("ascii")

    viewer3d_data = {
        "numSegments": n_segments,
        "numFaces": 20480,
        "frames": frame_data,
    }

    return viewer3d_data, glb_b64


# ── Full Pipeline ────────────────────────────────────────────────────────

def full_analysis(video_path: str, preds: np.ndarray, assets_dir: Path = None):
    """Full pipeline: scoring + brain images + 3D data.

    Returns dict with all data needed for the dashboard.
    """
    video_path = str(video_path)
    video_name = Path(video_path).stem
    video_duration = _get_video_duration(video_path)
    n_segments, n_vertices = preds.shape

    # Scoring
    result = compute_scores(preds, video_duration)

    # AI Analysis (Claude API) — replaces rule-based analysis
    ai_analysis = generate_ai_analysis(
        scores=result["scores"],
        diagnostics=result["diagnostics"],
        details=result["details"],
        system_timeseries=result["timeseries"]["systems"],
        window_activations=result["windowActivations"],
        video_name=video_name,
        video_duration=video_duration,
    )
    if ai_analysis:
        result["creativeAnalysis"] = ai_analysis

    # Brain images
    brain_images = generate_brain_images(preds, video_name)

    # 3D data (optional)
    viewer3d_data, glb_b64 = None, None
    if assets_dir and assets_dir.exists():
        viewer3d_data, glb_b64 = generate_3d_data(preds, assets_dir)

    result.update({
        "videoName": video_name,
        "videoFile": Path(video_path).name,
        "videoDuration": video_duration,
        "numSegments": n_segments,
        "brainImages": brain_images,
        "viewer3d": viewer3d_data,
        "glb": glb_b64,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

  
    return result


def generate_dashboard_html(data: dict) -> str:
    """Gera o HTML completo do dashboard a partir dos dados processados."""
    html = HTML_TEMPLATE.replace("PLACEHOLDER_JSON", json.dumps(data, ensure_ascii=False, cls=NumpyEncoder))
    html = html.replace("PLACEHOLDER_TITLE", data.get("videoName", "Análise"))
    return html


# ── HTML Template ────────────────────────────────────────────────────────────
# Template completo do dashboard Neuro Ads (self-contained)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html class="dark" lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neuro Ads — PLACEHOLDER_TITLE</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
<script>
tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "on-surface": "#f1f3fc", "surface": "#0a0e14", "on-surface-variant": "#a8abb3",
        "surface-container": "#151a21", "surface-container-low": "#0f141a",
        "surface-container-high": "#1b2028", "surface-container-highest": "#20262f",
        "outline": "#72757d", "outline-variant": "#44484f",
        "primary": "#94aaff", "primary-dim": "#3367ff", "primary-container": "#809bff",
        "secondary": "#e966ff", "tertiary": "#b5ffc2", "error": "#ff6e84",
      },
      fontFamily: { "headline": ["Space Grotesk"], "body": ["Inter"], "label": ["Inter"] },
    },
  },
}
</script>
<style>
body { font-family: 'Inter', sans-serif; background: #0a0e14; color: #f1f3fc; }
.font-headline { font-family: 'Space Grotesk', sans-serif; }
.material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; }
.glass-panel { background: rgba(21, 26, 33, 0.6); backdrop-filter: blur(20px); }
.neural-glow { box-shadow: 0 0 20px rgba(148, 170, 255, 0.15); }
input[type=range] { -webkit-appearance:none; width:100%; height:6px; background:#20262f; border-radius:3px; outline:none; cursor:pointer; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:18px; height:18px; border-radius:50%; background:#94aaff; cursor:pointer; border:3px solid #0a0e14; }
.metrics-table { width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }
.metrics-table th { text-align:left; padding:10px 14px; color:#a8abb3; font-weight:600; border-bottom:1px solid #20262f; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.metrics-table td { padding:10px 14px; border-bottom:1px solid rgba(32,38,47,.5); color:#a8abb3; }
.metrics-table td:first-child { color:#f1f3fc; font-weight:500; }
</style>
</head>
<body class="min-h-screen pb-12">
<header class="fixed top-0 z-50 w-full flex justify-between items-center px-6 py-4 bg-slate-950/60 backdrop-blur-xl shadow-[0px_24px_48px_rgba(0,0,0,0.4)]">
  <div class="flex items-center gap-2">
    <span class="material-symbols-outlined text-blue-400" style="font-variation-settings:'FILL' 1">psychology</span>
    <h1 class="text-xl font-bold tracking-tight text-blue-400 font-headline">Neuro Ads</h1>
  </div>
  <div class="text-[10px] text-on-surface-variant tracking-wider uppercase font-label" id="videoTitle"></div>
</header>

<main class="pt-24 px-5 max-w-5xl mx-auto space-y-6">

  <!-- Hero: Neuro Rank + Radar -->
  <section class="grid grid-cols-1 md:grid-cols-5 gap-4">
    <div class="md:col-span-3 glass-panel rounded-3xl p-6 flex flex-col justify-between relative overflow-hidden bg-gradient-to-br from-surface-container-low to-surface-container">
      <div class="absolute -top-4 -right-4 w-24 h-24 bg-primary/5 rounded-full blur-2xl"></div>
      <span class="text-xs font-label text-on-surface-variant uppercase tracking-widest mb-1">Neuro Rank</span>
      <div class="flex items-baseline gap-2">
        <span class="font-headline text-6xl font-bold text-on-surface leading-none" id="neuroRank">--</span>
      </div>
      <div class="mt-3 flex items-center gap-2">
        <div class="px-3 py-1 bg-primary/10 border border-primary/20 rounded-full">
          <span class="font-headline font-bold text-primary text-sm" id="grade">--</span>
        </div>
      </div>
      <p class="text-[10px] text-on-surface-variant mt-3 leading-relaxed" id="heroNote"></p>
    </div>
    <div class="md:col-span-2 glass-panel rounded-3xl p-4 flex items-center justify-center bg-surface-container-low">
      <canvas id="radarChart" class="max-h-[220px]"></canvas>
    </div>
  </section>

  <!-- Score Cards -->
  <section class="grid grid-cols-2 md:grid-cols-4 gap-4" id="scoreCards"></section>

  <!-- Timeline Chart -->
  <section class="bg-surface-container-low rounded-3xl p-6 overflow-hidden relative border border-outline-variant/10">
    <div class="absolute inset-0 opacity-5 pointer-events-none" style="background-image:radial-gradient(circle at 1px 1px,#94aaff 1px,transparent 0);background-size:32px 32px"></div>
    <div class="relative z-10">
      <div class="flex justify-between items-center mb-4">
        <h3 class="text-sm font-semibold text-on-surface-variant font-label uppercase tracking-wider">Engajamento Neural ao Longo do Tempo</h3>
        <span class="material-symbols-outlined text-primary text-sm">timeline</span>
      </div>
      <canvas id="timelineChart"></canvas>
    </div>
  </section>

  <!-- 3D Brain Viewer -->
  <section id="viewer3dSection" class="bg-surface-container-low rounded-3xl overflow-hidden border border-outline-variant/10">
    <div class="px-6 pt-5 pb-3">
      <div class="flex justify-between items-center">
        <div>
          <h3 class="text-sm font-semibold text-on-surface-variant font-label uppercase tracking-wider">Visualizador 3D da Ativação Cerebral</h3>
          <p class="text-[11px] text-outline mt-1">Arraste para girar. Deslize o tempo para animar.</p>
        </div>
        <div class="flex items-center gap-1">
          <span class="w-2 h-2 rounded-full bg-tertiary animate-pulse"></span>
          <span class="text-[10px] text-tertiary font-bold uppercase tracking-widest font-label">Interativo</span>
        </div>
      </div>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-[1fr_280px]">
      <div id="brain3dContainer" class="relative min-h-[420px] bg-[#0a0e14]">
        <div id="brain3dLoading" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-primary text-sm">Carregando...</div>
        <div id="brain3dTimeLabel" class="absolute bottom-4 left-1/2 -translate-x-1/2 bg-surface-container-high/80 backdrop-blur-md px-4 py-1.5 rounded-full border border-primary/20 text-primary font-headline font-bold text-sm pointer-events-none">0.0s</div>
      </div>
      <div class="bg-surface-container p-4 border-l border-outline-variant/10 flex flex-col gap-3">
        <div>
          <label class="text-[10px] text-on-surface-variant uppercase tracking-widest font-label flex justify-between mb-1">Tempo <span id="brain3dTimeValue" class="text-primary font-bold">0.0s</span></label>
          <input type="range" id="brain3dSlider" min="0" max="30" value="0" step="1">
        </div>
        <button id="brain3dPlayBtn" class="w-full py-2 rounded-xl bg-primary/20 text-primary font-bold text-xs uppercase tracking-wider border border-primary/30 hover:bg-primary/30 transition-all active:scale-95">Reproduzir</button>
        <div>
          <label class="text-[10px] text-on-surface-variant uppercase tracking-widest font-label flex justify-between mb-1">Rotação <span id="brain3dRotVal" class="text-primary">Ligada</span></label>
          <input type="range" id="brain3dRotSpeed" min="0" max="100" value="30" step="1">
        </div>
        <span class="text-[9px] text-on-surface-variant uppercase tracking-widest font-label mt-1">Visualização</span>
        <div id="brain3dMeshToggle" class="grid grid-cols-2 gap-1 bg-surface-container-low p-1 rounded-xl border border-outline-variant/5">
          <button class="btn active flex flex-col items-center py-1.5 rounded-lg bg-blue-500/20 text-blue-400 text-[9px] font-bold uppercase tracking-tighter transition-all active:scale-95" data-mode="normal"><span class="material-symbols-outlined text-[16px]">view_in_ar</span>Normal</button>
          <button class="btn flex flex-col items-center py-1.5 rounded-lg text-on-surface-variant text-[9px] font-bold uppercase tracking-tighter hover:bg-surface-container-high transition-all" data-mode="inflated"><span class="material-symbols-outlined text-[16px]">texture</span>Inflado</button>
        </div>
        <div id="brain3dHemiToggle" class="grid grid-cols-4 gap-1 bg-surface-container-low p-1 rounded-xl border border-outline-variant/5">
          <button class="btn active flex flex-col items-center py-1.5 rounded-lg bg-blue-500/20 text-blue-400 text-[9px] font-bold uppercase tracking-tighter transition-all" data-mode="both"><span class="material-symbols-outlined text-[14px]">hub</span>Ambos</button>
          <button class="btn flex flex-col items-center py-1.5 rounded-lg text-on-surface-variant text-[9px] font-bold uppercase tracking-tighter hover:bg-surface-container-high transition-all" data-mode="left"><span class="material-symbols-outlined text-[14px]">keyboard_arrow_left</span>Esq</button>
          <button class="btn flex flex-col items-center py-1.5 rounded-lg text-on-surface-variant text-[9px] font-bold uppercase tracking-tighter hover:bg-surface-container-high transition-all" data-mode="right"><span class="material-symbols-outlined text-[14px]">keyboard_arrow_right</span>Dir</button>
          <button class="btn flex flex-col items-center py-1.5 rounded-lg text-on-surface-variant text-[9px] font-bold uppercase tracking-tighter hover:bg-surface-container-high transition-all" data-mode="open"><span class="material-symbols-outlined text-[14px]">open_in_full</span>Aberto</button>
        </div>
        <div id="brain3dHeadToggle" class="grid grid-cols-2 gap-1 bg-surface-container-low p-1 rounded-xl border border-outline-variant/5">
          <button class="btn flex flex-col items-center py-1.5 rounded-lg text-on-surface-variant text-[9px] font-bold uppercase tracking-tighter hover:bg-surface-container-high transition-all" data-mode="head-on"><span class="material-symbols-outlined text-[14px]">face</span>Cabeça</button>
          <button class="btn active flex flex-col items-center py-1.5 rounded-lg bg-blue-500/20 text-blue-400 text-[9px] font-bold uppercase tracking-tighter transition-all" data-mode="head-off"><span class="material-symbols-outlined text-[14px]">visibility_off</span>Sem</button>
        </div>
        <div class="mt-auto bg-surface-container-low/40 rounded-xl p-3 border border-outline-variant/10">
          <span class="text-[9px] text-on-surface-variant uppercase tracking-widest font-label block mb-2">Intensidade</span>
          <div class="h-2.5 w-full rounded-full bg-gradient-to-r from-blue-900 via-red-600 via-orange-500 to-yellow-200 shadow-inner"></div>
          <div class="flex justify-between mt-1"><span class="text-[9px] text-blue-400 font-bold">Baixa</span><span class="text-[9px] text-yellow-300 font-bold">Alta</span></div>
        </div>
      </div>
    </div>
  </section>

  <!-- Systems Chart -->
  <section class="bg-surface-container-low rounded-3xl p-6 border border-outline-variant/10 space-y-4">
    <div class="flex justify-between items-center">
      <div>
        <h3 class="text-sm font-semibold text-on-surface-variant font-label uppercase tracking-wider">Ativação por Sistema Funcional</h3>
        <p class="text-[11px] text-outline mt-1">Atividade dos 6 sistemas neurais ao longo do vídeo</p>
      </div>
      <span class="material-symbols-outlined text-primary">legend_toggle</span>
    </div>
    <canvas id="systemsChart"></canvas>
  </section>

  <!-- Time Window Cards -->
  <section class="space-y-3" id="windowCardsSection">
    <h3 class="text-lg font-bold font-headline">Janelas Temporais</h3>
    <div class="space-y-3" id="windowCards"></div>
  </section>

  <!-- Diagnostics + Reading Guide -->
  <section class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <div class="space-y-4">
      <h3 class="text-xs font-bold text-blue-300/60 uppercase tracking-[0.25em] font-label">Diagnóstico Rápido</h3>
      <div id="diagList" class="space-y-3"></div>
    </div>
    <div class="space-y-4">
      <h3 class="text-xs font-bold text-blue-300/60 uppercase tracking-[0.25em] font-label">Régua de Leitura</h3>
      <div class="grid grid-cols-2 gap-3">
        <div class="bg-surface-container-low rounded-xl p-4 border border-outline-variant/5">
          <span class="text-xs font-bold text-[#FF6B35] font-headline">0 – 1,5s</span>
          <p class="text-[11px] text-on-surface-variant mt-1 leading-tight">Visual + Movimento devem subir rápido. Se não sobe, thumb-stop fraco.</p>
        </div>
        <div class="bg-surface-container-low rounded-xl p-4 border border-outline-variant/5">
          <span class="text-xs font-bold text-[#FF9F1C] font-headline">1,5 – 3s</span>
          <p class="text-[11px] text-on-surface-variant mt-1 leading-tight">Transição para linguagem e integração. Se só occipital, hook raso.</p>
        </div>
        <div class="bg-surface-container-low rounded-xl p-4 border border-outline-variant/5">
          <span class="text-xs font-bold text-[#2EC4B6] font-headline">3 – 8s</span>
          <p class="text-[11px] text-on-surface-variant mt-1 leading-tight">Linguagem + TPJ + Pré-frontal devem crescer. Promessa deve "fechar sentido".</p>
        </div>
        <div class="bg-surface-container-low rounded-xl p-4 border border-outline-variant/5">
          <span class="text-xs font-bold text-[#4361EE] font-headline">CTA/Marca</span>
          <p class="text-[11px] text-on-surface-variant mt-1 leading-tight">Melhor entrar perto do pico de integração multimodal, não sozinho no final.</p>
        </div>
      </div>
      <div class="space-y-2">
        <span class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest font-label">Indicadores Secundários</span>
        <div id="secondaryDiag"></div>
      </div>
    </div>
  </section>

  <!-- Static Brain Maps -->
  <section class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <div class="bg-surface-container-low rounded-3xl overflow-hidden border border-outline-variant/10">
      <div class="p-4"><span class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest font-label">Ativação Média</span></div>
      <img id="brainAvg" class="w-full" alt="">
    </div>
    <div class="bg-surface-container-low rounded-3xl overflow-hidden border border-outline-variant/10">
      <div class="p-4"><span class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest font-label">Pico de Ativação</span></div>
      <img id="brainPeak" class="w-full" alt="">
    </div>
  </section>

  <!-- Deep Analysis -->
  <section class="glass-panel rounded-3xl p-6 bg-gradient-to-r from-surface-container-low to-surface-container border border-outline-variant/10 space-y-4 [&_h4]:text-sm [&_h4]:font-bold [&_h4]:font-headline [&_h4]:text-primary [&_h4]:uppercase [&_h4]:tracking-wider [&_h4]:mt-6 [&_h4]:mb-3 [&_h4]:pb-2 [&_h4]:border-b [&_h4]:border-outline-variant/10 [&_p]:text-[13px] [&_p]:text-on-surface-variant [&_p]:leading-relaxed [&_p]:mb-3 [&_ol]:pl-5 [&_ol]:mb-3 [&_li]:text-[13px] [&_li]:text-on-surface-variant [&_li]:leading-relaxed [&_li]:mb-3 [&_strong]:text-on-surface">
    <div class="flex justify-between items-center">
      <h3 class="font-headline text-base font-bold">Análise Profunda do Criativo</h3>
      <span class="material-symbols-outlined text-on-surface-variant text-sm">north_east</span>
    </div>
    <div id="analysisContent"></div>
  </section>

</main>

<script>
const DATA = PLACEHOLDER_JSON;

function getGrade(s){if(s>=85)return{l:'A+',c:'#b5ffc2'};if(s>=75)return{l:'A',c:'#b5ffc2'};if(s>=65)return{l:'B+',c:'#94aaff'};if(s>=55)return{l:'B',c:'#f59e0b'};if(s>=45)return{l:'C+',c:'#f59e0b'};if(s>=35)return{l:'C',c:'#ff6e84'};return{l:'F',c:'#ff6e84'};}
function sCol(s){return s>=70?'#b5ffc2':s>=50?'#f59e0b':'#ff6e84';}

document.getElementById('videoTitle').textContent=DATA.videoName+' — '+DATA.videoDuration+'s';
document.getElementById('neuroRank').textContent=DATA.scores.neuroRank.toFixed(1);
const gr=getGrade(DATA.scores.neuroRank);
document.getElementById('grade').textContent='Nota '+gr.l;
document.getElementById('grade').style.color=gr.c;
document.getElementById('heroNote').textContent=DATA.nota;

const scoreDefs=[
  {key:'hook',label:'Impacto do Hook',icon:'bolt',iconCol:'text-tertiary',bgCol:'bg-tertiary/10',weight:'35%'},
  {key:'semantic',label:'Clareza Semântica',icon:'auto_awesome',iconCol:'text-primary',bgCol:'bg-primary/10',weight:'20%'},
  {key:'synergy',label:'Sinergia Multimodal',icon:'waves',iconCol:'text-secondary',bgCol:'bg-secondary/10',weight:'25%'},
  {key:'coherence',label:'Coerência do Anúncio',icon:'link',iconCol:'text-error',bgCol:'bg-error/10',weight:'20%'},
];
document.getElementById('scoreCards').innerHTML=scoreDefs.map(s=>{
  const v=DATA.scores[s.key],c=sCol(v);
  return `<div class="p-5 rounded-3xl bg-surface-container-low border border-transparent hover:border-primary/20 transition-all space-y-3">
    <div class="flex justify-between items-start">
      <div class="p-2 ${s.bgCol} rounded-xl"><span class="material-symbols-outlined ${s.iconCol} text-lg">${s.icon}</span></div>
      <span class="text-xs font-headline font-bold" style="color:${c}">${v.toFixed(1)}</span>
    </div>
    <div>
      <h3 class="text-xs font-label text-on-surface-variant mb-1">${s.label}</h3>
      <div class="w-full h-1 bg-surface-container-highest rounded-full overflow-hidden"><div class="h-full rounded-full" style="width:${v}%;background:${c}"></div></div>
      <span class="text-[9px] text-outline mt-1 block">Peso: ${s.weight}</span>
    </div>
  </div>`;
}).join('');

const chartOpts={responsive:true,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#72757d',font:{size:10}}},y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#72757d',font:{size:10}}}}};
const tl=DATA.timeseries.overall.map((_,i)=>i+'s');

new Chart(document.getElementById('timelineChart'),{type:'line',data:{labels:tl,datasets:[{data:DATA.timeseries.overall,borderColor:'#94aaff',backgroundColor:'rgba(148,170,255,.1)',fill:true,tension:.3,pointRadius:3,pointHoverRadius:6,borderWidth:2}]},options:{...chartOpts,plugins:{legend:{display:false},tooltip:{callbacks:{title:i=>'Segmento '+i[0].dataIndex,label:i=>'Ativação: '+i.raw.toFixed(4)}}}}});

new Chart(document.getElementById('radarChart'),{type:'radar',data:{labels:['Hook','Sinergia','Semântica','Coerência'],datasets:[{data:[DATA.scores.hook,DATA.scores.synergy,DATA.scores.semantic,DATA.scores.coherence],borderColor:'#94aaff',backgroundColor:'rgba(148,170,255,.12)',borderWidth:2,pointBackgroundColor:'#94aaff',pointRadius:4}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{r:{min:0,max:100,ticks:{stepSize:25,color:'#72757d',backdropColor:'transparent',font:{size:9}},grid:{color:'rgba(255,255,255,.06)'},pointLabels:{color:'#a8abb3',font:{size:10}},angleLines:{color:'rgba(255,255,255,.06)'}}}}});

const sysList=['visual','auditory','language','integration','prefrontal','insula'];
new Chart(document.getElementById('systemsChart'),{type:'line',data:{labels:tl,datasets:sysList.map(s=>({label:DATA.systems[s].label,data:DATA.timeseries.systems[s],borderColor:DATA.systems[s].color,backgroundColor:DATA.systems[s].color+'15',borderWidth:1.5,tension:.3,pointRadius:0,pointHoverRadius:5}))},options:{...chartOpts,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'top',labels:{boxWidth:10,color:'#a8abb3',font:{size:10}}}}}});

const winColors={hook:{border:'border-l-[#FF6B35]',text:'text-[#FF6B35]',name:'Hook (0–1,5s)'},transition:{border:'border-l-[#FF9F1C]',text:'text-[#FF9F1C]',name:'Transição (1,5–3s)'},message:{border:'border-l-[#2EC4B6]',text:'text-[#2EC4B6]',name:'Mensagem (3–8s)'},sustain:{border:'border-l-[#4361EE]',text:'text-[#4361EE]',name:'Sustentação (8s+)'}};
const allSys=Object.keys(DATA.systems),allWin=Object.keys(DATA.windows);
let allV=[];allWin.forEach(w=>allSys.forEach(s=>allV.push(DATA.windowActivations[w]?.[s]||0)));
const mxV=Math.max(...allV,.001);
document.getElementById('windowCards').innerHTML=allWin.map(w=>{
  const wc=winColors[w]||{border:'border-l-primary',text:'text-primary',name:w};
  const topSys=allSys.map(s=>({s,v:DATA.windowActivations[w]?.[s]||0})).sort((a,b)=>b.v-a.v).slice(0,4);
  const avg=topSys.reduce((a,x)=>a+x.v,0)/topSys.length;
  return `<div class="bg-surface-container rounded-xl overflow-hidden shadow-lg border-l-4 ${wc.border}">
    <div class="p-4 flex justify-between items-center">
      <div><span class="text-[10px] font-bold text-outline uppercase tracking-widest font-label">${DATA.windows[w].label}</span><br><span class="text-sm font-bold font-headline">${wc.name}</span></div>
      <span class="text-xl font-bold font-headline ${wc.text}">${(avg*1000).toFixed(1)}</span>
    </div>
    <div class="bg-surface-container-low px-4 py-2.5 flex gap-4 overflow-x-auto">${topSys.map(ts=>{
      const ratio=ts.v/mxV;const lvl=ratio>.7?'Alto':ratio>.4?'Médio':'Baixo';
      return `<div class="flex-shrink-0"><span class="text-[10px] block text-outline uppercase tracking-tighter">${DATA.systems[ts.s].label.split('(')[0].trim()}</span><span class="text-xs font-medium">${lvl}</span></div>`;
    }).join('')}</div>
  </div>`;
}).join('');

document.getElementById('diagList').innerHTML=DATA.diagnostics.map(d=>{
  const icons={success:'bolt',warning:'warning',error:'error',info:'info'};
  const cols={success:'text-tertiary bg-tertiary/10',warning:'text-yellow-400 bg-yellow-400/10',error:'text-error bg-error/10',info:'text-primary bg-primary/10'};
  return `<div class="bg-slate-950/40 backdrop-blur-xl border border-white/5 rounded-3xl p-4 flex items-center gap-4">
    <div class="w-10 h-10 rounded-xl ${cols[d.type]||cols.info} flex items-center justify-center flex-shrink-0"><span class="material-symbols-outlined">${icons[d.type]||'info'}</span></div>
    <div><h4 class="font-bold text-sm text-on-surface tracking-tight">${d.title}</h4><p class="text-[11px] text-on-surface-variant mt-0.5 leading-tight">${d.text}</p></div>
  </div>`;
}).join('');

const secSys=['ffa','ppa','vwfa','motion'];
const overallMean=DATA.timeseries.overall.reduce((a,b)=>a+b,0)/DATA.timeseries.overall.length;
document.getElementById('secondaryDiag').innerHTML=secSys.map(s=>{
  const i=DATA.systems[s],vs=DATA.timeseries.systems[s],m=vs.reduce((a,b)=>a+b,0)/vs.length,r=m/Math.max(overallMean,.001);
  const st=r>1.2?'ALTO':r>.8?'MÉDIO':'BAIXO',sc=r>1.2?'text-tertiary':r>.8?'text-on-surface-variant':'text-error';
  return `<div class="flex justify-between items-center py-2 border-b border-outline-variant/10"><span class="text-[11px] text-on-surface-variant"><span class="inline-block w-2 h-2 rounded-full mr-2" style="background:${i.color}"></span>${i.label}</span><span class="text-[11px] font-bold ${sc}">${st}</span></div>`;
}).join('');

if(DATA.brainImages){
  if(DATA.brainImages.avg)document.getElementById('brainAvg').src='data:image/png;base64,'+DATA.brainImages.avg;
  if(DATA.brainImages.peak)document.getElementById('brainPeak').src='data:image/png;base64,'+DATA.brainImages.peak;
}

document.getElementById('analysisContent').innerHTML=DATA.creativeAnalysis;

// Hide 3D section if no data
if(!DATA.viewer3d||!DATA.glb||!Object.keys(DATA.glb).length){document.getElementById('viewer3dSection').style.display='none';}
</script>

<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.181.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.181.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const V3D=DATA.viewer3d,GLB=DATA.glb;
if(!V3D||!GLB||!Object.keys(GLB).length){/* skip */}
else{
const ctr=document.getElementById('brain3dContainer');
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(30,ctr.clientWidth/ctr.clientHeight,0.1,1000);
cam.position.set(0,0,380);
const ren=new THREE.WebGLRenderer({antialias:true,alpha:true});
ren.setSize(ctr.clientWidth,ctr.clientHeight);
ren.setPixelRatio(Math.min(window.devicePixelRatio,2));
ren.setClearColor(0x0a0e14);
ren.outputColorSpace=THREE.SRGBColorSpace;
ctr.appendChild(ren.domElement);
const ctl=new OrbitControls(cam,ren.domElement);
ctl.enableDamping=true;ctl.dampingFactor=0.05;ctl.autoRotate=true;ctl.autoRotateSpeed=1.2;
ctl.minDistance=200;ctl.maxDistance=550;
ctl.minPolarAngle=Math.PI/2;ctl.maxPolarAngle=Math.PI/2;
scene.add(new THREE.AmbientLight(0xffffff,1.8));
const dl=new THREE.DirectionalLight(0xffffff,1.2);dl.position.set(100,200,150);scene.add(dl);
const dl2=new THREE.DirectionalLight(0xffffff,0.8);dl2.position.set(-100,100,-100);scene.add(dl2);
const dl3=new THREE.DirectionalLight(0xffffff,0.6);dl3.position.set(0,-150,100);scene.add(dl3);
const brainGrp=new THREE.Group();scene.add(brainGrp);
const ms={};let curFrame=0,playing=false,playInt=null,meshMode='normal',hemiMode='both',showHead=false;
function b64toBuf(b){const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a.buffer;}
function applyColors(mesh,rgb64){const rgb=new Uint8Array(b64toBuf(rgb64));const g=mesh.geometry,idx=g.index;if(!g.attributes.color)g.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count*3),3));const c=g.attributes.color;if(idx){const fc=idx.count/3;for(let f=0;f<fc&&f<V3D.numFaces;f++){const r=rgb[f*3]/255,gv=rgb[f*3+1]/255,b=rgb[f*3+2]/255;for(let v=0;v<3;v++){c.setXYZ(idx.getX(f*3+v),r,gv,b);}}}c.needsUpdate=true;}
function setFrame(fi){curFrame=Math.max(0,Math.min(fi,V3D.numSegments-1));const fd=V3D.frames[curFrame];for(const[h,b64] of Object.entries(fd)){const k=meshMode==='inflated'?h+'Inflated':h;if(ms[k])applyColors(ms[k],b64);}const t=(curFrame/V3D.numSegments*V3D.videoDuration).toFixed(1);document.getElementById('brain3dTimeLabel').textContent=t+'s';document.getElementById('brain3dTimeValue').textContent=t+'s';document.getElementById('brain3dSlider').value=curFrame;}
function updateVis(){for(const[k,m] of Object.entries(ms)){if(k==='head'){m.visible=showHead;continue;}const isInf=k.includes('Inflated'),isL=k.includes('left')||k.includes('Left'),isR=k.includes('right')||k.includes('Right');const isCur=meshMode==='inflated'?isInf:!isInf;let vis=isCur;if(hemiMode==='left')vis=vis&&isL;else if(hemiMode==='right')vis=vis&&isR;m.visible=vis;m.position.x=(hemiMode==='open'&&vis)?(isL?-15:15):0;}}
const ldr=new GLTFLoader();
async function loadGLB(name,b64){return new Promise(r=>{ldr.parse(b64toBuf(b64),'',gltf=>{gltf.scene.traverse(ch=>{if(ch.isMesh){if(name==='head'){ch.material=new THREE.MeshLambertMaterial({transparent:true,opacity:0.12,side:THREE.DoubleSide,depthWrite:false,color:0x8892a8});}else{ch.material=new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.FrontSide});}ms[name]=ch;brainGrp.add(ch);r(ch);}});});});}
async function init3d(){for(const n of['left','right','leftInflated','rightInflated','head']){if(GLB[n])await loadGLB(n,GLB[n]);}setFrame(0);updateVis();document.getElementById('brain3dLoading').style.display='none';(function anim(){requestAnimationFrame(anim);ctl.update();ren.render(scene,cam);})();}
const sl3d=document.getElementById('brain3dSlider');sl3d.max=V3D.numSegments-1;
sl3d.addEventListener('input',()=>setFrame(parseInt(sl3d.value)));
document.getElementById('brain3dPlayBtn').addEventListener('click',()=>{if(playing){playing=false;clearInterval(playInt);document.getElementById('brain3dPlayBtn').textContent='Reproduzir';}else{playing=true;document.getElementById('brain3dPlayBtn').textContent='Pausar';const iv=V3D.videoDuration/V3D.numSegments*1000;playInt=setInterval(()=>{const nx=(curFrame+1)%V3D.numSegments;setFrame(nx);sl3d.value=nx;if(nx===0){playing=false;clearInterval(playInt);document.getElementById('brain3dPlayBtn').textContent='Reproduzir';}},iv);}});
document.getElementById('brain3dRotSpeed').addEventListener('input',e=>{const v=parseInt(e.target.value);ctl.autoRotate=v>0;ctl.autoRotateSpeed=v/25;document.getElementById('brain3dRotVal').textContent=v>0?'Ligada':'Desligada';});
function setupBG(id,cb){document.getElementById(id).querySelectorAll('[data-mode]').forEach(b=>{b.addEventListener('click',()=>{document.getElementById(id).querySelectorAll('[data-mode]').forEach(x=>{x.classList.remove('bg-blue-500/20','text-blue-400');x.classList.add('text-on-surface-variant');});b.classList.remove('text-on-surface-variant');b.classList.add('bg-blue-500/20','text-blue-400');cb(b.dataset.mode);});});}
setupBG('brain3dMeshToggle',m=>{meshMode=m;setFrame(curFrame);updateVis();});
setupBG('brain3dHemiToggle',m=>{hemiMode=m;updateVis();});
setupBG('brain3dHeadToggle',m=>{showHead=m==='head-on';updateVis();});
new ResizeObserver(()=>{cam.aspect=ctr.clientWidth/ctr.clientHeight;cam.updateProjectionMatrix();ren.setSize(ctr.clientWidth,ctr.clientHeight);}).observe(ctr);
init3d();
}
</script>
</body>
</html>"""
