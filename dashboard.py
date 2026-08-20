"""Clinician-style Streamlit dashboard for the stroke-orthosis pipeline.

Run:  streamlit run dashboard.py
"""

import html

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

mne.set_log_level('ERROR')

from orthosis import (
    EEG_MAP,
    GATE1_FAIL_REASON,
    GATE2_FAIL_REASON,
    PARTICIPANTS_TSV,
    get_stroke_location,
    list_participant_ids,
    rescore_result,
    run_subject,
    stdev_threshold_gate1,
    stdev_threshold_gate2,
)

DEFAULT_K = 1.5
K_MIN = -2.5
K_MAX = 2.5

BG = '#050505'
PANEL = '#0e0e0e'
TEXT = '#f4f7fb'
MUTED = '#8b95a5'
RED = '#ff3b4e'
GREEN = '#3ee6a0'
CYAN = '#5ce1ff'
BLUE = CYAN
GRID = '#22262c'
HEAD = '#3a4250'
PASS_FILL = 'rgba(92,225,255,0.14)'
FAIL_FILL = 'rgba(255,59,78,0.14)'

EEG_POS = {
    'Fp1': (-0.28, 0.88), 'Fp2': (0.28, 0.88),
    'Fz': (0.0, 0.62), 'F3': (-0.40, 0.55), 'F4': (0.40, 0.55),
    'F7': (-0.68, 0.48), 'F8': (0.68, 0.48),
    'FCz': (0.0, 0.34), 'FC3': (-0.40, 0.30), 'FC4': (0.40, 0.30),
    'FT7': (-0.78, 0.22), 'FT8': (0.78, 0.22),
    'Cz': (0.0, 0.04), 'C3': (-0.40, 0.04), 'C4': (0.40, 0.04),
    'T3': (-0.88, 0.00), 'T4': (0.88, 0.00),
    'CPz': (0.0, -0.22), 'CP3': (-0.40, -0.22), 'CP4': (0.40, -0.22),
    'TP7': (-0.78, -0.28), 'TP8': (0.78, -0.28),
    'Pz': (0.0, -0.48), 'P3': (-0.40, -0.48), 'P4': (0.40, -0.48),
    'T5': (-0.68, -0.52), 'T6': (0.68, -0.52),
    'Oz': (0.0, -0.86), 'O1': (-0.28, -0.80), 'O2': (0.28, -0.80),
}

LEFT_MOTOR = ('FC3', 'C3', 'CP3')
RIGHT_MOTOR = ('FC4', 'C4', 'CP4')
LEFT_HAND_XY = (-1.62, -0.22)
RIGHT_HAND_XY = (1.62, -0.22)


def _bezier(p0, p1, p2, n=48):
    t = np.linspace(0, 1, n)
    p0, p1, p2 = np.asarray(p0), np.asarray(p1), np.asarray(p2)
    pts = (1 - t)[:, None] ** 2 * p0 + 2 * (1 - t)[:, None] * t[:, None] * p1 + t[:, None] ** 2 * p2
    return pts[:, 0], pts[:, 1]


def _centroid(names):
    pts = np.array([EEG_POS[n] for n in names])
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _pathway_for_paralysis(paralysis_side):
    """Lesion hemisphere motor strip → paralyzed (contralateral) hand."""
    if paralysis_side == 'right':
        return {
            'key': 'left_to_right',
            'electrodes': LEFT_MOTOR,
            'hand': 'right',
            'lesion': 'left',
            'label': 'Left motor strip (C3, FC3, CP3) → right hand',
        }
    if paralysis_side == 'left':
        return {
            'key': 'right_to_left',
            'electrodes': RIGHT_MOTOR,
            'hand': 'left',
            'lesion': 'right',
            'label': 'Right motor strip (C4, FC4, CP4) → left hand',
        }
    return None


def _path_from_click(kind, name):
    if kind == 'hand' and name == 'right':
        return 'left_to_right'
    if kind == 'hand' and name == 'left':
        return 'right_to_left'
    if kind == 'strip' and name == 'left':
        return 'left_to_right'
    if kind == 'strip' and name == 'right':
        return 'right_to_left'
    return None


def _hand_outline(cx, cy, mirror=False, scale=0.46):
    xs = np.array([
        0.00, 0.04, 0.10, 0.16, 0.20, 0.22, 0.18, 0.20, 0.36, 0.38, 0.22,
        0.24, 0.42, 0.44, 0.24, 0.22, 0.36, 0.36, 0.16, 0.12, 0.04, 0.00, 0.00,
    ])
    ys = np.array([
        -0.11, -0.20, -0.28, -0.30, -0.24, -0.14, -0.12, -0.12, -0.14, -0.06,
        -0.06, -0.02, -0.02, 0.06, 0.06, 0.10, 0.14, 0.22, 0.16, 0.20, 0.14, 0.10, -0.11,
    ])
    if mirror:
        xs = -xs
    return cx + xs * scale, cy + ys * scale


def make_pathway_figure(paralysis_side, lit_key):
    """Headset + both hands. Motor-strip electrodes are not individually clickable.
    Click a hand or a motor-strip cluster to switch pathway view."""
    paretic = _pathway_for_paralysis(paralysis_side)
    paretic_key = paretic['key'] if paretic else None
    if lit_key is None:
        lit_key = paretic_key

    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 180)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color=HEAD, width=3),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[1.0, 1.12], mode='lines',
        line=dict(color=HEAD, width=3), hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[-0.22, 0.22], y=[-1.02, -1.02], mode='lines',
        line=dict(color=HEAD, width=6), hoverinfo='skip', showlegend=False,
    ))

    def add_glow_path(x, y, color, active):
        width_glow = 16 if active else 3
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(color=color, width=width_glow),
            opacity=0.22 if active else 0.18,
            hoverinfo='skip', showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(color=color, width=3.2 if active else 1.4, dash='solid' if active else 'dot'),
            hoverinfo='skip', showlegend=False,
        ))

    left_c = _centroid(LEFT_MOTOR)
    right_c = _centroid(RIGHT_MOTOR)
    lx, ly = _bezier(left_c, (0.12, -0.72), RIGHT_HAND_XY)
    rx, ry = _bezier(right_c, (-0.12, -0.72), LEFT_HAND_XY)
    left_active = lit_key == 'left_to_right'
    right_active = lit_key == 'right_to_left'
    add_glow_path(lx, ly, RED if paretic_key == 'left_to_right' else CYAN, left_active)
    add_glow_path(rx, ry, RED if paretic_key == 'right_to_left' else CYAN, right_active)

    fig.add_annotation(
        x=0, y=-0.70, text='decussation (corticospinal tract)',
        showarrow=False, font=dict(size=10, color=MUTED),
    )

    other = [n for n in EEG_MAP if n not in LEFT_MOTOR and n not in RIGHT_MOTOR]
    fig.add_trace(go.Scatter(
        x=[EEG_POS[n][0] for n in other],
        y=[EEG_POS[n][1] for n in other],
        mode='markers+text',
        marker=dict(size=10, color='#4a5564', line=dict(color=BG, width=1)),
        text=other, textposition='bottom center',
        textfont=dict(size=8, color=MUTED),
        hovertemplate='%{text}<extra></extra>',
        name='other', showlegend=False,
    ))

    def add_motor(names, strip_id, active, color):
        mx = [EEG_POS[n][0] for n in names]
        my = [EEG_POS[n][1] for n in names]
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode='markers',
            marker=dict(size=36 if active else 22, color=color, opacity=0.22 if active else 0.08),
            customdata=[['strip', strip_id]] * len(names),
            hovertemplate='Motor strip · click to light pathway<extra></extra>',
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode='markers+text',
            marker=dict(
                size=18 if active else 13,
                color=color if active else '#3d4654',
                line=dict(color=color if active else '#5a6573', width=1.6),
            ),
            text=list(names), textposition='bottom center',
            textfont=dict(size=10, color=TEXT if active else MUTED, family='IBM Plex Mono, monospace'),
            hoverinfo='skip',
            showlegend=False,
        ))

    add_motor(
        LEFT_MOTOR, 'left', left_active,
        RED if paretic_key == 'left_to_right' else CYAN,
    )
    add_motor(
        RIGHT_MOTOR, 'right', right_active,
        RED if paretic_key == 'right_to_left' else CYAN,
    )

    def add_hand(side, active, color, paretic_hand):
        cx, cy = (RIGHT_HAND_XY if side == 'right' else LEFT_HAND_XY)
        hx, hy = _hand_outline(cx, cy, mirror=(side == 'left'))
        label = f"{'R' if side == 'right' else 'L'} HAND"
        if paretic_hand:
            label += '  · paretic'
        fig.add_trace(go.Scatter(
            x=list(hx), y=list(hy),
            fill='toself',
            fillcolor=color if active else 'rgba(58,66,80,0.28)',
            line=dict(color=color if active else '#5a6573', width=2.4 if active else 1.4),
            opacity=0.92 if active else 0.7,
            customdata=[['hand', side]] * len(hx),
            hovertemplate=label + '<extra></extra>',
            name=side + '_hand',
            showlegend=False,
        ))
        fig.add_annotation(
            x=cx, y=cy - 0.42,
            text=label,
            showarrow=False,
            font=dict(size=11, color=color if active else MUTED, family='IBM Plex Sans, sans-serif'),
        )

    add_hand(
        'right', left_active,
        RED if paretic_key == 'left_to_right' else CYAN,
        paretic and paretic['hand'] == 'right',
    )
    add_hand(
        'left', right_active,
        RED if paretic_key == 'right_to_left' else CYAN,
        paretic and paretic['hand'] == 'left',
    )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=8, r=8, t=40, b=8),
        xaxis=dict(visible=False, range=[-2.15, 2.15], constrain='domain'),
        yaxis=dict(visible=False, range=[-1.35, 1.22], constrain='domain'),
        height=460,
        title=dict(
            text='EEG pathway  ·  both hands shown  ·  click a hand or motor strip to switch view',
            font=dict(color=MUTED, size=13),
            x=0.0, xanchor='left',
        ),
        font=dict(color=TEXT),
        clickmode='event+select',
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig


def make_trial_structure_figure():
    """Interactive 0–8 s number line for one trial: instruction / imagery / break."""
    fig = go.Figure()
    segments = [
        (0, 2, '#3a4250', 'Instruction', '0–2 s  ·  cue  ·  not scored by the gates'),
        (2, 6, RED, 'Motor imagery', '2–6 s  ·  Gate 1 and Gate 2 evaluation window'),
        (6, 8, '#1d5c6b', 'Break / rest', '6–8 s  ·  Gate 1 calibration uses this window'),
    ]
    for x0, x1, color, name, hover in segments:
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[0.18, 0.18, 0.82, 0.82, 0.18],
            fill='toself',
            fillcolor=color,
            line=dict(color=color, width=1),
            name=name,
            hovertemplate=f'<b>{name}</b><br>{hover}<extra></extra>',
        ))
        fig.add_annotation(
            x=(x0 + x1) / 2, y=0.50, text=name,
            showarrow=False, font=dict(color=TEXT, size=12),
        )

    ticks = [0, 2, 6, 8]
    fig.add_trace(go.Scatter(
        x=ticks, y=[0, 0, 0, 0],
        mode='markers+text',
        marker=dict(size=10, color=TEXT),
        text=[f'{t} s' for t in ticks],
        textposition='bottom center',
        textfont=dict(size=12, color=MUTED),
        hovertemplate='%{x} s<extra></extra>',
        name='Time',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0, 8], y=[0, 0],
        mode='lines',
        line=dict(color='#3a4250', width=2),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_vline(x=2, line_width=1, line_dash='dot', line_color=MUTED)
    fig.add_vline(x=6, line_width=1, line_dash='dot', line_color=MUTED)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        height=200,
        margin=dict(l=16, r=16, t=28, b=36),
        font=dict(color=TEXT, size=12),
        title=dict(text='One trial on the time axis  ·  hover a phase', font=dict(size=13, color=MUTED), x=0, xanchor='left'),
        xaxis=dict(
            title='Time in trial (s)',
            range=[-0.35, 8.35],
            dtick=1,
            gridcolor=GRID,
            zeroline=False,
            color=MUTED,
        ),
        yaxis=dict(visible=False, range=[-0.45, 1.15]),
        legend=dict(orientation='h', y=1.18, x=1, xanchor='right', font=dict(size=11, color=MUTED), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest',
    )
    return fig


def _spread_y(xs, y0=0.0, row=0.11, min_frac=0.012):
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    ys = np.full(n, y0)
    if n == 0:
        return ys
    span = max(float(np.ptp(xs)), 1e-12)
    thresh = span * min_frac
    order = np.argsort(xs)
    placed_x, placed_y = [], []
    for i in order:
        x = xs[i]
        y = y0
        level = 0
        while any(abs(x - px) < thresh and abs(y - py) < row * 0.8 for px, py in zip(placed_x, placed_y)):
            level += 1
            y = y0 + ((level + 1) // 2) * row * (1 if level % 2 else -1)
        ys[i] = y
        placed_x.append(x)
        placed_y.append(y)
    return ys


def make_gate_figure(
    samples, mean, std, k, trial_value, mode, passed,
    gate_name, dist_caption, x_title, trial_label, x_scale=1.0, x_tickformat='.3f',
    require_positive=False,
):
    """Number line: calibration points, shaded pass region, trial of interest."""
    samples = np.asarray(samples, dtype=float).ravel() * x_scale
    mean = float(mean) * x_scale
    std = float(std) * x_scale if std is not None else 0.0
    k = float(k)
    trial_value = float(trial_value) * x_scale
    bound = mean - k * std
    accent = GREEN if passed else RED
    status = 'PASS' if passed else 'FAIL'
    if mode == 'below':
        rule = 'below the bound'
    elif require_positive:
        rule = 'above 0 and at/above the bound'
    else:
        rule = 'above the bound'

    span_pts = list(samples) + [trial_value, bound, mean]
    if require_positive:
        span_pts.append(0.0)
    lo, hi = min(span_pts), max(span_pts)
    pad = max((hi - lo) * 0.22, (std * 1.2 if std > 0 else abs(trial_value) * 0.1 + 1e-12))
    x_min, x_max = lo - pad, hi + pad

    fig = go.Figure()
    if mode == 'below':
        region_x0, region_x1 = x_min, bound
        region_label = 'Pass region (below bound)'
    else:
        region_x0 = max(bound, 0.0) if require_positive else bound
        region_x1 = x_max
        region_label = 'Pass region (LI > 0 and above bound)' if require_positive else 'Pass region (above bound)'
    fig.add_trace(go.Scatter(
        x=[region_x0, region_x1, region_x1, region_x0],
        y=[-0.55, -0.55, 0.55, 0.55],
        fill='toself',
        fillcolor=PASS_FILL if passed else FAIL_FILL,
        line=dict(width=0),
        name=region_label,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[0, 0],
        mode='lines',
        line=dict(color='#3a4250', width=2),
        hoverinfo='skip', showlegend=False,
    ))

    if len(samples):
        ys = _spread_y(samples)
        fig.add_trace(go.Scatter(
            x=samples,
            y=ys,
            mode='markers',
            marker=dict(size=13, color=CYAN, line=dict(color=TEXT, width=0.6)),
            name=f'Calibration trials (n={len(samples)})',
            hovertemplate='Calibration trial = %{x:.4g}<extra></extra>',
        ))

    fig.add_trace(go.Scatter(
        x=[mean, mean], y=[-0.55, 0.7],
        mode='lines',
        line=dict(color=BLUE, width=1.8, dash='dot'),
        name='Mean of those trials',
        hovertemplate=f'mean = {mean:.4g}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[bound, bound], y=[-0.55, 0.7],
        mode='lines',
        line=dict(color=RED, width=2.2, dash='dash'),
        name=f'Bound (mean − {k:.2f}·SD)',
        hovertemplate=f'bound = {bound:.4g}<extra>mean − {k:.2f}·SD</extra>',
    ))
    if require_positive:
        fig.add_trace(go.Scatter(
            x=[0.0, 0.0], y=[-0.55, 0.7],
            mode='lines',
            line=dict(color=MUTED, width=1.6, dash='dot'),
            name='LI = 0  (must be above)',
            hovertemplate='LI = 0<extra>trial must be > 0</extra>',
        ))
    fig.add_trace(go.Scatter(
        x=[trial_value], y=[0.0],
        mode='markers',
        marker=dict(size=18, color=RED, line=dict(color=TEXT, width=1.0)),
        name='This test trial',
        hovertemplate=f'{trial_label}<br>%{{x:.4g}}<extra>{status}</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[trial_value], y=[0.98],
        mode='text',
        text=[status],
        textfont=dict(color=accent, size=16),
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        height=340,
        margin=dict(l=16, r=16, t=90, b=88),
        font=dict(color=TEXT, size=12),
        title=dict(
            text=(
                f'<b>{gate_name}  ·  {status}</b>'
                f'<br><span style="color:{MUTED};font-size:12px">{dist_caption}</span>'
                f'<br><span style="color:{accent};font-size:12px">'
                f'Cyan dots = calibration. Red dot = this test trial, which must sit {rule}.</span>'
            ),
            x=0.0, xanchor='left',
            font=dict(size=16, color=TEXT),
        ),
        xaxis=dict(
            title=x_title,
            range=[x_min, x_max],
            gridcolor=GRID,
            zeroline=False,
            color=MUTED,
            tickformat=x_tickformat,
            exponentformat='none',
            showexponent='none',
        ),
        yaxis=dict(visible=False, range=[-0.75, 1.25]),
        legend=dict(
            orientation='h',
            yanchor='top', y=-0.32,
            xanchor='left', x=0,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color=MUTED),
        ),
        hovermode='closest',
        uirevision=f'{gate_name}-{k}-{trial_value}-{mean}',
    )
    return fig


@st.cache_data(show_spinner='Running pipeline for this participant…')
def load_subject(participant_id):
    return run_subject(participant_id, verbose=False)


def stroke_location_for(participant_id):
    metadata_df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
    return get_stroke_location(metadata_df, participant_id)


def _inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {BG}; color: {TEXT}; }}
        header[data-testid="stHeader"] {{ background: {BG}; }}
        section[data-testid="stSidebar"] {{ background: #080808; border-right: 1px solid {GRID}; }}
        h1, h2, h3, p, label, span {{ color: {TEXT} !important; }}
        .stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
        div[data-testid="stMetric"] {{
            background: {PANEL};
            padding: 0.65rem 0.85rem;
            border-radius: 12px;
            border: 1px solid {GRID};
        }}
        div[data-testid="stMetric"] label {{ color: {MUTED} !important; }}
        .card {{
            background: {PANEL};
            border: 1px solid {GRID};
            border-radius: 14px;
            padding: 14px 16px;
            color: {TEXT};
        }}
        .checkpoint {{
            background: {PANEL};
            border: 1px solid {GRID};
            border-radius: 14px;
            padding: 12px 14px;
            color: {TEXT};
            min-height: 96px;
        }}
        .checkpoint b {{ color: {CYAN}; }}
        .path-card {{
            background: {PANEL};
            border: 1px solid {RED};
            border-radius: 12px;
            padding: 10px 14px;
            color: {TEXT};
            margin: 8px 0 4px 0;
        }}
        .trial-grid-hint {{ color: {MUTED}; font-size: 13px; margin: 0 0 8px 0; }}
        .trial-footnote {{ color: {MUTED}; font-size: 12px; margin: -6px 0 10px 0; font-style: italic; }}
        .trial-face {{
            background: {PANEL};
            border: 2px solid {GRID};
            border-radius: 12px 12px 0 0;
            padding: 10px 12px 8px 12px;
            margin-bottom: -6px;
        }}
        .trial-head {{ font-weight: 700; color: {TEXT}; margin-bottom: 8px; font-size: 13px; }}
        .trial-gate {{
            padding: 2px 0 2px 12px;
            font-size: 13px;
            line-height: 1.55;
            letter-spacing: 0.02em;
        }}
        div[data-testid="stButton"] button {{
            background: {PANEL};
            color: {TEXT};
            border: 1px solid {GRID};
            border-radius: 12px;
        }}
        div[data-testid="stButton"] button:hover {{
            border-color: {CYAN};
            color: {CYAN};
        }}
        div[data-testid="column"] div[data-testid="stButton"] > button {{
            min-height: 36px;
            font-size: 12px;
            border-width: 2px;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_diagram_click(event):
    points = []
    if event is not None:
        selection = getattr(event, 'selection', None)
        if selection is not None:
            points = getattr(selection, 'points', None) or []
            if not points and isinstance(selection, dict):
                points = selection.get('points') or []
    if not points:
        return
    point = points[0]
    custom = point.get('customdata') if isinstance(point, dict) else getattr(point, 'customdata', None)
    if custom is None:
        return
    if isinstance(custom, (list, tuple)) and len(custom) >= 2:
        kind, name = custom[0], custom[1]
    else:
        return
    nxt = _path_from_click(kind, name)
    if nxt and nxt != st.session_state.get('lit_key'):
        st.session_state.lit_key = nxt
        st.rerun()


def _trial_card_html(trial, selected=False):
    opened = trial['opened']
    result = 'OPEN' if opened else 'LOCK'
    g1_ok = trial['gate1_passed']
    g2_ok = trial['gate2_passed']
    border = CYAN if selected else (GREEN if opened else RED)
    g1_color = GREEN if g1_ok else RED
    g2_color = GREEN if g2_ok else RED
    g1_icon = '●' if g1_ok else '●'
    g2_icon = '●' if g2_ok else '●'
    return (
        f"<div class='trial-face' style='border-color:{border}'>"
        f"<div class='trial-head'>Trial {trial['trial_num']}  ·  "
        f"<span style='color:{GREEN if opened else RED}'>{result}</span></div>"
        f"<div class='trial-gate' style='color:{g1_color}'>G1&nbsp;&nbsp;&nbsp;{g1_icon}&nbsp;&nbsp;{'PASS' if g1_ok else 'FAIL'}</div>"
        f"<div class='trial-gate' style='color:{g2_color}'>G2&nbsp;&nbsp;&nbsp;{g2_icon}&nbsp;&nbsp;{'PASS' if g2_ok else 'FAIL'}</div>"
        f"</div>"
    )


def main():
    st.set_page_config(page_title='BrainTrain', layout='wide')
    _inject_css()

    ids = list_participant_ids()
    if not ids:
        st.error('No sub-* folders with .mat files found.')
        return

    st.sidebar.title('Participant')
    participant_id = st.sidebar.selectbox('ID', ids, index=0)

    if st.session_state.get('loaded_pid') != participant_id:
        st.session_state.loaded_pid = participant_id
        st.session_state.k1 = DEFAULT_K
        st.session_state.k2 = DEFAULT_K
        st.session_state.selected_trial = None
        st.session_state.lit_key = None

    st.sidebar.markdown('---')
    st.sidebar.subheader('SD thresholds')
    st.sidebar.caption('Same inequalities as orthosis.py. Reset to 1.5 when you change participant.')
    k1 = st.sidebar.slider('Gate 1 — ERD  (k)', min_value=K_MIN, max_value=K_MAX, step=0.1, key='k1')
    k2 = st.sidebar.slider('Gate 2 — LI  (k)', min_value=K_MIN, max_value=K_MAX, step=0.1, key='k2')
    if st.sidebar.button('Reset to 1.5 SD'):
        st.session_state.k1 = DEFAULT_K
        st.session_state.k2 = DEFAULT_K
        st.rerun()

    raw = load_subject(participant_id)
    result = rescore_result(raw, k1, k2)

    paralysis = result.get('paralysis_side') or '—'
    handedness = result.get('handedness') or '—'
    lesion_side = {'right': 'left', 'left': 'right'}.get(paralysis, '—')
    paretic = _pathway_for_paralysis(paralysis)
    default_key = paretic['key'] if paretic else None
    if st.session_state.get('lit_key') is None:
        st.session_state.lit_key = default_key

    chans = result.get('paretic_channels') or []
    st.sidebar.markdown('---')
    st.sidebar.caption(
        f"Paretic channels: {' · '.join(chans) if chans else '—'}\n\n"
        f"Paretic hand: {paralysis.title() if paralysis != '—' else '—'}"
    )

    st.title('BrainTrain')
    st.caption(
        "An orthosis that opens when a participant's motor imagery passes both veto gates, "
        "signalling significant contralateral activity and promoting exercise of impaired pathways."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Participant', participant_id)
    m2.metric('Paralysis side', paralysis)
    m3.metric('Inferred lesion hemisphere', lesion_side)
    m4.metric('Handedness', handedness)
    stroke = stroke_location_for(participant_id) or '—'
    st.markdown(
        f"<div class='path-card' style='border-color:{GRID}'>"
        f"<b style='color:{CYAN}'>Stroke location</b><br>{html.escape(stroke)}</div>",
        unsafe_allow_html=True,
    )

    st.subheader('Pipeline checkpoints')
    c1, c2, c3, c4 = st.columns(4)
    n_orig = result.get('n_orig_trials')
    n_rej = result.get('n_rejected')
    with c1:
        st.markdown(
            f"<div class='checkpoint'><b>Epochs rejected</b><br>"
            f"{n_rej if n_rej is not None else '—'} / {n_orig if n_orig is not None else '—'}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        band = result.get('chosen_band') or '—'
        r2 = result.get('chosen_band_r2')
        hz = result.get('chosen_band_hz')
        hz_txt = f"{hz[0]}–{hz[1]} Hz" if hz else ''
        r2_txt = f"R² = {r2:.4f}" if r2 is not None else ''
        st.markdown(
            f"<div class='checkpoint'><b>Frequency band (best R²)</b><br>{band} {hz_txt}<br>{r2_txt}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        cal = result.get('cal_trial_nums') or []
        st.markdown(
            f"<div class='checkpoint'><b>Calibration trials</b><br>"
            f"{cal if cal else '—'}</div>",
            unsafe_allow_html=True,
        )
    with c4:
        tes = [int(t['trial_num']) for t in (result.get('trials') or [])]
        if not tes:
            tes = result.get('test_paretic_trial_nums') or []
        st.markdown(
            f"<div class='checkpoint'><b>Test trials (paretic)</b><br>"
            f"{tes if tes else '—'}</div>",
            unsafe_allow_html=True,
        )

    st.subheader('Trial structure')
    st.plotly_chart(
        make_trial_structure_figure(),
        use_container_width=True,
        config={'displayModeBar': False},
        key='trial-structure',
    )
    st.caption('Imagery epoch: tmin = 2 s, tmax = 6 s (gate evaluation window). Hover a phase for what it is used for.')

    map_col, grid_col = st.columns([1.12, 1], gap='large')

    with map_col:
        st.subheader('EEG pathway map')
        lit_key = st.session_state.get('lit_key') or default_key
        fig = make_pathway_figure(paralysis, lit_key)
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select='rerun',
            selection_mode='points',
            key=f'diagram-{participant_id}',
            config={'displayModeBar': False},
        )
        _apply_diagram_click(event)

        lit_key = st.session_state.get('lit_key') or default_key
        if paretic and lit_key == paretic['key']:
            st.markdown(
                f"<div class='path-card'><b style='color:{RED}'>Lit · controlled (paretic) hand</b><br>"
                f"{paretic['label']}<br>"
                f"Lesion hemisphere: <b>{paretic['lesion']}</b>  ·  paralyzed hand: <b>{paretic['hand']}</b></div>",
                unsafe_allow_html=True,
            )
        elif lit_key == 'left_to_right':
            st.info('View: left motor strip (C3, FC3, CP3) → right hand.')
        elif lit_key == 'right_to_left':
            st.info('View: right motor strip (C4, FC4, CP4) → left hand.')

        b1, b2, b3 = st.columns(3)
        if b1.button('Paretic / controlled', use_container_width=True) and paretic:
            st.session_state.lit_key = paretic['key']
            st.rerun()
        if b2.button('Left → right hand', use_container_width=True):
            st.session_state.lit_key = 'left_to_right'
            st.rerun()
        if b3.button('Right → left hand', use_container_width=True):
            st.session_state.lit_key = 'right_to_left'
            st.rerun()

    with grid_col:
        st.subheader('Testing trials')
        st.markdown(
            "<p class='trial-footnote'>(paretic trials from the second half of clean trials)</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='trial-grid-hint'>Each box shows G1 and G2 status. OPEN only if both pass. "
            "Click View distributions to see that trial on the Gate 1 / Gate 2 number lines.</p>",
            unsafe_allow_html=True,
        )

        if result.get('skipped'):
            st.error(result.get('skip_reason') or 'This participant was skipped.')
            st.info('No test-half paretic trials to score for open / lock.')
        else:
            trials = result.get('trials') or []
            if not trials:
                st.warning('No evaluated trials.')
            else:
                selected_num = st.session_state.get('selected_trial')
                n_cols = 4
                for row_start in range(0, len(trials), n_cols):
                    cols = st.columns(n_cols)
                    row = trials[row_start:row_start + n_cols]
                    for j, trial in enumerate(row):
                        with cols[j]:
                            is_sel = selected_num == trial['trial_num']
                            st.markdown(_trial_card_html(trial, selected=is_sel), unsafe_allow_html=True)
                            if st.button(
                                'Hide distributions' if is_sel else 'View distributions',
                                key=f"trial-{trial['trial_num']}",
                                use_container_width=True,
                                type='primary' if is_sel else 'secondary',
                            ):
                                if is_sel:
                                    st.session_state.selected_trial = None
                                else:
                                    st.session_state.selected_trial = trial['trial_num']
                                    if paretic:
                                        st.session_state.lit_key = paretic['key']
                                st.rerun()

                if selected_num is not None:
                    st.caption(f'Showing Trial {selected_num} detail below — click again to dismiss.')

    if result.get('skipped'):
        return

    trials = result.get('trials') or []
    selected_num = st.session_state.get('selected_trial')
    numbers = [t['trial_num'] for t in trials]
    if selected_num not in numbers:
        return

    trial = next(t for t in trials if t['trial_num'] == selected_num)
    g1p, g2p, opened = trial['gate1_passed'], trial['gate2_passed'], trial['opened']
    badge = 'OPEN' if opened else 'LOCK'
    badge_color = GREEN if opened else RED

    st.markdown('---')
    st.markdown(
        f"<div class='card'><b>Trial {trial['trial_num']} — Gate detail</b> "
        f"<span style='color:{badge_color};font-weight:700;margin-left:8px'>{badge}</span>"
        f"<br><span style='color:{MUTED}'>Gate 1: {'PASS' if g1p else 'FAIL'}  ·  "
        f"Gate 2: {'PASS' if g2p else 'FAIL'}  ·  k₁ = {k1:.2f}, k₂ = {k2:.2f}</span></div>",
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(
            make_gate_figure(
                result['gate1_break_powers'], result['gate1_mean'], result['gate1_std'],
                k1, trial['gate1_value'], mode='below', passed=g1p,
                gate_name='Gate 1 — Contralateral ERD',
                dist_caption='Cyan dots: break-period power from each calibration paretic-hand trial',
                x_title='Mean PSD  (µV²/Hz) on paretic motor-strip channels',
                trial_label='This trial · imagery power',
                x_scale=1e12,
                x_tickformat='.1f',
            ),
            use_container_width=True,
            config={'displayModeBar': False},
            key=f"g1-{participant_id}-{trial['trial_num']}-{k1:.2f}-{k2:.2f}",
        )
        st.metric('This trial · imagery power', f"{trial['gate1_value'] * 1e12:.1f} µV²/Hz")
        if not g1p:
            st.warning(result.get('gate1_fail_reason') or GATE1_FAIL_REASON)
        else:
            st.success('Gate 1 passed: imagery power is below the break-period bound (contralateral drop).')
    with p2:
        st.plotly_chart(
            make_gate_figure(
                result['gate2_healthy_lis'], result['gate2_mean'], result['gate2_std'],
                k2, trial['gate2_value'], mode='above', passed=g2p,
                gate_name='Gate 2 — Laterality index',
                dist_caption='Cyan dots: laterality index from each calibration healthy-hand imagery trial',
                x_title='Laterality index  (ipsi − contra) / (ipsi + contra)  ·  unitless',
                trial_label='This trial · imagery LI',
                x_scale=1.0,
                x_tickformat='.3f',
                require_positive=True,
            ),
            use_container_width=True,
            config={'displayModeBar': False},
            key=f"g2-{participant_id}-{trial['trial_num']}-{k1:.2f}-{k2:.2f}",
        )
        st.metric('This trial · laterality index', f"{trial['gate2_value']:.4f}")
        st.latex(r'\mathrm{LI} = \frac{P_{\mathrm{ipsi}} - P_{\mathrm{contra}}}{P_{\mathrm{ipsi}} + P_{\mathrm{contra}}}')
        if not g2p:
            st.warning(result.get('gate2_fail_reason') or GATE2_FAIL_REASON)
        else:
            st.success('Gate 2 passed: LI is above 0 and at/above the healthy-imagery floor.')


if __name__ == '__main__':
    main()
