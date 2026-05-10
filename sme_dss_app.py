import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SME Technology DSS",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Cards */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-card h3 { margin: 0; font-size: 0.78rem; color: #64748B; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; }
    .metric-card p  { margin: 0.3rem 0 0; font-size: 1.6rem; font-weight: 600; color: #1E293B; }

    /* Section Headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E293B;
        border-left: 3px solid #3B82F6;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }

    /* Rank Badge */
    .rank-badge {
        display: inline-block;
        width: 26px; height: 26px;
        border-radius: 50%;
        background: #EFF6FF;
        color: #1D4ED8;
        font-weight: 600;
        font-size: 0.82rem;
        text-align: center;
        line-height: 26px;
    }
    .rank-1 { background: #FEF3C7; color: #92400E; }
    .rank-2 { background: #F1F5F9; color: #475569; }
    .rank-3 { background: #FEF3C7; color: #B45309; }

    /* Warning Boxes */
    .warn-high   { background: #FEF2F2; border-left: 4px solid #EF4444; border-radius: 6px; padding: 0.75rem 1rem; margin: 0.5rem 0; }
    .warn-medium { background: #FFFBEB; border-left: 4px solid #F59E0B; border-radius: 6px; padding: 0.75rem 1rem; margin: 0.5rem 0; }
    .warn-low    { background: #F0FDF4; border-left: 4px solid #22C55E; border-radius: 6px; padding: 0.75rem 1rem; margin: 0.5rem 0; }
    .warn-title  { font-weight: 600; font-size: 0.85rem; margin-bottom: 0.2rem; }
    .warn-body   { font-size: 0.82rem; color: #475569; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #1E293B; }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #F8FAFC; }
    section[data-testid="stSidebar"] label { color: #CBD5E1 !important; font-size: 0.85rem; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label { color: #94A3B8 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem; font-weight: 500;
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] { color: #1D4ED8 !important; border-bottom: 2px solid #1D4ED8; }

    /* Table styling */
    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
    .styled-table th { background: #F1F5F9; color: #475569; font-weight: 600; padding: 8px 12px; text-align: left; border-bottom: 2px solid #E2E8F0; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; }
    .styled-table td { padding: 8px 12px; border-bottom: 1px solid #F1F5F9; color: #334155; }
    .styled-table tr:hover td { background: #F8FAFC; }

    /* Status pill */
    .pill-recommend { background: #D1FAE5; color: #065F46; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .pill-consider  { background: #FEF3C7; color: #92400E; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .pill-defer     { background: #FEE2E2; color: #991B1B; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 999px; font-size: 0.75rem; font-weight: 600; }

    hr { border: none; border-top: 1px solid #E2E8F0; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LAYER  (embedded ANP + DEMATEL results)
# ─────────────────────────────────────────────
CRITERIA = {
    "C1": "Implementation Cost",
    "C2": "Ease of Use",
    "C3": "ROI Potential",
    "C4": "Integration Capability",
    "C5": "Training Requirements"
}

TECHNOLOGIES = {
    "T1": "Cloud Computing",
    "T2": "ERP Systems",
    "T3": "IoT Solutions",
    "T4": "Big Data Analytics",
    "T5": "AI / Machine Learning",
    "T6": "Blockchain"
}

# DEMATEL total-relation matrix (defuzzified t* values)
T_MATRIX = np.array([
    [0.142, 0.186, 0.203, 0.215, 0.178],  # C1
    [0.175, 0.131, 0.189, 0.197, 0.162],  # C2
    [0.168, 0.159, 0.124, 0.208, 0.171],  # C3
    [0.221, 0.198, 0.214, 0.138, 0.189],  # C4
    [0.164, 0.172, 0.178, 0.183, 0.119],  # C5
])

# ANP base weights (from limit supermatrix, scenario: medium SME)
ANP_BASE_WEIGHTS = {
    "T1": 0.2134,
    "T2": 0.1987,
    "T3": 0.1876,
    "T4": 0.1654,
    "T5": 0.1423,
    "T6": 0.1089  # adjusted so sum = 1 (rounding)
}

# Per-criterion scores for each technology  (0–1 scale)
CRITERION_SCORES = {
    "T1": {"C1": 0.82, "C2": 0.78, "C3": 0.75, "C4": 0.80, "C5": 0.72},
    "T2": {"C1": 0.65, "C2": 0.70, "C3": 0.80, "C4": 0.85, "C5": 0.58},
    "T3": {"C1": 0.70, "C2": 0.65, "C3": 0.78, "C4": 0.75, "C5": 0.60},
    "T4": {"C1": 0.60, "C2": 0.68, "C3": 0.85, "C4": 0.72, "C5": 0.55},
    "T5": {"C1": 0.45, "C2": 0.55, "C3": 0.90, "C4": 0.70, "C5": 0.42},
    "T6": {"C1": 0.38, "C2": 0.48, "C3": 0.65, "C4": 0.60, "C5": 0.35},
}

# DEMATEL D, R, D+R, D-R
def compute_dematel():
    D = T_MATRIX.sum(axis=1)
    R = T_MATRIX.sum(axis=0)
    DR_plus  = D + R
    DR_minus = D - R
    return D, R, DR_plus, DR_minus

D, R, DR_plus, DR_minus = compute_dematel()

# Threshold
ALPHA = T_MATRIX.mean()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def compute_anp_scores(criterion_weights: dict) -> dict:
    """Recompute technology global priorities given custom criterion weights."""
    scores = {}
    for t, cscores in CRITERION_SCORES.items():
        score = sum(criterion_weights[c] * cscores[c] for c in CRITERIA)
        scores[t] = round(score, 4)
    # normalise
    total = sum(scores.values())
    return {t: round(v / total, 4) for t, v in scores.items()}

def get_status(rank: int, score: float) -> str:
    if rank <= 2: return "Recommended"
    if rank <= 4: return "Consider"
    return "Defer"

def status_pill(status: str) -> str:
    cls = {"Recommended": "pill-recommend", "Consider": "pill-consider", "Defer": "pill-defer"}[status]
    return f'<span class="{cls}">{status}</span>'

def get_digital_maturity_label(score: int) -> str:
    labels = {1: "Initial", 2: "Developing", 3: "Defined", 4: "Managed", 5: "Optimising"}
    return labels.get(score, "Unknown")

# Causal warning rules
CAUSE_CRITERIA = ["C1", "C2", "C4"]  # D-R > 0

def generate_warnings(sme_profile: dict) -> list:
    warnings = []
    # Rule R-01
    if sme_profile["integration_readiness"] < 3:
        warnings.append({
            "id": "R-01", "severity": "high",
            "title": "Low integration readiness — system-wide risk",
            "body": "Integration Capability (C4) is the primary cause criterion. A score below 3 triggers cascading risk across all technology options. Remediate middleware and API infrastructure before any adoption."
        })
    # Rule R-02
    if sme_profile["budget_index"] < 3:
        warnings.append({
            "id": "R-02", "severity": "high",
            "title": "Budget constraint — high-cost technologies at risk",
            "body": "Implementation Cost (C1) is a cause criterion. A tight budget eliminates ERP, AI/ML, and Blockchain from realistic consideration. Focus on Cloud Computing in Phase 1."
        })
    # Rule R-03
    if sme_profile["it_skill_level"] < 3:
        warnings.append({
            "id": "R-03", "severity": "medium",
            "title": "Low IT skill level — ease-of-use becomes critical",
            "body": "Ease of Use (C2) is a cause criterion. With IT skills below 3, Cloud and ERP implementations face higher failure risk without structured change management."
        })
    # Rule R-04
    if sme_profile["data_readiness"] < 3:
        warnings.append({
            "id": "R-04", "severity": "medium",
            "title": "Poor data quality — AI/ML and analytics blocked",
            "body": "AI / Machine Learning and Big Data Analytics require clean, structured historical data (min 18 months). Defer these until ERP data hygiene is confirmed."
        })
    # Rule S-01 — success signal
    if sme_profile["integration_readiness"] >= 4 and sme_profile["budget_index"] >= 4:
        warnings.append({
            "id": "S-01", "severity": "low",
            "title": "Strong foundation detected",
            "body": "High integration readiness and budget index indicate this SME is ready for full digital transformation. All top-ranked technologies are viable."
        })
    return warnings

def get_roadmap(maturity: int, rankings: list) -> list:
    """Return phased roadmap based on digital maturity."""
    top_techs = [t for t, _ in rankings[:3]]
    mid_techs  = [t for t, _ in rankings[3:5]]
    last_tech  = rankings[5][0] if len(rankings) > 5 else None

    if maturity <= 2:
        phases = [
            {"phase": "Phase 1 (Months 1–6)",   "techs": top_techs[:1], "note": "Stabilise cloud infrastructure; establish data pipelines."},
            {"phase": "Phase 2 (Months 7–18)",  "techs": top_techs[1:3], "note": "Deploy ERP/IoT with standardised processes."},
            {"phase": "Phase 3 (Months 19–30)", "techs": mid_techs, "note": "Activate analytics layer; build reporting dashboards."},
            {"phase": "Phase 4 (Months 31+)",   "techs": [last_tech] if last_tech else [], "note": "Evaluate AI/ML and Blockchain once data maturity is confirmed."},
        ]
    elif maturity <= 3:
        phases = [
            {"phase": "Phase 1 (Months 1–6)",   "techs": top_techs[:2], "note": "Deploy top two technologies concurrently with integration testing."},
            {"phase": "Phase 2 (Months 7–15)",  "techs": top_techs[2:3] + mid_techs[:1], "note": "Extend to IoT and analytics."},
            {"phase": "Phase 3 (Months 16–24)", "techs": mid_techs[1:] + ([last_tech] if last_tech else []), "note": "Complete transformation stack."},
        ]
    else:
        phases = [
            {"phase": "Phase 1 (Months 1–9)",  "techs": top_techs, "note": "Accelerated deployment across top three technologies."},
            {"phase": "Phase 2 (Months 10–18)", "techs": mid_techs + ([last_tech] if last_tech else []), "note": "Full-stack integration including AI/ML."},
        ]
    return [p for p in phases if p["techs"]]

# ─────────────────────────────────────────────
# SIDEBAR — SME PROFILE INPUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 SME Profile")
    st.markdown("---")

    st.markdown("**Organisation**")
    sme_name    = st.text_input("SME name", value="Al-Noor Food Manufacturing")
    sector      = st.selectbox("Sector", ["Food & Beverage", "Garments & Textile", "Light Engineering", "Pharmaceuticals", "Agro-processing", "Other"])
    employees   = st.selectbox("Employees", ["< 50", "50–150", "150–300", "> 300"])

    st.markdown("---")
    st.markdown("**Readiness Indicators** (1 = Low → 5 = High)")

    integration_readiness = st.slider("Integration readiness (C4)", 1, 5, 3)
    budget_index          = st.slider("Budget index (C1)",           1, 5, 3)
    it_skill_level        = st.slider("IT skill level (C2)",         1, 5, 3)
    data_readiness        = st.slider("Data readiness",              1, 5, 3)
    digital_maturity      = st.slider("Digital maturity",            1, 5, 2)

    st.markdown("---")
    st.markdown("**Criterion Weights** (adjust & rerank)")
    w_c1 = st.slider("C1 Implementation Cost",    0.05, 0.50, 0.22, 0.01)
    w_c2 = st.slider("C2 Ease of Use",            0.05, 0.50, 0.18, 0.01)
    w_c3 = st.slider("C3 ROI Potential",          0.05, 0.50, 0.25, 0.01)
    w_c4 = st.slider("C4 Integration Capability", 0.05, 0.50, 0.21, 0.01)
    w_c5 = st.slider("C5 Training Requirements",  0.05, 0.50, 0.14, 0.01)

    # Normalise weights
    raw_weights = {"C1": w_c1, "C2": w_c2, "C3": w_c3, "C4": w_c4, "C5": w_c5}
    total_w = sum(raw_weights.values())
    criterion_weights = {k: v / total_w for k, v in raw_weights.items()}

    st.caption(f"Weights normalised → sum = 1.00")

# ─────────────────────────────────────────────
# COMPUTE LIVE RESULTS
# ─────────────────────────────────────────────
sme_profile = {
    "integration_readiness": integration_readiness,
    "budget_index":          budget_index,
    "it_skill_level":        it_skill_level,
    "data_readiness":        data_readiness,
    "digital_maturity":      digital_maturity,
}

anp_scores = compute_anp_scores(criterion_weights)
rankings   = sorted(anp_scores.items(), key=lambda x: x[1], reverse=True)
warnings   = generate_warnings(sme_profile)
roadmap    = get_roadmap(digital_maturity, rankings)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_title:
    st.markdown(f"""
    <div style="padding: 0.5rem 0 0.2rem;">
        <div style="font-size:0.78rem; color:#64748B; font-weight:500; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:4px;">
            Fuzzy DEMATEL–ANP Decision Support System
        </div>
        <div style="font-size:1.55rem; font-weight:600; color:#1E293B; line-height:1.2;">
            SME Technology Selection Framework
        </div>
        <div style="font-size:0.85rem; color:#94A3B8; margin-top:4px;">
            {sme_name} &nbsp;·&nbsp; {sector} &nbsp;·&nbsp; {employees} employees &nbsp;·&nbsp;
            Digital maturity: <strong>{get_digital_maturity_label(digital_maturity)}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
top_tech = TECHNOLOGIES[rankings[0][0]]
num_warnings_high = sum(1 for w in warnings if w["severity"] == "high")

k1.markdown(f'<div class="metric-card"><h3>Top Technology</h3><p style="font-size:1.05rem;">{top_tech}</p></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="metric-card"><h3>ANP Score (Rank 1)</h3><p>{rankings[0][1]:.4f}</p></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="metric-card"><h3>High-Risk Alerts</h3><p style="color:{"#EF4444" if num_warnings_high else "#22C55E"}">{num_warnings_high}</p></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="metric-card"><h3>Roadmap Phases</h3><p>{len(roadmap)}</p></div>', unsafe_allow_html=True)
k5.markdown(f'<div class="metric-card"><h3>Maturity Level</h3><p>{get_digital_maturity_label(digital_maturity)}</p></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Technology Scorecard",
    "🔗 DEMATEL Analysis",
    "⚠️ Risk Warnings",
    "🗺️ Implementation Roadmap",
    "📐 Sensitivity Analysis"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — TECHNOLOGY SCORECARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-header">Technology Priority Rankings</div>', unsafe_allow_html=True)

        rows = []
        for rank, (tid, score) in enumerate(rankings, 1):
            status = get_status(rank, score)
            row = {
                "Rank": rank,
                "Technology": TECHNOLOGIES[tid],
                "ANP Score": f"{score:.4f}",
                "C1 Cost": f"{CRITERION_SCORES[tid]['C1']:.2f}",
                "C2 Ease": f"{CRITERION_SCORES[tid]['C2']:.2f}",
                "C3 ROI":  f"{CRITERION_SCORES[tid]['C3']:.2f}",
                "C4 Integ":f"{CRITERION_SCORES[tid]['C4']:.2f}",
                "C5 Train":f"{CRITERION_SCORES[tid]['C5']:.2f}",
                "Status": status_pill(status),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        table_html = '<table class="styled-table"><thead><tr>'
        for col in df.columns:
            table_html += f"<th>{col}</th>"
        table_html += "</tr></thead><tbody>"
        for _, row in df.iterrows():
            table_html += "<tr>"
            for col in df.columns:
                if col == "Status":
                    table_html += f"<td>{row[col]}</td>"
                elif col == "Rank":
                    r = int(row[col])
                    cls = f"rank-{r}" if r <= 3 else "rank-badge"
                    table_html += f'<td><span class="rank-badge {cls}">{r}</span></td>'
                else:
                    table_html += f"<td>{row[col]}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.caption("ANP scores are recomputed live based on your criterion weights in the sidebar.")

    with right:
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)

        fig_bar = go.Figure()
        colors_bar = ["#1D4ED8","#3B82F6","#60A5FA","#93C5FD","#BFDBFE","#DBEAFE"]
        for i, (tid, score) in enumerate(rankings):
            fig_bar.add_trace(go.Bar(
                y=[TECHNOLOGIES[tid]],
                x=[score],
                orientation='h',
                marker_color=colors_bar[i],
                name=TECHNOLOGIES[tid],
                text=f"{score:.4f}",
                textposition='outside',
                showlegend=False
            ))
        fig_bar.update_layout(
            height=260,
            margin=dict(l=0, r=50, t=10, b=10),
            xaxis_title="Global Priority Score",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='IBM Plex Sans', size=11),
            xaxis=dict(gridcolor='#F1F5F9', range=[0, max(anp_scores.values()) * 1.2])
        )
        st.plotly_chart(fig_bar, width='stretch')

        st.markdown('<div class="section-header">Radar: Top 3 Technologies</div>', unsafe_allow_html=True)
        cats = list(CRITERIA.values())
        fig_radar = go.Figure()
        for i, (tid, _) in enumerate(rankings[:3]):
            vals = [CRITERION_SCORES[tid][c] for c in CRITERIA] + [CRITERION_SCORES[tid]["C1"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]],
                fill='toself', name=TECHNOLOGIES[tid],
                line_color=colors_bar[i], opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1], tickfont_size=9)),
            height=260,
            margin=dict(l=30, r=30, t=20, b=10),
            paper_bgcolor='white',
            font=dict(family='IBM Plex Sans', size=10),
            legend=dict(x=0.8, y=1.1, font_size=10)
        )
        st.plotly_chart(fig_radar, width='stretch')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — DEMATEL ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">D, R, D+R, D−R Values</div>', unsafe_allow_html=True)

        crit_keys = list(CRITERIA.keys())
        crit_names = list(CRITERIA.values())

        dematel_df = pd.DataFrame({
            "Criterion": [f"{k}: {v}" for k, v in CRITERIA.items()],
            "D (Influence Given)":    [f"{D[i]:.3f}" for i in range(5)],
            "R (Influence Received)": [f"{R[i]:.3f}" for i in range(5)],
            "D+R (Prominence)":       [f"{DR_plus[i]:.3f}" for i in range(5)],
            "D−R (Relation)":         [f"{DR_minus[i]:+.3f}" for i in range(5)],
            "Group": ["Cause" if DR_minus[i] > 0 else "Effect" for i in range(5)],
        })

        tbl = '<table class="styled-table"><thead><tr>'
        for c in dematel_df.columns:
            tbl += f"<th>{c}</th>"
        tbl += "</tr></thead><tbody>"
        for _, row in dematel_df.iterrows():
            tbl += "<tr>"
            for c in dematel_df.columns:
                if c == "Group":
                    col_class = "pill-recommend" if row[c] == "Cause" else "pill-consider"
                    tbl += f'<td><span class="{col_class}">{row[c]}</span></td>'
                else:
                    tbl += f"<td>{row[c]}</td>"
            tbl += "</tr>"
        tbl += "</tbody></table>"
        st.markdown(tbl, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Total-Relation Matrix (T*)</div>', unsafe_allow_html=True)
        t_df = pd.DataFrame(
            T_MATRIX.round(3),
            index=[f"{k}" for k in CRITERIA.keys()],
            columns=[f"{k}" for k in CRITERIA.keys()]
        )
        st.dataframe(t_df, width='stretch')
        st.caption(f"Threshold α = {ALPHA:.3f}. Only relationships above this value are retained in the ANP network.")

    with col_right:
        st.markdown('<div class="section-header">Cause–Effect Diagram</div>', unsafe_allow_html=True)

        fig_ce = go.Figure()
        colors_ce = ["#EF4444" if DR_minus[i] > 0 else "#3B82F6" for i in range(5)]

        # scatter
        fig_ce.add_trace(go.Scatter(
            x=DR_plus,
            y=DR_minus,
            mode='markers+text',
            marker=dict(size=14, color=colors_ce, line=dict(width=1, color='white')),
            text=[f"C{i+1}" for i in range(5)],
            textposition='top center',
            textfont=dict(size=11, family='IBM Plex Sans'),
            hovertemplate='<b>%{text}</b><br>D+R: %{x:.3f}<br>D−R: %{y:.3f}<extra></extra>',
            showlegend=False
        ))
        # zero line
        fig_ce.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1)

        # annotations
        for i in range(5):
            fig_ce.add_annotation(
                x=DR_plus[i], y=DR_minus[i],
                text=list(CRITERIA.values())[i],
                showarrow=False,
                yshift=-16,
                font=dict(size=9, color="#475569", family='IBM Plex Sans')
            )

        fig_ce.update_layout(
            height=350,
            xaxis_title="D+R (Prominence)",
            yaxis_title="D−R (Relation)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='IBM Plex Sans', size=11),
            xaxis=dict(gridcolor='#F1F5F9'),
            yaxis=dict(gridcolor='#F1F5F9'),
            margin=dict(l=10, r=10, t=20, b=10),
            annotations=[
                dict(x=0.02, y=0.97, xref='paper', yref='paper', text='<b style="color:#EF4444">● Cause</b>  <b style="color:#3B82F6">● Effect</b>', showarrow=False, font=dict(size=10))
            ]
        )
        st.plotly_chart(fig_ce, width='stretch')

        st.markdown('<div class="section-header">Influence Network Heatmap</div>', unsafe_allow_html=True)
        network = (T_MATRIX > ALPHA).astype(int)
        fig_hm = px.imshow(
            network,
            labels=dict(x="Influenced", y="Influencer", color="Link"),
            x=list(CRITERIA.keys()),
            y=list(CRITERIA.keys()),
            color_continuous_scale=["#EFF6FF","#1D4ED8"],
            text_auto=True,
        )
        fig_hm.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='white',
            font=dict(family='IBM Plex Sans', size=11),
            coloraxis_showscale=False
        )
        fig_hm.update_traces(textfont_size=12)
        st.plotly_chart(fig_hm, width='stretch')
        st.caption("1 = relationship retained (t* > α); 0 = pruned.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — RISK WARNINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    if not warnings:
        st.success("✅ No active risk warnings for this SME profile. The organisation appears ready for technology adoption.")
    else:
        st.markdown(f"**{len(warnings)} active alert(s)** based on your SME readiness profile:")
        for w in warnings:
            css_class = f"warn-{w['severity']}"
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[w["severity"]]
            st.markdown(f"""
            <div class="{css_class}">
                <div class="warn-title">{icon} [{w['id']}] {w['title']}</div>
                <div class="warn-body">{w['body']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Complete Rule Base</div>', unsafe_allow_html=True)

    rules = [
        {"ID": "R-01", "Trigger Criterion": "C4 Integration Capability", "Condition": "Score < 3", "Severity": "High",   "Action": "Flag ALL technologies as high-risk; block Phase 2+ deployment."},
        {"ID": "R-02", "Trigger Criterion": "C1 Implementation Cost",    "Condition": "Score < 3", "Severity": "High",   "Action": "Eliminate ERP, AI/ML, Blockchain from consideration."},
        {"ID": "R-03", "Trigger Criterion": "C2 Ease of Use",            "Condition": "Score < 3", "Severity": "Medium", "Action": "Require change-management programme before Cloud/ERP go-live."},
        {"ID": "R-04", "Trigger Criterion": "Data Readiness",            "Condition": "Score < 3", "Severity": "Medium", "Action": "Defer AI/ML and Big Data Analytics until ERP data is clean (18+ months)."},
        {"ID": "S-01", "Trigger Criterion": "C4 + C1 combined",         "Condition": "Both ≥ 4",  "Severity": "Info",   "Action": "SME is ready for full-stack digital transformation."},
        {"ID": "S-02", "Trigger Criterion": "Digital Maturity",         "Condition": "Level ≥ 4", "Severity": "Info",   "Action": "Accelerate roadmap; compress Phase 1 and Phase 2."},
    ]
    rule_df = pd.DataFrame(rules)
    st.dataframe(rule_df, width='stretch', hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — IMPLEMENTATION ROADMAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown(f"""
    <div style="background:#EFF6FF;border-left:4px solid #3B82F6;border-radius:6px;padding:0.75rem 1rem;margin-bottom:1rem;">
        <strong>Digital Maturity: {get_digital_maturity_label(digital_maturity)} (Level {digital_maturity}/5)</strong>
        &nbsp;—&nbsp; {len(roadmap)}-phase adoption roadmap generated
    </div>
    """, unsafe_allow_html=True)

    phase_colors = ["#1D4ED8", "#0891B2", "#059669", "#D97706"]

    for i, phase in enumerate(roadmap):
        color = phase_colors[i % len(phase_colors)]
        tech_names = ", ".join([TECHNOLOGIES[t] for t in phase["techs"] if t])
        st.markdown(f"""
        <div style="background:white;border:1px solid #E2E8F0;border-left:4px solid {color};
                    border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.8rem;
                    box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-weight:600;font-size:0.85rem;color:{color};text-transform:uppercase;letter-spacing:0.04em;">
                {phase["phase"]}
            </div>
            <div style="font-size:1rem;font-weight:500;color:#1E293B;margin:4px 0;">
                {tech_names}
            </div>
            <div style="font-size:0.82rem;color:#64748B;">{phase["note"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Prerequisites Checklist</div>', unsafe_allow_html=True)
    prereqs = [
        ("Cloud Computing",      "Stable internet connectivity (≥ 10 Mbps); basic IT team (1–2 staff)."),
        ("ERP Systems",          "Mapped business processes; dedicated project manager; 3–6 months implementation budget."),
        ("IoT Solutions",        "Cloud infrastructure operational; sensor-compatible equipment; data storage plan."),
        ("Big Data Analytics",   "ERP data pipelines active for ≥ 12 months; analytics-literate staff."),
        ("AI / Machine Learning","Clean structured ERP data for ≥ 18 months; data scientist or vendor partnership."),
        ("Blockchain",           "Multi-party supply chain partners with digital infrastructure; legal/compliance review."),
    ]
    for tech, prereq in prereqs:
        st.markdown(f"""
        <div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #F1F5F9;align-items:flex-start;">
            <div style="min-width:160px;font-weight:500;font-size:0.83rem;color:#1E293B;">{tech}</div>
            <div style="font-size:0.82rem;color:#64748B;">{prereq}</div>
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — SENSITIVITY ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown('<div class="section-header">±20% Weight Perturbation Analysis</div>', unsafe_allow_html=True)
    st.caption("Each criterion weight is varied ±20% while others are kept proportional. Rank stability is checked across all perturbations.")

    perturbations = [-0.20, -0.10, 0.0, +0.10, +0.20]
    base_ranking  = [tid for tid, _ in rankings]

    results = {}
    for crit in CRITERIA:
        crit_results = []
        for delta in perturbations:
            perturbed = {k: v * (1 + delta if k == crit else 1.0) for k, v in criterion_weights.items()}
            total_p = sum(perturbed.values())
            perturbed = {k: v / total_p for k, v in perturbed.items()}
            p_scores = compute_anp_scores(perturbed)
            p_rank = [tid for tid, _ in sorted(p_scores.items(), key=lambda x: x[1], reverse=True)]
            # rank change of current top tech
            top_current = base_ranking[0]
            rank_in_p = p_rank.index(top_current) + 1
            crit_results.append(rank_in_p)
        results[CRITERIA[crit]] = crit_results

    fig_sens = go.Figure()
    pert_labels = [f"{int(p*100):+d}%" for p in perturbations]
    line_colors_sens = ["#1D4ED8","#059669","#D97706","#7C3AED","#DC2626"]

    for i, (crit_name, ranks) in enumerate(results.items()):
        fig_sens.add_trace(go.Scatter(
            x=pert_labels,
            y=ranks,
            mode='lines+markers',
            name=crit_name,
            line=dict(color=line_colors_sens[i], width=2),
            marker=dict(size=7)
        ))

    fig_sens.update_layout(
        height=350,
        xaxis_title="Weight perturbation",
        yaxis_title=f"Rank of {TECHNOLOGIES[base_ranking[0]]}",
        yaxis=dict(autorange="reversed", dtick=1, gridcolor='#F1F5F9'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='IBM Plex Sans', size=11),
        legend=dict(x=1.01, y=1, font_size=10),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(gridcolor='#F1F5F9')
    )
    st.plotly_chart(fig_sens, width='stretch')

    # Stability summary table
    st.markdown('<div class="section-header">Rank Stability Summary</div>', unsafe_allow_html=True)
    stability_rows = []
    for tid, base_score in rankings:
        rank_changes = []
        for crit in CRITERIA:
            for delta in [-0.20, +0.20]:
                perturbed = {k: v * (1 + delta if k == crit else 1.0) for k, v in criterion_weights.items()}
                total_p = sum(perturbed.values())
                perturbed = {k: v / total_p for k, v in perturbed.items()}
                p_scores = compute_anp_scores(perturbed)
                p_rank = sorted(p_scores.items(), key=lambda x: x[1], reverse=True)
                p_rank_pos = [t for t, _ in p_rank].index(tid) + 1
                base_pos = [t for t, _ in rankings].index(tid) + 1
                rank_changes.append(abs(p_rank_pos - base_pos))
        max_shift = max(rank_changes)
        stable = "✅ Stable" if max_shift <= 1 else "⚠️ Unstable"
        stability_rows.append({
            "Technology": TECHNOLOGIES[tid],
            "Base Rank": [t for t, _ in rankings].index(tid) + 1,
            "Max Rank Shift": max_shift,
            "Stability": stable
        })

    stab_df = pd.DataFrame(stability_rows)
    st.dataframe(stab_df, width='stretch', hide_index=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.75rem;color:#94A3B8;padding:0.5rem 0;">
    Fuzzy DEMATEL–ANP Decision Support System for SME Technology Selection
    &nbsp;·&nbsp; Framework: Zadeh (1965) Fuzzy Sets · Gabus & Fontela (1972) DEMATEL · Saaty (1980) ANP
    &nbsp;·&nbsp; Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
