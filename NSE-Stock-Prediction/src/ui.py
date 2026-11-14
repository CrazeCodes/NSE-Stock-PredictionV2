"""Shared UI helpers. Theme colours come from Streamlit's theme (.streamlit/config.toml),
so the app follows the viewer's light/dark choice (Settings menu)."""
import streamlit as st

from .config import APP_NAME, APP_TAGLINE, DISCLAIMER

PALETTE = {"actual": "#2E86DE", "rf": "#F39C12", "lr": "#2ECC71",
           "forecast": "#E74C3C", "neutral": "#95A5A6"}
MODEL_COLORS = {"Random_Forest": PALETTE["rf"], "Linear_Regression": PALETTE["lr"]}

_CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px;}
h1 {font-size: 2rem !important; letter-spacing: -0.01em;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
"""


def setup_page(title: str, icon: str = "📈") -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} | {APP_NAME}", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"### 📈 {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.divider()
        st.caption("Experimental portfolio project. Not financial advice.")


def header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def disclaimer() -> None:
    st.warning(DISCLAIMER, icon="⚠️")


def show(fig, height: int = 460) -> None:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def rupee(value: float) -> str:
    return f"₹{value:,.2f}"
