import streamlit as st
from pathlib import Path

LOGO_PATH = Path(__file__).parents[1] / "assets" / "logo.png"


def logo_sidebar() -> None:
    """Render the shared sidebar with logo."""
    with st.sidebar:
        st.image(str(LOGO_PATH), width=80)
        st.markdown("---")