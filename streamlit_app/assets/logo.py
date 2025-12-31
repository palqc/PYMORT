import base64
from pathlib import Path

import streamlit as st

LOGO_PATH = Path(__file__).parent / "logo.png"


def add_logo_top_right():
    with open(LOGO_PATH, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
            [data-testid="stHeader"] {{
                background: transparent;
            }}

            .app-logo {{
                position: fixed;
                top: 50px;
                right: 20px;
                z-index: 100;
            }}
        </style>

        <div class="app-logo">
            <img src="data:image/png;base64,{encoded}" width="120">
        </div>
        """,
        unsafe_allow_html=True,
    )
