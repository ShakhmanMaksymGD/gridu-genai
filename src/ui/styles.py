"""
UI Styles Module

Contains CSS styles and styling utilities for the Streamlit app.
"""
import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .prompt-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .params-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
    }
    .data-preview {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
    }
    /* Black button styling */
    .stButton > button {
        background-color: #000000 !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #333333 !important;
        color: white !important;
    }
    /* Style file uploader to look like a button */
    div[data-testid="stFileUploader"] > label > div {
        background-color: #000000 !important;
        color: white !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        cursor: pointer !important;
    }
    

        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        text-align: center !important;
        cursor: pointer !important;
    }
    div[data-testid="stFileUploader"] > label > div:hover {
        background-color: #333333 !important;
    }
    div[data-testid="stFileUploader"] > label > div[data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background-color: #000000 !important;
    }
    div[data-testid="stHorizontalBlock"] {
      align-items: center;
    }
    div[data-testid="stCode"] > pre {
        background: #000;
        color: #fff;
    }
    button[data-testid="stCodeCopyButton"] {
        color: #fff !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_coming_soon_section(title: str, description: str):
    """Render a coming soon section with consistent styling."""
    st.markdown(f"""
    <div style='text-align: center; padding: 4rem 2rem; background: #f8f9fa; border-radius: 8px; margin: 2rem 0;'>
        <h3 style='color: #6c757d; margin-bottom: 1rem;'>🚧 {title}</h3>
        <p style='color: #868e96; font-size: 1.1rem;'>
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)