"""
Application Pages

Contains the main page components and navigation.
"""
import streamlit as st
from src.ui.styles import render_coming_soon_section
from src.ui.file_upload import FileUploadManager
from src.ui.data_generation import DataGenerationManager
from src.ui.chat_ui import ChatUIManager


def render_data_generation_page():
    """Render the main data generation page."""
    
    # Main prompt area
    st.markdown("**Prompt**")
    user_instructions = st.text_input(
        "User instructions",
        placeholder="Enter your prompt here...",
        key="main_prompt",
        label_visibility="collapsed"
    )
    
    # File upload section
    FileUploadManager.render_file_upload()
    
    # Advanced parameters
    temperature, max_tokens = DataGenerationManager.render_parameters_section()
    
    # Data preview section (only show if data exists)
    DataGenerationManager.render_data_preview_section()
    
    # Main Generate button (only show if schema is loaded and no data exists)
    if st.session_state.schema and not st.session_state.generated_data:
        if st.button("Generate", type="primary", width="stretch"):
            DataGenerationManager.generate_data(20, temperature, user_instructions, 42, max_tokens)


def render_talk_to_data_page():
    """Render the talk to your data page with full chat functionality."""
    ChatUIManager.render_chat_interface()


def render_sidebar_navigation():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("<h2 style='margin-bottom: 2rem;'>Data Assistant</h2 >", unsafe_allow_html=True)
        
        # Navigation with selectbox to maintain state
        selected_tab = st.selectbox(
            "Choose Section:",
            ["📃 Data Generation", "💬 Talk to your data"],
            key="main_navigation"
        )
    
    return selected_tab