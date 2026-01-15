"""
Refactored Main Streamlit Application

This is the main entry point for the Synthetic Data Generation application.
Refactored for better maintainability and separation of concerns.
"""
import streamlit as st
import sys
import os

# Set up the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.session_manager import SessionManager
from src.ui.styles import apply_custom_css
from src.ui.pages import render_data_generation_page, render_talk_to_data_page, render_sidebar_navigation


def configure_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Data Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def main():
    """Main application function."""
    # Configure page
    configure_page()
    
    # Apply custom styling
    apply_custom_css()
    
    # Initialize session state
    SessionManager.initialize_session_state()
    
    # Initialize components
    if not SessionManager.initialize_components():
        st.stop()
    
    # Create session manager instance for UI components
    if 'session_manager' not in st.session_state:
        class SessionManagerInstance:
            @property
            def chat_interface(self):
                return st.session_state.get('chat_interface', None)
        st.session_state.session_manager = SessionManagerInstance()
    
    # Render sidebar navigation
    selected_tab = render_sidebar_navigation()
    
    # Render main content based on selected tab
    if selected_tab == "📃 Data Generation":
        render_data_generation_page()
    elif selected_tab == "💬 Talk to your data":
        render_talk_to_data_page()


if __name__ == "__main__":
    main()