"""
Application Pages

Contains the main page components and navigation.
"""
import streamlit as st
from src.ui.styles import render_coming_soon_section
from src.ui.file_upload import FileUploadManager
from src.ui.data_generation import DataGenerationManager


def render_data_generation_page():
    """Render the main data generation page."""
    
    # Main prompt area
    st.markdown("**Prompt**")
    user_instructions = st.text_input(
        "",
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
        if st.button("Generate", type="primary", use_container_width=True):
            DataGenerationManager.generate_data(20, temperature, user_instructions, 42, max_tokens)
    
    else:
        st.info("Upload a DDL schema or select a sample to get started.")


def render_talk_to_data_page():
    """Render the talk to your data page."""
    st.markdown("<div class='main-header'>💬 Talk to Your Data</div>", unsafe_allow_html=True)
    
    # Coming soon section
    render_coming_soon_section(
        "Feature Coming Soon",
        "Conversational AI interface will be available in Phase 2 & 3"
    )
    
    # Show available data if any exists
    if st.session_state.generated_data:
        st.markdown("**Available Tables:**")
        
        for table_name, df in st.session_state.generated_data.items():
            with st.expander(f"📊 {table_name} ({len(df)} rows)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Data Types", len(df.dtypes.unique()))
                
                # Show sample
                st.markdown("**Sample Data:**")
                st.dataframe(df.head(3), use_container_width=True)


def render_sidebar_navigation():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("<h3 style='margin-bottom: 2rem;'>Assistant</h3>", unsafe_allow_html=True)
        
        # Navigation with selectbox to maintain state
        selected_tab = st.selectbox(
            "Choose Section:",
            ["📃 Data Generation", "💬 Talk to your data"],
            key="main_navigation"
        )
    
    return selected_tab