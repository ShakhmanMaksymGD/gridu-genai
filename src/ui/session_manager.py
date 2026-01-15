"""
Session State Manager

Handles initialization and management of Streamlit session state.
"""
import streamlit as st
from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
from src.database.postgres_handler import DatabaseHandler


class SessionManager:
    """Manages application session state."""
    
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables."""
        if 'schema' not in st.session_state:
            st.session_state.schema = None
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = {}
        if 'raw_ai_responses' not in st.session_state:
            st.session_state.raw_ai_responses = {}
        if 'current_file_name' not in st.session_state:
            st.session_state.current_file_name = None
        if 'db_handler' not in st.session_state:
            st.session_state.db_handler = None
        if 'data_generator' not in st.session_state:
            st.session_state.data_generator = None
        if 'chat_interface' not in st.session_state:
            st.session_state.chat_interface = None
    
    @staticmethod
    def initialize_components() -> bool:
        """Initialize database handler and data generator."""
        try:
            if st.session_state.db_handler is None:
                st.session_state.db_handler = DatabaseHandler()
                
            if st.session_state.data_generator is None:
                # Pass the database engine to enable foreign key validation
                db_engine = st.session_state.db_handler.engine if st.session_state.db_handler else None
                st.session_state.data_generator = SyntheticDataGenerator(db_engine=db_engine)
                
            return True
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            return False
    
    @staticmethod
    def reset_data_state():
        """Reset all generated data and related state."""
        st.session_state.generated_data = {}
        st.session_state.raw_ai_responses = {}
    
    @staticmethod
    def reset_file_state():
        """Reset file-related state."""
        st.session_state.current_file_name = None
        st.session_state.schema = None
        SessionManager.reset_data_state()