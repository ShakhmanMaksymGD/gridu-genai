"""
File Upload Component

Handles DDL file upload and processing.
"""
import streamlit as st
from src.utils.ddl_parser import parse_ddl_content
from src.utils.langfuse_observer import log_user_action
from src.ui.session_manager import SessionManager


class FileUploadManager:
    """Manages DDL file upload and processing."""
    
    @staticmethod
    def render_file_upload():
        """Render file upload UI and handle file processing."""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload DDL Schema",
                type=['sql', 'ddl'],
                key="ddl_file_uploader",
                label_visibility="hidden"
            )
        with col2:
            st.caption("Supported formats: SQL, JSON")
        
        FileUploadManager._process_uploaded_file(uploaded_file)
    
    @staticmethod
    def _process_uploaded_file(uploaded_file):
        """Process the uploaded file and update session state."""
        if uploaded_file is not None:
            # Check if this is a new file (reset data if file changed)
            current_file_name = uploaded_file.name
            
            if st.session_state.current_file_name != current_file_name:
                # New file selected - reset all generated data
                SessionManager.reset_data_state()
                st.session_state.current_file_name = current_file_name
                # Force rerun to hide the table section
                st.rerun()
                
            try:
                content = uploaded_file.read().decode('utf-8')
                schema = parse_ddl_content(content)
                st.session_state.schema = schema
                
                # Log user action
                log_user_action("schema_uploaded", {
                    "file_name": uploaded_file.name,
                    "file_size": uploaded_file.size,
                    "tables_count": len(schema.tables),
                    "table_names": list(schema.tables.keys())
                })
                
                st.success(f"✅ Successfully parsed {len(schema.tables)} tables")
            except Exception as e:
                st.error(f"❌ Failed to parse DDL: {e}")
        else:
            # No file uploaded - reset everything
            if (hasattr(st.session_state, 'current_file_name') and 
                st.session_state.current_file_name is not None):
                SessionManager.reset_file_state()
    
    @staticmethod
    def load_sample_schema(name: str, path: str):
        """Load a sample schema file."""
        try:
            with open(path, 'r') as f:
                content = f.read()
            schema = parse_ddl_content(content)
            st.session_state.schema = schema
            st.success(f"✅ Loaded {name} schema with {len(schema.tables)} tables")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to load sample: {e}")