"""
Main Streamlit Application

This is the main entry point for the Synthetic Data Generation application.
"""
import streamlit as st
import asyncio
import os
import tempfile
import zipfile
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any

# Set up the path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.ddl_parser import parse_ddl_content
from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
from src.database.postgres_handler import DatabaseHandler, DataExporter
from config.settings import settings


# Page configuration
st.set_page_config(
    page_title="Data Generation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
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


def initialize_components():
    """Initialize database handler and data generator."""
    try:
        if st.session_state.db_handler is None:
            st.session_state.db_handler = DatabaseHandler()
            
        if st.session_state.data_generator is None:
            st.session_state.data_generator = SyntheticDataGenerator()
            
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return False


def main():
    """Main application function."""
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("<h3 style='margin-bottom: 2rem;'>Assistant</h3>", unsafe_allow_html=True)
        
        # Navigation with selectbox to maintain state
        selected_tab = st.selectbox(
            "Choose Section:",
            ["📃 Data Generation", "💬 Talk to your data"],
            key="main_navigation"
        )
    
    # Main content area
    if selected_tab == "📃 Data Generation":
        render_data_generation_tab()
    elif selected_tab == "💬 Talk to your data":
        render_talk_to_data_tab()


def render_data_generation_tab():
    """Render the data generation tab."""
    
    # Main prompt area - full width
    st.markdown("**Prompt**")
    user_instructions = st.text_input(
        "",
        placeholder="Enter your prompt here...",
        key="main_prompt",
        label_visibility="collapsed"
    )
    
    # Upload DDL Schema button with supported formats
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

    # Handle file upload
    if uploaded_file is not None:
        # Check if this is a new file (reset data if file changed)
        current_file_name = uploaded_file.name
        if 'current_file_name' not in st.session_state:
            st.session_state.current_file_name = None
        
        if st.session_state.current_file_name != current_file_name:
            # New file selected - reset all generated data
            st.session_state.generated_data = {}
            st.session_state.raw_ai_responses = {}
            st.session_state.current_file_name = current_file_name
            # Force rerun to hide the table section
            st.rerun()
            
        try:
            content = uploaded_file.read().decode('utf-8')
            schema = parse_ddl_content(content)
            st.session_state.schema = schema
            st.success(f"✅ Successfully parsed {len(schema.tables)} tables")
        except Exception as e:
            st.error(f"❌ Failed to parse DDL: {e}")
    else:
        # No file uploaded - reset everything
        if 'current_file_name' in st.session_state and st.session_state.current_file_name is not None:
            st.session_state.generated_data = {}
            st.session_state.raw_ai_responses = {}
            st.session_state.current_file_name = None
            st.session_state.schema = None
    
    # Advanced Parameters section
    st.markdown("**Advanced Parameters**")
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=settings.default_temperature,
            step=0.1
        )
    
    with param_col2:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=10,
            max_value=8192,
            value=1000,  # Default value as requested
            step=10
        )
    
    # Generated Table section
    if st.session_state.generated_data:
        st.markdown("**Data Preview**")
        
        # Table selector
        table_names = list(st.session_state.generated_data.keys())
        selected_table = st.selectbox(
            "Select table",
            table_names,
            key="table_selector"
        )
        
        if selected_table:
            # Show raw AI response if available
            # if st.session_state.raw_ai_responses and selected_table in st.session_state.raw_ai_responses:
            #     with st.expander(f"🤖 Raw AI Response for {selected_table}", expanded=False):
            #         raw_response = st.session_state.raw_ai_responses[selected_table]
            #         st.code(raw_response, language="text")
                    
            #         # Show response stats
            #         col1, col2, col3 = st.columns(3)
            #         with col1:
            #             st.metric("Response Length", f"{len(raw_response)} chars")
            #         with col2:
            #             lines = raw_response.count('\n') + 1
            #             st.metric("Lines", lines)
            #         with col3:
            #             csv_start = raw_response.find('```csv')
            #             csv_end = raw_response.find('```', csv_start + 6)
            #             if csv_start != -1 and csv_end != -1:
            #                 csv_content = raw_response[csv_start+6:csv_end].strip()
            #                 csv_lines = csv_content.count('\n') if csv_content else 0
            #                 st.metric("CSV Data Lines", csv_lines)
            #             else:
            #                 st.metric("CSV Data Lines", "Not found")
            
            df = st.session_state.generated_data[selected_table]
            st.dataframe(df, use_container_width=True, height=300)
            
            # Quick edit section
            edit_col1, edit_col2 = st.columns([4, 1])
            
            with edit_col1:
                quick_edit = st.text_input(
                    "",
                    placeholder="Edit quick edit instructions...",
                    key="quick_edit_input",
                    label_visibility="collapsed"
                )
            
            with edit_col2:
                if st.button("Submit", key="quick_edit_generate", use_container_width=True):
                    if quick_edit.strip():
                        modify_table_data(selected_table, quick_edit)
    
    # Main Generate button (only show if schema is loaded)
    elif st.session_state.schema:
        if st.button("Generate", type="primary", use_container_width=True):
            generate_data(20, temperature, user_instructions, 42, max_tokens)
    
    else:
        st.info("Upload a DDL schema or select a sample to get started.")


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


async def generate_data_with_raw_responses(schema, num_rows, user_instructions, generation_params):
    """Generate data and capture raw AI responses for debugging."""
    try:
        from src.data_generation.synthetic_data_generator import DataGenerationContext
        
        # Create generation context
        context = DataGenerationContext(
            schema=schema,
            generated_tables={},
            user_instructions=user_instructions,
            generation_params=generation_params
        )
        
        generated_data = {}
        raw_responses = {}
        
        # Generate data for each table and capture raw responses
        tables = list(schema.tables.keys())
        
        for table_name in tables:
            try:
                table = schema.tables[table_name]
                
                # Generate data for this table
                table_data, raw_response = await st.session_state.data_generator.generate_table_data_with_raw(
                    table, num_rows, context
                )
                
                if table_data is not None:
                    generated_data[table_name] = table_data
                    context.generated_tables[table_name] = table_data
                    
                if raw_response:
                    raw_responses[table_name] = raw_response
                    
            except Exception as e:
                st.error(f"Error generating {table_name}: {e}")
                raw_responses[table_name] = f"Error: {e}"
                
        return generated_data, raw_responses
        
    except Exception as e:
        st.error(f"Error in data generation: {e}")
        return {}, {}


def generate_data(num_rows: int, temperature: float, user_instructions: str, seed: int, max_tokens: int):
    """Generate synthetic data."""
    if st.session_state.schema is None:
        st.error("Please upload a DDL schema first")
        return
    
    try:
        with st.spinner("🔄 Generating synthetic data..."):
            # Set up generation parameters
            generation_params = {
                'temperature': temperature,
                'seed': seed,
                'max_tokens': max_tokens
            }
            
            # Run async data generation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Generate data and capture raw responses
            generated_data, raw_responses = loop.run_until_complete(
                generate_data_with_raw_responses(
                    st.session_state.schema,
                    num_rows,
                    user_instructions,
                    generation_params
                )
            )
            
            loop.close()
            
            if generated_data:
                st.session_state.generated_data = generated_data
                st.session_state.raw_ai_responses = raw_responses
                st.success(f"✅ Generated data for {len(generated_data)} tables")
                
                # Store in database
                if st.session_state.db_handler:
                    store_in_database()
                
                st.rerun()
            else:
                st.error("❌ Failed to generate data")
                
    except Exception as e:
        st.error(f"❌ Error during data generation: {e}")


def modify_table_data(table_name: str, modification_instructions: str):
    """Modify existing table data."""
    if not modification_instructions.strip():
        st.warning("Please provide modification instructions")
        return
    
    try:
        with st.spinner(f"🔄 Modifying {table_name}..."):
            current_data = st.session_state.generated_data[table_name]
            
            # Check if the modify_table_data method exists
            if not hasattr(st.session_state.data_generator, 'modify_table_data'):
                st.error("❌ Modify table functionality not implemented yet")
                return
            
            # Run async data modification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                modified_data = loop.run_until_complete(
                    st.session_state.data_generator.modify_table_data(
                        table_name,
                        current_data,
                        modification_instructions,
                        st.session_state.schema
                    )
                )
            except Exception as generation_error:
                st.error(f"❌ Error during data modification: {generation_error}")
                return
            finally:
                loop.close()
            
            if modified_data is not None:
                st.session_state.generated_data[table_name] = modified_data
                st.success(f"✅ Modified {table_name}")
                
                # Update database
                if st.session_state.db_handler:
                    store_table_in_database(table_name, modified_data)
                
                st.rerun()
            else:
                st.error(f"❌ Failed to modify {table_name} - method returned None")
                
    except Exception as e:
        st.error(f"❌ Error modifying data: {e}")


def store_in_database():
    """Store generated data in PostgreSQL database."""
    try:
        with st.spinner("💾 Storing data in database..."):
            # Create database if not exists
            st.session_state.db_handler.create_database_if_not_exists()
            
            # Create tables
            success = st.session_state.db_handler.create_schema_tables(st.session_state.schema)
            
            if success:
                # Insert data
                success = st.session_state.db_handler.bulk_insert_data(
                    st.session_state.generated_data, 
                    st.session_state.schema
                )
                
                if success:
                    st.success("✅ Data stored in database")
                else:
                    st.error("❌ Failed to store data in database")
            else:
                st.error("❌ Failed to create database tables")
                
    except Exception as e:
        st.error(f"❌ Database storage error: {e}")


def store_table_in_database(table_name: str, data: pd.DataFrame):
    """Store single table data in database."""
    try:
        # Clear existing data
        st.session_state.db_handler.clear_table(table_name)
        
        # Insert new data
        success = st.session_state.db_handler.insert_dataframe(table_name, data)
        
        if success:
            st.success(f"✅ Updated {table_name} in database")
        else:
            st.error(f"❌ Failed to update {table_name} in database")
            
    except Exception as e:
        st.error(f"❌ Database update error: {e}")


def export_options_section():
    """Render export options."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download CSV", use_container_width=True):
            download_csv()
    
    with col2:
        if st.button("📦 Download ZIP", use_container_width=True):
            download_zip()
    
    with col3:
        if st.button("💾 Save to Database", use_container_width=True):
            store_in_database()


def download_csv():
    """Prepare CSV downloads for individual tables."""
    if not st.session_state.generated_data:
        st.warning("No data to download")
        return
    
    st.subheader("CSV Downloads")
    
    for table_name, df in st.session_state.generated_data.items():
        csv = df.to_csv(index=False)
        
        st.download_button(
            label=f"📄 Download {table_name}.csv",
            data=csv,
            file_name=f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"download_{table_name}"
        )


def download_zip():
    """Create and download a ZIP file with all tables."""
    if not st.session_state.generated_data:
        st.warning("No data to download")
        return
    
    try:
        # Create temporary ZIP file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for table_name, df in st.session_state.generated_data.items():
                    csv_content = df.to_csv(index=False)
                    zipf.writestr(f"{table_name}.csv", csv_content)
            
            # Read the ZIP file for download
            with open(tmp_zip.name, 'rb') as zip_file:
                zip_data = zip_file.read()
            
            st.download_button(
                label="📦 Download All Tables (ZIP)",
                data=zip_data,
                file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            
        # Clean up temp file
        os.unlink(tmp_zip.name)
        
    except Exception as e:
        st.error(f"❌ Failed to create ZIP file: {e}")


def render_talk_to_data_tab():
    """Render the talk to your data tab."""
    st.markdown("<div class='main-header'>💬 Talk to Your Data</div>", unsafe_allow_html=True)
    
    # Coming soon placeholder that matches the design
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; background: #f8f9fa; border-radius: 8px; margin: 2rem 0;'>
        <h3 style='color: #6c757d; margin-bottom: 1rem;'>🚧 Feature Coming Soon</h3>
        <p style='color: #868e96; font-size: 1.1rem;'>
            Conversational AI interface will be available in Phase 2 & 3
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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


if __name__ == "__main__":
    main()