"""
Data Generation Components

Handles data generation UI and business logic.
"""
import asyncio
import streamlit as st
from config.settings import settings
from src.data_generation.synthetic_data_generator import DataGenerationContext
from src.utils.langfuse_observer import log_user_action, data_observer


class DataGenerationManager:
    """Manages data generation processes."""
    
    @staticmethod
    def render_parameters_section():
        """Render advanced parameters section."""
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
                value=1000,
                step=10
            )
        
        return temperature, max_tokens
    
    @staticmethod
    def render_data_preview_section():
        """Render the data preview section."""
        if not st.session_state.generated_data:
            return
        
        st.markdown("**Data Preview**")
        
        # Table selector
        table_names = list(st.session_state.generated_data.keys())
        selected_table = st.selectbox(
            "Select table",
            table_names,
            key="table_selector"
        )
        
        if selected_table:
            DataGenerationManager._render_table_data(selected_table)
            DataGenerationManager._render_quick_edit_section(selected_table)
    
    @staticmethod
    def _render_table_data(selected_table: str):
        """Render individual table data."""
        df = st.session_state.generated_data[selected_table]
        st.dataframe(df, use_container_width=True, height=300)
    
    @staticmethod
    def _render_quick_edit_section(selected_table: str):
        """Render quick edit section for table modifications."""
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
                    DataGenerationManager.modify_table_data(selected_table, quick_edit)
    
    @staticmethod
    def generate_data(num_rows: int, temperature: float, user_instructions: str, seed: int, max_tokens: int):
        """Generate synthetic data."""
        if st.session_state.schema is None:
            st.error("Please upload a DDL schema first")
            return
        
        # Log data generation request
        log_user_action("data_generation_started", {
            "num_rows": num_rows,
            "temperature": temperature,
            "user_instructions": user_instructions,
            "max_tokens": max_tokens,
            "num_tables": len(st.session_state.schema.tables),
            "table_names": list(st.session_state.schema.tables.keys())
        })
        
        try:
            with st.spinner("🔄 Generating synthetic data..."):
                # Set up generation parameters
                generation_params = {
                    'temperature': temperature,
                    'seed': seed,
                    'max_tokens': max_tokens
                }
                
                # Run async data generation
                generated_data, raw_responses = DataGenerationManager._run_async_generation(
                    st.session_state.schema,
                    num_rows,
                    user_instructions,
                    generation_params
                )
                
                if generated_data:
                    st.session_state.generated_data = generated_data
                    st.session_state.raw_ai_responses = raw_responses
                    st.success(f"✅ Generated data for {len(generated_data)} tables")
                    
                    # Store in database
                    if st.session_state.db_handler:
                        DataGenerationManager._store_in_database()
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to generate data")
                    
        except Exception as e:
            st.error(f"❌ Error during data generation: {e}")
    
    @staticmethod
    def _run_async_generation(schema, num_rows, user_instructions, generation_params):
        """Run async data generation in a new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                DataGenerationManager._generate_data_with_raw_responses(
                    schema, num_rows, user_instructions, generation_params
                )
            )
            return result
        finally:
            loop.close()
    
    @staticmethod
    async def _generate_data_with_raw_responses(schema, num_rows, user_instructions, generation_params):
        """Generate data and capture raw AI responses for debugging."""
        try:
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
    
    @staticmethod
    def modify_table_data(table_name: str, modification_instructions: str):
        """Modify existing table data."""
        if not modification_instructions.strip():
            st.warning("Please provide modification instructions")
            return
        
        # Log table modification request
        log_user_action("table_modification_started", {
            "table_name": table_name,
            "instructions": modification_instructions,
            "original_rows": len(st.session_state.generated_data.get(table_name, []))
        })
        
        try:
            with st.spinner(f"🔄 Modifying {table_name}..."):
                current_data = st.session_state.generated_data[table_name]
                
                # Check if the modify_table_data method exists
                if not hasattr(st.session_state.data_generator, 'modify_table_data'):
                    st.error("❌ Modify table functionality not implemented yet")
                    return
                
                # Run async data modification
                modified_data = DataGenerationManager._run_async_modification(
                    table_name, current_data, modification_instructions
                )
                
                if modified_data is not None:
                    st.session_state.generated_data[table_name] = modified_data
                    st.success(f"✅ Modified {table_name}")
                    
                    # Log successful modification
                    data_observer.log_table_modification(
                        table_name=table_name,
                        instructions=modification_instructions,
                        success=True
                    )
                    
                    # Update database
                    if st.session_state.db_handler:
                        DataGenerationManager._store_table_in_database(table_name, modified_data)
                    
                    st.rerun()
                else:
                    st.error(f"❌ Failed to modify {table_name} - method returned None")
                    
                    # Log failed modification
                    data_observer.log_table_modification(
                        table_name=table_name,
                        instructions=modification_instructions,
                        success=False
                    )
                    
        except Exception as e:
            st.error(f"❌ Error modifying data: {e}")
    
    @staticmethod
    def _run_async_modification(table_name: str, current_data, modification_instructions: str):
        """Run async table modification."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                st.session_state.data_generator.modify_table_data(
                    table_name,
                    current_data,
                    modification_instructions,
                    st.session_state.schema
                )
            )
        except Exception as e:
            st.error(f"❌ Error during data modification: {e}")
            return None
        finally:
            loop.close()
    
    @staticmethod
    def _store_in_database():
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
    
    @staticmethod
    def _store_table_in_database(table_name: str, data):
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