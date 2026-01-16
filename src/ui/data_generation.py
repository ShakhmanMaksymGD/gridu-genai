"""
Data Generation Components

Handles data generation UI and business logic.
"""
import asyncio
import streamlit as st
from config.settings import settings
from src.data_generation.synthetic_data_generator import DataGenerationContext


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
        st.dataframe(df, width="stretch", height=300)
    
    @staticmethod
    def _render_quick_edit_section(selected_table: str):
        """Render quick edit section for table modifications."""
        # Initialize modification loading state if not present
        if 'table_modification_loading' not in st.session_state:
            st.session_state.table_modification_loading = False
        
        # Initialize pending modification instruction
        if 'pending_modification_instruction' not in st.session_state:
            st.session_state.pending_modification_instruction = None
            
        edit_col1, edit_col2 = st.columns([4, 1])
        
        with edit_col1:
            quick_edit = st.text_input(
                "Quick edit instructions",
                placeholder="Edit quick edit instructions...",
                key="quick_edit_input",
                label_visibility="collapsed"
            )
        
        with edit_col2:
            # Disable button during loading
            button_disabled = st.session_state.table_modification_loading
            button_text = "Modifying..." if st.session_state.table_modification_loading else "Submit"
            
            if st.button(button_text, key="quick_edit_generate", width="stretch", disabled=button_disabled):
                if quick_edit.strip():
                    st.session_state.pending_modification_instruction = quick_edit.strip()
                    st.session_state.table_modification_loading = True
                    st.rerun()
        
        # Handle table modification when loading state is active and we have a pending instruction
        if (st.session_state.table_modification_loading and 
            st.session_state.pending_modification_instruction):
            
            DataGenerationManager.modify_table_data(
                selected_table, 
                st.session_state.pending_modification_instruction
            )
            
            # Clear the pending instruction and loading state
            st.session_state.pending_modification_instruction = None
            st.session_state.table_modification_loading = False
            
            # Rerun to refresh the UI with updated data and reset button state
            st.rerun()
    
    @staticmethod
    def generate_data(num_rows: int, temperature: float, user_instructions: str, seed: int, max_tokens: int):
        """Generate synthetic data."""
        if st.session_state.schema is None:
            st.error("Please upload a DDL schema first")
            return
        
        try:
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
                
                # Store in database - ensure all tables are saved
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
        
        try:
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
                # Update session state first
                st.session_state.generated_data[table_name] = modified_data
                st.success(f"✅ Modified {table_name}")
                
                # Update database gracefully
                if st.session_state.db_handler:
                    try:
                        DataGenerationManager._update_table_in_database_gracefully(table_name, modified_data)
                    except Exception as db_error:
                        error_msg = str(db_error).lower()
                        if "foreign key" in error_msg or "violates" in error_msg:
                            st.info(f"ℹ️ {table_name} successfully modified in UI. Database not updated due to foreign key relationships.")
                        else:
                            st.warning(f"⚠️ Table modified in UI but database update failed: {str(db_error)[:100]}...")
                        print(f"Database update error: {db_error}")
                
                # Don't call st.rerun() here - let the quick edit section handle state transitions
            else:
                st.error(f"❌ Failed to modify {table_name} - method returned None")
                    
        except Exception as e:
            st.error(f"❌ Error modifying data: {e}")
            print(f"Table modification error: {e}")
    
    @staticmethod
    def _run_async_modification(table_name: str, current_data, modification_instructions: str):
        """Run async table modification with improved error handling."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            print(f"Starting async modification for table: {table_name}")
            print(f"Modification instructions: {modification_instructions[:100]}...")
            
            if not hasattr(st.session_state.data_generator, 'modify_table_data'):
                raise Exception("Data generator does not have modify_table_data method")
            
            result = loop.run_until_complete(
                st.session_state.data_generator.modify_table_data(
                    table_name,
                    current_data,
                    modification_instructions,
                    st.session_state.schema
                )
            )
            
            print(f"✅ Async modification completed for {table_name}")
            return result
            
        except Exception as e:
            print(f"❌ Error during async data modification for {table_name}: {e}")
            st.error(f"❌ Error during data modification: {str(e)[:100]}...")
            return None
        finally:
            loop.close()
    
    @staticmethod
    def _update_table_in_database_gracefully(table_name: str, data):
        """Gracefully update table data in database with better error handling."""
        try:
            print(f"Updating table {table_name} in database...")
            
            # Check if database connection is available
            if not st.session_state.db_handler:
                raise Exception("Database handler not available")
            
            # Instead of deleting, let's try to update existing records
            # This approach preserves foreign key relationships
            try:
                # Use UPSERT approach - PostgreSQL specific
                success = st.session_state.db_handler.upsert_dataframe(table_name, data)
                
                if not success:
                    # Fallback: try individual updates
                    success = DataGenerationManager._update_records_individually(table_name, data)
                
                if success:
                    print(f"✅ Successfully updated {table_name} in database ({len(data)} rows)")
                else:
                    raise Exception("Failed to update table data")
                
            except Exception as db_error:
                error_msg = str(db_error).lower()
                if "foreign key" in error_msg or "violates" in error_msg:
                    print(f"⚠️ Foreign key constraint prevents updating {table_name} in database")
                    st.warning(f"⚠️ {table_name} updated in UI only. Cannot update database due to foreign key relationships.")
                else:
                    raise db_error
                
        except Exception as e:
            print(f"❌ Database update error for {table_name}: {e}")
            raise e
    
    @staticmethod
    def _update_records_individually(table_name: str, data):
        """Update records one by one using UPDATE statements."""
        try:
            print(f"Attempting individual record updates for {table_name}")
            
            # Get the first column as the key (usually ID)
            if data.empty:
                return True
                
            columns = data.columns.tolist()
            key_column = columns[0]  # Assume first column is primary key
            
            with st.session_state.db_handler.get_connection() as conn:
                for _, row in data.iterrows():
                    # Build UPDATE statement
                    key_value = row[key_column]
                    
                    # Build SET clause for other columns
                    set_clauses = []
                    for col in columns[1:]:  # Skip the key column
                        set_clauses.append(f"{col} = :{col}")
                    
                    if set_clauses:  # Only if there are columns to update
                        update_sql = f"""
                        UPDATE {table_name} 
                        SET {', '.join(set_clauses)}
                        WHERE {key_column} = :{key_column}
                        """
                        
                        # Convert row to dict and ensure proper types
                        params = {}
                        for col in columns:
                            value = row[col]
                            # Convert pandas/numpy types to native Python types
                            if hasattr(value, 'item'):
                                value = value.item()
                            params[col] = value
                        
                        from sqlalchemy import text
                        conn.execute(text(update_sql), params)
                
                conn.commit()
                print(f"✅ Updated {len(data)} records in {table_name}")
                return True
                
        except Exception as e:
            print(f"❌ Individual update failed for {table_name}: {e}")
            return False
    
    @staticmethod
    def _store_in_database():
        """Store generated data in PostgreSQL database."""
        try:
            with st.spinner("💾 Storing data in database..."):
                # Create database if not exists
                st.session_state.db_handler.create_database_if_not_exists()
                
                # Try to create tables - if it fails due to schema issues, we'll still show the data
                try:
                    success = st.session_state.db_handler.create_schema_tables(st.session_state.schema)
                    
                    if success:
                        # Insert data using bulk insert method
                        success = st.session_state.db_handler.bulk_insert_data(
                            st.session_state.generated_data, 
                            st.session_state.schema
                        )
                        
                        if success:
                            total_rows = sum(len(data) for data in st.session_state.generated_data.values())
                            st.success(f"✅ All {len(st.session_state.generated_data)} tables saved to database ({total_rows} total rows)")
                        else:
                            st.warning(f"⚠️ Generated data successfully but failed to store in database. Data is still available in the UI.")
                    else:
                        st.warning("⚠️ Failed to create database tables due to schema issues. Generated data is available in the UI but not saved to database.")
                        
                except Exception as schema_error:
                    # Schema has issues (like missing primary keys), but we still have the generated data
                    st.warning(f"⚠️ Database schema issue: {str(schema_error)[:100]}... Generated data is available in the UI.")
                    print(f"Schema error: {schema_error}")
                    
        except Exception as e:
            st.warning(f"⚠️ Database storage error: {str(e)[:100]}... Generated data is still available in the UI.")
            print(f"Database storage error: {e}")
    
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