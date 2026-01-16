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
                
                # Store in database first, then update UI based on success
                database_saved = False
                if st.session_state.db_handler:
                    database_saved = DataGenerationManager._store_in_database_with_verification()
                
                if database_saved:
                    st.success(f"✅ Generated and saved {len(generated_data)} tables to database")
                else:
                    st.warning(f"⚠️ Generated {len(generated_data)} tables (UI only - database save failed)")
                
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
                # Update database first to ensure consistency
                database_updated = False
                if st.session_state.db_handler:
                    try:
                        print(f"🔄 Starting database update for {table_name}")
                        database_updated = DataGenerationManager._update_table_in_database_safely(table_name, modified_data)
                        print(f"📊 Database update result for {table_name}: {database_updated}")
                        
                        if database_updated:
                            # Verify the update by re-fetching from database
                            try:
                                print(f"🔍 Re-fetching data from database for {table_name}")
                                fresh_data = st.session_state.db_handler.get_table_data(table_name)
                                st.session_state.generated_data[table_name] = fresh_data
                                st.success(f"✅ Modified {table_name} successfully and synced with database")
                                print(f"✅ Successfully synced {table_name} with database")
                            except Exception as fetch_error:
                                # Fallback if fetch fails
                                print(f"⚠️ Failed to re-fetch {table_name} from database: {fetch_error}")
                                st.session_state.generated_data[table_name] = modified_data
                                st.success(f"✅ Modified {table_name} (using cached data)")
                        else:
                            st.warning(f"⚠️ {table_name} modified in UI only - database sync failed")
                            st.session_state.generated_data[table_name] = modified_data
                            print(f"❌ Database update failed for {table_name}")
                    except Exception as db_error:
                        st.error(f"❌ Database update failed: {str(db_error)[:100]}...")
                        st.session_state.generated_data[table_name] = modified_data
                        print(f"❌ Database update exception for {table_name}: {db_error}")
                        import traceback
                        traceback.print_exc()
                else:
                    # No database handler, just update UI
                    st.session_state.generated_data[table_name] = modified_data
                    st.info("ℹ️ Table modified in UI only (no database connection)")
                
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
    def _update_table_in_database_safely(table_name: str, data) -> bool:
        """Safely update table data in database with comprehensive error handling."""
        try:
            print(f"🔄 Updating table {table_name} in database...")
            
            # Check if database connection is available
            if not st.session_state.db_handler:
                print("❌ Database handler not available")
                return False
            
            # Validate data
            if data is None or data.empty:
                print("❌ No data to update")
                return False
            
            print(f"📊 Data to update: {len(data)} rows with columns: {list(data.columns)}")
            
            # Strategy 1: Try clearing and inserting (most reliable for independent tables)
            try:
                print(f"🗑️ Clearing existing data from {table_name}")
                clear_success = st.session_state.db_handler.clear_table(table_name)
                
                if clear_success:
                    print(f"✅ Successfully cleared {table_name}")
                    print(f"📥 Inserting {len(data)} rows into {table_name}")
                    
                    insert_success = st.session_state.db_handler.insert_dataframe(table_name, data)
                    
                    if insert_success:
                        print(f"✅ Successfully inserted data into {table_name}")
                        
                        # Verify the data was saved correctly
                        saved_count = DataGenerationManager._verify_table_saved(table_name, len(data))
                        
                        if saved_count == len(data):
                            print(f"✅ Verification successful: {table_name} has {saved_count} rows")
                            return True
                        else:
                            print(f"❌ Verification failed: expected {len(data)}, got {saved_count}")
                            return False
                    else:
                        print(f"❌ Failed to insert data into {table_name}")
                        return False
                else:
                    print(f"❌ Failed to clear {table_name}, trying individual updates")
                    return DataGenerationManager._update_records_with_individual_updates(table_name, data)
                    
            except Exception as e:
                error_msg = str(e).lower()
                print(f"❌ Clear+Insert strategy failed for {table_name}: {e}")
                
                # Strategy 2: Try individual row updates if clear+insert fails
                print(f"🔄 Switching to individual row updates for {table_name}")
                return DataGenerationManager._update_records_with_individual_updates(table_name, data)
                
        except Exception as e:
            print(f"❌ Critical database update error for {table_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def _update_records_with_individual_updates(table_name: str, data) -> bool:
        """Update table using individual UPDATE statements to preserve foreign key relationships."""
        try:
            print(f"🔄 Attempting individual row updates for {table_name}")
            
            if data.empty:
                return True
                
            columns = data.columns.tolist()
            key_column = columns[0]  # Assume first column is primary key
            
            updated_count = 0
            
            # Use autocommit mode to avoid transaction issues
            connection = st.session_state.db_handler.engine.connect()
            
            try:
                for _, row in data.iterrows():
                    try:
                        # Build UPDATE statement for non-key columns
                        key_value = str(row[key_column])  # Convert to string for TEXT fields
                        
                        set_clauses = []
                        params = {'key_value': key_value}
                        
                        for col in columns[1:]:  # Skip the key column
                            value = row[col]
                            # Convert pandas/numpy types to native Python types
                            if hasattr(value, 'item'):
                                value = value.item()
                            
                            # Convert to string for TEXT fields (all columns in our schema are TEXT)
                            value = str(value) if value is not None else None
                            
                            set_clauses.append(f"{col} = :param_{col}")
                            params[f'param_{col}'] = value
                        
                        if set_clauses:  # Only if there are columns to update
                            update_sql = f"""
                            UPDATE {table_name} 
                            SET {', '.join(set_clauses)}
                            WHERE {key_column} = :key_value
                            """
                            
                            from sqlalchemy import text
                            
                            # Use a separate transaction for each update to avoid transaction aborts
                            with connection.begin() as trans:
                                result = connection.execute(text(update_sql), params)
                                updated_count += result.rowcount
                                trans.commit()
                                
                            print(f"✅ Updated row {key_value}")
                        
                    except Exception as row_error:
                        print(f"❌ Failed to update row {row[key_column]}: {row_error}")
                        # Continue with other rows instead of stopping
                        continue
                
                print(f"✅ Updated {updated_count} out of {len(data)} records in {table_name}")
                
                # Return True if we updated at least some records
                return updated_count > 0
                
            finally:
                connection.close()
                
        except Exception as e:
            print(f"❌ Individual update failed for {table_name}: {e}")
            return False
    
    @staticmethod
    def _store_in_database_with_verification() -> bool:
        """Store generated data in PostgreSQL database with verification."""
        try:
            with st.spinner("💾 Storing data in database..."):
                # Create database if not exists
                st.session_state.db_handler.create_database_if_not_exists()
                
                # Track success for all operations
                schema_created = False
                tables_saved = {}
                
                # Try to create tables
                try:
                    schema_created = st.session_state.db_handler.create_schema_tables(st.session_state.schema)
                    if not schema_created:
                        st.error("❌ Failed to create database schema")
                        return False
                except Exception as schema_error:
                    st.error(f"❌ Schema creation failed: {str(schema_error)[:100]}...")
                    print(f"Schema error: {schema_error}")
                    return False
                
                # Store each table individually with verification
                total_tables = len(st.session_state.generated_data)
                tables_successful = 0
                
                for table_name, table_data in st.session_state.generated_data.items():
                    try:
                        # Clear existing data first
                        st.session_state.db_handler.clear_table(table_name)
                        
                        # Insert new data
                        success = st.session_state.db_handler.insert_dataframe(table_name, table_data)
                        
                        if success:
                            # Verify the data was actually saved by counting rows
                            saved_count = DataGenerationManager._verify_table_saved(table_name, len(table_data))
                            
                            if saved_count == len(table_data):
                                tables_saved[table_name] = True
                                tables_successful += 1
                                print(f"✅ Successfully saved {table_name}: {saved_count} rows")
                            else:
                                tables_saved[table_name] = False
                                print(f"❌ Row count mismatch for {table_name}: expected {len(table_data)}, got {saved_count}")
                        else:
                            tables_saved[table_name] = False
                            print(f"❌ Failed to insert data for {table_name}")
                            
                    except Exception as table_error:
                        tables_saved[table_name] = False
                        print(f"❌ Error saving {table_name}: {table_error}")
                
                # Report results
                if tables_successful == total_tables:
                    total_rows = sum(len(data) for data in st.session_state.generated_data.values())
                    st.success(f"✅ All {total_tables} tables saved to database ({total_rows} total rows)")
                    return True
                elif tables_successful > 0:
                    failed_tables = [name for name, success in tables_saved.items() if not success]
                    st.warning(f"⚠️ {tables_successful}/{total_tables} tables saved. Failed: {', '.join(failed_tables)}")
                    return False
                else:
                    st.error("❌ No tables were saved to database")
                    return False
                    
        except Exception as e:
            st.error(f"❌ Database storage error: {str(e)[:100]}...")
            print(f"Database storage error: {e}")
            return False
    
    @staticmethod
    def _verify_table_saved(table_name: str, expected_count: int) -> int:
        """Verify that table data was saved correctly by counting rows."""
        try:
            with st.session_state.db_handler.get_connection() as conn:
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = result.fetchone()[0]
                return count
        except Exception as e:
            print(f"❌ Failed to verify {table_name}: {e}")
            return 0
    
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