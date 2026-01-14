"""
Synthetic Data Generator Module

This module provides functionality to generate realistic synthetic data
using Google's Gemini 2.0 Flash with function calling and structured output.
"""
import json
import random
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from pydantic import BaseModel, Field

from ..utils.ddl_parser import Schema, Table, Column, ConstraintType
from ..utils.langfuse_observer import data_observer, trace_llm_generation
from config.settings import settings


class TableDataRequest(BaseModel):
    """Request model for generating table data."""
    table_name: str = Field(description="Name of the table")
    num_rows: int = Field(description="Number of rows to generate", gt=0)
    user_instructions: Optional[str] = Field(description="User-specific instructions for data generation")
    existing_data: Optional[Dict[str, List[Any]]] = Field(description="Existing data from referenced tables")


class GeneratedTableData(BaseModel):
    """Response model for generated table data."""
    table_name: str
    columns: List[str]
    data: List[List[Any]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DataGenerationContext:
    """Context for data generation process."""
    schema: Schema
    generated_tables: Dict[str, pd.DataFrame]
    user_instructions: str
    generation_params: Dict[str, Any]
    
    def get_referenced_data(self, table_name: str) -> Dict[str, List[Any]]:
        """Get data from referenced tables for foreign key constraints."""
        table = self.schema.tables[table_name]
        referenced_data = {}
        
        for constraint in table.constraints:
            if constraint.constraint_type == ConstraintType.FOREIGN_KEY:
                ref_table = constraint.referenced_table
                ref_columns = constraint.referenced_columns
                
                if ref_table in self.generated_tables:
                    ref_df = self.generated_tables[ref_table]
                    for ref_col in ref_columns:
                        if ref_col in ref_df.columns:
                            referenced_data[f"{ref_table}.{ref_col}"] = ref_df[ref_col].tolist()
        
        return referenced_data


class SyntheticDataGenerator:
    """Main synthetic data generator using Gemini 2.0 Flash."""
    
    def __init__(self):
        self.model = None
        self.initialize_gemini()
        
    def initialize_gemini(self):
        """Initialize Gemini model with API key."""
        genai.configure(api_key=settings.gemini_api_key)
        
        # Configure generation parameters
        generation_config = {
            "temperature": settings.default_temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        # Initialize model without function calling for now to avoid schema issues
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config
        )
    
    @trace_llm_generation("gemini-2.0-flash-exp", "schema_data_generation")
    async def generate_schema_data(
        self, 
        schema: Schema, 
        num_rows_per_table: int = 10,
        user_instructions: str = "",
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Generate data for all tables in the schema."""
        
        if generation_params is None:
            generation_params = {}
        
        # Start Langfuse session tracking
        schema_name = getattr(schema, 'name', 'unknown_schema')
        data_observer.start_generation_session(
            schema_name=schema_name,
            num_tables=len(schema.tables),
            rows_per_table=num_rows_per_table
        )
        
        context = DataGenerationContext(
            schema=schema,
            generated_tables={},
            user_instructions=user_instructions,
            generation_params=generation_params
        )
        
        # Get tables in dependency order
        creation_order = schema.get_creation_order()
        
        print(f"Processing {len(creation_order)} tables in order: {creation_order}")
        print(f"DEBUG: num_rows_per_table = {num_rows_per_table}")
        
        for table_name in creation_order:
            print(f"Generating data for table: {table_name}")
            print(f"DEBUG: Using num_rows = {num_rows_per_table} for table {table_name}")
            
            start_time = asyncio.get_event_loop().time()
            
            try:
                table_data = await self._generate_table_data(
                    table_name, 
                    context, 
                    num_rows_per_table
                )
                
                end_time = asyncio.get_event_loop().time()
                generation_time = end_time - start_time
                
                if table_data is not None:
                    context.generated_tables[table_name] = table_data
                    print(f"✅ Generated {len(table_data)} rows for {table_name}")
                    
                    # Log successful table generation
                    data_observer.log_table_generation(
                        table_name=table_name,
                        num_rows=len(table_data),
                        generation_time=generation_time,
                        success=True
                    )
                else:
                    print(f"❌ Failed to generate data for {table_name} - returned None")
                    data_observer.log_table_generation(
                        table_name=table_name,
                        num_rows=0,
                        generation_time=generation_time,
                        success=False
                    )
            except Exception as e:
                end_time = asyncio.get_event_loop().time()
                generation_time = end_time - start_time
                print(f"❌ Error generating data for {table_name}: {e}")
                
                # Log failed table generation
                data_observer.log_table_generation(
                    table_name=table_name,
                    num_rows=0,
                    generation_time=generation_time,
                    success=False
                )
                continue  # Continue with next table instead of stopping
        
        # End the generation session
        data_observer.end_generation_session()
        
        print(f"Completed generation for {len(context.generated_tables)} out of {len(creation_order)} tables")
        return context.generated_tables
    
    async def generate_table_data_with_raw(
        self, 
        table: Table, 
        num_rows: int, 
        context: DataGenerationContext
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """Generate data for a single table and return both processed data and raw response."""
        
        table_name = table.name
        print(f"DEBUG: generate_table_data_with_raw called with num_rows = {num_rows} for table {table_name}")
        
        referenced_data = context.get_referenced_data(table_name)
        
        # Prepare prompt for data generation
        prompt = self._build_generation_prompt(
            table, 
            num_rows, 
            context.user_instructions,
            referenced_data
        )
        
        print(f"DEBUG: Built prompt for {table_name} requesting {num_rows} rows")
        
        try:
            # Generate data using Gemini
            response = await self._call_gemini_with_retry(prompt, context.generation_params)
            
            if response and response.candidates:
                # Parse the text response and extract CSV data
                response_text = response.candidates[0].content.parts[0].text
                
                print(f"DEBUG: AI Response for {table_name} (length: {len(response_text)} chars):")
                print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                
                # Try to extract CSV from the response
                df = self._parse_csv_from_response(response_text, table)
                
                if df is not None and not df.empty:
                    # Post-process data to ensure it meets constraints
                    df = self._post_process_data(df, table, context)
                    return df, response_text
                else:
                    return None, response_text
            else:
                return None, "No response from AI model"
        
        except Exception as e:
            print(f"Error generating data for {table_name}: {e}")
            return None, f"Error: {e}"

    async def _generate_table_data(
        self, 
        table_name: str, 
        context: DataGenerationContext, 
        num_rows: int
    ) -> Optional[pd.DataFrame]:
        """Generate data for a single table."""
        
        print(f"DEBUG: _generate_table_data called with num_rows = {num_rows} for table {table_name}")
        
        table = context.schema.tables[table_name]
        referenced_data = context.get_referenced_data(table_name)
        
        # Prepare prompt for data generation
        prompt = self._build_generation_prompt(
            table, 
            num_rows, 
            context.user_instructions,
            referenced_data
        )
        
        print(f"DEBUG: Built prompt for {table_name} requesting {num_rows} rows")
        print(f"DEBUG: Prompt contains text: '...Generate EXACTLY {num_rows} rows...'")
        
        try:
            # Generate data using Gemini
            response = await self._call_gemini_with_retry(prompt, context.generation_params)
            
            if response and response.candidates:
                # Parse the text response and extract CSV data
                response_text = response.candidates[0].content.parts[0].text
                
                # Try to extract CSV from the response
                df = self._parse_csv_from_response(response_text, table)
                
                if df is not None and not df.empty:
                    # Post-process data to ensure it meets constraints
                    df = self._post_process_data(df, table, context)
                    return df
        
        except Exception as e:
            print(f"Error generating data for {table_name}: {e}")
            return None
        
        return None
    
    def _build_generation_prompt(
        self, 
        table: Table, 
        num_rows: int,
        user_instructions: str,
        referenced_data: Dict[str, List[Any]]
    ) -> str:
        """Build the prompt for data generation."""
        
        # Collect table information
        table_info = {
            "name": table.name,
            "columns": [],
            "constraints": [],
            "foreign_keys": []
        }
        
        # Add column information
        for col_name, column in table.columns.items():
            col_info = {
                "name": col_name,
                "type": column.data_type,
                "nullable": column.is_nullable,
                "max_length": column.max_length,
                "auto_increment": column.auto_increment,
                "enum_values": column.enum_values
            }
            table_info["columns"].append(col_info)
        
        # Add constraint information
        primary_keys = []
        for constraint in table.constraints:
            if constraint.constraint_type == ConstraintType.PRIMARY_KEY:
                primary_keys.extend(constraint.columns)
            elif constraint.constraint_type == ConstraintType.FOREIGN_KEY:
                fk_info = {
                    "columns": constraint.columns,
                    "referenced_table": constraint.referenced_table,
                    "referenced_columns": constraint.referenced_columns
                }
                table_info["foreign_keys"].append(fk_info)
        
        table_info["primary_keys"] = primary_keys
        
        # Build the prompt
        prompt = f"""
        Generate realistic synthetic data for the following database table:
        
        Table Information:
        {json.dumps(table_info, indent=2)}
        
        CRITICAL REQUIREMENTS:
        - Generate EXACTLY {num_rows} rows of data (no more, no less)
        - The CSV must have {num_rows} data rows plus 1 header row (total {num_rows + 1} lines)
        - **EVERY FIELD IN EVERY ROW MUST BE COMPLETELY FILLED - NO EMPTY OR INCOMPLETE FIELDS**
        - Ensure data types match the column specifications
        - Respect all constraints (primary keys, foreign keys, nullability, etc.)
        - Make the data realistic and coherent
        - For foreign keys, use values from the referenced data provided below
        
        Referenced Data (for foreign key constraints):
        {json.dumps(referenced_data, indent=2) if referenced_data else "None"}
        
        User Instructions:
        {user_instructions if user_instructions else "Generate realistic, diverse data appropriate for the table context."}
        
        OUTPUT FORMAT REQUIREMENTS:
        - Provide EXACTLY {num_rows} rows of data in CSV format
        - Include a header row with column names
        - **COMPLETE ALL FIELDS - Do not leave any field empty or partial (e.g., avoid "3,1,Tester," - complete it as "3,1,Tester,40.5,2023-01-15,2023-03-30")**
        - **COUNT YOUR FIELDS: Each data row must have EXACTLY {len(table_info['columns'])} values (one per column)**
        - Wrap the CSV data in ```csv and ``` markdown code blocks
        - Do NOT use quotes around field values unless absolutely necessary
        - If a field contains commas, wrap ONLY that field in double quotes
        - Use simple, clean formatting: column1,column2,column3
        - Do NOT include any explanatory text outside the CSV block
        
        CORRECT EXAMPLE:
        ```csv
        employee_id,project_id,role,hours_worked,start_date,end_date
        1,1,Developer,40.00,2023-01-15,2023-03-15
        2,2,Project Manager,20.00,2023-02-01,2023-04-30
        3,1,Tester,35.50,2023-01-10,2023-02-28
        ```
        
        INCORRECT EXAMPLE (DO NOT DO THIS):
        ```csv
        employee_id,project_id,role,hours_worked,start_date,end_date
        1,1,Developer,40.00,2023-01-15,2023-03-15
        2,2,Project Manager,20.00,2023-02-01,2023-04-30
        3,1,Tester,
        ```
        
        VALIDATION CHECKLIST:
        ✓ {num_rows} data rows generated
        ✓ Header row included  
        ✓ **ALL FIELDS IN ALL ROWS COMPLETELY FILLED**
        ✓ **Each row has exactly {len(table_info['columns'])} values**
        ✓ All constraints respected
        ✓ Realistic data values
        ✓ Proper CSV formatting
        
        Data type formatting:
        - Dates: YYYY-MM-DD format
        - Timestamps: YYYY-MM-DD HH:MM:SS format  
        - ENUM columns: Only use values from {[col.enum_values for col in table.columns.values() if col.enum_values]}
        
        IMPORTANT NOTES:
        - Auto-increment columns should start from 1 and increment sequentially
        - Foreign key values MUST exist in the referenced data provided
        - Primary keys must be unique across all {num_rows} rows
        - Generate exactly {num_rows} complete rows - this is mandatory
        """
        
        return prompt
    
    async def _call_gemini_with_retry(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, max_retries: int = 3):
        """Call Gemini API with retry logic and optional generation parameters."""
        
        # Update generation config if parameters provided
        if generation_params:
            generation_config = {
                "temperature": generation_params.get('temperature', settings.default_temperature),
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": generation_params.get('max_tokens', 1000),
            }
            
            # Create model with updated config
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config
            )
        else:
            model = self.model
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt
                )
                return response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def _parse_csv_from_response(self, response_text: str, table: Table) -> Optional[pd.DataFrame]:
        """Parse CSV data from the Gemini response text."""
        import io
        import re
        
        try:
            # Look for CSV data in the response
            # Try to find content between ```csv and ``` or just extract lines that look like CSV
            csv_match = re.search(r'```csv\n(.*?)\n```', response_text, re.DOTALL)
            if csv_match:
                csv_data = csv_match.group(1)
            else:
                # Look for content that starts with column names
                lines = response_text.split('\n')
                csv_lines = []
                found_header = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line contains column names from the table
                    if any(col_name in line for col_name in table.columns.keys()):
                        found_header = True
                        csv_lines.append(line)
                    elif found_header and ',' in line:
                        csv_lines.append(line)
                    elif found_header and not ',' in line:
                        break
                
                if csv_lines:
                    csv_data = '\n'.join(csv_lines)
                else:
                    return None
            
            # Try multiple parsing strategies
            df = None
            
            # Strategy 1: Standard pandas CSV parsing
            try:
                df = pd.read_csv(io.StringIO(csv_data))
            except Exception as e1:
                print(f"Standard CSV parsing failed: {e1}")
                
                # Strategy 2: Try with different quoting
                try:
                    df = pd.read_csv(io.StringIO(csv_data), quoting=3)  # QUOTE_NONE
                except Exception as e2:
                    print(f"No-quote CSV parsing failed: {e2}")
                    
                    # Strategy 3: Manual line-by-line parsing
                    try:
                        lines = csv_data.strip().split('\n')
                        if len(lines) < 2:
                            return None
                            
                        headers = [h.strip('"').strip() for h in lines[0].split(',')]
                        data_rows = []
                        
                        for line in lines[1:]:
                            if line.strip():
                                # Simple split and clean
                                row = [cell.strip('"').strip() for cell in line.split(',')]
                                
                                # If row is shorter than headers, pad with appropriate defaults
                                while len(row) < len(headers):
                                    row.append('')  # Add empty string for missing fields
                                
                                # If row is longer than headers, truncate
                                if len(row) > len(headers):
                                    row = row[:len(headers)]
                                
                                data_rows.append(row)
                        
                        if data_rows:
                            df = pd.DataFrame(data_rows, columns=headers)
                    except Exception as e3:
                        print(f"Manual parsing failed: {e3}")
                        return None
            
            if df is None or df.empty:
                print(f"CSV parsing resulted in empty DataFrame for table")
                return None
                
            print(f"Successfully parsed CSV with {len(df)} rows and columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            print(f"Error parsing CSV from response: {e}")
            print(f"Response text preview: {response_text[:500]}...")
            return None
    
    def _post_process_data(
        self, 
        df: pd.DataFrame, 
        table: Table, 
        context: DataGenerationContext
    ) -> pd.DataFrame:
        """Post-process generated data to ensure it meets all constraints."""
        
        # Ensure auto-increment columns are properly set
        for col_name, column in table.columns.items():
            if column.auto_increment and col_name in df.columns:
                df[col_name] = range(1, len(df) + 1)
        
        # Validate and fix foreign key references
        for constraint in table.constraints:
            if constraint.constraint_type == ConstraintType.FOREIGN_KEY:
                ref_table = constraint.referenced_table
                ref_columns = constraint.referenced_columns
                local_columns = constraint.columns
                
                if (ref_table in context.generated_tables and 
                    all(col in df.columns for col in local_columns)):
                    
                    ref_df = context.generated_tables[ref_table]
                    
                    for i, local_col in enumerate(local_columns):
                        ref_col = ref_columns[i] if i < len(ref_columns) else ref_columns[0]
                        
                        if ref_col in ref_df.columns:
                            valid_values = ref_df[ref_col].dropna().unique().tolist()
                            if valid_values:
                                # Ensure all foreign key values are valid
                                df.loc[df[local_col].notna(), local_col] = [
                                    random.choice(valid_values) 
                                    for _ in range(sum(df[local_col].notna()))
                                ]
        
        # Ensure unique constraints
        for constraint in table.constraints:
            if constraint.constraint_type == ConstraintType.UNIQUE:
                for col in constraint.columns:
                    if col in df.columns:
                        # Remove duplicates by making values unique
                        df[col] = pd.Series(range(len(df))).astype(str) + "_" + df[col].astype(str)
        
        # Handle data type conversions
        for col_name, column in table.columns.items():
            if col_name in df.columns:
                df[col_name] = self._convert_column_type(df[col_name], column)
        
        return df
    
    def _convert_column_type(self, series: pd.Series, column: Column) -> pd.Series:
        """Convert series to appropriate data type based on column definition."""
        
        # Ensure data_type is a string
        if isinstance(column.data_type, list):
            data_type = column.data_type[0] if column.data_type else "TEXT"
        else:
            data_type = str(column.data_type)
            
        if data_type in ['INT', 'INTEGER', 'BIGINT']:
            return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
        
        elif data_type in ['DECIMAL', 'FLOAT', 'DOUBLE']:
            return pd.to_numeric(series, errors='coerce')
        
        elif data_type in ['DATE']:
            return pd.to_datetime(series, errors='coerce').dt.date
        
        elif data_type in ['DATETIME', 'TIMESTAMP']:
            return pd.to_datetime(series, errors='coerce')
        
        elif data_type in ['VARCHAR', 'TEXT', 'CHAR']:
            if column.max_length:
                return series.astype(str).str[:column.max_length]
            return series.astype(str)
        
        elif data_type == 'ENUM':
            # Ensure values are from the allowed enum values
            if column.enum_values:
                return series.apply(
                    lambda x: x if x in column.enum_values else random.choice(column.enum_values)
                )
        
        return series
    
    async def modify_table_data(
        self, 
        table_name: str, 
        current_data: pd.DataFrame,
        modification_instructions: str,
        schema: Schema
    ) -> Optional[pd.DataFrame]:
        """Modify existing table data based on user instructions."""
        
        table = schema.tables[table_name]
        
        prompt = f"""
        Modify the following table data based on the user's instructions:
        
        Table: {table_name}
        Current Data (first 5 rows as example):
        {current_data.head().to_json(orient='records', indent=2)}
        
        Table Schema:
        - Columns: {list(table.columns.keys())}
        - Constraints: {[str(c) for c in table.constraints]}
        
        Modification Instructions:
        {modification_instructions}
        
        Please generate the modified data while maintaining:
        - All existing constraints
        - Data integrity and relationships
        - Appropriate data types
        - The same number of rows: {len(current_data)}
        
        Return the data as CSV format with header row containing the column names:
        {','.join(table.columns.keys())}
        
        Output ONLY the CSV data, starting with the header row.
        """
        
        try:
            response = await self._call_gemini_with_retry(prompt, {'temperature': 0.3, 'max_tokens': 1000})
            
            if response and response.candidates and response.candidates[0].content:
                response_text = response.candidates[0].content.parts[0].text
                
                # Parse CSV data using existing method
                modified_df = self._parse_csv_from_response(response_text, table)
                
                if modified_df is not None:
                    # Ensure same structure as original
                    modified_df = modified_df.reindex(columns=current_data.columns, fill_value=None)
                    
                    # Ensure same number of rows if possible
                    if len(modified_df) > len(current_data):
                        modified_df = modified_df.head(len(current_data))
                    
                    return modified_df
        
        except Exception as e:
            print(f"Error modifying data for {table_name}: {e}")
            return None
        
        return None


# Example usage
async def example_usage():
    """Example of how to use the SyntheticDataGenerator."""
    
    # This would typically be called from the main application
    from ..utils.ddl_parser import parse_ddl_content
    
    # Sample DDL
    ddl = """
    CREATE TABLE Companies (
        company_id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(255) NOT NULL,
        industry VARCHAR(100)
    );
    """
    
    schema = parse_ddl_content(ddl)
    generator = SyntheticDataGenerator()
    
    # Generate data
    generated_data = await generator.generate_schema_data(
        schema, 
        num_rows_per_table=10,
        user_instructions="Generate tech companies with realistic names and industries"
    )
    
    for table_name, df in generated_data.items():
        print(f"\n{table_name}:")
        print(df.head())


if __name__ == "__main__":
    asyncio.run(example_usage())