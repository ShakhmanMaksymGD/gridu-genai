"""
Chat Interface Module

Handles the conversational AI interface for querying data with natural language.
Includes SQL generation, query execution, and visualization capabilities.
"""
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

from ..database.postgres_handler import DatabaseHandler
from ..utils.langfuse_observer import observer
from ..utils.session_utils import get_constant_session_id
from config.settings import settings

@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    sql_query: Optional[str] = None
    query_result: Optional[pd.DataFrame] = None
    visualization: Optional[str] = None  # Base64 encoded image
    error: Optional[str] = None


class SecurityGuard:
    """Implements security and guardrails for chat queries."""
    
    # Dangerous SQL patterns to block
    BLOCKED_PATTERNS = [
        r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\b',
        r'\b(EXEC|EXECUTE|xp_|sp_)\b',
        r'(--|/\*|\*/)',
        r'(\bUNION\b.*\bSELECT\b)',
        r'(\bOR\b.*=.*=)',  # Basic SQL injection pattern
    ]
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore\s+(?:previous|all|above)\s+instructions',
        r'forget\s+(?:previous|all|above)\s+instructions',
        r'system\s*:?\s*',
        r'</?\s*system\s*>',
        r'you\s+are\s+(?:now|a)\s+',
        r'pretend\s+(?:to\s+be|you\s+are)',
    ]
    
    @classmethod
    def is_safe_query(cls, query: str) -> Tuple[bool, Optional[str]]:
        """Check if a query is safe to execute."""
        query_upper = query.upper()
        
        # Check for dangerous SQL patterns
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                return False, f"Query contains dangerous operation: {pattern}"
        
        # Ensure it's a SELECT query
        if not query_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        return True, None
    
    @classmethod
    def detect_prompt_injection(cls, user_input: str) -> Tuple[bool, Optional[str]]:
        """Detect potential prompt injection attempts."""
        input_lower = user_input.lower()
        
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return True, f"Potential prompt injection detected: {pattern}"
        
        return False, None


class DataVisualizer:
    """Handles data visualization generation."""
    
    @staticmethod
    def create_visualization(df: pd.DataFrame, chart_type: str = "auto", title: str = "") -> Optional[str]:
        """Create visualization from DataFrame and return as base64 string."""
        if df.empty:
            return None
        
        try:
            # Analyze original data types
            original_types = DataVisualizer._detect_data_types(df)
            print(f"📊 Original data types detected: {original_types}")
            
            # Preprocess data to convert strings to appropriate numeric types
            df_processed = DataVisualizer._preprocess_dataframe(df)
            
            # Analyze processed data types
            processed_types = DataVisualizer._detect_data_types(df_processed)
            print(f"📊 After preprocessing: {processed_types}")
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Determine chart type if auto
            if chart_type == "auto":
                chart_type = DataVisualizer._determine_chart_type(df_processed)
                print(f"📈 Auto-selected chart type: {chart_type}")
            
            # Create visualization based on type
            if chart_type == "bar":
                DataVisualizer._create_bar_chart(df_processed, ax, title)
            elif chart_type == "line":
                DataVisualizer._create_line_chart(df_processed, ax, title)
            elif chart_type == "scatter":
                DataVisualizer._create_scatter_plot(df_processed, ax, title)
            elif chart_type == "histogram":
                DataVisualizer._create_histogram(df_processed, ax, title)
            else:
                # Default to bar chart
                DataVisualizer._create_bar_chart(df_processed, ax, title)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            print(f"✅ Visualization created successfully ({len(plot_data)} bytes)")
            return base64.b64encode(plot_data).decode()
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            import traceback
            print(f"❌ Visualization error traceback: {traceback.format_exc()}")
            plt.close()
            return None
    
    @staticmethod
    def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame to convert string numbers to numeric types."""
        df_copy = df.copy()
        
        for col in df_copy.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                continue
                
            # Try to convert to numeric
            try:
                # First, handle common cases where numbers might be stored as strings
                if df_copy[col].dtype == 'object':
                    # Clean the column: strip whitespace, handle nulls
                    cleaned_series = df_copy[col].astype(str).str.strip()
                    
                    # Replace common non-numeric indicators
                    cleaned_series = cleaned_series.replace(['', 'null', 'NULL', 'None', 'nan', 'NaN'], pd.NA)
                    
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If we successfully converted most values (more than 50%), use numeric
                    non_null_original = df_copy[col].notna().sum()
                    non_null_converted = numeric_series.notna().sum()
                    
                    if non_null_original > 0 and (non_null_converted / non_null_original) >= 0.5:
                        df_copy[col] = numeric_series
                        print(f"✅ Converted column '{col}' to numeric ({non_null_converted}/{non_null_original} values)")
                    else:
                        print(f"ℹ️ Keeping column '{col}' as categorical ({non_null_converted}/{non_null_original} convertible to numeric)")
                        
            except Exception as e:
                print(f"⚠️ Could not convert column '{col}' to numeric: {e}")
                continue
        
        return df_copy
    
    @staticmethod
    def _detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Analyze and report the detected data types in the DataFrame."""
        type_info = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                type_info[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                type_info[col] = "datetime"
            elif df[col].dtype == 'bool':
                type_info[col] = "boolean"
            else:
                # Check if it might be convertible to numeric
                try:
                    non_null_count = df[col].notna().sum()
                    if non_null_count > 0:
                        cleaned = df[col].astype(str).str.strip().replace(['', 'null', 'NULL', 'None', 'nan', 'NaN'], pd.NA)
                        numeric_converted = pd.to_numeric(cleaned, errors='coerce').notna().sum()
                        conversion_rate = numeric_converted / non_null_count if non_null_count > 0 else 0
                        
                        if conversion_rate >= 0.8:
                            type_info[col] = f"text_numeric ({conversion_rate:.1%} convertible)"
                        elif conversion_rate >= 0.3:
                            type_info[col] = f"mixed ({conversion_rate:.1%} numeric)"
                        else:
                            type_info[col] = "text_categorical"
                    else:
                        type_info[col] = "empty"
                except:
                    type_info[col] = "text_categorical"
        
        return type_info
    
    @staticmethod
    def _determine_chart_type(df: pd.DataFrame) -> str:
        """Automatically determine the best chart type for the data."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) == 1 and len(categorical_cols) == 1:
            return "bar"
        elif len(numeric_cols) >= 2:
            return "scatter"
        elif len(numeric_cols) == 1:
            return "histogram"
        else:
            return "bar"
    
    @staticmethod
    def _create_bar_chart(df: pd.DataFrame, ax, title: str):
        """Create a bar chart."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
            
            # Clean data: remove rows with NaN values in key columns
            df_clean = df.dropna(subset=[x_col, y_col])
            
            # Limit to top 20 categories for readability
            if len(df_clean) > 20:
                df_plot = df_clean.nlargest(20, y_col)
            else:
                df_plot = df_clean
            
            # Ensure we have data to plot
            if len(df_plot) > 0:
                sns.barplot(data=df_plot, x=x_col, y=y_col, ax=ax)
                ax.set_title(title or f"{y_col} by {x_col}")
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(title or "No Data Available")
        else:
            # Fallback: just plot the first numeric column
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                df_clean = df.dropna(subset=[col])
                if len(df_clean) > 0:
                    df_clean[col].plot(kind='bar', ax=ax)
                    ax.set_title(title or f"{col} Distribution")
                else:
                    ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(title or "No Data Available")
            else:
                # No numeric columns found - show first few rows as text
                ax.text(0.5, 0.5, f'No numeric data found for visualization.\nColumns: {list(df.columns)}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                ax.set_title(title or "Data Preview")
    
    @staticmethod
    def _create_line_chart(df: pd.DataFrame, ax, title: str):
        """Create a line chart."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            df_clean = df.dropna(subset=[x_col, y_col])
            if len(df_clean) > 0:
                df_clean.plot(x=x_col, y=y_col, kind='line', ax=ax, marker='o')
                ax.set_title(title or f"{y_col} vs {x_col}")
            else:
                ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(title or "No Data Available")
        elif len(numeric_cols) == 1:
            col = numeric_cols[0]
            df_clean = df.dropna(subset=[col])
            if len(df_clean) > 0:
                df_clean[col].plot(kind='line', ax=ax, marker='o')
                ax.set_title(title or f"{col} Trend")
            else:
                ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(title or "No Data Available")
        else:
            ax.text(0.5, 0.5, f'No numeric data found for line chart.\nColumns: {list(df.columns)}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(title or "Data Preview")
    
    @staticmethod
    def _create_scatter_plot(df: pd.DataFrame, ax, title: str):
        """Create a scatter plot."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            df_clean = df.dropna(subset=[x_col, y_col])
            if len(df_clean) > 0:
                sns.scatterplot(data=df_clean, x=x_col, y=y_col, ax=ax, alpha=0.7)
                ax.set_title(title or f"{y_col} vs {x_col}")
            else:
                ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(title or "No Data Available")
        else:
            ax.text(0.5, 0.5, f'Need at least 2 numeric columns for scatter plot.\nFound: {len(numeric_cols)} numeric columns', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(title or "Insufficient Numeric Data")
    
    @staticmethod
    def _create_histogram(df: pd.DataFrame, ax, title: str):
        """Create a histogram."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            df_clean = df.dropna(subset=[col])
            if len(df_clean) > 0:
                # Determine appropriate number of bins based on data size
                n_bins = min(20, max(5, int(len(df_clean) / 10)))
                df_clean[col].hist(bins=n_bins, ax=ax, alpha=0.7, edgecolor='black')
                ax.set_title(title or f"{col} Distribution")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                # Add some statistics to the plot
                mean_val = df_clean[col].mean()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(title or "No Data Available")
        else:
            ax.text(0.5, 0.5, f'No numeric data found for histogram.\nColumns: {list(df.columns)}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(title or "Data Preview")


class ChatInterface:
    """Main chat interface for natural language data querying."""
    
    def __init__(self, db_handler: DatabaseHandler):
        self.db_handler = db_handler
        self.model = None
        self.conversation_history: List[ChatMessage] = []
        self.available_tables = {}
        self.current_session_id = None
        self.current_session_title = "New Conversation"
        # Use constant session ID for testing
        self.browser_session_id = get_constant_session_id()
        
        # Initialize chat history manager
        from .chat_history import ChatHistoryManager
        self.history_manager = ChatHistoryManager(db_handler)
        
        # Create tables first, before any session operations
        self.history_manager.create_tables()
        
        self.initialize_gemini()
        self.load_schema_info()
        
        # Initialize with constant session ID for testing
        self._initialize_constant_session()
    
    def initialize_gemini(self):
        """Initialize Gemini model for chat."""
        genai.configure(api_key=settings.gemini_api_key)
        
        # Define function for SQL generation
        sql_generation_func = FunctionDeclaration(
            name="generate_sql_query",
            description="Generate a SQL query based on the user's natural language request",
            parameters={
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "The SQL SELECT query to answer the user's question"
                    },
                    "explanation": {
                        "type": "string", 
                        "description": "Brief explanation of what the query does"
                    },
                    "visualization_type": {
                        "type": "string",
                        "description": "Suggested visualization type: bar, line, scatter, histogram, or none",
                        "enum": ["bar", "line", "scatter", "histogram", "none"]
                    }
                },
                "required": ["sql_query", "explanation"]
            }
        )
        
        # Create tool
        sql_tool = Tool(function_declarations=[sql_generation_func])
        
        # Initialize model with function calling
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            tools=[sql_tool],
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            }
        )
    
    def load_schema_info(self):
        """Load information about available tables and their structure."""
        try:
            tables = self.db_handler.get_all_tables()
            
            for table_name in tables:
                table_info = self.db_handler.get_table_info(table_name)
                self.available_tables[table_name] = table_info
                
            print(f"Loaded schema info for {len(self.available_tables)} tables")
            
        except Exception as e:
            print(f"Error loading schema info: {e}")
    
    def get_schema_context(self) -> str:
        """Get schema context for the AI model."""
        if not self.available_tables:
            return "No tables available."
        
        context = "Available Database Tables and Columns:\n\n"
        
        for table_name, table_info in self.available_tables.items():
            context += f"Table: {table_name}\n"
            context += f"Rows: {table_info.get('row_count', 'Unknown')}\n"
            context += "Columns:\n"
            
            for col_info in table_info.get('columns', []):
                col_name = col_info['column_name']
                col_type = col_info['data_type']
                nullable = "NULL" if col_info['is_nullable'] == 'YES' else "NOT NULL"
                context += f"  - {col_name}: {col_type} {nullable}\n"
            
            context += "\n"
        
        return context
    
    async def process_message(self, user_message: str) -> ChatMessage:
        """Process a user message and generate a response."""
        # Security checks
        is_injection, injection_reason = SecurityGuard.detect_prompt_injection(user_message)
        if is_injection:
            
            return ChatMessage(
                role="assistant",
                content="⚠️ I detected a potential security issue with your request. Please rephrase your question about the data.",
                timestamp=datetime.utcnow(),
                error=injection_reason
            )
        
        # Generate SQL response using AI
        try:            
            # Generate AI response
            response = await self._generate_sql_response(user_message)
            
            # Extract function call result
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call'):
                        function_call = part.function_call
                        if function_call.name == "generate_sql_query":
                            args = function_call.args
                            
                            # Extract parameters
                            sql_query = args.get("sql_query", "")
                            explanation = args.get("explanation", "")
                            viz_type = args.get("visualization_type", "none")
                            
                            # Security check on generated SQL
                            is_safe, safety_reason = SecurityGuard.is_safe_query(sql_query)
                            if not is_safe:
                                return ChatMessage(
                                    role="assistant",
                                    content=f"⚠️ The generated query was blocked for security reasons: {safety_reason}",
                                    timestamp=datetime.utcnow(),
                                    error=f"Security violation: {safety_reason}"
                                )
                            
                            # Execute query and return response
                            return await self._execute_query_and_respond(
                                user_question=user_message,
                                sql_query=sql_query,
                                explanation=explanation,
                                viz_type=viz_type,
                            )
            
            return ChatMessage(
                role="assistant",
                content="I'm sorry, I couldn't understand your request. Could you please rephrase your question about the data?",
                timestamp=datetime.utcnow(),
                error="No valid function call generated"
            )
            
        except Exception as e:            
            return ChatMessage(
                role="assistant",
                content=f"Sorry, I encountered an error while processing your request: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _generate_sql_response(self, user_message: str) -> Any:
        """Generate SQL response using Gemini with function calling."""
        schema_context = self.get_schema_context()
        
        # Get recent conversation context
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "\n\nRecent Conversation:\n"
            # Include last 3 exchanges for context
            recent_messages = self.conversation_history[-6:]  # Last 3 user-assistant pairs
            
            for msg in recent_messages:
                role_display = "User" if msg.role == "user" else "Assistant"
                conversation_context += f"{role_display}: {msg.content[:300]}...\n"
                
                # Include SQL query and results if available
                if msg.sql_query:
                    conversation_context += f"Previous SQL: {msg.sql_query}\n"
                if msg.query_result is not None and not msg.query_result.empty:
                    conversation_context += f"Previous Results: {len(msg.query_result)} rows\n"
                    # Show sample of results for context
                    sample_data = msg.query_result.head(2).to_string(index=False)
                    conversation_context += f"Sample Data:\n{sample_data}\n"
                conversation_context += "\n"
        
        prompt = f"""
        You are a helpful SQL assistant that answers questions about data using SQL queries.
        
        Database Schema:
        {schema_context}
        {conversation_context}
        Current User Question: {user_message}
        
        Guidelines:
        - Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
        - Use proper SQL syntax for PostgreSQL
        - Join tables when necessary to answer the question
        - Use aggregation functions (COUNT, SUM, AVG, etc.) when appropriate
        - Limit results to reasonable numbers (add LIMIT clause if needed)
        - If the question asks for visualization, suggest an appropriate chart type
        - Pay attention to the recent conversation context and previous queries
        - If the user is asking to fix or modify a previous query, reference the previous results and SQL
        - Handle NULL values appropriately (use WHERE column IS NOT NULL if needed to exclude NULL values)
        
        Use the generate_sql_query function to provide your response.
        """
        
        return await asyncio.to_thread(
            self.model.generate_content,
            prompt
        )
    
    async def _execute_query_and_respond(
        self, 
        user_question: str,
        sql_query: str, 
        explanation: str, 
        viz_type: str,
    ) -> ChatMessage:
        """Execute SQL query and generate response with optional visualization."""
        
        try:
            # Execute the query
            result_df = self.db_handler.execute_query(sql_query)
            
            if result_df.empty:
                content = f"**Query Result:** No data found.\n\n**SQL Query:**\n```sql\n{sql_query}\n```\n\n**Explanation:** {explanation}"
            else:
                # Format the response
                content = f"**{explanation}**\n\n"
                content += f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n"
                content += f"**Results:** {len(result_df)} rows found"
            
            # Generate visualization if requested
            visualization = None
            if viz_type != "none" and not result_df.empty:
                title = f"Results: {user_question}"
                visualization = DataVisualizer.create_visualization(result_df, viz_type, title)
            
            # Log to Langfuse with comprehensive data
            observer.log_generation(
                name="ai_conversation",
                model="gemini-2.0-flash-exp",
                input_messages=[{
                    "role": "user", 
                    "content": user_question,
                    "context": "Database query with available tables: " + ", ".join(self.available_tables.keys())
                }],
                output_text=content,
                usage={
                    "prompt_tokens": len(user_question.split()),
                    "completion_tokens": len(content.split()),
                    "total_tokens": len(user_question.split()) + len(content.split())
                },
                metadata={
                    "conversation_type": "data_query",
                    "sql_query": sql_query,
                    "explanation": explanation,
                    "query_performance": {
                        "rows_returned": len(result_df),
                        "columns_count": len(result_df.columns) if not result_df.empty else 0,
                        "has_visualization": visualization is not None,
                        "visualization_type": viz_type
                    },
                    "data_summary": {
                        "table_shape": f"{len(result_df)} rows x {len(result_df.columns) if not result_df.empty else 0} columns",
                        "column_names": list(result_df.columns) if not result_df.empty else [],
                        "sample_data": result_df.head(2).to_dict('records') if not result_df.empty else []
                    },
                    "ai_output": {
                        "full_response": content,
                        "response_length": len(content),
                        "contains_sql": sql_query is not None,
                        "contains_data": not result_df.empty,
                        "contains_visualization": visualization is not None
                    },
                    "session_context": {
                        "session_id": self.current_session_id,
                        "session_title": self.current_session_title,
                        "conversation_turn": len(self.conversation_history) + 1
                    }
                }
            )
            # Flush to ensure data is sent to Langfuse immediately
            observer.flush()
            
            return ChatMessage(
                role="assistant",
                content=content,
                timestamp=datetime.utcnow(),
                sql_query=sql_query,
                query_result=result_df,
                visualization=visualization
            )
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}\n\n**SQL Query:**\n```sql\n{sql_query}\n```"            
            
            return ChatMessage(
                role="assistant", 
                content=error_msg,
                timestamp=datetime.utcnow(),
                sql_query=sql_query,
                error=str(e)
            )
    
    def add_message_to_history(self, message: ChatMessage):
        """Add a message to the conversation history and save to database."""
        # Ensure we have a current session
        if not self.current_session_id:
            self.start_new_session()
        
        # Set session_id for the message
        message.session_id = self.current_session_id
        
        # Add to in-memory history
        self.conversation_history.append(message)
        
        # Save to database
        self.history_manager.save_message(
            session_id=self.current_session_id,
            role=message.role,
            content=message.content,
            sql_query=message.sql_query,
            query_result=message.query_result,
            visualization=message.visualization,
            error=message.error
        )
        
        # Keep only last 50 messages in memory to prevent memory issues
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def get_conversation_context(self, last_n: int = 5) -> str:
        """Get recent conversation context for the AI model."""
        if not self.conversation_history:
            return ""
        
        context = "Recent conversation:\n"
        recent_messages = self.conversation_history[-last_n:]
        
        for msg in recent_messages:
            context += f"{msg.role}: {msg.content[:200]}...\n"
        
        return context
    
    def _initialize_constant_session(self):
        """Initialize or load the constant test session."""
        try:
            # Try to load the existing constant session
            success = self.load_session(self.browser_session_id)
            if success:
                print(f"✅ Loaded existing session: {self.current_session_title}")
            else:
                # Create new session with constant ID
                self.current_session_id = self.browser_session_id
                self.current_session_title = "Test Session"
                self.conversation_history = []
                
                # Ensure session exists in database
                try:
                    from .chat_history import ChatHistoryManager
                    result_id = self.history_manager.create_session_with_id(
                        self.browser_session_id, 
                        self.current_session_title, 
                        self.browser_session_id
                    )
                    print(f"✅ Created new constant session: {result_id}")
                except Exception as e:
                    print(f"❌ Failed to create session: {e}")
                    print(f"❌ Error type: {type(e).__name__}")
                    import traceback
                    print(f"❌ Full traceback: {traceback.format_exc()}")
                    
                    # Try to create session without the specific method
                    try:
                        print("🔄 Trying alternative session creation...")
                        self.start_new_session(self.current_session_title)
                        print(f"✅ Alternative session creation successful: {self.current_session_id}")
                    except Exception as e2:
                        print(f"❌ Alternative session creation also failed: {e2}")
                    
        except Exception as e:
            print(f"⚠️ Error initializing session: {e}")
            # Fallback - just set the session ID
            self.current_session_id = self.browser_session_id
            self.current_session_title = "Test Session" 
            self.conversation_history = []
    
    def start_new_session(self, title: Optional[str] = None):
        """Start a new chat session."""
        if not title:
            title = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        self.current_session_id = self.history_manager.create_session(title)
        self.current_session_title = title
        self.conversation_history = []
        
        print(f"✅ Started new chat session: {title}")
    
    def load_session(self, session_id: str):
        """Load an existing chat session."""
        try:
            # Get session info
            sessions = self.history_manager.get_all_sessions()
            session_info = next((s for s in sessions if s['session_id'] == session_id), None)
            
            if not session_info:
                print(f"❌ Session {session_id} not found")
                return False
            
            # Load messages
            messages_data = self.history_manager.get_session_messages(session_id)
            
            # Convert to ChatMessage objects
            messages = []
            for msg_data in messages_data:
                message = ChatMessage(
                    role=msg_data['role'],
                    content=msg_data['content'],
                    timestamp=msg_data['timestamp'],
                    session_id=session_id,
                    message_id=msg_data['message_id'],
                    sql_query=msg_data['sql_query'],
                    query_result=msg_data['query_result'],
                    visualization=msg_data['visualization'],
                    error=msg_data['error']
                )
                messages.append(message)
            
            # Update current session
            self.current_session_id = session_id
            self.current_session_title = session_info['title']
            self.conversation_history = messages
            
            print(f"✅ Loaded session: {session_info['title']} ({len(messages)} messages)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading session: {e}")
            return False
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all available chat sessions."""
        return self.history_manager.get_all_sessions()