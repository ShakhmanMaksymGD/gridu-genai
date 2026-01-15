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
from ..utils.langfuse_observer import observer, log_user_action
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
            print("DEBUG: DataFrame is empty")
            return None
        
        try:
            print(f"DEBUG: Creating visualization with chart_type={chart_type}, df shape={df.shape}")
            print(f"DEBUG: DataFrame columns: {list(df.columns)}")
            print(f"DEBUG: DataFrame dtypes BEFORE conversion: {df.dtypes.to_dict()}")
            
            # Convert numeric-like columns to proper numeric types
            df = df.copy()  # Don't modify original
            for col in df.columns:
                # Try to convert to numeric, ignore errors for truly non-numeric columns
                if df[col].dtype == 'object':  # String columns that might be numeric
                    print(f"DEBUG: Attempting to convert {col} from object to numeric")
                    print(f"DEBUG: Sample values for {col}: {df[col].head().tolist()}")
                    numeric_converted = pd.to_numeric(df[col], errors='coerce')
                    # Only use conversion if we didn't lose too much data (less than 50% NaNs)
                    if numeric_converted.notna().sum() >= len(df) * 0.5:
                        df[col] = numeric_converted
                        print(f"DEBUG: Successfully converted {col} to numeric")
                    else:
                        print(f"DEBUG: Kept {col} as object (too many conversion errors)")
            
            print(f"DEBUG: DataFrame dtypes AFTER conversion: {df.dtypes.to_dict()}")
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Determine chart type if auto
            if chart_type == "auto":
                chart_type = DataVisualizer._determine_chart_type(df)
                print(f"DEBUG: Auto-determined chart type: {chart_type}")
            
            # Create visualization based on type
            if chart_type == "bar":
                DataVisualizer._create_bar_chart(df, ax, title)
            elif chart_type == "line":
                DataVisualizer._create_line_chart(df, ax, title)
            elif chart_type == "scatter":
                DataVisualizer._create_scatter_plot(df, ax, title)
            elif chart_type == "histogram":
                DataVisualizer._create_histogram(df, ax, title)
            else:
                # Default to bar chart
                DataVisualizer._create_bar_chart(df, ax, title)
            
            print("DEBUG: Chart created, converting to base64...")
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            base64_result = base64.b64encode(plot_data).decode()
            print(f"DEBUG: Base64 conversion successful, length: {len(base64_result)}")
            
            return base64_result
            
        except Exception as e:
            print(f"ERROR: Error creating visualization: {e}")
            import traceback
            print(f"ERROR: Full traceback: {traceback.format_exc()}")
            plt.close()
            return None
    
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
        print(f"DEBUG: _create_bar_chart called with df shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        print(f"DEBUG: Numeric columns: {list(numeric_cols)}")
        print(f"DEBUG: Categorical columns: {list(categorical_cols)}")
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
            
            print(f"DEBUG: Using x_col={x_col}, y_col={y_col}")
            
            # Limit to top 20 categories for readability
            if len(df) > 20:
                df_plot = df.nlargest(20, y_col)
            else:
                df_plot = df
            
            # Ensure numeric column is properly converted
            df_plot = df_plot.copy()
            df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
            
            print(f"DEBUG: After numeric conversion, y_col values: {df_plot[y_col].head().tolist()}")
            print(f"DEBUG: y_col sum: {df_plot[y_col].sum()}")
            
            # Remove rows with NaN values in y column
            df_plot = df_plot.dropna(subset=[y_col])
            
            print(f"DEBUG: After dropna, df_plot shape: {df_plot.shape}")
            
            if len(df_plot) == 0 or df_plot[y_col].sum() == 0:
                print("DEBUG: No valid data or zero sum, showing error message")
                ax.text(0.5, 0.5, 'No valid data for chart', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title or "No Data Available")
                return
            
            print("DEBUG: Creating seaborn barplot...")
            sns.barplot(data=df_plot, x=x_col, y=y_col, ax=ax)
            ax.set_title(title or f"{y_col} by {x_col}")
            ax.tick_params(axis='x', rotation=45)
            print("DEBUG: Seaborn barplot created successfully")
        else:
            print("DEBUG: Using fallback numeric plot")
            # Fallback: just plot the first numeric column
            if len(numeric_cols) > 0:
                col_data = pd.to_numeric(df[numeric_cols[0]], errors='coerce').dropna()
                if len(col_data) > 0:
                    col_data.plot(kind='bar', ax=ax)
                    ax.set_title(title or f"{numeric_cols[0]} Distribution")
                else:
                    ax.text(0.5, 0.5, 'No valid numeric data', 
                           ha='center', va='center', transform=ax.transAxes)
    
    @staticmethod
    def _create_line_chart(df: pd.DataFrame, ax, title: str):
        """Create a line chart."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            df.plot(x=x_col, y=y_col, kind='line', ax=ax)
            ax.set_title(title or f"{y_col} vs {x_col}")
        elif len(numeric_cols) == 1:
            df[numeric_cols[0]].plot(kind='line', ax=ax)
            ax.set_title(title or f"{numeric_cols[0]} Trend")
    
    @staticmethod
    def _create_scatter_plot(df: pd.DataFrame, ax, title: str):
        """Create a scatter plot."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(title or f"{y_col} vs {x_col}")
    
    @staticmethod
    def _create_histogram(df: pd.DataFrame, ax, title: str):
        """Create a histogram."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            df[col].hist(bins=20, ax=ax)
            ax.set_title(title or f"{col} Distribution")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")


class ChatInterface:
    """Main chat interface for natural language data querying."""
    
    def __init__(self, db_handler: DatabaseHandler):
        self.db_handler = db_handler
        self.model = None
        self.conversation_history: List[ChatMessage] = []
        self.available_tables = {}
        self.current_session_id = None
        self.current_session_title = "New Conversation"
        
        # Initialize chat history manager
        from .chat_history import ChatHistoryManager
        self.history_manager = ChatHistoryManager(db_handler)
        
        self.initialize_gemini()
        self.load_schema_info()
        
        # Load the most recent session automatically
        self._load_most_recent_session()
    
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
                error="Security violation detected"
            )
        
        try:
            # Create trace for this conversation
            trace_id = observer.create_trace(
                name="chat_query",
                metadata={
                    "user_query": user_message,
                    "session_id": self.current_session_id,
                    "session_title": self.current_session_title,
                    "available_tables": list(self.available_tables.keys()),
                    "conversation_length": len(self.conversation_history)
                }
            )
            
            # Generate SQL query using Gemini
            response = await self._generate_sql_response(user_message)
            
            if response and response.candidates:
                # Check if function calling was used
                candidate = response.candidates[0]
                
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            # Extract function call result
                            function_call = part.function_call
                            if function_call.name == "generate_sql_query":
                                args = function_call.args
                                sql_query = args.get('sql_query', '')
                                explanation = args.get('explanation', '')
                                viz_type = args.get('visualization_type', 'none')
                                
                                # Security check for generated SQL
                                is_safe, safety_reason = SecurityGuard.is_safe_query(sql_query)
                                if not is_safe:
                                    return ChatMessage(
                                        role="assistant",
                                        content=f"I cannot execute this query for security reasons: {safety_reason}",
                                        timestamp=datetime.utcnow(),
                                        error=safety_reason
                                    )
                                
                                # Execute query and create response
                                return await self._execute_query_and_respond(
                                    user_message, sql_query, explanation, viz_type, trace_id
                                )
                        else:
                            # Regular text response (no function call)
                            text_content = part.text if hasattr(part, 'text') else str(part)
                            return ChatMessage(
                                role="assistant",
                                content=text_content,
                                timestamp=datetime.utcnow()
                            )
            
            # Fallback response
            return ChatMessage(
                role="assistant",
                content="I couldn't process your request. Please try rephrasing your question about the data.",
                timestamp=datetime.utcnow(),
                error="No valid response generated"
            )
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return ChatMessage(
                role="assistant",
                content=f"Sorry, I encountered an error: {str(e)}",
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
        trace_id: Optional[str] = None
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
            if trace_id:
                # Prepare detailed output data
                output_data = {
                    "response_content": content,
                    "sql_query": sql_query,
                    "execution_result": {
                        "rows_returned": len(result_df),
                        "columns": list(result_df.columns) if not result_df.empty else [],
                        "data_types": str(result_df.dtypes.to_dict()) if not result_df.empty else {},
                        "sample_data": result_df.head(3).to_dict('records') if not result_df.empty else []
                    },
                    "visualization": {
                        "type": viz_type,
                        "created": visualization is not None,
                        "size_bytes": len(visualization) if visualization else 0
                    }
                }
                
                observer.log_generation(
                    trace_id=trace_id,
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
            
            # Log error to Langfuse if trace exists
            if 'trace_id' in locals():
                observer.log_generation(
                    trace_id=trace_id,
                    name="ai_conversation_error",
                    model="gemini-2.0-flash-exp",
                    input_messages=[{"role": "user", "content": user_question}],
                    output_text=error_msg,
                    usage={
                        "prompt_tokens": len(user_question.split()),
                        "completion_tokens": len(error_msg.split()),
                        "total_tokens": len(user_question.split()) + len(error_msg.split())
                    },
                    metadata={
                        "conversation_type": "data_query_error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "sql_query": sql_query,
                        "ai_output": {
                            "full_response": error_msg,
                            "response_length": len(error_msg),
                            "is_error": True
                        },
                        "session_context": {
                            "session_id": self.current_session_id,
                            "session_title": self.current_session_title,
                            "conversation_turn": len(self.conversation_history) + 1
                        }
                    }
                )
                # Flush to ensure error is logged to Langfuse immediately
                observer.flush()
            
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
    
    def _load_most_recent_session(self):
        """Load the most recent chat session automatically."""
        try:
            sessions = self.history_manager.get_all_sessions()
            if sessions:
                # Load the most recently updated session
                recent_session = sessions[0]  # Already sorted by updated_at DESC
                session_id = recent_session['session_id']
                
                # Load the session
                success = self.load_session(session_id)
                if success:
                    print(f"✅ Auto-loaded recent session: {self.current_session_title}")
                else:
                    print("❌ Failed to auto-load recent session")
            else:
                print("ℹ️ No previous sessions found - starting fresh")
        except Exception as e:
            print(f"⚠️ Could not auto-load recent session: {e}")
    
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
    
    def update_session_title(self, new_title: str) -> bool:
        """Update the current session title."""
        if self.current_session_id:
            success = self.history_manager.update_session_title(self.current_session_id, new_title)
            if success:
                self.current_session_title = new_title
            return success
        return False
    
    def delete_current_session(self) -> bool:
        """Delete the current chat session."""
        if self.current_session_id:
            success = self.history_manager.delete_session(self.current_session_id)
            if success:
                self.current_session_id = None
                self.current_session_title = "New Conversation"
                self.conversation_history = []
            return success
        return False
    
    def clear_history(self):
        """Clear current conversation history (but keep session in DB)."""
        self.conversation_history = []
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """Get statistics about the chat session."""
        total_messages = len(self.conversation_history)
        user_messages = len([m for m in self.conversation_history if m.role == "user"])
        successful_queries = len([m for m in self.conversation_history if m.sql_query and not m.error])
        
        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "successful_queries": successful_queries,
            "error_rate": (total_messages - successful_queries) / max(total_messages, 1),
            "current_session_id": self.current_session_id,
            "current_session_title": self.current_session_title
        }