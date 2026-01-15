"""
Chat UI Component

Handles the chat interface for talking to your data.
"""
import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
from typing import Optional

from src.chat.chat_interface import ChatInterface, ChatMessage
from src.utils.langfuse_observer import log_user_action


class ChatUIManager:
    """Manages the chat UI and interactions."""
    
    @staticmethod
    def render_chat_interface():
        """Render the main chat interface."""
        st.markdown("<div class='main-header'>💬 Talk to Your Data</div>", unsafe_allow_html=True)
        
        # Check if we have data to query
        if not ChatUIManager._has_data_available():
            ChatUIManager._render_no_data_message()
            return
        
        # Initialize chat interface if needed
        if not ChatUIManager._initialize_chat_interface():
            st.error("❌ Failed to initialize chat interface")
            return
        
        # Double-check that chat interface is properly initialized
        if 'chat_interface' not in st.session_state or st.session_state.chat_interface is None:
            st.error("❌ Chat interface initialization failed")
            return
        
        # Render chat history
        ChatUIManager._render_chat_history()
        
        # Render input area
        ChatUIManager._render_chat_input()
        
        # Render sidebar with data info
        ChatUIManager._render_data_info_sidebar()
    
    @staticmethod
    def _has_data_available() -> bool:
        """Check if there's data available to query."""
        # Check if we have generated data in session state
        if st.session_state.generated_data:
            return True
        
        # Check if we have data in the database
        if st.session_state.db_handler:
            try:
                tables = st.session_state.db_handler.get_all_tables()
                return len(tables) > 0
            except Exception:
                return False
        
        return False
    
    @staticmethod
    def _initialize_chat_interface() -> bool:
        """Initialize the chat interface if not already done."""
        if 'chat_interface' not in st.session_state or st.session_state.chat_interface is None:
            try:
                # Check if we have a valid database handler
                if st.session_state.db_handler is None:
                    print("Database handler not available for chat interface")
                    return False
                
                # Import here to avoid circular imports
                from src.chat.chat_interface import ChatInterface
                st.session_state.chat_interface = ChatInterface(st.session_state.db_handler)
                
                # Initialize chat history tables if they don't exist
                st.session_state.chat_interface.history_manager.create_tables()
                
                print("✅ Chat interface initialized successfully with database tables")
                return True
            except Exception as e:
                print(f"❌ Error initializing chat interface: {e}")
                st.session_state.chat_interface = None
                return False
        return True
    
    @staticmethod
    def _render_no_data_message():
        """Render message when no data is available."""
        st.info("""
        📊 **No Data Available**
        
        To start chatting with your data, you need to:
        1. Go to the **Data Generation** tab
        2. Upload a DDL schema or select a sample
        3. Generate synthetic data
        4. Return here to query your data with natural language
        """)
        
        # Show sample questions for when data is available
        st.markdown("**Example questions you can ask when data is available:**")
        sample_questions = [
            "Show me the top 10 companies by revenue",
            "What's the average salary by department?",
            "How many employees work in each city?",
            "Create a bar chart of sales by month",
            "Which customers have the most orders?"
        ]
        
        for question in sample_questions:
            st.markdown(f"• *{question}*")
    
    @staticmethod
    def _render_chat_history():
        """Render the conversation history."""
        if 'chat_interface' not in st.session_state or st.session_state.chat_interface is None:
            return
        
        chat_interface = st.session_state.chat_interface
        
        # Create a container for messages
        messages_container = st.container()
        
        with messages_container:
            if not chat_interface.conversation_history:
                st.markdown("""
                👋 **Welcome! I'm your data assistant.**
                
                Ask me questions about your data in natural language. I can:
                - Answer questions by generating SQL queries
                - Show data in tables
                - Create visualizations and charts
                - Help you explore your database
                
                **Try asking something like:**
                - "Show me all the data in the companies table"
                - "What are the top 5 customers by order value?"
                - "Create a chart showing sales over time"
                """)
            else:
                # Display conversation history
                for message in chat_interface.conversation_history:
                    ChatUIManager._render_message(message)
    
    @staticmethod
    def _render_message(message: ChatMessage):
        """Render a single chat message."""
        if message.role == "user":
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                # Show the main content
                st.markdown(message.content)
                
                # Show data table if available
                if message.query_result is not None:
                    try:
                        if not message.query_result.empty:
                            st.dataframe(message.query_result, width="stretch")
                    except Exception as e:
                        # Handle cases where query_result might not be a proper DataFrame
                        print(f"Warning: query_result display issue: {e}")
                        if hasattr(message, 'query_result') and message.query_result is not None:
                            st.write("Data available but display error occurred")
                
                # Show visualization if available
                if message.visualization:
                    try:
                        import base64
                        img_data = base64.b64decode(message.visualization)
                        st.image(img_data)
                    except Exception as e:
                        st.error(f"Error displaying visualization: {e}")
                
                # Show error if any
                if message.error:
                    st.error(f"Error: {message.error}")
    
    @staticmethod
    def _render_chat_input():
        """Render the chat input area."""
        # Create input form
        with st.form(key="chat_input_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask a question about your data...",
                    placeholder="e.g., Show me the top 10 companies by revenue",
                    label_visibility="collapsed",
                    key="chat_user_input"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send", type="primary", width="stretch")
            
            # Handle form submission
            if submit_button and user_input.strip():
                ChatUIManager._process_user_message(user_input.strip())
    
    @staticmethod
    def _process_user_message(user_input: str):
        """Process user message and generate response."""
        if 'chat_interface' not in st.session_state or st.session_state.chat_interface is None:
            st.error("Chat interface not initialized")
            return
        
        chat_interface = st.session_state.chat_interface
        
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=user_input,
            timestamp=datetime.utcnow()
        )
        chat_interface.add_message_to_history(user_message)
        
        # Show processing indicator
        with st.spinner("🤔 Thinking and generating SQL query..."):
            # Process message asynchronously
            try:
                # Run the async function in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    assistant_response = loop.run_until_complete(
                        chat_interface.process_message(user_input)
                    )
                finally:
                    loop.close()
                
                # Add assistant response to history
                chat_interface.add_message_to_history(assistant_response)
                
                # Force refresh to show new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing message: {e}")
                
                # Add error message to history
                error_response = ChatMessage(
                    role="assistant",
                    content=f"Sorry, I encountered an error while processing your request: {e}",
                    timestamp=datetime.utcnow(),
                    error=str(e)
                )
                chat_interface.add_message_to_history(error_response)
    
    @staticmethod
    def _render_data_info_sidebar():
        """Render data information and session management in the sidebar."""
        with st.sidebar:
            # Session management
            st.header("💬 Chat Sessions")
            
            if hasattr(st.session_state, 'session_manager') and \
               hasattr(st.session_state.session_manager, 'chat_interface') and \
               st.session_state.session_manager.chat_interface:
                chat_interface = st.session_state.session_manager.chat_interface
                
                # Current session info
                st.subheader("Current Session")
                current_title = getattr(chat_interface, 'current_session_title', 'New Conversation')
                st.write(f"**{current_title}**")
                
                # New session button
                if st.button("🆕 New Session", width="stretch"):
                    new_title = f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
                    chat_interface.start_new_session(new_title)
                    st.rerun()
                
                # Load existing sessions
                st.subheader("📚 Chat History")
                try:
                    sessions = chat_interface.get_all_sessions()
                    if sessions:
                        # Create options for selectbox
                        session_options = {}
                        current_session_id = getattr(chat_interface, 'current_session_id', None)
                        
                        for session in sessions:
                            session_id = session['session_id']
                            title = session['title']
                            created_at = session['created_at']
                            
                            # Format display text
                            if isinstance(created_at, str):
                                try:
                                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                except:
                                    pass
                            
                            display_text = f"{title} ({created_at.strftime('%m/%d %H:%M') if hasattr(created_at, 'strftime') else created_at})"
                            session_options[display_text] = session_id
                        
                        # Find current selection
                        current_selection = None
                        for display_text, session_id in session_options.items():
                            if session_id == current_session_id:
                                current_selection = display_text
                                break
                        
                        # Session selector
                        if session_options:
                            selected_display = st.selectbox(
                                "Select conversation:",
                                options=list(session_options.keys()),
                                index=list(session_options.keys()).index(current_selection) if current_selection else 0,
                                key="session_selector"
                            )
                            
                            selected_session_id = session_options[selected_display]
                            
                            # Load session if different from current
                            if selected_session_id != current_session_id:
                                if st.button("📂 Load Session", width="stretch"):
                                    chat_interface.load_session(selected_session_id)
                                    st.rerun()
                        
                        # Session actions
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✏️ Rename", width="stretch"):
                                st.session_state.show_rename_dialog = True
                        with col2:
                            if st.button("🗑️ Delete", width="stretch"):
                                if chat_interface.delete_current_session():
                                    st.success("Session deleted!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete session")
                        
                        # Rename dialog
                        if st.session_state.get('show_rename_dialog', False):
                            with st.form("rename_session"):
                                new_title = st.text_input("New session title:", value=current_title)
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.form_submit_button("💾 Save"):
                                        if chat_interface.update_session_title(new_title):
                                            st.success("Title updated!")
                                            st.session_state.show_rename_dialog = False
                                            st.rerun()
                                        else:
                                            st.error("Failed to update title")
                                with col2:
                                    if st.form_submit_button("❌ Cancel"):
                                        st.session_state.show_rename_dialog = False
                                        st.rerun()
                    
                    else:
                        st.info("No chat history yet. Start a conversation!")
                        
                except Exception as e:
                    st.error(f"Error loading chat history: {e}")
                
                st.divider()
                
            # Data information section
            st.header("📊 Available Data")
            
            if hasattr(st.session_state, 'session_manager') and \
               hasattr(st.session_state.session_manager, 'chat_interface') and \
               st.session_state.session_manager.chat_interface:
                chat_interface = st.session_state.session_manager.chat_interface
                
                if chat_interface.available_tables:
                    for table_name, table_info in chat_interface.available_tables.items():
                        with st.expander(f"📋 {table_name}"):
                            st.markdown(f"**Rows:** {table_info.get('row_count', 'Unknown')}")
                            
                            columns = table_info.get('columns', [])
                            if columns:
                                st.markdown("**Columns:**")
                                for col in columns[:10]:  # Limit to first 10 columns
                                    col_name = col['column_name']
                                    col_type = col['data_type']
                                    st.markdown(f"• `{col_name}` ({col_type})")
                                
                                if len(columns) > 10:
                                    st.markdown(f"*...and {len(columns) - 10} more columns*")
                else:
                    st.info("No table information available")
            else:
                st.info("Chat interface not available")
            
            # Chat statistics
            ChatUIManager._render_chat_stats()
    
    @staticmethod
    def _render_chat_stats():
        """Render chat session statistics."""
        if not hasattr(st.session_state, 'session_manager') or \
           not hasattr(st.session_state.session_manager, 'chat_interface') or \
           st.session_state.session_manager.chat_interface is None:
            return
        
        chat_interface = st.session_state.session_manager.chat_interface
        stats = chat_interface.get_chat_stats()
        
        st.header("📈 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats['total_messages'])
            st.metric("Success Rate", f"{(1-stats.get('error_rate', 0))*100:.1f}%")
        with col2:
            st.metric("Queries", stats['successful_queries'])
            st.metric("User Messages", stats['user_messages'])
        
        # Clear current chat button
        if st.button("🗑️ Clear Current Chat", width="stretch"):
            chat_interface.clear_history()
            st.rerun()
    
    @staticmethod
    def render_sample_questions():
        """Render sample questions to help users get started."""
        if not ChatUIManager._has_data_available():
            return
        
        st.markdown("### 💡 Sample Questions")
        
        sample_questions = [
            "Show me all the data in the first table",
            "What tables are available in the database?", 
            "Count the number of records in each table",
            "Show me the top 10 records by the first numeric column",
            "Create a visualization of the data distribution"
        ]
        
        # Create clickable sample questions
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_q_{i}", width="stretch"):
                # Auto-fill the question
                st.session_state.chat_user_input = question
                ChatUIManager._process_user_message(question)