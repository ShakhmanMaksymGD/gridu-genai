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
from src.utils.session_utils import get_constant_session_id


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
                
                # Generate persistent browser session ID
                if 'browser_session_id' not in st.session_state:
                    st.session_state.browser_session_id = get_constant_session_id()
                
                # Import here to avoid circular imports
                from src.chat.chat_interface import ChatInterface
                st.session_state.chat_interface = ChatInterface(
                    st.session_state.db_handler
                )
                
                # Initialize chat history tables if they don't exist
                st.session_state.chat_interface.history_manager.create_tables()
                
                print(f"✅ Chat interface initialized with browser session: {st.session_state.browser_session_id}")
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
            "Show me the top 5 employees by salary",
            "What's the average salary by department?",
            "How many employees work in each city?",
            "Create a bar chart of sales by month",
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
                - "Show me the top 5 employees by salary",
                - "What's the average salary by department?",
                - "How many employees work in each city?",
                - "Create a bar chart of sales by month",
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
        # Initialize chat loading state if not present
        if 'chat_message_loading' not in st.session_state:
            st.session_state.chat_message_loading = False
            
        # Create input form
        with st.form(key="chat_input_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask a question about your data...",
                    placeholder="e.g., Show me the top 5 employees by salary",
                    label_visibility="collapsed",
                    key="chat_user_input"
                )
            
            with col2:
                # Disable button during loading
                button_disabled = st.session_state.chat_message_loading
                button_text = "Thinking..." if st.session_state.chat_message_loading else "Send"
                
                submit_button = st.form_submit_button(button_text, type="primary", width="stretch", disabled=button_disabled)
            
            # Handle form submission
            if submit_button and user_input.strip():
                st.session_state.pending_user_message = user_input.strip()
                st.session_state.chat_message_loading = True
                st.rerun()
        
        # Handle message processing when loading state is active
        if st.session_state.chat_message_loading:
            # Get the user input from session state since form was cleared
            if hasattr(st.session_state, 'pending_user_message'):
                try:
                    ChatUIManager._process_user_message(st.session_state.pending_user_message)
                except Exception as e:
                    st.error(f"Failed to process message: {e}")
                finally:
                    # Always clear loading state and pending message, even on errors
                    del st.session_state.pending_user_message
                    st.session_state.chat_message_loading = False
            else:
                # No pending message, clear loading state
                st.session_state.chat_message_loading = False
    
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
        
        # Process message asynchronously
        try:
            # Run the async function in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                assistant_response = loop.run_until_complete(
                    chat_interface.process_message(user_input)
                )
                
                # Add assistant response to history
                chat_interface.add_message_to_history(assistant_response)
                
                # Force refresh to show new messages
                st.rerun()
                
            finally:
                loop.close()
                
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
            # Disable sample question buttons during chat loading
            button_disabled = st.session_state.get('chat_message_loading', False)
            if st.button(question, key=f"sample_q_{i}", width="stretch", disabled=button_disabled):
                # Auto-fill the question
                st.session_state.chat_user_input = question
                st.session_state.pending_user_message = question
                st.session_state.chat_message_loading = True
                st.rerun()