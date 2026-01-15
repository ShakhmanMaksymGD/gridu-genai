"""
Chat History Database Handler

Handles persistence of chat conversations and messages in PostgreSQL.
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import text

from ..database.postgres_handler import DatabaseHandler


class ChatHistoryManager:
    """Manages chat conversation history in the database."""
    
    def __init__(self, db_handler: DatabaseHandler):
        self.db_handler = db_handler
        # Note: create_tables() is called explicitly from chat_interface.py
    
    def create_tables(self):
        """Create chat history tables if they don't exist."""
        try:
            # Create chat sessions table
            sessions_ddl = """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR(36) PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                metadata JSONB
            );
            """
            
            # Create chat messages table
            messages_ddl = """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id VARCHAR(36) PRIMARY KEY,
                session_id VARCHAR(36) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sql_query TEXT,
                query_result JSONB,
                visualization TEXT,
                error TEXT,
                metadata JSONB
            );
            """
            
            # Create indexes
            indexes_ddl = """
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);
            """
            
            with self.db_handler.get_connection() as conn:
                conn.execute(text(sessions_ddl))
                conn.execute(text(messages_ddl))
                conn.execute(text(indexes_ddl))
                conn.commit()
            
            print("✅ Chat history tables created/verified")
            
        except Exception as e:
            print(f"❌ Error creating chat history tables: {e}")
    
    def create_session(self, title: str, metadata: Optional[Dict] = None) -> str:
        """Create a new chat session and return session_id."""
        session_id = str(uuid.uuid4())
        
        try:
            query = """
            INSERT INTO chat_sessions (session_id, title, metadata)
            VALUES (:session_id, :title, :metadata)
            """
            
            with self.db_handler.get_connection() as conn:
                conn.execute(
                    text(query),
                    {
                        'session_id': session_id,
                        'title': title,
                        'metadata': json.dumps(metadata or {})
                    }
                )
                conn.commit()
            
            print(f"✅ Created chat session: {title}")
            return session_id
            
        except Exception as e:
            print(f"❌ Error creating chat session: {e}")
            return str(uuid.uuid4())  # Fallback to memory-only session
    
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sql_query: Optional[str] = None,
        query_result: Optional[pd.DataFrame] = None,
        visualization: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save a chat message to the database."""
        try:
            message_id = str(uuid.uuid4())
            
            # Convert DataFrame to JSON for storage
            result_json = None
            if query_result is not None and not query_result.empty:
                try:
                    result_json = query_result.to_json(orient='records', date_format='iso')
                except Exception:
                    result_json = str(query_result.shape)  # Fallback to shape info
            
            query = """
            INSERT INTO chat_messages (
                message_id, session_id, role, content, sql_query, 
                query_result, visualization, error, metadata
            ) VALUES (:message_id, :session_id, :role, :content, :sql_query, :query_result, :visualization, :error, :metadata)
            """
            
            with self.db_handler.get_connection() as conn:
                conn.execute(
                    text(query),
                    {
                        'message_id': message_id,
                        'session_id': session_id,
                        'role': role,
                        'content': content,
                        'sql_query': sql_query,
                        'query_result': result_json,
                        'visualization': visualization,
                        'error': error,
                        'metadata': json.dumps(metadata or {})
                    }
                )
                conn.commit()
            
            # Update session message count and timestamp
            self._update_session_stats(session_id)
            return True
            
        except Exception as e:
            print(f"❌ Error saving chat message: {e}")
            return False
    
    def _update_session_stats(self, session_id: str):
        """Update session statistics."""
        try:
            query = """
            UPDATE chat_sessions 
            SET updated_at = CURRENT_TIMESTAMP,
                message_count = (
                    SELECT COUNT(*) FROM chat_messages 
                    WHERE session_id = :session_id
                )
            WHERE session_id = :session_id
            """
            
            with self.db_handler.get_connection() as conn:
                conn.execute(
                    text(query),
                    {'session_id': session_id}
                )
                conn.commit()
                
        except Exception as e:
            print(f"Warning: Could not update session stats: {e}")
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all chat sessions ordered by most recent."""
        try:
            query = """
            SELECT session_id, title, created_at, updated_at, message_count
            FROM chat_sessions
            ORDER BY updated_at DESC
            """
            
            with self.db_handler.get_connection() as conn:
                result = conn.execute(text(query))
                sessions = []
                
                for row in result:
                    sessions.append({
                        'session_id': row[0],
                        'title': row[1],
                        'created_at': row[2],
                        'updated_at': row[3],
                        'message_count': row[4] or 0
                    })
                
                return sessions
                
        except Exception as e:
            print(f"❌ Error getting chat sessions: {e}")
            return []
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Get all messages for a specific session."""
        try:
            query = """
            SELECT message_id, role, content, timestamp, sql_query, 
                   query_result, visualization, error, metadata
            FROM chat_messages
            WHERE session_id = :session_id
            ORDER BY timestamp ASC
            """
            
            with self.db_handler.get_connection() as conn:
                result = conn.execute(
                    text(query),
                    {'session_id': session_id}
                )
                
                messages = []
                for row in result:
                    # Parse query_result JSON back to DataFrame
                    query_result = None
                    if row[5]:  # query_result
                        try:
                            result_data = row[5]  # Already parsed by PostgreSQL JSONB
                            
                            # If it's a string, parse it; if it's already a list/dict, use directly
                            if isinstance(result_data, str):
                                result_data = json.loads(result_data)
                            
                            if result_data:
                                query_result = pd.DataFrame(result_data)
                                print(f"✅ Loaded DataFrame with shape: {query_result.shape}")
                        except Exception as e:
                            print(f"❌ Error parsing query_result: {e}")
                            pass
                    
                    messages.append({
                        'message_id': row[0],
                        'role': row[1],
                        'content': row[2],
                        'timestamp': row[3],
                        'sql_query': row[4],
                        'query_result': query_result,
                        'visualization': row[6],
                        'error': row[7],
                        'metadata': json.loads(row[8] or '{}')
                    })
                
                return messages
                
        except Exception as e:
            print(f"❌ Error getting session messages: {e}")
            return []
    
    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """Update the title of a chat session."""
        try:
            query = """
            UPDATE chat_sessions 
            SET title = :title, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = :session_id
            """
            
            with self.db_handler.get_connection() as conn:
                conn.execute(
                    text(query),
                    {'title': new_title, 'session_id': session_id}
                )
                conn.commit()
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating session title: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages."""
        try:
            query = "DELETE FROM chat_sessions WHERE session_id = :session_id"
            
            with self.db_handler.get_connection() as conn:
                conn.execute(
                    text(query),
                    {'session_id': session_id}
                )
                conn.commit()
            
            print(f"✅ Deleted chat session: {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting chat session: {e}")
            return False
    
    def search_messages(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search for messages containing a specific term."""
        try:
            query = """
            SELECT cm.*, cs.title as session_title
            FROM chat_messages cm
            JOIN chat_sessions cs ON cm.session_id = cs.session_id
            WHERE cm.content ILIKE :search_term OR cm.sql_query ILIKE :search_term
            ORDER BY cm.timestamp DESC
            LIMIT :limit
            """
            
            search_pattern = f"%{search_term}%"
            
            with self.db_handler.get_connection() as conn:
                result = conn.execute(
                    text(query),
                    {
                        'search_term': search_pattern,
                        'limit': limit
                    }
                )
                
                messages = []
                for row in result:
                    messages.append({
                        'message_id': row[0],
                        'session_id': row[1],
                        'role': row[2],
                        'content': row[3],
                        'timestamp': row[4],
                        'sql_query': row[5],
                        'session_title': row[-1]
                    })
                
                return messages
                
        except Exception as e:
            print(f"❌ Error searching messages: {e}")
            return []
    
    def get_session_stats(self) -> Dict:
        """Get overall chat statistics."""
        try:
            query = """
            SELECT 
                COUNT(*) as total_sessions,
                SUM(message_count) as total_messages,
                MAX(updated_at) as last_activity,
                COUNT(CASE WHEN message_count > 0 THEN 1 END) as active_sessions
            FROM chat_sessions
            """
            
            with self.db_handler.get_connection() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
                return {
                    'total_sessions': row[0] or 0,
                    'total_messages': row[1] or 0,
                    'last_activity': row[2],
                    'active_sessions': row[3] or 0
                }
                
        except Exception as e:
            print(f"❌ Error getting session stats: {e}")
            return {
                'total_sessions': 0,
                'total_messages': 0,
                'last_activity': None,
                'active_sessions': 0
            }