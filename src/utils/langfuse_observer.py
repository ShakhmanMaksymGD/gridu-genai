"""
Langfuse Observability Integration

This module provides integration with Langfuse for monitoring AI conversation generation.
"""
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    print("⚠️ Langfuse not installed - observability features disabled")
    LANGFUSE_AVAILABLE = False
    # Create dummy classes for when langfuse is not available
    class Langfuse:
        pass
    
from config.settings import settings


class LangfuseObserver:
    """Main class for Langfuse observability integration."""
    
    def __init__(self):
        self.langfuse_client = None
        self.initialize_langfuse()
    
    def initialize_langfuse(self):
        """Initialize Langfuse client if credentials are provided."""
        if not LANGFUSE_AVAILABLE:
            print("⚠️ Langfuse not available - observability disabled")
            return
            
        try:
            if (settings.langfuse_secret_key and 
                settings.langfuse_public_key and
                settings.langfuse_host):
                
                self.langfuse_client = Langfuse(
                    secret_key=settings.langfuse_secret_key,
                    public_key=settings.langfuse_public_key,
                    host=settings.langfuse_host
                )
                
                print("✅ Langfuse client initialized successfully")
            else:
                print("⚠️ Langfuse credentials not provided - observability disabled")
                
        except Exception as e:
            print(f"❌ Failed to initialize Langfuse: {e}")
            self.langfuse_client = None
    
    def is_enabled(self) -> bool:
        """Check if Langfuse observability is enabled."""
        return LANGFUSE_AVAILABLE and self.langfuse_client is not None
    
    def create_trace(
        self,
        name: str,
        input: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Create a new trace and return its ID."""
        if not self.is_enabled():
            print(f"🔍 Langfuse not enabled, skipping trace creation for {name}")
            return None
        
        try:
            trace_id = self.langfuse_client.create_trace_id()
            print(f"🔍 Creating trace in Langfuse: {name} with ID: {trace_id}")
            
            # Update the trace with metadata
            self.langfuse_client.update_current_trace(
                name=name,
                input={"message": input},
                userId=user_id,
                sessionId=session_id,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )
            
            print(f"✅ Successfully created trace: {name} with ID: {trace_id}")
            return trace_id
            
        except Exception as e:
            print(f"❌ Failed to create trace in Langfuse: {e}")
            traceback.print_exc()
            # Return a dummy trace_id so logging can continue
            return str(uuid.uuid4())
    
    def log_generation(
        self,
        name: str,
        model: str,
        input_messages: List[Dict],
        output_text: str,
        usage: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Log an LLM generation event."""
        if not self.is_enabled():
            print(f"🔍 Langfuse not enabled, skipping generation logging for {name}")
            return None
        
        try:
            print(f"🔍 Logging generation to Langfuse: {name}")
            print(f"🔍 Model: {model}")
            print(f"🔍 Input messages: {len(input_messages)} messages")
            print(f"🔍 Output text length: {len(output_text) if output_text else 0}")
            
            # Create a generation event using the Langfuse SDK
            generation = self.langfuse_client.start_generation(
                name=name,
                model=model,
                input=input_messages,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )
            
            # Update with output and usage
            generation.update(
                output=output_text,
                usage=usage or {}
            )
            
            # End the generation
            generation.end()
            
        except Exception as e:
            print(f"❌ Failed to log generation to Langfuse: {e}")
            traceback.print_exc()
            return None
    
    def flush(self):
        """Flush any pending events to Langfuse."""
        if self.is_enabled():
            try:
                print("🔍 Flushing Langfuse events...")
                self.langfuse_client.flush()
                print("✅ Successfully flushed Langfuse events")
            except Exception as e:
                print(f"❌ Failed to flush Langfuse events: {e}")
                traceback.print_exc()


# Global observer instance
observer = LangfuseObserver()