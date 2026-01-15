"""
Langfuse Observability Integration

This module provides integration with Langfuse for monitoring and observability
of LLM interactions, data generation processes, and system performance.
"""
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from functools import wraps
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
    
import pandas as pd

from config.settings import settings


class LangfuseObserver:
    """Main class for Langfuse observability integration."""
    
    def __init__(self):
        self.langfuse_client = None
        self.session_id = str(uuid.uuid4())
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
    
    def create_trace(self, name: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Optional[str]:
        """Create a new trace for a user session or operation."""
        if not self.is_enabled():
            print(f"🔍 Langfuse not enabled, skipping trace creation for {name}")
            return None
        
        try:
            print(f"🔍 Creating Langfuse trace: {name}")
            
            # Generate a trace ID
            trace_id = str(uuid.uuid4())
            
            # Create an initial event (without trace_id parameter)
            self.langfuse_client.create_event(
                name=f"start_{name}",
                input={
                    "user_id": user_id or "anonymous", 
                    "session_id": self.session_id,
                    "operation": name,
                    "trace_id": trace_id  # Include in metadata instead
                },
                metadata={
                    "trace_id": trace_id,
                    **(metadata or {})
                }
            )
            
            print(f"✅ Successfully created trace: {name} with ID: {trace_id}")
            return trace_id
            
        except Exception as e:
            print(f"❌ Failed to create trace: {e}")
            traceback.print_exc()
            return None
    
    def create_span(self, trace_id: str, name: str, input_data: Any = None, metadata: Optional[Dict] = None) -> Optional[str]:
        """Create a span within a trace."""
        if not self.is_enabled() or not trace_id:
            return None
        
        try:
            span = self.langfuse_client.start_span(
                name=name,
                input=self._serialize_input(input_data),
                metadata=metadata or {},
                trace_id=trace_id
            )
            
            return span.id if hasattr(span, 'id') else str(uuid.uuid4())
            
        except Exception as e:
            print(f"Failed to create span: {e}")
            return None
    
    def end_span(self, span_id: str, output_data: Any = None, metadata: Optional[Dict] = None, status: str = "success"):
        """End a span with output data and metadata."""
        if not self.is_enabled() or not span_id:
            return
        
        try:
            self.langfuse_client.update_current_span(
                output=self._serialize_output(output_data),
                metadata=metadata or {},
                level="DEFAULT" if status == "success" else "ERROR"
            )
            
        except Exception as e:
            print(f"Failed to end span: {e}")
    
    def log_generation(
        self,
        trace_id: str,
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
            print(f"🔍 Trace ID: {trace_id}, Model: {model}")
            print(f"🔍 Input messages: {len(input_messages)} messages")
            print(f"🔍 Output text length: {len(output_text) if output_text else 0}")
            
            # Create a generation event using start_generation
            generation_id = str(uuid.uuid4())
            
            generation = self.langfuse_client.start_generation(
                name=name,
                model=model,
                input=input_messages,
                metadata={
                    "model": model,
                    "usage": usage or {},
                    "timestamp": datetime.utcnow().isoformat(),
                    "trace_id": trace_id,  # Include in metadata for tracking
                    "generation_id": generation_id,
                    **(metadata or {})
                }
            )
            
            # Update the generation with output and usage
            generation.update(
                output=output_text,
                usage=usage or {}
            )
            
            # End the generation
            generation.end()
            
            print(f"✅ Successfully logged generation: {name} with ID: {generation_id}")
            return generation_id
            
        except Exception as e:
            print(f"❌ Failed to log generation to Langfuse: {e}")
            traceback.print_exc()
            return None
    
    def log_error(self, trace_id: str, error: Exception, context: Optional[Dict] = None):
        """Log an error event."""
        if not self.is_enabled() or not trace_id:
            return
        
        try:
            print(f"🔍 Logging error to Langfuse: {type(error).__name__}")
            
            self.langfuse_client.create_event(
                name="error",
                input=context or {},
                output={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc()
                },
                metadata={
                    "severity": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "trace_id": trace_id  # Include in metadata for tracking
                }
            )
            
            print(f"✅ Successfully logged error: {type(error).__name__}")
            
        except Exception as e:
            print(f"❌ Failed to log error: {e}")
            traceback.print_exc()
    
    def log_performance_metrics(self, trace_id: str, metrics: Dict[str, Any]):
        """Log performance metrics."""
        if not self.is_enabled() or not trace_id:
            return
        
        try:
            self.langfuse_client.event(
                trace_id=trace_id,
                name="performance_metrics",
                input=metrics,
                metadata={
                    "type": "metrics",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            print(f"Failed to log metrics: {e}")
    
    def _serialize_input(self, data: Any) -> Any:
        """Serialize input data for logging."""
        if data is None:
            return None
        
        if isinstance(data, (str, int, float, bool, list, dict)):
            return data
        
        if isinstance(data, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": data.shape,
                "columns": data.columns.tolist(),
                "sample": data.head().to_dict('records')
            }
        
        # For other types, convert to string
        return str(data)
    
    def _serialize_output(self, data: Any) -> Any:
        """Serialize output data for logging."""
        return self._serialize_input(data)  # Same logic for now
    
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


def trace_operation(name: str, user_id: Optional[str] = None):
    """Decorator to trace function operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = observer.create_trace(
                name=name,
                user_id=user_id,
                metadata={
                    "function": func.__name__,
                    "module": func.__module__
                }
            )
            
            span_id = observer.create_span(
                trace_id=trace_id,
                name=f"{func.__name__}_execution",
                input_data={"args": args, "kwargs": kwargs}
            )
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                observer.end_span(
                    span_id=span_id,
                    output_data=result,
                    metadata={"execution_time_seconds": execution_time},
                    status="success"
                )
                
                observer.log_performance_metrics(
                    trace_id=trace_id,
                    metrics={
                        "execution_time": execution_time,
                        "function": func.__name__
                    }
                )
                
                return result
                
            except Exception as e:
                observer.log_error(trace_id, e, {"function": func.__name__})
                observer.end_span(
                    span_id=span_id,
                    metadata={"error": str(e)},
                    status="error"
                )
                raise
            
            finally:
                observer.flush()
        
        return wrapper
    return decorator


def trace_llm_generation(model_name: str, operation_name: str):
    """Decorator specifically for tracing LLM generation operations."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_id = observer.create_trace(
                name=f"llm_generation_{operation_name}",
                metadata={
                    "model": model_name,
                    "operation": operation_name,
                    "function": func.__name__
                }
            )
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Log the generation
                observer.log_generation(
                    trace_id=trace_id,
                    name=operation_name,
                    model=model_name,
                    input_messages=[{"role": "user", "content": str(args)}],
                    output_text=str(result),
                    usage={"execution_time": execution_time},
                    metadata={"function": func.__name__}
                )
                
                return result
                
            except Exception as e:
                observer.log_error(trace_id, e, {"operation": operation_name})
                raise
            
            finally:
                observer.flush()
        
        def sync_wrapper(*args, **kwargs):
            trace_id = observer.create_trace(
                name=f"llm_generation_{operation_name}",
                metadata={
                    "model": model_name,
                    "operation": operation_name,
                    "function": func.__name__
                }
            )
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Log the generation
                observer.log_generation(
                    trace_id=trace_id,
                    name=operation_name,
                    model=model_name,
                    input_messages=[{"role": "user", "content": str(args)}],
                    output_text=str(result),
                    usage={"execution_time": execution_time},
                    metadata={"function": func.__name__}
                )
                
                return result
                
            except Exception as e:
                observer.log_error(trace_id, e, {"operation": operation_name})
                raise
            
            finally:
                observer.flush()
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class DataGenerationObserver:
    """Specialized observer for data generation operations."""
    
    def __init__(self):
        self.current_trace_id = None
        self.current_session = {
            "schema_tables": 0,
            "rows_generated": 0,
            "generation_start": None,
            "generation_end": None
        }
    
    def start_generation_session(self, schema_name: str, num_tables: int, rows_per_table: int):
        """Start a new data generation session."""
        self.current_trace_id = observer.create_trace(
            name="data_generation_session",
            metadata={
                "schema_name": schema_name,
                "num_tables": num_tables,
                "rows_per_table": rows_per_table,
                "session_type": "batch_generation"
            }
        )
        
        self.current_session = {
            "schema_tables": num_tables,
            "rows_generated": 0,
            "generation_start": datetime.utcnow(),
            "generation_end": None
        }
        
        return self.current_trace_id
    
    def log_table_generation(self, table_name: str, num_rows: int, generation_time: float, success: bool):
        """Log the generation of a single table."""
        if not self.current_trace_id:
            return
        
        self.current_session["rows_generated"] += num_rows if success else 0
        
        observer.langfuse_client.event(
            trace_id=self.current_trace_id,
            name="table_generated",
            input={
                "table_name": table_name,
                "requested_rows": num_rows
            },
            output={
                "success": success,
                "actual_rows": num_rows if success else 0,
                "generation_time": generation_time
            },
            metadata={
                "table": table_name,
                "status": "success" if success else "failed"
            }
        ) if observer.is_enabled() else None
    
    def log_table_modification(self, table_name: str, instructions: str, success: bool):
        """Log table data modification."""
        if not self.current_trace_id:
            return
        
        observer.langfuse_client.event(
            trace_id=self.current_trace_id,
            name="table_modified",
            input={
                "table_name": table_name,
                "instructions": instructions
            },
            output={
                "success": success
            },
            metadata={
                "table": table_name,
                "operation": "modification"
            }
        ) if observer.is_enabled() else None
    
    def end_generation_session(self):
        """End the current data generation session."""
        if not self.current_trace_id:
            return
        
        self.current_session["generation_end"] = datetime.utcnow()
        
        total_time = None
        if self.current_session["generation_start"]:
            total_time = (
                self.current_session["generation_end"] - 
                self.current_session["generation_start"]
            ).total_seconds()
        
        observer.log_performance_metrics(
            self.current_trace_id,
            {
                "total_tables": self.current_session["schema_tables"],
                "total_rows_generated": self.current_session["rows_generated"],
                "total_generation_time": total_time,
                "avg_time_per_table": total_time / self.current_session["schema_tables"] if total_time and self.current_session["schema_tables"] > 0 else None
            }
        )
        
        observer.flush()


# Global data generation observer
data_observer = DataGenerationObserver()


# Convenience functions
def log_user_action(action: str, details: Dict[str, Any]):
    """Log a user action for analytics."""
    if not observer.is_enabled():
        return
        
    try:
        observer.langfuse_client.create_event(
            name=f"user_action_{action}",
            input=details,
            metadata={
                "action": action,
                "category": "user_interaction",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        print(f"Failed to log user action: {e}")
    
    observer.flush()


def log_system_event(event_type: str, details: Dict[str, Any]):
    """Log a system event."""
    trace_id = observer.create_trace(
        name="system_event",
        metadata={
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    observer.langfuse_client.event(
        trace_id=trace_id,
        name=event_type,
        input=details,
        metadata={
            "category": "system"
        }
    ) if observer.is_enabled() else None
    
    observer.flush()


# Example usage
if __name__ == "__main__":
    # Test the observer
    @trace_operation("test_function")
    def test_function(x, y):
        return x + y
    
    result = test_function(1, 2)
    print(f"Result: {result}")
    
    # Test user action logging
    log_user_action("schema_uploaded", {
        "file_name": "test.sql",
        "tables_count": 5
    })
    
    # Test system event logging
    log_system_event("database_connected", {
        "host": "localhost",
        "database": "test_db"
    })