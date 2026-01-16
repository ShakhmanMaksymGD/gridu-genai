"""
Session Utilities

Provides shared session management utilities across the application.
"""


def get_constant_session_id() -> str:
    """Get a constant session ID for single-user testing."""
    # Use a constant session ID for testing - single session for all users
    session_id = "test_session_001"
    print(f"📋 Using constant test session: {session_id}")
    return session_id