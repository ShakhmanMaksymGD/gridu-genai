"""
Configuration module for the Synthetic Data Generation Application.
"""
import os
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings."""
    
    model_config = ConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra='ignore'  # Ignore extra fields from .env
    )
    
    # Google Cloud Configuration
    google_cloud_project: str = ""
    google_application_credentials: Optional[str] = None
    gemini_api_key: str = ""
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "synthetic_data_app"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    
    # Langfuse Configuration
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    
    # Application Configuration
    debug: bool = True
    data_storage_path: str = "./data"
    temp_upload_path: str = "./temp_uploads"
    
    # Default generation parameters
    default_rows_per_table: int = 1000
    default_temperature: float = 0.7
    max_retries: int = 3


# Global settings instance
settings = Settings()