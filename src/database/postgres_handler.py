"""
PostgreSQL Database Handler Module

This module provides functionality to connect to PostgreSQL, create tables,
insert data, and perform other database operations for the synthetic data application.
"""
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table as SqlAlchemyTable
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..utils.ddl_parser import Schema, Table, Column, ConstraintType
from config.settings import settings


logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Handles PostgreSQL database operations."""
    
    def __init__(self):
        self.connection_string = self._build_connection_string()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from settings."""
        return (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}@"
            f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine and session."""
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=300
            )
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    @contextmanager
    def get_session(self):
        """Get database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist."""
        # Connect to PostgreSQL server (without specific database)
        server_connection_string = (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}@"
            f"{settings.postgres_host}:{settings.postgres_port}/postgres"
        )
        
        try:
            engine = create_engine(server_connection_string)
            with engine.connect() as conn:
                # Check if database exists
                result = conn.execute(text(
                    f"SELECT 1 FROM pg_database WHERE datname = '{settings.postgres_db}'"
                ))
                
                if not result.fetchone():
                    # Database doesn't exist, create it
                    conn.execute(text("COMMIT"))  # End any transaction
                    conn.execute(text(f"CREATE DATABASE {settings.postgres_db}"))
                    logger.info(f"Created database: {settings.postgres_db}")
                else:
                    logger.info(f"Database {settings.postgres_db} already exists")
            
            engine.dispose()
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    def convert_ddl_to_postgresql(self, schema: Schema) -> str:
        """Convert parsed schema to PostgreSQL-compatible DDL."""
        ddl_statements = []
        
        # Get tables in creation order
        creation_order = schema.get_creation_order()
        
        for table_name in creation_order:
            table = schema.tables[table_name]
            ddl = self._generate_table_ddl(table)
            ddl_statements.append(ddl)
        
        return "\n\n".join(ddl_statements)
    
    def _generate_table_ddl(self, table: Table) -> str:
        """Generate PostgreSQL DDL for a single table."""
        lines = [f"CREATE TABLE {table.name} ("]
        
        # Column definitions
        column_lines = []
        for column in table.columns.values():
            col_def = self._generate_column_definition(column)
            column_lines.append(f"    {col_def}")
        
        # Constraint definitions
        constraint_lines = []
        for constraint in table.constraints:
            if constraint.constraint_type == ConstraintType.PRIMARY_KEY:
                cols = ", ".join(constraint.columns)
                constraint_lines.append(f"    PRIMARY KEY ({cols})")
            elif constraint.constraint_type == ConstraintType.UNIQUE:
                cols = ", ".join(constraint.columns)
                constraint_lines.append(f"    UNIQUE ({cols})")
            elif constraint.constraint_type == ConstraintType.FOREIGN_KEY:
                local_cols = ", ".join(constraint.columns)
                ref_cols = ", ".join(constraint.referenced_columns)
                constraint_lines.append(
                    f"    FOREIGN KEY ({local_cols}) REFERENCES {constraint.referenced_table}({ref_cols})"
                )
        
        # Combine all definitions
        all_lines = column_lines + constraint_lines
        lines.append(",\n".join(all_lines))
        lines.append(");")
        
        return "\n".join(lines)
    
    def _generate_column_definition(self, column: Column) -> str:
        """Generate column definition for PostgreSQL."""
        parts = [column.name]
        
        # Data type conversion
        pg_type = self._convert_data_type(column)
        parts.append(pg_type)
        
        # Nullable constraint
        if not column.is_nullable:
            parts.append("NOT NULL")
        
        # Default value
        if column.default_value:
            parts.append(f"DEFAULT {column.default_value}")
        
        return " ".join(parts)
    
    def _convert_data_type(self, column: Column) -> str:
        """Convert column data type to PostgreSQL equivalent."""
        # Ensure data_type is a string
        if isinstance(column.data_type, list):
            data_type = column.data_type[0] if column.data_type else "TEXT"
        else:
            data_type = str(column.data_type)
            
        data_type = data_type.upper()
        
        if data_type in ['INT', 'INTEGER']:
            if column.auto_increment:
                return "SERIAL"
            return "INTEGER"
        
        elif data_type in ['BIGINT']:
            if column.auto_increment:
                return "BIGSERIAL"
            return "BIGINT"
        
        elif data_type in ['VARCHAR']:
            if column.max_length:
                return f"VARCHAR({column.max_length})"
            return "VARCHAR(255)"
        
        elif data_type in ['TEXT']:
            return "TEXT"
        
        elif data_type in ['DECIMAL']:
            if column.precision and column.scale:
                return f"DECIMAL({column.precision}, {column.scale})"
            return "DECIMAL"
        
        elif data_type in ['FLOAT', 'DOUBLE']:
            return "DOUBLE PRECISION"
        
        elif data_type in ['DATE']:
            return "DATE"
        
        elif data_type in ['DATETIME', 'TIMESTAMP']:
            return "TIMESTAMP"
        
        elif data_type in ['BOOLEAN', 'BOOL']:
            return "BOOLEAN"
        
        elif data_type == 'ENUM':
            if column.enum_values:
                values = "', '".join(column.enum_values)
                return f"VARCHAR(50) CHECK ({column.name} IN ('{values}'))"
            return "VARCHAR(50)"
        
        else:
            return "TEXT"  # Fallback
    
    def create_schema_tables(self, schema: Schema) -> bool:
        """Create all tables from the schema in the database."""
        try:
            ddl = self.convert_ddl_to_postgresql(schema)
            
            with self.get_connection() as conn:
                # Drop existing tables if they exist (in reverse order)
                creation_order = schema.get_creation_order()
                for table_name in reversed(creation_order):
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                
                # Execute DDL
                conn.execute(text(ddl))
                conn.commit()
            
            logger.info(f"Created {len(schema.tables)} tables successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create schema tables: {e}")
            return False
    
    def insert_dataframe(self, table_name: str, df: pd.DataFrame) -> bool:
        """Insert DataFrame data into a table."""
        try:
            # Clean the DataFrame
            df_clean = self._clean_dataframe_for_postgres(df)
            
            # Insert data
            df_clean.to_sql(
                table_name.lower(),
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f"Inserted {len(df_clean)} rows into {table_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {e}")
            return False
    
    def _clean_dataframe_for_postgres(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame for PostgreSQL insertion."""
        df_clean = df.copy()
        
        # Replace NaN values with None
        df_clean = df_clean.where(pd.notna(df_clean), None)
        
        # Convert datetime columns to proper format
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try to convert to datetime if it looks like a date
                try:
                    pd.to_datetime(df_clean[col], errors='raise')
                    df_clean[col] = pd.to_datetime(df_clean[col])
                except:
                    pass
        
        return df_clean
    
    def bulk_insert_data(self, table_data: Dict[str, pd.DataFrame], schema: Schema) -> bool:
        """Insert data for multiple tables in dependency order."""
        try:
            creation_order = schema.get_creation_order()
            
            for table_name in creation_order:
                if table_name in table_data:
                    success = self.insert_dataframe(table_name, table_data[table_name])
                    if not success:
                        logger.error(f"Failed to insert data for {table_name}")
                        return False
            
            logger.info("Successfully inserted all table data")
            return True
        
        except Exception as e:
            logger.error(f"Failed to bulk insert data: {e}")
            return False
    
    def get_table_data(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve data from a table."""
        try:
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn)
                return df
        
        except Exception as e:
            logger.error(f"Failed to retrieve data from {table_name}: {e}")
            return pd.DataFrame()
    
    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        try:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
            
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                tables = [row[0] for row in result]
                return tables
        
        except Exception as e:
            logger.error(f"Failed to get table list: {e}")
            return []
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a custom query and return results as DataFrame."""
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn)
                return df
        
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return pd.DataFrame()
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a table."""
        try:
            info_query = f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            
            with self.get_connection() as conn:
                columns_df = pd.read_sql(info_query, conn)
                count_result = conn.execute(text(count_query))
                row_count = count_result.fetchone()[0]
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "columns": columns_df.to_dict('records')
            }
        
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            return {}
    
    def clear_table(self, table_name: str) -> bool:
        """Clear all data from a table."""
        try:
            with self.get_connection() as conn:
                conn.execute(text(f"DELETE FROM {table_name}"))
                conn.commit()
                logger.info(f"Cleared all data from {table_name}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to clear table {table_name}: {e}")
            return False
    
    def drop_all_tables(self) -> bool:
        """Drop all tables in the database."""
        try:
            tables = self.get_all_tables()
            
            with self.get_connection() as conn:
                for table in tables:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                conn.commit()
            
            logger.info(f"Dropped {len(tables)} tables")
            return True
        
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            return False


class DataExporter:
    """Handles data export functionality."""
    
    def __init__(self, db_handler: DatabaseHandler):
        self.db_handler = db_handler
    
    def export_table_to_csv(self, table_name: str, output_path: str) -> bool:
        """Export table data to CSV file."""
        try:
            df = self.db_handler.get_table_data(table_name)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {table_name} to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export {table_name}: {e}")
            return False
    
    def export_all_tables_to_csv(self, output_dir: str) -> bool:
        """Export all tables to CSV files in a directory."""
        import os
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            tables = self.db_handler.get_all_tables()
            
            exported_files = []
            for table_name in tables:
                output_path = os.path.join(output_dir, f"{table_name}.csv")
                if self.export_table_to_csv(table_name, output_path):
                    exported_files.append(output_path)
            
            logger.info(f"Exported {len(exported_files)} tables to CSV")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export tables: {e}")
            return False
    
    def create_zip_export(self, output_path: str) -> bool:
        """Create a ZIP file containing all table data as CSV files."""
        import os
        import tempfile
        import zipfile
        
        try:
            tables = self.db_handler.get_all_tables()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Export all tables to temporary directory
                for table_name in tables:
                    csv_path = os.path.join(temp_dir, f"{table_name}.csv")
                    self.export_table_to_csv(table_name, csv_path)
                
                # Create ZIP file
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for table_name in tables:
                        csv_path = os.path.join(temp_dir, f"{table_name}.csv")
                        if os.path.exists(csv_path):
                            zipf.write(csv_path, f"{table_name}.csv")
            
            logger.info(f"Created ZIP export: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create ZIP export: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Test the database handler
    db = DatabaseHandler()
    
    # Test connection
    if db.test_connection():
        print("Database connection successful")
    else:
        print("Database connection failed")
    
    # Get all tables
    tables = db.get_all_tables()
    print(f"Found {len(tables)} tables: {tables}")