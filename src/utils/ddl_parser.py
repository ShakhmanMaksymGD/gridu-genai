"""
DDL Schema Parser Module

This module provides functionality to parse DDL (Data Definition Language) files
and extract table structures, columns, data types, constraints, and relationships.

Uses SQLGlot-based parser for reliable DDL parsing.
"""
import re
import sqlglot
from sqlglot import exp, parse_one
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ConstraintType(Enum):
    """Constraint types."""
    PRIMARY_KEY = "PRIMARY_KEY"
    FOREIGN_KEY = "FOREIGN_KEY" 
    UNIQUE = "UNIQUE"
    CHECK = "CHECK"


class DataType(Enum):
    """Supported data types."""
    INTEGER = "INTEGER"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    DATE = "DATE"
    DATETIME = "DATETIME"
    DECIMAL = "DECIMAL"
    BOOLEAN = "BOOLEAN"
    ENUM = "ENUM"
    TIMESTAMP = "TIMESTAMP"


@dataclass
class Column:
    """Represents a database column."""
    name: str
    data_type: str
    is_nullable: bool = True
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    default_value: Optional[str] = None
    auto_increment: bool = False
    enum_values: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"Column(name='{self.name}', type='{self.data_type}')"


@dataclass
class Constraint:
    """Represents a database constraint."""
    name: Optional[str]
    constraint_type: ConstraintType
    columns: List[str]
    referenced_table: Optional[str] = None
    referenced_columns: List[str] = field(default_factory=list)
    check_condition: Optional[str] = None
    
    def __str__(self):
        return f"Constraint(type={self.constraint_type.value}, columns={self.columns})"


@dataclass
class Table:
    """Represents a database table."""
    name: str
    columns: Dict[str, Column] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    
    def add_column(self, column: Column):
        """Add a column to the table."""
        self.columns[column.name] = column
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the table."""
        self.constraints.append(constraint)
    
    def get_primary_key_columns(self) -> List[str]:
        """Get primary key columns."""
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.PRIMARY_KEY:
                return constraint.columns
        return []
    
    def get_foreign_keys(self) -> List[Constraint]:
        """Get all foreign key constraints."""
        return [c for c in self.constraints if c.constraint_type == ConstraintType.FOREIGN_KEY]
    
    def __str__(self):
        return f"Table(name='{self.name}', columns={len(self.columns)}, constraints={len(self.constraints)})"


@dataclass  
class Schema:
    """Represents a database schema."""
    name: str = "default"
    tables: Dict[str, Table] = field(default_factory=dict)
    
    def add_table(self, table: Table):
        """Add a table to the schema."""
        self.tables[table.name] = table
    
    def get_table_names(self) -> List[str]:
        """Get all table names."""
        return list(self.tables.keys())
    
    def get_creation_order(self) -> List[str]:
        """Get tables in dependency order for creation."""
        # Simple topological sort based on foreign key dependencies
        remaining = set(self.tables.keys())
        ordered = []
        
        while remaining:
            # Find tables with no unresolved dependencies
            ready = []
            for table_name in remaining:
                table = self.tables[table_name]
                dependencies = set()
                
                for constraint in table.constraints:
                    if (constraint.constraint_type == ConstraintType.FOREIGN_KEY and 
                        constraint.referenced_table):
                        dependencies.add(constraint.referenced_table)
                
                # Check if all dependencies are already ordered
                if dependencies.issubset(set(ordered)):
                    ready.append(table_name)
            
            if not ready:
                # Handle circular dependencies by just taking any remaining table
                ready = [next(iter(remaining))]
            
            ordered.extend(ready)
            remaining -= set(ready)
        
        return ordered
    
    def __str__(self):
        return f"Schema(name='{self.name}', tables={len(self.tables)})"


def parse_ddl_content(ddl_content: str) -> Schema:
    """Parse DDL content using SQLGlot transpilation and simple extraction."""
    schema = Schema()
    
    try:
        # First transpile from MySQL to PostgreSQL for normalization
        transpiled_statements = sqlglot.transpile(ddl_content, read="mysql", write="postgres")
        full_sql = "\\n".join(transpiled_statements)
    except Exception as e:
        print(f"Warning: SQLGlot transpilation failed: {e}")
        full_sql = ddl_content

    # Extract CREATE TABLE statements using a simpler approach
    # Split on CREATE TABLE and process each section
    create_sections = re.split(r'CREATE\s+TABLE', full_sql, flags=re.IGNORECASE)
    
    for i, section in enumerate(create_sections[1:], 1):  # Skip first empty section
        # Extract table name and definition with better parentheses handling
        # Look for table_name followed by opening paren, then find the matching closing paren
        name_match = re.match(r'\s+(\w+)\s*\(', section)
        if name_match:
            table_name = name_match.group(1)
            
            # Find the table definition by counting parentheses
            start_pos = name_match.end() - 1  # Position of opening paren
            paren_count = 0
            end_pos = start_pos
            
            for j, char in enumerate(section[start_pos:], start_pos):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_pos = j
                        break
            
            # Extract the content between the parentheses
            table_def = section[start_pos + 1:end_pos]
            
            table = Table(name=table_name)
            
            # Parse columns and constraints from the table definition
            try:
                _parse_table_definition(table, table_def)
                schema.add_table(table)
            except Exception as e:
                # Continue with next table instead of stopping
                continue
    
    return schema


def _parse_table_definition(table: Table, table_def: str):
    """Parse table definition content."""
    # Split by commas but respect parentheses
    items = _split_by_commas(table_def)
    
    for item in items:
        item = item.strip()
        
        # Check if it's a constraint or column
        if _is_constraint(item):
            _parse_constraint(table, item)
        else:
            _parse_column(table, item)


def _split_by_commas(content: str) -> List[str]:
    """Split content by commas, respecting parentheses."""
    items = []
    current_item = ""
    paren_depth = 0
    
    for char in content:
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == ',' and paren_depth == 0:
            items.append(current_item.strip())
            current_item = ""
            continue
        
        current_item += char
    
    if current_item.strip():
        items.append(current_item.strip())
    
    return items


def _is_constraint(item: str) -> bool:
    """Check if an item is a constraint definition."""
    item_upper = item.upper().strip()
    
    # Check if it starts with constraint keywords (standalone constraints)
    constraint_starts = [
        'PRIMARY KEY',
        'FOREIGN KEY', 
        'UNIQUE',
        'CHECK',
        'CONSTRAINT'
    ]
    
    for keyword in constraint_starts:
        if item_upper.startswith(keyword):
            return True
            
    # For PRIMARY KEY, also check if it's a multi-column constraint like "PRIMARY KEY (col1, col2)"
    if 'PRIMARY KEY' in item_upper and '(' in item_upper:
        # Find PRIMARY KEY followed by parentheses
        if re.search(r'PRIMARY\s+KEY\s*\(', item_upper):
            return True
    
    return False


def _parse_column(table: Table, column_def: str):
    """Parse a column definition."""
    # Remove comments
    column_def = re.sub(r'--.*$', '', column_def).strip()
    
    # Extract column name and data type
    parts = column_def.split()
    if len(parts) < 2:
        return
        
    column_name = parts[0]
    data_type_part = parts[1]
    
    # Parse data type 
    data_type, max_length, precision, scale = _parse_data_type(data_type_part)
    
    # Create column
    column = Column(
        name=column_name,
        data_type=data_type,
        max_length=max_length,
        precision=precision,
        scale=scale
    )
    
    # Check for column constraints
    column_def_upper = column_def.upper()
    
    if 'NOT NULL' in column_def_upper:
        column.is_nullable = False
    
    if any(keyword in column_def_upper for keyword in ['AUTO_INCREMENT', 'GENERATED BY DEFAULT AS IDENTITY']):
        column.auto_increment = True
        column.is_nullable = False
    
    if 'PRIMARY KEY' in column_def_upper:
        column.is_nullable = False
        # Add primary key constraint
        pk_constraint = Constraint(
            name=None,
            constraint_type=ConstraintType.PRIMARY_KEY,
            columns=[column_name]
        )
        table.add_constraint(pk_constraint)
    
    table.add_column(column)


def _parse_data_type(data_type_str: str) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """Parse data type string."""
    # Handle common type mappings
    type_mapping = {
        'SERIAL': 'INT',
        'INTEGER': 'INT',
        'BIGINT': 'INT', 
        'NUMERIC': 'DECIMAL',
        'FLOAT': 'DECIMAL',
        'DOUBLE': 'DECIMAL',
        'CHAR': 'VARCHAR',
        'CHARACTER': 'VARCHAR',
        'BOOL': 'BOOLEAN'
    }
    
    # Extract base type and parameters
    match = re.match(r'(\\w+)(?:\\(([^)]+)\\))?', data_type_str)
    if not match:
        return "TEXT", None, None, None
    
    base_type = match.group(1).upper()
    params_str = match.group(2)
    
    # Map type
    mapped_type = type_mapping.get(base_type, base_type)
    
    # Parse parameters
    max_length = None
    precision = None
    scale = None
    
    if params_str:
        params = [p.strip() for p in params_str.split(',')]
        if mapped_type in ['VARCHAR', 'CHAR']:
            try:
                max_length = int(params[0])
            except (ValueError, IndexError):
                pass
        elif mapped_type == 'DECIMAL':
            try:
                precision = int(params[0])
                if len(params) > 1:
                    scale = int(params[1])
            except (ValueError, IndexError):
                pass
    
    return mapped_type, max_length, precision, scale


def _parse_constraint(table: Table, constraint_def: str):
    """Parse a constraint definition."""
    constraint_def_upper = constraint_def.upper()
    
    # Primary key constraint
    if 'PRIMARY KEY' in constraint_def_upper:
        # Extract column names
        match = re.search(r'PRIMARY\\s+KEY\\s*\\(([^)]+)\\)', constraint_def, re.IGNORECASE)
        if match:
            columns = [col.strip() for col in match.group(1).split(',')]
            constraint = Constraint(
                name=None,
                constraint_type=ConstraintType.PRIMARY_KEY,
                columns=columns
            )
            table.add_constraint(constraint)
    
    # Foreign key constraint
    elif 'FOREIGN KEY' in constraint_def_upper:
        # Extract local columns, referenced table and columns
        fk_pattern = r'FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\)'
        match = re.search(fk_pattern, constraint_def, re.IGNORECASE)
        if match:
            local_columns = [col.strip() for col in match.group(1).split(',')]
            ref_table = match.group(2)
            ref_columns = [col.strip() for col in match.group(3).split(',')]
            
            constraint = Constraint(
                name=None,
                constraint_type=ConstraintType.FOREIGN_KEY,
                columns=local_columns,
                referenced_table=ref_table,
                referenced_columns=ref_columns
            )
            table.add_constraint(constraint)
    
    # Unique constraint
    elif constraint_def_upper.startswith('UNIQUE'):
        match = re.search(r'UNIQUE\\s*\\(([^)]+)\\)', constraint_def, re.IGNORECASE)
        if match:
            columns = [col.strip() for col in match.group(1).split(',')]
            constraint = Constraint(
                name=None,
                constraint_type=ConstraintType.UNIQUE,
                columns=columns
            )
            table.add_constraint(constraint)


def parse_ddl_file(file_path: str) -> Schema:
    """Parse DDL file and return schema."""
    with open(file_path, 'r', encoding='utf-8') as f:
        ddl_content = f.read()
    return parse_ddl_content(ddl_content)


# Re-export everything for backward compatibility
__all__ = [
    'parse_ddl_content',
    'parse_ddl_file',
    'Schema', 
    'Table',
    'Column', 
    'Constraint',
    'ConstraintType',
    'DataType'
]