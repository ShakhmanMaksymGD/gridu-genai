"""
DDL Schema Parser Module

This module provides functionality to parse DDL (Data Definition Language) files
and extract table structures, columns, data types, constraints, and relationships.
"""
import re
import sqlparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


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


class ConstraintType(Enum):
    """Constraint types."""
    PRIMARY_KEY = "PRIMARY_KEY"
    FOREIGN_KEY = "FOREIGN_KEY"
    UNIQUE = "UNIQUE"
    NOT_NULL = "NOT_NULL"
    CHECK = "CHECK"
    DEFAULT = "DEFAULT"


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
    tables: Dict[str, Table] = field(default_factory=dict)
    
    def add_table(self, table: Table):
        """Add a table to the schema."""
        self.tables[table.name] = table
    
    def get_table_dependencies(self) -> List[Tuple[str, str]]:
        """Get table dependencies based on foreign keys."""
        dependencies = []
        for table in self.tables.values():
            for fk in table.get_foreign_keys():
                if fk.referenced_table and fk.referenced_table != table.name:
                    dependencies.append((fk.referenced_table, table.name))
        return dependencies
    
    def get_creation_order(self) -> List[str]:
        """Get tables in creation order (respecting foreign key dependencies)."""
        dependencies = self.get_table_dependencies()
        table_names = list(self.tables.keys())
        ordered = []
        
        # Simple topological sort
        while table_names:
            # Find tables with no unresolved dependencies
            ready = []
            for table in table_names:
                has_unresolved_deps = any(
                    dep[0] in table_names for dep in dependencies 
                    if dep[1] == table
                )
                if not has_unresolved_deps:
                    ready.append(table)
            
            if not ready:
                # Handle circular dependencies by adding remaining tables
                ready = table_names[:]
            
            ordered.extend(ready)
            for table in ready:
                table_names.remove(table)
        
        return ordered


class DDLParser:
    """Parser for DDL statements."""
    
    def __init__(self):
        self.schema = Schema()
    
    def parse_ddl(self, ddl_content: str) -> Schema:
        """Parse DDL content and return a Schema object."""
        self.schema = Schema()
        
        # Parse SQL statements using sqlparse
        statements = sqlparse.split(ddl_content)
        
        for statement in statements:
            if statement.strip():
                self._parse_statement(statement.strip())
        
        return self.schema
    
    def _parse_statement(self, statement: str):
        """Parse a single DDL statement."""
        parsed = sqlparse.parse(statement)[0]
        
        # Check if it's a CREATE TABLE statement
        if self._is_create_table_statement(parsed):
            self._parse_create_table(statement)
    
    def _is_create_table_statement(self, parsed) -> bool:
        """Check if the parsed statement is a CREATE TABLE statement."""
        tokens = list(parsed.flatten())
        create_found = False
        table_found = False
        
        for token in tokens:
            if token.ttype is sqlparse.tokens.Keyword or token.ttype is sqlparse.tokens.Keyword.DDL:
                if token.value.upper() == 'CREATE':
                    create_found = True
                elif token.value.upper() == 'TABLE' and create_found:
                    table_found = True
                    break
        
        return create_found and table_found
    
    def _parse_create_table(self, statement: str):
        """Parse a CREATE TABLE statement."""
        # Extract table name
        table_name_match = re.search(r'CREATE\s+TABLE\s+(\w+)', statement, re.IGNORECASE)
        if not table_name_match:
            return
        
        table_name = table_name_match.group(1)
        table = Table(name=table_name)
        
        # Extract column definitions and constraints
        # Find the content within parentheses
        paren_match = re.search(r'\((.*)\)', statement, re.DOTALL)
        if not paren_match:
            return
        
        content = paren_match.group(1)
        
        # Split by commas, but be careful of commas within function calls
        items = self._split_table_definition(content)
        
        for item in items:
            item = item.strip()
            if self._is_constraint_definition(item):
                constraint = self._parse_constraint(item)
                if constraint:
                    table.add_constraint(constraint)
            else:
                column = self._parse_column_definition(item)
                if column:
                    table.add_column(column)
                    # Check for inline constraints
                    inline_constraints = self._parse_inline_constraints(item, column.name)
                    for constraint in inline_constraints:
                        table.add_constraint(constraint)
        
        self.schema.add_table(table)
    
    def _split_table_definition(self, content: str) -> List[str]:
        """Split table definition by commas, respecting parentheses."""
        items = []
        current_item = ""
        paren_level = 0
        
        for char in content:
            if char == '(':
                paren_level += 1
            elif char == ')':
                paren_level -= 1
            elif char == ',' and paren_level == 0:
                items.append(current_item.strip())
                current_item = ""
                continue
            
            current_item += char
        
        if current_item.strip():
            items.append(current_item.strip())
        
        return items
    
    def _is_constraint_definition(self, item: str) -> bool:
        """Check if an item is a constraint definition."""
        constraint_keywords = ['PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT']
        item_upper = item.upper()
        return any(keyword in item_upper for keyword in constraint_keywords)
    
    def _parse_column_definition(self, definition: str) -> Optional[Column]:
        """Parse a column definition."""
        # Match column name and data type
        match = re.match(r'(\w+)\s+(\w+(?:\([^)]+\))?)', definition, re.IGNORECASE)
        if not match:
            return None
        
        column_name = match.group(1)
        data_type_full = match.group(2)
        
        # Extract data type and parameters
        data_type, max_length, precision, scale, enum_values = self._parse_data_type(data_type_full)
        
        # Ensure data_type is a string
        if not isinstance(data_type, str):
            data_type = str(data_type) if data_type else "TEXT"
        
        column = Column(
            name=column_name,
            data_type=data_type,
            max_length=max_length,
            precision=precision,
            scale=scale,
            enum_values=enum_values
        )
        
        # Check for column attributes
        definition_upper = definition.upper()
        
        if 'NOT NULL' in definition_upper:
            column.is_nullable = False
        
        if 'AUTO_INCREMENT' in definition_upper:
            column.auto_increment = True
        
        # Check for default value
        default_match = re.search(r'DEFAULT\s+([^,\s]+)', definition, re.IGNORECASE)
        if default_match:
            column.default_value = default_match.group(1)
        
        return column
    
    def _parse_data_type(self, data_type_str: str) -> Tuple[str, Optional[int], Optional[int], Optional[int], List[str]]:
        """Parse data type string and extract type, length, precision, scale."""
        # Handle ENUM separately
        if data_type_str.upper().startswith('ENUM'):
            enum_match = re.match(r'ENUM\s*\(([^)]+)\)', data_type_str, re.IGNORECASE)
            if enum_match:
                enum_values = [val.strip().strip("'\"") for val in enum_match.group(1).split(',')]
                return 'ENUM', None, None, None, enum_values
        
        # Handle other data types with parameters
        type_match = re.match(r'(\w+)(?:\(([^)]+)\))?', data_type_str)
        if not type_match:
            return data_type_str, None, None, None, []
        
        base_type = type_match.group(1).upper()
        params = type_match.group(2)
        
        max_length = None
        precision = None
        scale = None
        
        if params:
            if ',' in params:
                # DECIMAL(10,2) format
                parts = [p.strip() for p in params.split(',')]
                if len(parts) >= 2:
                    precision = int(parts[0]) if parts[0].isdigit() else None
                    scale = int(parts[1]) if parts[1].isdigit() else None
            else:
                # VARCHAR(255) format
                if params.isdigit():
                    max_length = int(params)
        
        return base_type, max_length, precision, scale, []
    
    def _parse_inline_constraints(self, definition: str, column_name: str) -> List[Constraint]:
        """Parse inline constraints from column definition."""
        constraints = []
        definition_upper = definition.upper()
        
        if 'PRIMARY KEY' in definition_upper:
            constraints.append(Constraint(
                name=None,
                constraint_type=ConstraintType.PRIMARY_KEY,
                columns=[column_name]
            ))
        
        if 'UNIQUE' in definition_upper:
            constraints.append(Constraint(
                name=None,
                constraint_type=ConstraintType.UNIQUE,
                columns=[column_name]
            ))
        
        # Handle REFERENCES (foreign key)
        fk_match = re.search(r'REFERENCES\s+(\w+)\s*\((\w+)\)', definition, re.IGNORECASE)
        if fk_match:
            ref_table = fk_match.group(1)
            ref_column = fk_match.group(2)
            constraints.append(Constraint(
                name=None,
                constraint_type=ConstraintType.FOREIGN_KEY,
                columns=[column_name],
                referenced_table=ref_table,
                referenced_columns=[ref_column]
            ))
        
        return constraints
    
    def _parse_constraint(self, definition: str) -> Optional[Constraint]:
        """Parse a standalone constraint definition."""
        definition = definition.strip()
        definition_upper = definition.upper()
        
        # Primary Key
        pk_match = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', definition, re.IGNORECASE)
        if pk_match:
            columns = [col.strip() for col in pk_match.group(1).split(',')]
            return Constraint(
                name=None,
                constraint_type=ConstraintType.PRIMARY_KEY,
                columns=columns
            )
        
        # Foreign Key
        fk_match = re.search(r'FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\)', definition, re.IGNORECASE)
        if fk_match:
            columns = [col.strip() for col in fk_match.group(1).split(',')]
            ref_table = fk_match.group(2)
            ref_columns = [col.strip() for col in fk_match.group(3).split(',')]
            return Constraint(
                name=None,
                constraint_type=ConstraintType.FOREIGN_KEY,
                columns=columns,
                referenced_table=ref_table,
                referenced_columns=ref_columns
            )
        
        # Unique
        unique_match = re.search(r'UNIQUE\s*\(([^)]+)\)', definition, re.IGNORECASE)
        if unique_match:
            columns = [col.strip() for col in unique_match.group(1).split(',')]
            return Constraint(
                name=None,
                constraint_type=ConstraintType.UNIQUE,
                columns=columns
            )
        
        return None


def parse_ddl_file(file_path: str) -> Schema:
    """Parse a DDL file and return a Schema object."""
    parser = DDLParser()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        ddl_content = file.read()
    
    return parser.parse_ddl(ddl_content)


def parse_ddl_content(content: str) -> Schema:
    """Parse DDL content and return a Schema object."""
    parser = DDLParser()
    return parser.parse_ddl(content)


# Example usage and testing
if __name__ == "__main__":
    # Test with the company schema
    test_ddl = """
    CREATE TABLE Companies (
        company_id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(255) NOT NULL,
        industry VARCHAR(100)
    );

    CREATE TABLE Departments (
        department_id INT PRIMARY KEY AUTO_INCREMENT,
        company_id INT NOT NULL,
        name VARCHAR(100) NOT NULL,
        FOREIGN KEY (company_id) REFERENCES Companies(company_id)
    );
    """
    
    schema = parse_ddl_content(test_ddl)
    print(f"Parsed {len(schema.tables)} tables:")
    for table_name, table in schema.tables.items():
        print(f"  {table}")
        for col_name, col in table.columns.items():
            print(f"    {col}")
        for constraint in table.constraints:
            print(f"    {constraint}")