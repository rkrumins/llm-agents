import re
from typing import List, Tuple, Set
import pandas as pd


def analyze_column_values(df: pd.DataFrame, column_name: str) -> Tuple[List[any], float, float]:
    """Analyze column values for patterns and statistics."""
    values = df[column_name].tolist()
    unique_ratio = len(df[column_name].unique()) / len(df) if len(df) > 0 else 0
    null_ratio = df[column_name].isnull().sum() / len(df) if len(df) > 0 else 0
    return values, unique_ratio, null_ratio


def detect_relationships(table1: pd.DataFrame, table2: pd.DataFrame) -> List[Tuple[str, str]]:
    """Detect potential relationships between two tables based on content."""
    relationships = []
    for col1 in table1.columns:
        for col2 in table2.columns:

            # Check if column names suggest a relationship
            if (col1 == col2 or
                    f"{table2.name}_id" == col1 or
                    f"{table1.name}_id" == col2):
                # Check if values overlap
                common_values = set(str(table1[col1])) & set(str(table2[col2]))
                if common_values:
                    relationships.append((col1, col2))
    return relationships


def extract_columns_used(sql: str) -> List[str]:
    """Extract column names used in the SQL query."""
    # Simple regex to extract columns
    column_pattern = r'SELECT\s+(.*?)\s+FROM'
    match = re.search(column_pattern, sql, re.IGNORECASE | re.DOTALL)
    if match:
        columns = match.group(1).split(',')
        return [col.strip().split('.')[-1].split(' AS ')[0] for col in columns]
    return []


def extract_joins(sql: str) -> List[str]:
    """Extract JOIN clauses from the SQL query."""
    join_pattern = r'(LEFT|RIGHT|INNER|OUTER|CROSS)?\s*JOIN\s+(\w+)\s+ON\s+(.*?)(?=(?:LEFT|RIGHT|INNER|OUTER|CROSS)?\s*JOIN|\s*WHERE|\s*GROUP|\s*ORDER|\s*LIMIT|$)'
    return re.findall(join_pattern, sql, re.IGNORECASE | re.DOTALL)


def extract_where_conditions(sql: str) -> List[str]:
    """Extract WHERE conditions from the SQL query."""
    where_pattern = r'WHERE\s+(.*?)(?=\s+GROUP|\s+ORDER|\s+LIMIT|$)'
    match = re.search(where_pattern, sql, re.IGNORECASE | re.DOTALL)
    if match:
        conditions = match.group(1).split('AND')
        return [cond.strip() for cond in conditions]
    return []
