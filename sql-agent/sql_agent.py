import sqlite3
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import re


@dataclass
class TableSchema:
    name: str
    columns: List[Dict[str, str]]
    sample_data: pd.DataFrame


class SQLAgent:
    def __init__(self, db_path: str, llm):
        """
        Initialize the SQL Agent with a database connection.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.tables = self._load_database_schema()
        self.llm = llm

    def _load_database_schema(self) -> Dict[str, TableSchema]:
        """
        Load the schema and sample data for all tables in the database.

        Returns:
            Dictionary mapping table names to their schemas and sample data
        """
        cursor = self.conn.cursor()

        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {}

        for (table_name,) in cursor.fetchall():
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = []
            for col in cursor.fetchall():
                columns.append({
                    "name": col[1],
                    "type": col[2],
                    "nullable": not col[3],
                    "primary_key": bool(col[5])
                })

            # Get sample data (first 5 rows)
            sample_data = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", self.conn)

            tables[table_name] = TableSchema(
                name=table_name,
                columns=columns,
                sample_data=sample_data
            )

        return tables

    def _generate_schema_context(self) -> str:
        """
        Generate a natural language description of the database schema.

        Returns:
            String containing the schema description
        """
        context = "Database Schema:\n\n"

        for table_name, schema in self.tables.items():
            context += f"Table: {table_name}\n"
            context += "Columns:\n"

            for col in schema.columns:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                pk = " (PRIMARY KEY)" if col["primary_key"] else ""
                context += f"- {col['name']}: {col['type']} {nullable}{pk}\n"

            context += "\nSample data:\n"
            context += schema.sample_data.to_string() + "\n\n"

        return context

    def _generate_prompt(self, query: str) -> str:
        """
        Generate a prompt for the LLM including schema context and query.

        Args:
            query: Natural language query

        Returns:
            Complete prompt string
        """
        schema_context = self._generate_schema_context()

        prompt = f"""Given the following database schema and sample data:

{schema_context}

Convert this question to SQL: "{query}"

Rules:
1. Use only tables and columns that exist in the schema
2. Return a valid SQL query that will answer the question
3. Use proper SQL syntax and formatting
4. Consider the sample data when writing the query
5. Use appropriate JOIN conditions based on the schema
6. Add comments to explain complex parts of the query

SQL Query:

Only return SQL query in the response without any formatting"""

        return prompt

    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the generated SQL query.

        Args:
            sql: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check SQL syntax
            cursor = self.conn.cursor()
            cursor.execute("EXPLAIN QUERY PLAN " + sql)
            return True, None
        except sqlite3.Error as e:
            return False, str(e)

    def _clean_sql(self, sql: str) -> str:
        """
        Clean and format the generated SQL query.

        Args:
            sql: Raw SQL query

        Returns:
            Cleaned and formatted SQL query
        """
        # Remove extra whitespace
        sql = " ".join(sql.split())

        # Add proper line breaks and indentation
        keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]
        for keyword in keywords:
            sql = sql.replace(keyword, f"\n{keyword}")

        # Add indentation
        lines = sql.split("\n")
        formatted_lines = []
        for line in lines:
            if any(line.strip().startswith(keyword) for keyword in keywords):
                formatted_lines.append(line)
            else:
                formatted_lines.append("    " + line)

        return "\n".join(formatted_lines).strip()

    def generate_sql(self, query: str) -> Dict[str, str]:
        """
        Convert a natural language query to SQL.

        Args:
            query: Natural language query

        Returns:
            Dictionary containing the generated SQL and any relevant metadata
        """
        # Here you would typically call your LLM with the prompt
        # For this example, we'll use a placeholder that demonstrates the structure
        prompt = self._generate_prompt(query)

        sql = self.llm.invoke(prompt)
        sql_query = sql.content

        # Validate and clean the SQL
        is_valid, error = self._validate_sql(sql_query)
        if not is_valid:
            return {
                "success": False,
                "error": error,
                "sql": None
            }

        cleaned_sql = self._clean_sql(sql_query)

        return {
            "success": True,
            "sql": cleaned_sql,
            "tables_used": self._extract_tables_used(cleaned_sql),
            "explanation": self._generate_query_explanation(cleaned_sql)
        }

    def _extract_tables_used(self, sql: str) -> List[str]:
        """Extract table names used in the SQL query."""
        tables = []
        for table in self.tables.keys():
            if re.search(r'\b' + table + r'\b', sql):
                tables.append(table)
        return tables

    def _generate_query_explanation(self, sql: str) -> str:
        """Generate a natural language explanation of the SQL query."""
        # This would typically use an LLM to generate an explanation
        # For now, we'll return a simple placeholder
        return "This query retrieves data from the specified tables with the given conditions."

    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute the generated SQL query and return results.

        Args:
            sql: SQL query to execute

        Returns:
            DataFrame containing query results
        """
        try:
            return pd.read_sql(sql, self.conn)
        except sqlite3.Error as e:
            raise Exception(f"Error executing query: {str(e)}")

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
