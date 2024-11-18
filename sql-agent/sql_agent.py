import sqlite3
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
import re
import sql_table_analyzer_utils
from models import ColumnInfo, TableSchema, QueryMetadata, SQLQueryResult, ColumnType, ColumnSemantics, ColumnStats


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

    def _analyze_column(self, df: pd.DataFrame, column_name: str, sql_type: str, sample_values_count: int = 5) -> ColumnInfo:
        """Analyze a single column for patterns and characteristics"""
        series = df[column_name]

        # Calculate basic stats
        stats = ColumnStats(
            distinct_count=series.nunique(),
            null_count=series.isnull().sum(),
            unique_ratio=series.nunique() / len(df) if len(df) > 0 else 0,
            null_ratio=series.isnull().sum() / len(df) if len(df) > 0 else 0,
            sample_values=series.head(sample_values_count).tolist()
        )

        # Calculate numeric stats if applicable
        if np.issubdtype(series.dtype, np.number):
            non_null = series.dropna()
            if len(non_null) > 0:
                stats.min_value = float(non_null.min())
                stats.max_value = float(non_null.max())
                stats.avg_value = float(non_null.mean())

        # Get most common values
        value_counts = series.value_counts().head(5)
        stats.common_values = [(val, count) for val, count in value_counts.items()]

        # Determine column type
        if column_name.lower().endswith('_id'):
            col_type = ColumnType.ID
        elif pd.api.types.is_numeric_dtype(series):
            col_type = ColumnType.NUMERIC
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = ColumnType.DATE
        elif pd.api.types.is_bool_dtype(series):
            col_type = ColumnType.BOOLEAN
        else:
            col_type = ColumnType.TEXT

        # Create semantics
        semantics = ColumnSemantics(
            is_identifier=col_type == ColumnType.ID,
            is_metric=pd.api.types.is_numeric_dtype(series),
            is_categorical=stats.unique_ratio < 0.1,
            is_temporal=pd.api.types.is_datetime64_any_dtype(series),
            is_descriptive=column_name.lower() in ['name', 'description', 'title']
        )

        return ColumnInfo(
            name=column_name,
            type=col_type,
            nullable=stats.null_count > 0,
            primary_key=stats.unique_ratio == 1.0 and stats.null_count == 0,
            stats=stats,
            semantics=semantics
        )

    def _load_database_schema(self, no_of_sample_data_rows=1000) -> Dict[str, TableSchema]:
        """
        Load the schema and sample data for all tables in the database.

        Returns:
            Dictionary mapping table names to their schemas and sample data
        """
        cursor = self.conn.cursor()

        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {}

        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

            for (table_name,) in cursor.fetchall():
                # Get sample data
                # df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {no_of_sample_data_rows}", self.conn)

                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = {}

                for col in cursor.fetchall():
                    column_name = col[1]
                    column_type = col[2]
                    columns[column_name] = ColumnInfo(
                        name=column_name,
                        type=column_type,
                        nullable=not col[3],
                        primary_key=bool(col[5])
                    )

                # for col in cursor.fetchall():
                #     coumn_name = col[1]
                #     column_data_type = col[2]
                #     col_info = self._analyze_column(df, coumn_name, column_data_type)
                #     columns[coumn_name] = col_info

                # Get foreign key information
                cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                for fk in cursor.fetchall():
                    if fk[3] in columns:
                        columns[fk[3]].foreign_key = (fk[2], fk[4])

                # Get strategic sample
                row_count = self._get_row_count(table_name)

                if row_count <= no_of_sample_data_rows:
                    sample_query = f"SELECT * FROM {table_name}"
                else:
                    sample_query = f"""
                            SELECT * FROM {table_name} 
                            WHERE ROWID IN (
                                SELECT ROWID FROM {table_name} 
                                WHERE ROWID IN (
                                    SELECT ABS(RANDOM() % {row_count}) + 1
                                    FROM (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5)
                                )
                            )
                            LIMIT 5
                            """

                sample_data = pd.read_sql(sample_query, self.conn)

                # Analyze column values
                for col_name in columns:
                    if col_name in sample_data.columns:
                        values, unique_ratio, null_ratio = sql_table_analyzer_utils.analyze_column_values(
                            sample_data, col_name
                        )
                        columns[col_name].sample_values = values
                        columns[col_name].unique_ratio = unique_ratio
                        columns[col_name].null_ratio = null_ratio

                tables[table_name] = TableSchema(
                    name=table_name,
                    columns=columns,
                    sample_data=sample_data,
                    row_count=row_count,
                    relationships={}
                )

            return tables

        except Exception as e:
            raise ValueError(f"Error loading database schema: {str(e)}")

    def _analyze_relationships(self) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
        """
        Analyze and store relationships between tables.

        Returns:
            Dictionary mapping table pairs to their relationships
        """
        relationships = defaultdict(lambda: defaultdict(list))

        try:
            # Add explicit relationships from foreign keys
            for table_name, table_schema in self.tables.items():
                for col_name, col_info in table_schema.columns.items():
                    if col_info.foreign_key:
                        referenced_table, referenced_col = col_info.foreign_key
                        relationships[table_name][referenced_table].append((col_name, referenced_col))
                        self.tables[table_name].relationships[referenced_table] = [(col_name, referenced_col)]

            # Detect implicit relationships
            for table1_name, table2_name in combinations(self.tables.keys(), 2):
                if table2_name not in relationships[table1_name]:
                    detected_relationships = sql_table_analyzer_utils.detect_relationships(
                        self.tables[table1_name].sample_data,
                        self.tables[table2_name].sample_data
                    )
                    if detected_relationships:
                        relationships[table1_name][table2_name].extend(detected_relationships)
                        self.tables[table1_name].relationships[table2_name] = detected_relationships

            return relationships

        except Exception as e:
            raise ValueError(f"Error analyzing relationships: {str(e)}")

    def _find_join_path(self, start_table: str, end_table: str) -> List[Tuple[str, str, str, str]]:
        """
        Find the shortest path to join two tables.

        Returns:
            List of (table1, column1, table2, column2) representing the join path
        """
        if start_table == end_table:
            return []

        visited = {start_table}
        queue = [(start_table, [])]

        while queue:
            current_table, path = queue.pop(0)

            for next_table, relationships in self.tables[current_table].relationships.items():
                if next_table not in visited:
                    if next_table == end_table:
                        # Found the target table
                        return path + [(current_table, relationships[0][0],
                                        next_table, relationships[0][1])]

                    visited.add(next_table)
                    queue.append((next_table, path + [(current_table, relationships[0][0],
                                                       next_table, relationships[0][1])]))

        return None  # No path found

    def _generate_join_conditions(self, required_tables: Set[str]) -> List[str]:
        """
        Generate optimal JOIN conditions for the required tables.

        Args:
            required_tables: Set of tables that need to be joined

        Returns:
            List of JOIN clauses
        """
        if len(required_tables) <= 1:
            return []

        # Find the optimal join order based on table sizes and relationships
        tables_list = list(required_tables)
        join_conditions = []

        # Start with the largest table as the base
        base_table = max(tables_list, key=lambda t: self.tables[t].row_count)
        remaining_tables = set(tables_list) - {base_table}

        while remaining_tables:
            best_next_table = None
            best_join_path = None

            for table in remaining_tables:
                join_path = self._find_join_path(base_table, table)
                if join_path is not None:
                    if best_next_table is None or len(join_path) < len(best_join_path):
                        best_next_table = table
                        best_join_path = join_path

            if best_next_table is None:
                # No direct join path found, try to find an intermediate table
                for intermediate in self.tables.keys():
                    if intermediate not in remaining_tables and intermediate != base_table:
                        path1 = self._find_join_path(base_table, intermediate)
                        if path1:
                            for table in remaining_tables:
                                path2 = self._find_join_path(intermediate, table)
                                if path2:
                                    best_next_table = table
                                    best_join_path = path1 + path2
                                    break
                            if best_next_table:
                                break

            if best_next_table and best_join_path:
                for t1, c1, t2, c2 in best_join_path:
                    join_conditions.append(f"LEFT JOIN {t2} ON {t1}.{c1} = {t2}.{c2}")
                remaining_tables.remove(best_next_table)
            else:
                # If no join path found, use CROSS JOIN as last resort
                next_table = remaining_tables.pop()
                join_conditions.append(f"CROSS JOIN {next_table}")

        return join_conditions

    def _generate_schema_context(self, query: str) -> str:
        """
        Generate a natural language description of the database schema.

        Returns:
            String containing the schema description
        """

        # Extract required tables
        required_tables = self._extract_required_tables(query)

        # Generate focused schema context only for relevant tables
        schema_context = "Relevant Tables for your query:\n\n"

        for table_name in required_tables:
            schema = self.tables[table_name]
            schema_context += f"Table: {table_name} ({schema.row_count} total rows)\n"
            schema_context += "Columns:\n"

            for col_name, col_info in schema.columns.items():
                schema_context += f"- {col_name}: {col_info.type}"
                if col_info.sample_values:
                    schema_context += f" (Sample values: {', '.join(map(str, col_info.sample_values))})"
                schema_context += "\n"

        # Add join information
        join_conditions = self._generate_join_conditions(required_tables)
        if join_conditions:
            schema_context += "\nSuggested Joins:\n"
            schema_context += "\n".join(join_conditions) + "\n"

        return schema_context

    def _generate_prompt(self, query: str, dialect: str = "SQLite") -> str:
        """
        Generate a prompt for the LLM including schema context and query.

        Args:
            query: Natural language query

        Returns:
            Complete prompt string
        """
        schema_context = self._generate_schema_context(query)

        prompt = f"""
You are a {dialect} expert.

Please help to generate a {dialect} query to answer the question. Your response should ONLY be based on the given context and must follow the rules.

Given the relevant database schema and sample data:

{schema_context}

Convert this question to SQL: "{query}"

Rules:
1. Use only the tables and columns shown above
2. Use the suggested joins when combining tables
3. Consider the sample values when writing conditions
4. Use appropriate aggregations based on the question
5. Add comments to explain complex parts
6. Consider table sizes when ordering joins
7. If the provided context is sufficient, please generate a valid query without any explanations for the question. The query should start with a comment containing the question being asked.
8. If the provided context is insufficient, please explain why it can't be generated.
9. Please use the most relevant table(s).
10. Please format the query before responding.
11. Please always respond with a valid well-formed JSON object with the following format
12. You MUST evaluate thoroughly the question against the relevant database schema and be creative about answering it for what selection of fields to query
13. If question contains a geographical region, broad category or anything generic, you MUST review the relevant database schema and adapt the result for that 
14. If relevant, you can include multiple filter conditions if they are applicable to the question

Only return SQL query in the response without any formatting. You MUST NOT skip over Rules.


SQL Query:"""

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

    def generate_sql(self, query: str) -> SQLQueryResult:
        """
        Convert a natural language query to SQL.

        Args:
            query: Natural language query

        Returns:
            Dictionary containing the generated SQL and any relevant metadata
        """
        start_time = datetime.now()

        try:
            required_tables = self._extract_required_tables(query)
            # print(required_tables)
            prompt = self._generate_prompt(query)
            print("DEBUG: Prompt set to: {}".format(prompt))

            sql = self.llm.invoke(prompt)
            sql_query = sql.content
            print("DEBUG: SQL query set to: {}".format(sql_query))

            llm_validated_sql_query = sql_query
            # llm_validated_sql_query = self._validate_sql_query_via_agent(sql_query)

            # Validate and extract metadata
            metadata = QueryMetadata(
                tables_used=list(required_tables),
                columns_used=sql_table_analyzer_utils.extract_columns_used(llm_validated_sql_query),
                joins_used=sql_table_analyzer_utils.extract_joins(llm_validated_sql_query),
                where_conditions=sql_table_analyzer_utils.extract_where_conditions(llm_validated_sql_query),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

            print("Metadata: \n" + str(metadata.model_dump()))

            return SQLQueryResult(
                success=True,
                sql=llm_validated_sql_query,
                metadata=metadata,
                explanation=self._generate_query_explanation(llm_validated_sql_query)
            )

        except Exception as e:
            print(e)
            return SQLQueryResult(
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _extract_tables_used(self, sql: str) -> List[str]:
        """Extract table names used in the SQL query."""
        tables = []
        for table in self.tables.keys():
            if re.search(r'\b' + table + r'\b', sql):
                tables.append(table)
        return tables

    def _extract_required_tables(self, query: str) -> Set[str]:
        """
        Extract tables that might be needed for the query based on context.

        Args:
            query: Natural language query

        Returns:
            Set of table names
        """
        required_tables = set()

        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()

        # Look for table names in the query
        for table_name in self.tables.keys():
            # Check for both plural and singular forms
            if table_name.lower() in query_lower or f"{table_name}s".lower() in query_lower:
                required_tables.add(table_name)

        # Look for column names in the query
        for table_name, schema in self.tables.items():
            for col_name in schema.columns.keys():
                if col_name.lower() in query_lower:
                    required_tables.add(table_name)

        return required_tables

    def _generate_query_explanation(self, sql: str) -> str:
        """Generate a natural language explanation of the SQL query."""
        # This would typically use an LLM to generate an explanation
        # For now, we'll return a simple placeholder
        return "This query retrieves data from the specified tables with the given conditions."

    def _validate_sql_query_via_agent(self, sql: str, dialect="sqllite"):

        validation_sql_prompt = f"""Double check the user's {dialect} query for common mistakes, including:
        - Only return SQL Query not anything else like ```sql ... ```
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates\
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins
        - Do not apply any LIMIT on results
        
        If there are any of the above mistakes, rewrite the query.
        If there are no mistakes, just reproduce the original query with no further commentary.
        
        Output the final SQL query only for the following SQL query: {sql}"""

        cleaned_sql_query = self.llm.invoke(validation_sql_prompt)
        return cleaned_sql_query.content

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

    def _get_row_count(self, table_name: str) -> int:
        """Get the total number of rows in a table."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        except Exception as e:
            raise ValueError(f"Error getting row count for table {table_name}: {str(e)}")

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
