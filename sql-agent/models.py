from enum import Enum
from typing import Optional, Tuple, Any, List, Dict
from pydantic import BaseModel, Field, field_validator


class ColumnType(str, Enum):
    """Standardized column types for better analysis"""
    NUMERIC = "numeric"
    TEXT = "text"
    DATE = "date"
    BOOLEAN = "boolean"
    ID = "id"
    FOREIGN_KEY = "foreign_key"
    UNKNOWN = "unknown"


class ColumnSemantics(BaseModel):
    """Semantic information about a column"""
    is_identifier: bool = False
    is_metric: bool = False
    is_categorical: bool = False
    is_temporal: bool = False
    is_descriptive: bool = False
    common_patterns: List[str] = Field(default_factory=list)
    value_range: Optional[Tuple[Any, Any]] = None
    related_concepts: List[str] = Field(default_factory=list)


class ColumnStats(BaseModel):
    """Statistical information about a column"""
    distinct_count: int
    null_count: int
    unique_ratio: float
    null_ratio: float
    sample_values: List[Any]
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_value: Optional[float] = None
    common_values: List[Tuple[Any, int]] = Field(default_factory=list)

    @field_validator('unique_ratio', 'null_ratio')
    def validate_ratio(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Ratio must be between 0 and 1')
        return v

class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[Tuple[str, str]] = Field(default=None, description="(referenced_table, referenced_column)")
    sample_values: Optional[List[Any]] = None
    unique_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    null_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    stats: Optional[ColumnStats] = None
    semantics: Optional[ColumnSemantics] = None

    @field_validator('unique_ratio', 'null_ratio')
    def validate_ratio(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError('Ratio must be between 0 and 1')
        return v

    class Config:
        arbitrary_types_allowed = True


class TableSchema(BaseModel):
    name: str
    columns: Dict[str, ColumnInfo]
    sample_data: Any  # pandas DataFrame
    row_count: int = Field(ge=0)
    relationships: Dict[str, List[Tuple[str, str]]] = Field(
        default_factory=dict,
        description="Dictionary mapping related tables to list of (local_col, foreign_col) pairs"
    )

    class Config:
        arbitrary_types_allowed = True

    @field_validator('row_count')
    def validate_row_count(cls, v):
        if v < 0:
            raise ValueError('Row count must be non-negative')
        return v


class QueryMetadata(BaseModel):
    """Metadata about a generated SQL query"""
    tables_used: List[str]
    columns_used: List[str]
    estimated_rows: Optional[int] = None
    joins_used: List[str] = Field(default_factory=list)
    where_conditions: List[str] = Field(default_factory=list)
    execution_time: Optional[float] = None


class SQLQueryResult(BaseModel):
    """Result of SQL query generation"""
    success: bool
    sql: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[QueryMetadata] = None
    explanation: Optional[str] = None

    # @field_validator('sql')
    # def validate_sql(self, v, values):
    #     if values.get('success') and not v:
    #         raise ValueError('SQL must be provided when success is True')
    #     return v
    #
    # @field_validator('error')
    # def validate_error(self, v, values):
    #     if not values.get('success') and not v:
    #         raise ValueError('Error must be provided when success is False')
    #     return v
