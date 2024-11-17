from typing import Optional, Tuple, Any, List, Dict
from pydantic import BaseModel, Field, field_validator


class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[Tuple[str, str]] = Field(default=None, description="(referenced_table, referenced_column)")
    sample_values: Optional[List[Any]] = None
    unique_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    null_ratio: Optional[float] = Field(default=None, ge=0, le=1)

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
