import json
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool, Tool
from redis_cache import load_rows, load_row

session_mapping: Dict[int, int] = {}

class _Empty(BaseModel):  ...
class _MapIn(BaseModel):
    mapping: Dict[int, int] = Field(..., description="index→row")

class _IdsIn(BaseModel):
    indices: List[int]

def list_all_products(_: None = None) -> str:
    return json.dumps(load_rows(), ensure_ascii=False)

def store_mapping(mapping: Dict[int, int]) -> str:
    session_mapping.clear()
    session_mapping.update(mapping)
    return "stored"

def get_fields_by_index(indices: List[int]) -> str:
    rows = [session_mapping.get(i, i) for i in indices]
    res = {str(r): load_row(r) for r in rows}
    return json.dumps(res, ensure_ascii=False)

def catalog_tools() -> List[Tool]:
    return [
        StructuredTool.from_function("list_all_products", list_all_products,
            args_schema=_Empty, description="→ JSON {row:модель}"),
        StructuredTool.from_function("store_mapping", store_mapping,
            args_schema=_MapIn, description="Сохраняет соответствие index→row"),
        StructuredTool.from_function("get_fields_by_index", get_fields_by_index,
            args_schema=_IdsIn, description="indices → JSON полей модели"),
    ]
