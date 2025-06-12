# tools.py
import json
from typing import Dict, List

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool, Tool

from gsheet import GoogleSheets

gs = GoogleSheets()

session_mapping: Dict[int, int] = {}


def _invert_mapping_if_needed(mapping: Dict[int, int]) -> Dict[int, int]:
    if not mapping:
        return mapping
    keys, vals = list(mapping.keys()), list(mapping.values())
    if min(keys) > 7 and max(vals) <= 7:
        return {v: k for k, v in mapping.items()}
    return mapping


def list_all_products(_: None = None) -> str:
    data = gs.get_all_models()
    return json.dumps(data, ensure_ascii=False)


class EmptyInput(BaseModel):
    pass


class MapInput(BaseModel):
    mapping: Dict[int, int] = Field(..., description="index→row")


def store_mapping(mapping: Dict[int, int]) -> str:
    corrected = _invert_mapping_if_needed(mapping)
    session_mapping.clear()
    session_mapping.update(corrected)
    return "stored"


class IdsInput(BaseModel):
    indices: List[int] = Field(
        ..., 
        description="Список чисел — либо индексов из предыдущего листинга, либо row-номеров прямо из таблицы"
    )


def get_fields_by_index(indices: List[int]) -> str:
    if session_mapping and all(i in session_mapping.values() for i in indices):
        rows = indices
    else:
        rows = [session_mapping[i] for i in indices if i in session_mapping]

    headers = list(gs.col_letter.keys())
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        result[str(row)] = gs.get_product_fields(row, headers)
    return json.dumps(result, ensure_ascii=False)


def get_tools() -> List[Tool]:
    list_all_products_tool = StructuredTool.from_function(
        name="list_all_products",
        func=list_all_products,
        args_schema=EmptyInput,
        description="{} → JSON {row:название модели}"
    )
    store_mapping_tool = StructuredTool.from_function(
        name="store_mapping",
        func=store_mapping,
        args_schema=MapInput,
        description='{"mapping":{"1":13,"2":14}} → сохранить index→row'
    )
    get_fields_tool = StructuredTool.from_function(
        name="get_fields_by_index",
        func=get_fields_by_index,
        args_schema=IdsInput,
        description='{"indices":[1,3]} → JSON с полями модели'
    )
    return [
        list_all_products_tool,
        store_mapping_tool,
        get_fields_tool,
    ]
