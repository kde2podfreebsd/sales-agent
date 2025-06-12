import os, json, redis
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
r = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

KEY_ROWS      = "asic:rows"       
KEY_ROW_FMT   = "asic:row:{row}"   
KEY_TIMESTAMP = "asic:last_sync"   

def cache_rows(rows: Dict[int, str]):
    r.set(KEY_ROWS, json.dumps(rows, ensure_ascii=False))

def cache_row(row: int, data: Dict[str, Any]):
    r.set(KEY_ROW_FMT.format(row=row), json.dumps(data, ensure_ascii=False))

def load_rows() -> Dict[int, str]:
    raw = r.get(KEY_ROWS) or "{}"
    return json.loads(raw)

def load_row(row: int) -> Dict[str, Any]:
    raw = r.get(KEY_ROW_FMT.format(row=row)) or "{}"
    return json.loads(raw)
