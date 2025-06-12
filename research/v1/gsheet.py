# gsheet.py
import os
import string
from functools import lru_cache
from typing import Dict, List

import gspread
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials

load_dotenv()


class GoogleSheets:
    def __init__(self) -> None:
        creds_path = os.path.join(os.path.dirname(__file__), "technologydynamicsasiccalc-76e05fa1a200.json")
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path)
        self.client = gspread.authorize(creds)
        sheet_id = os.getenv("GOOGLE_SHEET_ID")
        self.sheet = self.client.open_by_key(sheet_id).sheet1

        headers = self.sheet.get("A1:Z1")[0]
        self.col_letter: Dict[str, str] = {
            name: string.ascii_uppercase[i] for i, name in enumerate(headers)
        }

    def get_product_fields(self, row: int, fields: List[str]) -> Dict[str, str]:
        ranges = [f"{self.col_letter[f]}{row}:{self.col_letter[f]}{row}" for f in fields]
        values = self.sheet.batch_get(ranges)
        flat = [col[0][0] if col and col[0] else "" for col in values]
        return dict(zip(fields, flat))

    @lru_cache(maxsize=1)
    def get_all_models(self) -> Dict[int, str]:
        data = self.sheet.get("A2:A10000")
        return {idx + 2: row[0] for idx, row in enumerate(data) if row}
