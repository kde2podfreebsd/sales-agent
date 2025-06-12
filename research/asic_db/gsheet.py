import os
import re
import string
from functools import lru_cache
from typing import Dict, List

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

load_dotenv()
BASEDIR = os.path.dirname(os.path.abspath(__file__))


class GoogleSheets:

    def __init__(self) -> None:
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            f"{BASEDIR}/technologydynamicsasiccalc-76e05fa1a200.json"
        )
        self.client = gspread.authorize(creds)
        self.sheet_id = os.getenv("GOOGLE_SHEET_ID")
        self._sheet = self.client.open_by_key(self.sheet_id).sheet1

        headers = self._sheet.get("A1:Z1")[0]
        self.col_letter: Dict[str, str] = {
            name: string.ascii_uppercase[i] for i, name in enumerate(headers)
        }

    def get_product_fields(self, row: int, fields: List[str]) -> Dict[str, str]:
        letters = [self.col_letter[f] for f in fields]
        ranges = [f"{c}{row}:{c}{row}" for c in letters]
        values = self._sheet.batch_get(ranges)
        flat = [v[0][0] if v and v[0] else "" for v in values]
        return dict(zip(fields, flat))

    @lru_cache(maxsize=1)
    def _get_col_a(self) -> Dict[int, str]:
        data = self._sheet.get("A2:A10000")
        return {idx: row[0] for idx, row in enumerate(data, start=2) if row}
