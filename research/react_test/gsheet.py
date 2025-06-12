import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.dirname(os.path.abspath(__file__))

class GoogleSheets:
    def __init__(self):
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            f"{basedir}/technologydynamicsasiccalc-76e05fa1a200.json"
        )
        self.client = gspread.authorize(creds)
        self.sheet_id = os.getenv("GOOGLE_SHEET_ID")

    def _open_sheet(self):
        return self.client.open_by_key(self.sheet_id)

    def get_products_name(self) -> dict[int, str]:
        sheet = self._open_sheet().sheet1
        values = sheet.get("A2:A10000")
        return {
            idx: row[0] 
            for idx, row in enumerate(values, start=2) 
            if row and row[0].strip()
        }

    def get_product_info(self, row_number: int) -> dict[str, str]:
        ws = self._open_sheet().sheet1
        headers = ws.get("A1:Z1")[0]
        row = ws.get(f"A{row_number}:Z{row_number}")[0]
        row += [""] * (len(headers) - len(row))
        return dict(zip(headers, row))

if __name__ == "__main__":
    g = GoogleSheets()
    #print(g.get_products_name())
    #print(g.get_product_info(90))