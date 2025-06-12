import os, string, time, json, datetime as dt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from redis_cache import cache_rows, cache_row, KEY_TIMESTAMP, r

load_dotenv()

def sync():
    creds_path = os.path.join(
        os.path.dirname(__file__),
        "technologydynamicsasiccalc-76e05fa1a200.json"
    )
    creds   = ServiceAccountCredentials.from_json_keyfile_name(creds_path)
    client  = gspread.authorize(creds)
    sheet   = client.open_by_key(os.getenv("GOOGLE_SHEET_ID")).sheet1

    headers = sheet.get("A1:Z1")[0]
    cols    = {h: string.ascii_uppercase[i] for i, h in enumerate(headers)}

    models = sheet.get("A2:A10000")
    rows_dict = {idx + 2: row[0] for idx, row in enumerate(models) if row}
    cache_rows(rows_dict)

    ranges = [f"{cols[h]}2:{cols[h]}10000" for h in headers]
    columns = sheet.batch_get(ranges)

    for i, row_idx in enumerate(rows_dict.keys()):
        data = {headers[col]: columns[col][i][0] if columns[col][i] else ""
                for col in range(len(headers))}
        cache_row(row_idx, data)

    r.set(KEY_TIMESTAMP, dt.datetime.utcnow().isoformat())
    print("✔ Google Sheet synced → Redis :", len(rows_dict), "rows")

if __name__ == "__main__":
    while True:
        try:
            sync()
        except Exception as e:
            print("⚠️  sync error:", e)
        time.sleep(600)
