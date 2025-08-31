# source_probe.py
# -------------------------------------------------------------------
# Quick probe to see which NSE source works and has the columns we need.
#
# Usage examples:
#   python source_probe.py --date 2025-08-31
#   python source_probe.py --date 2024-06-20 --max_back 21
#   python source_probe.py --date 2025-08-31 --local_zip "C:\\path\\to\\fo31AUG2025bhav.csv.zip"
#
# Deps: requests pandas

import io
import os
import zipfile
import argparse
import datetime as dt
from typing import List, Tuple, Optional, Dict

import requests
import pandas as pd

HOSTS = ["https://archives.nseindia.com", "https://nsearchives.nseindia.com"]

def _month_abbr_upper(d: dt.date) -> str:
    return d.strftime("%b").upper()

def _ddmmmyyyy_upper(d: dt.date) -> str:
    return d.strftime("%d%b%Y").upper()

def _yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def log(msg: str):
    print(msg, flush=True)

def prefetch_cookie(session: requests.Session, headers: dict):
    try:
        session.get("https://www.nseindia.com/", headers=headers, timeout=15)
    except Exception:
        pass

# ----------------------
# Source A: Option Chain
# ----------------------
def test_optionchain(symbol: str = "NIFTY") -> Dict:
    """Free, today's chain only."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain,*/*",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate, br",
    }
    s = requests.Session()
    prefetch_cookie(s, headers)
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    try:
        log(f"[optionchain] GET {url}")
        r = s.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()

        exps = j.get("records", {}).get("expiryDates", [])
        data = j.get("records", {}).get("data", [])
        rows = []
        for row in data:
            K = row.get("strikePrice")
            ce = row.get("CE") or {}
            pe = row.get("PE") or {}
            C = ce.get("closePrice") or ce.get("lastPrice")
            P = pe.get("closePrice") or pe.get("lastPrice")
            if K and C and P:
                rows.append((float(K), float(C), float(P)))
        ok = len(rows) >= 6
        cols = ["strikePrice", "CE.closePrice/lastPrice", "PE.closePrice/lastPrice"]
        today = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date().isoformat()
        return {
            "source": "optionchain-json",
            "ok": ok,
            "rows": len(rows),
            "columns_present": cols,
            "note": f"today only; trade_date={today}",
        }
    except Exception as e:
        return {"source": "optionchain-json", "ok": False, "error": str(e)}

# ----------------------
# Source B: Old archives
# ----------------------
def old_archive_urls_for_date(d: dt.date) -> List[str]:
    y = d.strftime("%Y")
    m_abbr = _month_abbr_upper(d)
    ddmmmyyyy = _ddmmmyyyy_upper(d)
    path = f"/content/historical/DERIVATIVES/{y}/{m_abbr}/fo{ddmmmyyyy}bhav.csv.zip"
    return [host + path for host in HOSTS]

def fetch_bhavcopy_old(date: dt.date, max_back: int = 14) -> Tuple[Optional[pd.DataFrame], Optional[dt.date], Optional[str], Optional[str]]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate, br",
    }
    s = requests.Session()
    prefetch_cookie(s, headers)
    for i in range(max_back + 1):
        d_try = date - dt.timedelta(days=i)
        for url in old_archive_urls_for_date(d_try):
            try:
                log(f"[archives] GET {url}")
                r = s.get(url, headers=headers, timeout=30)
                if r.status_code == 200 and r.content:
                    try:
                        zf = zipfile.ZipFile(io.BytesIO(r.content))
                        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                        with zf.open(name) as f:
                            df = pd.read_csv(f)
                        return df, d_try, url, None
                    except zipfile.BadZipFile:
                        return None, d_try, url, "200 but not a ZIP (blocked/changed)"
                else:
                    log(f"  status {r.status_code}")
            except Exception as e:
                log(f"  error {e}")
    return None, None, None, "not found within backoff window"

# ----------------------
# Source C: UDiFF paths
# ----------------------
def udiff_urls_for_date(d: dt.date) -> List[str]:
    ymd = _yyyymmdd(d)
    candidates = [
        f"/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip",
        f"/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.gz",
    ]
    return [host + path for host in HOSTS for path in candidates]

def fetch_bhavcopy_udiff(date: dt.date, max_back: int = 3) -> Tuple[Optional[pd.DataFrame], Optional[dt.date], Optional[str], Optional[str]]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate, br",
    }
    s = requests.Session()
    prefetch_cookie(s, headers)
    import gzip
    for i in range(max_back + 1):
        d_try = date - dt.timedelta(days=i)
        for url in udiff_urls_for_date(d_try):
            try:
                log(f"[udiff] GET {url}")
                r = s.get(url, headers=headers, timeout=30)
                if r.status_code == 200 and r.content:
                    raw = io.BytesIO(r.content)
                    try:
                        if url.endswith(".zip"):
                            zf = zipfile.ZipFile(raw)
                            name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                            with zf.open(name) as f:
                                df = pd.read_csv(f)
                        else:
                            with gzip.GzipFile(fileobj=raw) as gz:
                                df = pd.read_csv(gz)
                        return df, d_try, url, None
                    except Exception as e:
                        return None, d_try, url, f"parse error: {e}"
                else:
                    log(f"  status {r.status_code}")
            except Exception as e:
                log(f"  error {e}")
    return None, None, None, "not found/blocked"

# ----------------------
# Local ZIP
# ----------------------
def load_local_zip(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path, "r") as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(name) as f:
            return pd.read_csv(f)

# ----------------------
# Check bhavcopy columns and ability to build a chain
# ----------------------
def check_bhavcopy_chain(df: pd.DataFrame, trade_date: dt.date, symbol: str = "NIFTY") -> Dict:
    d = df.copy()
    d.columns = [c.upper() for c in d.columns]

    def pick(*names):
        for nm in names:
            if nm in d.columns:
                return nm
        return None

    instr = pick("INSTRUMENT","INSTTYPE","INSTR")
    sym   = pick("SYMBOL","SYMB")
    exp   = pick("EXPIRY_DT","EXPIRY","EXPIRYDT","EXPDATE")
    strike= pick("STRIKE_PR","STRIKE","STRIKEPRICE")
    otyp  = pick("OPTION_TYP","OPTION_TYPE","OPTIONTYPE","OPT_TYPE")
    price = pick("SETTLE_PR","CLOSE","SETTLEPRICE","SETLPRC","SETTEL_PR")

    missing = [nm for nm,v in [("INSTRUMENT",instr),("SYMBOL",sym),("EXPIRY_DT",exp),
                               ("STRIKE_PR",strike),("OPTION_TYP",otyp)] if v is None]
    if missing:
        return {"ok": False, "reason": f"missing columns: {missing}", "columns": list(d.columns)}

    d = d[(d[instr].astype(str).str.upper().str.contains("OPT")) & (d[sym].astype(str).str.upper() == symbol)]
    if d.empty:
        return {"ok": False, "reason": f"no {symbol} OPT* rows", "columns": list(df.columns)}

    d["__EXP__"] = pd.to_datetime(d[exp], dayfirst=True, errors="coerce").dt.date
    expiries = sorted([e for e in d["__EXP__"].dropna().unique().tolist() if e >= trade_date])
    if not expiries:
        expiries = sorted([e for e in d["__EXP__"].dropna().unique().tolist()])
        if not expiries:
            return {"ok": False, "reason": "no valid expiry dates", "columns": list(df.columns)}
    expiry = expiries[0]
    de = d[d["__EXP__"] == expiry].copy()

    de["__STRIKE__"] = pd.to_numeric(de[strike], errors="coerce")
    if price is None:
        return {"ok": False, "reason": "no price column among [SETTLE_PR,CLOSE,...]", "columns": list(df.columns)}
    de["__PRICE__"]  = pd.to_numeric(de[price], errors="coerce")
    de = de.dropna(subset=["__STRIKE__","__PRICE__",otyp])

    calls = de[de[otyp].astype(str).str.upper().isin(["CE","C"])][["__STRIKE__","__PRICE__"]].rename(
        columns={"__STRIKE__":"strike","__PRICE__":"call_price"})
    puts  = de[de[otyp].astype(str).str.upper().isin(["PE","P"])][["__STRIKE__","__PRICE__"]].rename(
        columns={"__STRIKE__":"strike","__PRICE__":"put_price"})
    chain = pd.merge(calls, puts, on="strike", how="inner")
    chain = chain[(chain["call_price"]>0) & (chain["put_price"]>0)].drop_duplicates("strike").sort_values("strike")

    return {
        "ok": len(chain) >= 6,
        "reason": None if len(chain)>=6 else f"too few usable CE/PE pairs ({len(chain)})",
        "expiry": expiry,
        "rows": len(chain),
        "price_col": price,
        "columns": list(df.columns),
        "chain_head": chain.head(5).to_dict(orient="records")
    }

# ----------------------
# MAIN
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="Probe NSE sources for NIFTY option-chain feasibility")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD (used for archives/udiff)")
    ap.add_argument("--symbol", default="NIFTY", help="OPTIDX symbol (default NIFTY)")
    ap.add_argument("--max_back", type=int, default=14, help="Backoff days for archives/udiff")
    ap.add_argument("--local_zip", type=str, default=None, help="Path to a local bhavcopy ZIP to test")
    args = ap.parse_args()

    target_date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()

    print("\n=== Testing: Option Chain (today only) ===")
    res_opt = test_optionchain(args.symbol)
    print(res_opt)

    print("\n=== Testing: Old Archives (pre-UDiFF) ===")
    df_a, d_used_a, url_a, err_a = fetch_bhavcopy_old(target_date, max_back=args.max_back)
    if df_a is None:
        print({"source":"archives","ok":False,"error":err_a})
    else:
        print(f"  Got file for {d_used_a} from {url_a}")
        print("  Columns:", list(df_a.columns)[:12], "...")
        chk = check_bhavcopy_chain(df_a, d_used_a, args.symbol)
        print("  Chain check:", chk)

    print("\n=== Testing: UDiFF paths (post-2024) ===")
    df_u, d_used_u, url_u, err_u = fetch_bhavcopy_udiff(target_date, max_back=3)
    if df_u is None:
        print({"source":"udiff","ok":False,"error":err_u})
    else:
        print(f"  Got file for {d_used_u} from {url_u}")
        print("  Columns:", list(df_u.columns)[:12], "...")
        chk = check_bhavcopy_chain(df_u, d_used_u, args.symbol)
        print("  Chain check:", chk)

    if args.local_zip:
        print("\n=== Testing: Local ZIP ===")
        try:
            df_l = load_local_zip(args.local_zip)
            print("  Columns:", list(df_l.columns)[:12], "...")
            chk = check_bhavcopy_chain(df_l, target_date, args.symbol)
            print("  Chain check:", chk)
        except Exception as e:
            print({"source":"local_zip","ok":False,"error":str(e)})

if __name__ == "__main__":
    main()
