# nifty_bl_range_eod_nse_fixed.py
# --------------------------------
# NIFTY Range/Breakout via Breeden–Litzenberger (free data) — robust version
#
# What’s new vs your last script:
# - Sources: optionchain (today, free JSON), archives (old path pre-UDiFF), udiff (new file names),
#   plus --local-zip for manual bhavcopy files.
# - Better errors/logs; same BL math/plots.
#
# Usage:
#   python nifty_bl_range_eod_nse_fixed.py --date 2025-08-31 --source optionchain
#   python nifty_bl_range_eod_nse_fixed.py --date 2024-06-20 --source archives
#   python nifty_bl_range_eod_nse_fixed.py --date 2025-08-20 --local-zip "C:\\fo20AUG2025bhav.csv.zip"
#
# Deps: requests pandas numpy matplotlib

import io
import os
import math
import zipfile
import argparse
import datetime as dt
from typing import Tuple, Dict, Optional, List

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# Black–Scholes helper bits
# ===========================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_call_from_forward(F0: float, K: float, T: float, sigma: float, D: float) -> float:
    if sigma <= 1e-8 or T <= 0:
        return max(D * (F0 - K), 0.0)
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F0 / K) / vol_sqrt_t) + 0.5 * vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return D * (F0 * _norm_cdf(d1) - K * _norm_cdf(d2))

def implied_vol_from_call(F0: float, K: float, T: float, D: float, C_obs: float,
                          tol: float = 1e-6, max_iter: int = 100) -> float:
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        C_mid = _bs_call_from_forward(F0, K, T, mid, D)
        if abs(C_mid - C_obs) < tol:
            return mid
        if C_mid > C_obs:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

# ===========================
# Utilities
# ===========================

HOSTS = ["https://archives.nseindia.com", "https://nsearchives.nseindia.com"]

def _month_abbr_upper(d: dt.date) -> str:
    return d.strftime("%b").upper()

def _ddmmmyyyy_upper(d: dt.date) -> str:
    return d.strftime("%d%b%Y").upper()

def _yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def _log(s: str):
    print(s, flush=True)

def _prefetch_cookie(session: requests.Session, headers: dict):
    try:
        session.get("https://www.nseindia.com/", headers=headers, timeout=15)
    except Exception:
        pass

# ===========================
# Source A: old FO archives (pre-UDiFF; mostly ≤ 2024-07-05)
# ===========================

def _old_archive_urls_for_date(d: dt.date) -> List[str]:
    y = d.strftime("%Y")
    m_abbr = _month_abbr_upper(d)
    ddmmmyyyy = _ddmmmyyyy_upper(d)
    # Classic derivatives path (often discontinued post-UDiFF)
    path = f"/content/historical/DERIVATIVES/{y}/{m_abbr}/fo{ddmmmyyyy}bhav.csv.zip"
    return [host + path for host in HOSTS]

def fetch_bhavcopy_old(date: dt.date, session: requests.Session, headers: dict,
                       max_back: int = 14) -> Tuple[pd.DataFrame, dt.date, str]:
    for i in range(max_back + 1):
        d_try = date - dt.timedelta(days=i)
        for url in _old_archive_urls_for_date(d_try):
            try:
                _log(f"  Trying: {url}")
                resp = session.get(url, headers=headers, timeout=30)
                if resp.status_code == 200 and resp.content:
                    try:
                        zf = zipfile.ZipFile(io.BytesIO(resp.content))
                        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                        with zf.open(csv_name) as f:
                            df = pd.read_csv(f)
                        return df, d_try, url
                    except zipfile.BadZipFile:
                        _log("    200 but not a ZIP (blocked/changed).")
                else:
                    _log(f"    Status: {resp.status_code}")
            except requests.RequestException as e:
                _log(f"    Request failed: {e}")
    raise RuntimeError("Old archive pattern failed within backoff window.")

# ===========================
# Source B: UDiFF file names (may be public for CM; FO often restricted)
# ===========================

def _udiff_urls_for_date(d: dt.date) -> List[str]:
    # Try both .zip and .csv.gz under /content/fo/ with PascalCase name
    ymd = _yyyymmdd(d)
    candidates = [
        f"/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip",
        f"/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.gz",
    ]
    urls = []
    for host in HOSTS:
        for path in candidates:
            urls.append(host + path)
    return urls

def fetch_bhavcopy_udiff(date: dt.date, session: requests.Session, headers: dict,
                         max_back: int = 3) -> Tuple[pd.DataFrame, dt.date, str]:
    for i in range(max_back + 1):
        d_try = date - dt.timedelta(days=i)
        for url in _udiff_urls_for_date(d_try):
            try:
                _log(f"  Trying UDiFF: {url}")
                resp = session.get(url, headers=headers, timeout=30)
                if resp.status_code == 200 and resp.content:
                    raw = io.BytesIO(resp.content)
                    # Handle zip or gz
                    if url.endswith(".zip"):
                        zf = zipfile.ZipFile(raw)
                        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                        with zf.open(csv_name) as f:
                            df = pd.read_csv(f)
                    else:
                        # csv.gz
                        import gzip
                        with gzip.GzipFile(fileobj=raw) as gz:
                            df = pd.read_csv(gz)
                    return df, d_try, url
                else:
                    _log(f"    Status: {resp.status_code}")
            except Exception as e:
                _log(f"    UDiFF fetch failed: {e}")
    raise RuntimeError("UDiFF pattern failed (may be restricted to members).")

# ===========================
# Source C: live Option Chain JSON (free; TODAY only)
# ===========================

def fetch_optionchain_today(symbol: str = "NIFTY") -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Pulls NSE 'option-chain-indices' JSON for the given symbol.
    Returns a merged CE/PE dataframe (strike, call_price, put_price) and meta with
    trade_date (today IST), expiry, and T in years.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain,*/*",
        "Referer": f"https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate, br",
    }
    s = requests.Session()
    _prefetch_cookie(s, headers)
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    _log(f"  Hitting: {url}")
    r = s.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    j = r.json()

    # Pick nearest expiry (first in list usually)
    exps = j.get("records", {}).get("expiryDates", [])
    if not exps:
        raise ValueError("No expiryDates in option chain response.")
    expiry_str = exps[0]  # e.g., '28-Aug-2025'
    try:
        expiry_dt = dt.datetime.strptime(expiry_str, "%d-%b-%Y").date()
    except Exception:
        # Fallback: try other common format
        expiry_dt = pd.to_datetime(expiry_str, errors="coerce").date()

    data = j.get("records", {}).get("data", [])
    rows = []
    for row in data:
        K = row.get("strikePrice")
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}
        # prefer closePrice; fallback to lastPrice
        C = ce.get("closePrice") or ce.get("lastPrice")
        P = pe.get("closePrice") or pe.get("lastPrice")
        if K is None or C in (None, 0, "", " - ") or P in (None, 0, "", " - "):
            continue
        rows.append((float(K), float(C), float(P)))

    if not rows:
        raise ValueError("No usable CE/PE pairs from option chain.")

    chain = pd.DataFrame(rows, columns=["strike", "call_price", "put_price"]).drop_duplicates("strike")
    chain = chain.sort_values("strike").reset_index(drop=True)

    # Time to expiry in years (rough; intraday)
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date()
    T_days = max((expiry_dt - today).days, 1)  # floor at 1 day
    T = T_days / 365.0

    meta = {
        "trade_date": today.isoformat(),
        "expiry": expiry_dt.isoformat(),
        "T": T,
        "source": "optionchain-json",
    }
    return chain, meta

# ===========================
# Local ZIP parser
# ===========================

def load_local_bhavcopy_zip(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"--local-zip not found: {path}")
    with zipfile.ZipFile(path, 'r') as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    return df

# ===========================
# FO bhavcopy → NIFTY chain builder
# (handles classic columns; UDiFF columns vary by release – best effort)
# ===========================

def build_nifty_chain_from_bhavcopy(df: pd.DataFrame, trade_date: dt.date,
                                    symbol: str = "NIFTY") -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = df.copy()
    d.columns = [c.upper() for c in d.columns]

    # Column harmonization (classic files)
    # Required logical fields:
    #   INSTRUMENT ('OPTIDX'), SYMBOL ('NIFTY'), EXPIRY_DT, STRIKE_PR, OPTION_TYP ('CE'/'PE'),
    #   price column: SETTLE_PR or CLOSE
    # Try to detect UDiFF-like alternative names if present.
    colmap = {}
    def pick(*names):
        for nm in names:
            if nm in d.columns:
                return nm
        return None

    instr = pick("INSTRUMENT", "INSTTYPE", "INSTR")
    sym   = pick("SYMBOL", "SYMB")
    exp   = pick("EXPIRY_DT", "EXPIRY", "EXPIRYDT", "EXPDATE")
    strike= pick("STRIKE_PR", "STRIKE", "STRIKEPRICE")
    otyp  = pick("OPTION_TYP", "OPTION_TYPE", "OPTIONTYPE", "OPT_TYPE")
    price = pick("SETTLE_PR", "CLOSE", "SETTEL_PR", "SETTLEPRICE", "SETLPRC")

    missing = [nm for nm,v in [("INSTRUMENT",instr),("SYMBOL",sym),("EXPIRY_DT",exp),
                               ("STRIKE_PR",strike),("OPTION_TYP",otyp)] if v is None]
    if missing:
        raise ValueError(f"Bhavcopy missing columns: {missing}")

    if price is None:
        raise ValueError("Bhavcopy has no price column among [SETTLE_PR, CLOSE, ...].")

    # Filter to NIFTY index options
    d = d[(d[instr].astype(str).str.upper().str.contains("OPT")) & (d[sym].astype(str).str.upper() == symbol)]
    if d.empty:
        raise ValueError(f"No {symbol} OPT* rows in this file.")

    # Parse expiry and choose nearest >= trade_date
    d["__EXP__"] = pd.to_datetime(d[exp], dayfirst=True, errors="coerce").dt.date
    expiries = sorted([e for e in d["__EXP__"].dropna().unique().tolist() if e >= trade_date])
    if not expiries:
        # fallback: choose min expiry available
        expiries = sorted([e for e in d["__EXP__"].dropna().unique().tolist()])
        if not expiries:
            raise ValueError("No valid expiry dates in file.")
    expiry_chosen = expiries[0]
    de = d[d["__EXP__"] == expiry_chosen].copy()

    # Clean numeric fields
    de["__STRIKE__"] = pd.to_numeric(de[strike], errors="coerce")
    de["__PRICE__"]  = pd.to_numeric(de[price], errors="coerce")
    de = de.dropna(subset=["__STRIKE__", "__PRICE__", otyp])

    calls = de[de[otyp].astype(str).str.upper().isin(["CE","C"])][["__STRIKE__", "__PRICE__"]].rename(
        columns={"__STRIKE__":"strike","__PRICE__":"call_price"})
    puts  = de[de[otyp].astype(str).str.upper().isin(["PE","P"])][["__STRIKE__", "__PRICE__"]].rename(
        columns={"__STRIKE__":"strike","__PRICE__":"put_price"})

    chain = pd.merge(calls, puts, on="strike", how="inner")
    chain = chain[(chain["call_price"] > 0) & (chain["put_price"] > 0)]
    chain = chain.drop_duplicates(subset=["strike"]).sort_values("strike").reset_index(drop=True)

    if len(chain) < 6:
        raise ValueError(f"Chain too small after cleaning ({len(chain)} rows). Try another date/file.")

    # Time to expiry T (years)
    T_days = max((expiry_chosen - trade_date).days, 1)
    T = T_days / 365.0

    meta = {"T": T, "trade_date": trade_date.isoformat(), "expiry": expiry_chosen.isoformat(), "price_col": price}
    return chain, meta

# ===========================
# Parity, IVs, smile & BL CDF
# ===========================

def infer_forward_and_discount(df: pd.DataFrame, T: float) -> Tuple[float, float]:
    y = df["call_price"].values - df["put_price"].values
    X = df["strike"].values
    A = np.vstack([np.ones_like(X), X]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    D = -b
    if D <= 0 or D > 2:
        # Reasonable fallback (7% pa)
        D = math.exp(-0.07 * max(T, 1e-6))
        F0 = np.mean(X + y / D)
        return float(F0), float(D)
    F0 = a / D
    return float(F0), float(D)

def prices_to_ivs(df: pd.DataFrame, F0: float, D: float, T: float) -> pd.Series:
    ivs = []
    for K, C in zip(df["strike"].values, df["call_price"].values):
        try:
            iv = implied_vol_from_call(F0, K, T, D, C)
        except Exception:
            iv = np.nan
        ivs.append(iv)
    s = pd.Series(ivs, index=df.index, name="iv").astype(float)
    return s

def fit_iv_quadratic(df: pd.DataFrame, F0: float) -> Tuple[np.ndarray, np.ndarray]:
    m = np.log(df["strike"].values / F0)
    y = df["iv"].values
    if np.sum(np.isfinite(y)) < 5:
        raise ValueError("Too few valid IVs to fit smile.")
    coeffs = np.polyfit(m[np.isfinite(y)], y[np.isfinite(y)], deg=2)
    return coeffs, m

def iv_from_poly(coeffs: np.ndarray, K: float, F0: float) -> float:
    m = math.log(K / F0)
    return max(1e-4, float(np.polyval(coeffs, m)))

def build_call_curve(coeffs: np.ndarray, F0: float, D: float, T: float, K_min: float, K_max: float, n: int = 1201) -> Tuple[np.ndarray, np.ndarray]:
    K_grid = np.linspace(K_min, K_max, n)
    C_grid = np.array([_bs_call_from_forward(F0, K, T, iv_from_poly(coeffs, K, F0), D) for K in K_grid])
    return K_grid, C_grid

def cdf_from_call_curve(K_grid: np.ndarray, C_grid: np.ndarray, D: float) -> np.ndarray:
    dK = K_grid[1] - K_grid[0]
    dC_dK = np.gradient(C_grid, dK)
    F = 1.0 + (1.0 / D) * dC_dK
    F = np.clip(F, 0.0, 1.0)
    F_monotone = np.maximum.accumulate(F)
    if F_monotone[-1] > 0:
        F_monotone = F_monotone / F_monotone[-1]
    return F_monotone

def choose_band_bounds(F0: float, D: float, coeffs: np.ndarray, T: float, k: float = 1.0) -> Tuple[float, float, float]:
    sigma_atm = float(np.polyval(coeffs, 0.0))
    b = k * sigma_atm * math.sqrt(T)
    S0_est = F0 * D
    L = S0_est * (1.0 - b)
    U = S0_est * (1.0 + b)
    return S0_est, L, U

def band_breakout_probabilities(K_grid: np.ndarray, F_cdf: np.ndarray, L: float, U: float):
    F_L = float(np.interp(L, K_grid, F_cdf))
    F_U = float(np.interp(U, K_grid, F_cdf))
    p_range = max(0.0, F_U - F_L)
    p_down = max(0.0, F_L)
    p_up = max(0.0, 1.0 - F_U)
    s = p_range + p_down + p_up
    if s > 0:
        p_range, p_down, p_up = p_range/s, p_down/s, p_up/s
    return {"p_range": p_range, "p_down": p_down, "p_up": p_up}

def final_prediction(p, thresh_range: float = 0.55) -> str:
    if p["p_range"] >= thresh_range:
        return f"Predict: RANGE (≥ band), confidence ~{p['p_range']:.1%}"
    if p["p_up"] >= p["p_down"]:
        return f"Predict: UPSIDE BREAKOUT (> U), prob ~{p['p_up']:.1%}"
    else:
        return f"Predict: DOWNSIDE BREAKOUT (< L), prob ~{p['p_down']:.1%}"

# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser(description="NIFTY Range/Breakout via BL using free NSE data (robust)")
    parser.add_argument("--date", required=True, help="Target date (YYYY-MM-DD). In optionchain mode, used only for display.")
    parser.add_argument("--symbol", default="NIFTY", help="OPTIDX symbol (default: NIFTY)")
    parser.add_argument("--k", type=float, default=1.0, help="Band width multiplier (×ATM implied move)")
    parser.add_argument("--source", choices=["auto","optionchain","archives","udiff"], default="auto",
                        help="Data source. 'optionchain' = today (free JSON), 'archives' = old path, 'udiff' = new file names, 'auto' = pick best.")
    parser.add_argument("--max-back", type=int, default=14, help="Backoff days for archives/udiff (default 14)")
    parser.add_argument("--local-zip", type=str, default=None, help="Path to a local bhavcopy ZIP (skips download)")
    args = parser.parse_args()

    target_date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    today_ist = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date()

    # 1) Acquire chain + meta
    chain = None
    meta = None
    src_used = None

    if args.local-zip:
        _log(f"Loading local ZIP: {args.local-zip}")
        df_raw = load_local_bhavcopy_zip(args.local-zip)
        chain, meta = build_nifty_chain_from_bhavcopy(df_raw, target_date, symbol=args.symbol)
        src_used = f"local:{os.path.basename(args.local-zip)}"
    else:
        # Build a session and common headers
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.nseindia.com/",
            "Connection": "keep-alive",
            "Accept-Encoding": "gzip, deflate, br",
        }
        s = requests.Session()
        _prefetch_cookie(s, headers)

        def try_archives():
            df_raw, used_date, src = fetch_bhavcopy_old(target_date, s, headers, max_back=args.max-back)
            ch, me = build_nifty_chain_from_bhavcopy(df_raw, used_date, symbol=args.symbol)
            return ch, me, src

        def try_udiff():
            df_raw, used_date, src = fetch_bhavcopy_udiff(target_date, s, headers, max_back=args.max-back)
            ch, me = build_nifty_chain_from_bhavcopy(df_raw, used_date, symbol=args.symbol)
            return ch, me, src

        def try_optionchain():
            ch, me = fetch_optionchain_today(args.symbol)
            return ch, me, "optionchain-json"

        # Source picking
        if args.source == "optionchain":
            _log("Source: optionchain (today’s live JSON; not historical EOD).")
            chain, meta, src_used = try_optionchain()
            if target_date != today_ist:
                _log(f"  Note: requested {target_date} but using TODAY {meta['trade_date']} from live chain.")
        elif args.source == "archives":
            _log("Source: old archives (pre-UDiFF).")
            chain, meta, src_used = try_archives()
            if meta["trade_date"] != target_date:
                _log(f"  Using previous business day {meta['trade_date']}.")
        elif args.source == "udiff":
            _log("Source: UDiFF FO file names (may be restricted).")
            chain, meta, src_used = try_udiff()
            if meta["trade_date"] != target_date:
                _log(f"  Using previous business day {meta['trade_date']}.")
        else:
            # auto
            # If date ≤ 2024-07-05 → old archives first; else try UDiFF then fallback to optionchain.
            cutoff = dt.date(2024, 7, 5)
            if target_date <= cutoff:
                _log("Auto: trying old archives (pre-UDiFF) …")
                try:
                    chain, meta, src_used = try_archives()
                except Exception as e_arch:
                    _log(f"  Archives failed: {e_arch}. Trying optionchain as fallback …")
                    chain, meta, src_used = try_optionchain()
            else:
                _log("Auto: trying UDiFF …")
                try:
                    chain, meta, src_used = try_udiff()
                except Exception as e_ud:
                    _log(f"  UDiFF failed: {e_ud}. Trying optionchain (today) …")
                    chain, meta, src_used = try_optionchain()
                    if target_date != today_ist:
                        _log(f"  Note: requested {target_date} but using TODAY {meta['trade_date']} from live chain.")

    # 2) Focus on central strikes; infer parity; IVs; smile
    T = float(meta["T"])
    F0_tmp, D_tmp = infer_forward_and_discount(chain, T)
    S0_tmp = F0_tmp * D_tmp
    mask = (chain["strike"] >= 0.85 * S0_tmp) & (chain["strike"] <= 1.15 * S0_tmp)
    chain_central = chain[mask].copy()
    if len(chain_central) < 8:
        chain_central = chain.copy()

    _log("Inferring forward/discount (put–call parity) …")
    F0, D = infer_forward_and_discount(chain_central, T)
    if not (0 < D < 2):
        raise ValueError(f"Bad discount factor inferred: {D}")

    _log("Converting prices to IVs and fitting smile …")
    chain_central["iv"] = prices_to_ivs(chain_central, F0, D, T)
    chain_central = chain_central.dropna(subset=["iv"])
    if len(chain_central) < 6:
        raise ValueError("Too few valid IVs after inversion. Try another date/source.")
    coeffs, _ = fit_iv_quadratic(chain_central, F0)

    _log("Building smooth call curve and BL CDF …")
    K_min, K_max = chain_central["strike"].min(), chain_central["strike"].max()
    K_grid, C_grid = build_call_curve(coeffs, F0, D, T, K_min, K_max, n=1201)
    F_cdf = cdf_from_call_curve(K_grid, C_grid, D)

    _log("Computing band probabilities and final prediction …")
    S0_est, L, U = choose_band_bounds(F0, D, coeffs, T, k=args.k)
    probs = band_breakout_probabilities(K_grid, F_cdf, L, U)
    pred = final_prediction(probs, thresh_range=0.55)

    # 3) Report
    print("\n=== Data source & dates ===")
    print(f"Source           : {src_used}")
    print(f"Trade date used  : {meta['trade_date']}")
    print(f"Expiry chosen    : {meta['expiry']}")
    print(f"Rows in chain    : {len(chain)} (central fit on {len(chain_central)})")

    print("\n=== Inferred market inputs ===")
    print(f"Discount factor D: {D:.6f}  (implied r ≈ {-(math.log(D)/T):.4%} annualized)")
    print(f"Forward F0       : {F0:.2f}")
    print(f"Spot estimate S0 : {S0_est:.2f}")

    sigma_atm = float(np.polyval(coeffs, 0.0))
    print(f"ATM implied vol  : {sigma_atm:.2%} for T = {T:.5f} years")

    print("\n=== Band (k × ATM move; k={}) ===".format(args.k))
    print(f"L ≈ {L:.2f}   U ≈ {U:.2f}")

    print("\n=== Probabilities (risk-neutral) ===")
    print(f"P_range  (L ≤ S_T ≤ U): {probs['p_range']:.2%}")
    print(f"P_up     (S_T > U)    : {probs['p_up']:.2%}")
    print(f"P_down   (S_T < L)    : {probs['p_down']:.2%}")

    print("\n=== Final model output ===")
    print(pred)

    # 4) Visuals & exports
    os.makedirs("out_nifty_bl", exist_ok=True)

    # Smile: raw IVs vs fitted IVs (x = log-moneyness)
    plt.figure()
    m_vals = np.log(chain_central["strike"].values / F0)
    plt.scatter(m_vals, chain_central["iv"].values, label="Raw IVs")
    m_dense = np.linspace(m_vals.min(), m_vals.max(), 300)
    iv_fit = np.polyval(coeffs, m_dense)
    plt.plot(m_dense, iv_fit, label="Fitted IV (quadratic)")
    plt.xlabel("log-moneyness m = ln(K/F0)")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.title(f"{args.symbol} IV Smile — {meta['trade_date']} → {meta['expiry']}")
    plt.tight_layout()
    smile_path = os.path.join("out_nifty_bl", f"{args.symbol}_smile_{meta['trade_date']}_{meta['expiry']}.png")
    plt.savefig(smile_path, dpi=160)
    plt.show()

    # CDF with band
    plt.figure()
    plt.plot(K_grid, F_cdf, label="Risk-neutral CDF via BL")
    plt.axvline(L, linestyle="--", label="L (band lower)")
    plt.axvline(U, linestyle="--", label="U (band upper)")
    plt.xlabel("Terminal Price S_T (≈ strike K)")
    plt.ylabel("CDF  F_Q(S_T ≤ K)")
    plt.legend()
    plt.title(f"{args.symbol} BL CDF & Range Band — {meta['trade_date']} → {meta['expiry']}")
    plt.tight_layout()
    cdf_path = os.path.join("out_nifty_bl", f"{args.symbol}_cdf_{meta['trade_date']}_{meta['expiry']}.png")
    plt.savefig(cdf_path, dpi=160)
    plt.show()

    # Save smooth curve & CDF
    out = pd.DataFrame({"K": K_grid, "C_smooth": C_grid, "F_cdf": F_cdf})
    csv_path = os.path.join("out_nifty_bl", f"{args.symbol}_call_curve_and_cdf_{meta['trade_date']}_{meta['expiry']}.csv")
    out.to_csv(csv_path, index=False)

    print("\nSaved:")
    print(" ", smile_path)
    print(" ", cdf_path)
    print(" ", csv_path)

if __name__ == "__main__":
    main()
