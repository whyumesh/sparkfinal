#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SlotMAX for Division 42. Same logic; 4 brands: GANATON, GANATON TOTAL, DUPHALAC, ELDICET.
Output: Report month (Feb) in template schema -> out_jan_div42.csv.

What it does:
1) Only 4 brands count: GANATON, GANATON TOTAL, DUPHALAC, ELDICET. All other brands ignored.
2) Only doctors who have ALL 4 brands (at least one non-zero Rx across Nov-Feb) are included.
3) Data restricted to 4 months: Nov, Dec, Jan (baseline) and Feb (report month).
4) Detect BrandN / Rx/MonthN slot pairs; for each (Doctor, MonthPeriod) and each slot N:
   pick the row with maximum Rx/MonthN (tie-break on Date desc, Filed Date desc if present),
   take the BrandN from that same row.
5) Aggregate to (Doctor, MonthPeriod, Brand) keeping max Rx across slots; keep only the 4 brands.
6) Pivot to product columns (GANATON, GANATON TOTAL, DUPHALAC, ELDICET).
7) Variance flags: Nov-Jan average total Rx vs February total Rx per doctor.
   Variance >25% = |Feb - avg(Nov,Dec,Jan)|/avg > 25%. Increasing/Decreasing at +/-25% threshold.
8) Output: only report month (Feb) rows, only doctors with all 4 brands. Keeps 0's, blanks, NaN. CSV in EXACT template order.

Use --report-month 2026-02 to set report month (Feb); baseline is then Nov-Dec-Jan. If omitted, defaults to 2026-02.
"""

import re
import csv
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------
# Division 42 config
# ---------------------------
PROJECT_DIR = Path(__file__).resolve().parent
ALLOWED_BRANDS = ["GANATON", "GANATON TOTAL", "DUPHALAC", "ELDICET"]
DEFAULT_INPUT = PROJECT_DIR / "DCR_RAW_STANDARDIZED_4div_2025-11-01_2026-02-28_Div42.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "out_jan_div42_tbm.csv"
DEFAULT_REPORT_MONTH = "2026-02"
DEFAULT_TEMPLATE = PROJECT_DIR / "Copy of Data Dump.csv"
DEFAULT_HIERARCHY = PROJECT_DIR / "hierarchy.csv"

# Map input CSV column names to template column names (so output has correct headers and data)
INPUT_TO_TEMPLATE_COLUMN_MAP = {
    "Full Name": "User: Full Name",
    "Account ID_18": "Account: ID_18",
    "Account Name": "Account: Account Name",
    "Patch Name": "Patch: Patch Name",
    "Assignment": "Assignment: Assignment",
    "Frequency": "Assignment: Frequency",
    "Status": "Assignment: Status",
}


# ---------------------------
# Helpers
# ---------------------------

def clean_columns(cols: List[str]) -> List[str]:
    return [str(c).replace("\r", "").strip() for c in cols]


def resolve_col(existing_cols: List[str], desired: str) -> Optional[str]:
    """Case-insensitive match of a desired column name against existing columns."""
    desired_clean = desired.replace("\r", "").strip().lower()
    for c in existing_cols:
        if str(c).replace("\r", "").strip().lower() == desired_clean:
            return c
    return None


def detect_slot_pairs(columns: List[str]) -> Tuple[List[int], Dict[int, str], Dict[int, str]]:
    """
    Detect slot pairs:
      Brand1, Brand 1, brand1, etc.
      Rx/Month1, Rx / Month 1, Rx Month1, etc.
    """
    brand_re = re.compile(r"^Brand\s*(\d+)$", flags=re.I)
    rx_re = re.compile(r"^Rx\s*/?\s*Month\s*(\d+)$", flags=re.I)

    brand_col = {}
    rx_col = {}

    for c in columns:
        s = str(c).replace("\r", "").strip()

        mb = brand_re.match(s)
        if mb:
            brand_col[int(mb.group(1))] = c
            continue

        mr = rx_re.match(s)
        if mr:
            rx_col[int(mr.group(1))] = c

    slots = sorted(set(brand_col.keys()).intersection(rx_col.keys()))
    return slots, brand_col, rx_col


def norm_brand(series: pd.Series) -> pd.Series:
    x = series.astype("string").fillna("").str.replace("\r", "", regex=False)
    x = x.str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    x = x.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
    return x


def safe_pct_change(prev_arr: np.ndarray, curr_arr: np.ndarray) -> np.ndarray:
    """
    Vectorized safe % change:
    - prev == 0 and curr == 0 => 0
    - prev == 0 and curr > 0  => +inf (treated as big jump)
    - else normal pct change
    """
    prev_arr = prev_arr.astype(float)
    curr_arr = curr_arr.astype(float)
    pct = np.empty(len(curr_arr), dtype=float)

    zero_prev = (prev_arr == 0.0)
    pct[zero_prev] = np.where(curr_arr[zero_prev] == 0.0, 0.0, np.inf)
    pct[~zero_prev] = ((curr_arr[~zero_prev] - prev_arr[~zero_prev]) / prev_arr[~zero_prev]) * 100.0
    return pct


def read_template_header(template_csv: str, encoding: str = "utf-8") -> List[str]:
    """
    Reads the first line of template CSV as raw header fields.
    Preserves trailing empty columns (i.e. ending commas).
    Strips BOM so first column name matches (e.g. "Division" not "\ufeffDivision").
    """
    with open(template_csv, "r", encoding=encoding, newline="") as f:
        header_line = f.readline()
    header_line = header_line.rstrip("\n").rstrip("\r")
    if header_line.startswith("\ufeff"):
        header_line = header_line[1:]
    return header_line.split(",")


def infer_product_columns_from_template(template_fields: List[str]) -> List[str]:
    """
    In template, product columns appear between 'MONTH' and 'Variance >25%'.
    Example in provided template: UDILIV, COLOSPA, FLORACHAMP, EZYBIXY
    """
    # Find indices safely
    def idx_of(name: str) -> int:
        for i, v in enumerate(template_fields):
            if v.strip().lower() == name.strip().lower():
                return i
        return -1

    i_month = idx_of("MONTH")
    i_var = idx_of("Variance >25%")

    if i_month == -1 or i_var == -1 or i_var <= i_month:
        # fallback: known set (still safe)
        return []

    return [c for c in template_fields[i_month + 1:i_var] if c.strip() != ""]


def build_output_fields(template_fields: List[str], product_cols: List[str]) -> List[str]:
    """
    Build division-specific output header: template leading + product_cols (ALLOWED_BRANDS) + template trailing.
    Product columns sit between MONTH and Variance >25%.
    """
    def idx_of(name: str) -> int:
        for i, v in enumerate(template_fields):
            if v.strip().lower() == name.strip().lower():
                return i
        return -1

    i_month = idx_of("MONTH")
    i_var = idx_of("Variance >25%")
    if i_month == -1 or i_var == -1 or i_var <= i_month:
        return template_fields
    return template_fields[: i_month + 1] + product_cols + template_fields[i_var:]


def format_bool_excel_style(x) -> str:
    """Template uses TRUE/FALSE."""
    if pd.isna(x):
        return ""
    return "TRUE" if bool(x) else "FALSE"


def parse_report_month(report_month_str: Optional[str]) -> Optional[pd.Period]:
    """Parse --report-month (e.g. '2026-01' or '2026-01-01') to pandas Period monthly."""
    if not report_month_str or not str(report_month_str).strip():
        return None
    s = str(report_month_str).strip()
    try:
        dt = pd.to_datetime(s, errors="raise")
        return dt.to_period("M")
    except Exception:
        return None


def get_baseline_months(report_jan: pd.Period) -> List[pd.Period]:
    """Return [Oct, Nov, Dec] (3 months before report month Jan)."""
    return [report_jan - 3, report_jan - 2, report_jan - 1]


def load_hierarchy_lookup(
    hierarchy_csv: str,
    encoding: str = "utf-8",
) -> Optional[pd.DataFrame]:
    """
    Load hierarchy CSV and build one row per (Division, User: Alias) with columns
    to fill template: Division, Territory Code, User: Alias, User: Full Name (from TBM Name),
    and any ABM Code, ABM Name, ZBM Code, ZBM Name if present.
    ZBM -> ABM -> TBM hierarchy: we key by TBM (User: Alias) and bring in parent info.
    """
    try:
        h = pd.read_csv(hierarchy_csv, encoding=encoding, low_memory=True)
    except Exception:
        return None
    h.columns = clean_columns(list(h.columns))
    cols = list(h.columns)
    div_col = resolve_col(cols, "Division") or resolve_col(cols, "User: Division")
    alias_col = resolve_col(cols, "User: Alias")
    tbm_name_col = resolve_col(cols, "TBM Name")
    territory_col = resolve_col(cols, "Territory Code")
    if alias_col is None:
        return None
    keep = []
    rename_map = {}
    if div_col:
        keep.append(div_col)
        rename_map[div_col] = "Division"
    keep.append(alias_col)
    if alias_col != "User: Alias":
        rename_map[alias_col] = "User: Alias"
    if tbm_name_col:
        keep.append(tbm_name_col)
        rename_map[tbm_name_col] = "User: Full Name"
    if territory_col:
        keep.append(territory_col)
        if territory_col != "Territory Code":
            rename_map[territory_col] = "Territory Code"
    for name in ["ABM Code", "ABM Name", "ZBM Code", "ZBM Name"]:
        src = resolve_col(cols, name)
        if src and src not in keep:
            keep.append(src)
    keep = [c for c in keep if c in h.columns]
    h = h[keep].copy()
    h = h.rename(columns=rename_map)
    h["Division"] = h["Division"].astype(str).str.strip()
    h["User: Alias"] = h["User: Alias"].astype(str).str.strip()
    hierarchy_lookup = h.drop_duplicates(subset=["Division", "User: Alias"], keep="first").copy()
    return hierarchy_lookup


def _is_empty_val(val) -> bool:
    if pd.isna(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() == "nan"


def _norm_alias(val) -> str:
    """Normalize User: Alias for lookup (strip and remove trailing .0)."""
    if pd.isna(val):
        return ""
    return re.sub(r"\.0$", "", str(val).strip())


def _extract_tbm_from_territory_code(val) -> str:
    """Extract TBM (code starting with IT) from Territory Code for unique-per-TBM-per-month grain."""
    if pd.isna(val) or str(val).strip() == "":
        return ""
    parts = re.split(r"[\s;]+", str(val).strip())
    for p in parts:
        p = p.strip().rstrip(";")
        if p.upper().startswith("IT"):
            return p
    return ""


def fill_empty_from_hierarchy(
    out: pd.DataFrame,
    hierarchy_lookup: pd.DataFrame,
    template_fields: List[str],
) -> pd.DataFrame:
    """Fill empty cells in out from hierarchy_lookup, keyed by Division and User: Alias (or User: Alias only if Division empty)."""
    if hierarchy_lookup.empty:
        return out
    alias_col_out = resolve_col(list(out.columns), "User: Alias")
    if alias_col_out is None:
        return out
    out = out.copy()
    if alias_col_out != "User: Alias":
        out = out.rename(columns={alias_col_out: "User: Alias"})
    out["User: Alias"] = out["User: Alias"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if "Division" not in out.columns:
        out["Division"] = ""
    out["Division"] = out["Division"].astype(str).str.strip().replace("nan", "")
    h = hierarchy_lookup.copy()
    h["User: Alias"] = h["User: Alias"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    h["Division"] = h["Division"].astype(str).str.strip()
    use_alias_only = (out["Division"].isin(["", "nan"])).all()
    if use_alias_only:
        h = h.drop_duplicates(subset=["User: Alias"], keep="first")
    fill_cols = [c for c in h.columns if c in template_fields and c not in ["Division", "User: Alias"]]
    if "Division" in h.columns and "Division" in template_fields:
        fill_cols = ["Division"] + fill_cols
    fill_cols = [c for c in fill_cols if c in h.columns]
    if not fill_cols:
        return out
    # Build lookup key -> row (dict) for fast lookup
    if use_alias_only:
        h["User: Alias"] = h["User: Alias"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        key_to_row = h.set_index("User: Alias")[fill_cols].to_dict("index")
        for idx in out.index:
            alias = _norm_alias(out.at[idx, "User: Alias"])
            if not alias or alias not in key_to_row:
                continue
            row_vals = key_to_row[alias]
            for c in fill_cols:
                if c not in out.columns:
                    out[c] = pd.NA
                if _is_empty_val(out.at[idx, c]) and c in row_vals:
                    out.at[idx, c] = row_vals[c]
    else:
        fill_cols_no_key = [c for c in fill_cols if c not in ("Division", "User: Alias")]
        if not fill_cols_no_key:
            return out
        key_to_row = h.set_index(["Division", "User: Alias"])[fill_cols_no_key].to_dict("index")
        for idx in out.index:
            div = out.at[idx, "Division"]
            alias = _norm_alias(out.at[idx, "User: Alias"])
            key = (str(div).strip(), alias)
            if not alias or key not in key_to_row:
                continue
            row_vals = key_to_row[key]
            for c in fill_cols_no_key:
                if c not in out.columns:
                    out[c] = pd.NA
                if _is_empty_val(out.at[idx, c]) and c in row_vals:
                    out.at[idx, c] = row_vals[c]
    return out


def load_customer_code_abm_zbm_lookup(
    hierarchy_csv: str,
    encoding: str = "utf-8",
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Load hierarchy CSV and build lookup by Account: Customer Code -> ABM Code, ABM Name, ZBM Code, ZBM Name.
    Returns dict mapping normalized customer code to {ABM Code, ABM Name, ZBM Code, ZBM Name}, or None if columns missing.
    """
    try:
        h = pd.read_csv(hierarchy_csv, encoding=encoding, low_memory=True)
    except Exception:
        return None
    h.columns = clean_columns(list(h.columns))
    cols = list(h.columns)
    cc_col = resolve_col(cols, "Account: Customer Code") or resolve_col(cols, "Customer Code")
    abm_code = resolve_col(cols, "ABM Code")
    abm_name = resolve_col(cols, "ABM Name")
    zbm_code = resolve_col(cols, "ZBM Code")
    zbm_name = resolve_col(cols, "ZBM Name")
    if not cc_col or not all([abm_code, abm_name, zbm_code, zbm_name]):
        return None
    need = [cc_col, abm_code, abm_name, zbm_code, zbm_name]
    h = h[need].dropna(subset=[cc_col]).copy()
    if h.empty:
        return {}
    h["_cc"] = h[cc_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    h = h.drop_duplicates(subset=["_cc"], keep="first")
    lookup = {}
    for _, r in h.iterrows():
        cc = str(r["_cc"]).strip()
        if not cc:
            continue
        lookup[cc] = {
            "ABM Code": r[abm_code],
            "ABM Name": r[abm_name],
            "ZBM Code": r[zbm_code],
            "ZBM Name": r[zbm_name],
        }
    return lookup


def fill_abm_zbm_by_customer_code(
    out: pd.DataFrame,
    customer_code_lookup: Dict[str, Dict[str, Any]],
    customer_code_col: str = "Account: Customer Code",
) -> pd.DataFrame:
    """Fill empty ABM Code, ABM Name, ZBM Code, ZBM Name in out by looking up Account: Customer Code in hierarchy."""
    if not customer_code_lookup:
        return out
    cc_resolved = resolve_col(list(out.columns), customer_code_col)
    if cc_resolved is None:
        return out
    fill_cols = ["ABM Code", "ABM Name", "ZBM Code", "ZBM Name"]
    for c in fill_cols:
        if c not in out.columns:
            out[c] = pd.NA
    for idx in out.index:
        cc_val = out.at[idx, cc_resolved]
        cc = re.sub(r"\.0$", "", str(cc_val).strip()) if pd.notna(cc_val) else ""
        if not cc or cc not in customer_code_lookup:
            continue
        row_vals = customer_code_lookup[cc]
        for c in fill_cols:
            if _is_empty_val(out.at[idx, c]) and c in row_vals and not _is_empty_val(row_vals[c]):
                out.at[idx, c] = row_vals[c]
    return out


def fill_hierarchy_from_raw_territory_codes(
    out: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    territory_code_col: str = "Territory Code",
    customer_code_col: str = "Account: Customer Code",
    raw_full_name_col: str = "Full Name",
) -> pd.DataFrame:
    """
    Fill hierarchy using only the raw DCR 'Territory Code' (no external hierarchy file).

    User rules:
    - IT* => TBM
    - IA* => ABM
    - RG* => ZBM
    - TBM works under ABM; ABM works under ZBM

    We infer parent mappings via maximum shared customer overlap in the raw DCR:
    - IT -> IA by most shared 'Account: Customer Code'
    - IA -> RG by most shared 'Account: Customer Code'
    Names come from raw 'Full Name' (most frequent for a given Territory Code).
    """
    if out.empty or raw_df is None or raw_df.empty:
        return out
    if territory_code_col not in out.columns or territory_code_col not in raw_df.columns:
        return out
    if customer_code_col not in raw_df.columns:
        return out

    def _norm_terr(x) -> str:
        if pd.isna(x):
            return ""
        return str(x).strip().rstrip(";").strip()

    cols_needed = [territory_code_col, customer_code_col]
    if raw_full_name_col in raw_df.columns:
        cols_needed.append(raw_full_name_col)
    t = raw_df[cols_needed].copy()
    t[territory_code_col] = t[territory_code_col].map(_norm_terr)
    t[customer_code_col] = t[customer_code_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    t = t[(t[territory_code_col] != "") & (t[customer_code_col] != "")].drop_duplicates()

    code_to_name: Dict[str, str] = {}
    if raw_full_name_col in t.columns:
        name_df = t[[territory_code_col, raw_full_name_col]].dropna().copy()
        name_df[raw_full_name_col] = name_df[raw_full_name_col].astype(str).str.strip()
        if not name_df.empty:
            vc = name_df.groupby([territory_code_col, raw_full_name_col]).size().reset_index(name="_n")
            best = vc.sort_values([territory_code_col, "_n"], ascending=[True, False]).drop_duplicates(territory_code_col)
            code_to_name = dict(zip(best[territory_code_col].tolist(), best[raw_full_name_col].tolist()))

    def _best_mapping(child_prefix: str, parent_prefix: str) -> Dict[str, str]:
        child = t[t[territory_code_col].str.startswith(child_prefix)][[customer_code_col, territory_code_col]].drop_duplicates()
        parent = t[t[territory_code_col].str.startswith(parent_prefix)][[customer_code_col, territory_code_col]].drop_duplicates()
        if child.empty or parent.empty:
            return {}
        pairs = child.merge(parent, on=customer_code_col, how="inner", suffixes=("_child", "_parent"))
        if pairs.empty:
            return {}
        counts = pairs.groupby([f"{territory_code_col}_child", f"{territory_code_col}_parent"]).size().reset_index(name="_n")
        best = counts.sort_values([f"{territory_code_col}_child", "_n"], ascending=[True, False]).drop_duplicates(f"{territory_code_col}_child")
        return dict(zip(best[f"{territory_code_col}_child"].tolist(), best[f"{territory_code_col}_parent"].tolist()))

    it_to_ia = _best_mapping("IT", "IA")
    ia_to_rg = _best_mapping("IA", "RG")

    out = out.copy()
    out_terr = out[territory_code_col].map(_norm_terr)

    designation = np.where(
        out_terr.str.startswith("IT"),
        "TBM",
        np.where(out_terr.str.startswith("IA"), "ABM", np.where(out_terr.str.startswith("RG"), "ZBM", "")),
    )
    if "User: Designation" in out.columns:
        cur = out["User: Designation"].astype(str).str.strip()
        out["User: Designation"] = np.where(cur == "", designation, out["User: Designation"])
    else:
        out["User: Designation"] = designation

    abm_code = np.where(out_terr.str.startswith("IA"), out_terr, np.where(out_terr.str.startswith("IT"), out_terr.map(it_to_ia), ""))
    abm_ser = pd.Series(abm_code, index=out.index)
    zbm_via_abm = abm_ser.map(lambda c: ia_to_rg.get(str(c), "") if c else "")
    zbm_code = np.where(
        out_terr.str.startswith("RG"),
        out_terr,
        np.where(out_terr.str.startswith("IA"), out_terr.map(ia_to_rg), np.where(out_terr.str.startswith("IT"), zbm_via_abm, "")),
    )

    if "ABM Code" in out.columns:
        cur = out["ABM Code"].astype(str).str.strip()
        out["ABM Code"] = np.where(cur == "", abm_code, out["ABM Code"])
    else:
        out["ABM Code"] = abm_code
    if "ZBM Code" in out.columns:
        cur = out["ZBM Code"].astype(str).str.strip()
        out["ZBM Code"] = np.where(cur == "", zbm_code, out["ZBM Code"])
    else:
        out["ZBM Code"] = zbm_code

    abm_name = abm_ser.map(lambda c: code_to_name.get(str(c), "") if c else "")
    zbm_name = pd.Series(zbm_code, index=out.index).map(lambda c: code_to_name.get(str(c), "") if c else "")
    if "ABM Name" in out.columns:
        cur = out["ABM Name"].astype(str).str.strip()
        out["ABM Name"] = np.where(cur == "", abm_name, out["ABM Name"])
    else:
        out["ABM Name"] = abm_name
    if "ZBM Name" in out.columns:
        cur = out["ZBM Name"].astype(str).str.strip()
        out["ZBM Name"] = np.where(cur == "", zbm_name, out["ZBM Name"])
    else:
        out["ZBM Name"] = zbm_name

    # Territory Code column: business rule is TBM only — show only the code starting with IT
    def _extract_it_only(val):
        if pd.isna(val) or str(val).strip() == "":
            return ""
        parts = re.split(r"[\s;]+", str(val).strip())
        for p in parts:
            p = p.strip().rstrip(";")
            if p.upper().startswith("IT"):
                return p
        return ""

    out[territory_code_col] = out[territory_code_col].map(_extract_it_only)

    return out


# ---------------------------
# Core computation
# ---------------------------

def add_variance_flags(
    df: pd.DataFrame,
    doctor_col: str,
    monthperiod_col: str,
    product_cols: List[str]
) -> pd.DataFrame:
    """
    Adds flags:
      Variance >25%, Increasing, Decreasing, All Zero
    Based on month-to-month % change of TOTAL Rx (sum across product columns) per doctor.
    """
    out = df.copy()

    # Numeric coercion for products
    for c in product_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        else:
            out[c] = 0

    out["_total_rx"] = out[product_cols].sum(axis=1)
    out["All Zero"] = (out["_total_rx"] == 0)

    # Ensure proper ordering by month period
    out = out.sort_values([doctor_col, monthperiod_col], kind="mergesort").reset_index(drop=True)

    out["_prev_total_rx"] = out.groupby(doctor_col)["_total_rx"].shift(1)

    prev = out["_prev_total_rx"].fillna(0).to_numpy(dtype=float)
    curr = out["_total_rx"].to_numpy(dtype=float)
    pct = safe_pct_change(prev, curr)

    has_prev = out["_prev_total_rx"].notna().to_numpy()

    # ✅ FIXED boolean logic (your original script had broken operators here)
    variance = (np.isinf(pct) | (np.abs(pct) > 25.0)) & has_prev
    increasing = ((np.isinf(pct) & (pct > 0)) | (pct > 25.0)) & has_prev
    decreasing = (pct < -25.0) & has_prev

    out["Variance >25%"] = variance
    out["Increasing"] = increasing
    out["Decreasing"] = decreasing

    out = out.drop(columns=["_total_rx", "_prev_total_rx"], errors="ignore")
    return out


def add_variance_flags_octdec_vs_jan(
    df: pd.DataFrame,
    doctor_col: str,
    monthperiod_col: str,
    report_jan: pd.Period,
    baseline_months: List[pd.Period],
    product_cols: List[str],
    tbm_col: Optional[str] = None,
) -> pd.DataFrame:
    """Variance flags: Nov-Jan avg vs Feb; when tbm_col set, grain is (TBM, doctor)."""
    out = df.copy()
    for c in product_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        else:
            out[c] = 0
    out["_total_rx"] = out[product_cols].sum(axis=1)
    nov_period, dec_period, jan_period = baseline_months[0], baseline_months[1], baseline_months[2]
    if tbm_col and tbm_col in out.columns:
        group_keys = [tbm_col, doctor_col]
        all_pairs = out[group_keys].drop_duplicates()
        all_index = all_pairs.set_index(group_keys).index
        nov_totals = out.loc[out[monthperiod_col] == nov_period].groupby(group_keys)["_total_rx"].sum().reindex(all_index).fillna(0)
        dec_totals = out.loc[out[monthperiod_col] == dec_period].groupby(group_keys)["_total_rx"].sum().reindex(all_index).fillna(0)
        jan_totals = out.loc[out[monthperiod_col] == jan_period].groupby(group_keys)["_total_rx"].sum().reindex(all_index).fillna(0)
        report_totals = out.loc[out[monthperiod_col] == report_jan].groupby(group_keys)["_total_rx"].sum().reindex(all_index).fillna(0)
        avg_baseline = (nov_totals + dec_totals + jan_totals) / 3.0
        feb_total = report_totals
        mask_both_zero = (avg_baseline == 0) & (feb_total == 0)
        mask_baseline_zero_feb_positive = (avg_baseline == 0) & (feb_total > 0)
        percent_diff = np.where(mask_both_zero, 0.0, np.where(mask_baseline_zero_feb_positive, np.nan, ((feb_total - avg_baseline) / avg_baseline) * 100.0))
        variance_ser = pd.Series(np.where(mask_both_zero, False, np.where(mask_baseline_zero_feb_positive, True, np.abs(percent_diff) > 25.0)), index=report_totals.index)
        increasing_ser = pd.Series(np.where(mask_both_zero, False, np.where(mask_baseline_zero_feb_positive, True, percent_diff >= 25.0)), index=report_totals.index)
        decreasing_ser = pd.Series(np.where(mask_both_zero, False, np.where(mask_baseline_zero_feb_positive, False, percent_diff <= -25.0)), index=report_totals.index)
        all_zero_ser = (feb_total == 0)
        sum_last_3_months = nov_totals + dec_totals + jan_totals
        is_report = out[monthperiod_col] == report_jan
        out["Sum of last 3 months"] = out.apply(lambda r: sum_last_3_months.get((r[tbm_col], r[doctor_col]), 0), axis=1)
        out["All Zero"] = out.apply(lambda r: all_zero_ser.get((r[tbm_col], r[doctor_col]), True), axis=1)
        out["Variance >25%"] = out.apply(lambda r: variance_ser.get((r[tbm_col], r[doctor_col]), False), axis=1)
        out["Increasing"] = out.apply(lambda r: increasing_ser.get((r[tbm_col], r[doctor_col]), False), axis=1)
        out["Decreasing"] = out.apply(lambda r: decreasing_ser.get((r[tbm_col], r[doctor_col]), False), axis=1)
        out.loc[~is_report, "All Zero"] = False
        out.loc[~is_report, "Variance >25%"] = False
        out.loc[~is_report, "Increasing"] = False
        out.loc[~is_report, "Decreasing"] = False
    else:
        all_doctors = out[doctor_col].unique()
        nov_totals = out.loc[out[monthperiod_col] == nov_period].groupby(doctor_col)["_total_rx"].sum().reindex(all_doctors).fillna(0)
        dec_totals = out.loc[out[monthperiod_col] == dec_period].groupby(doctor_col)["_total_rx"].sum().reindex(all_doctors).fillna(0)
        jan_totals = out.loc[out[monthperiod_col] == jan_period].groupby(doctor_col)["_total_rx"].sum().reindex(all_doctors).fillna(0)
        report_totals = out.loc[out[monthperiod_col] == report_jan].groupby(doctor_col)["_total_rx"].sum().reindex(all_doctors).fillna(0)
        avg_baseline = (nov_totals + dec_totals + jan_totals) / 3.0
        feb_total = report_totals
        mask_both_zero = (avg_baseline == 0) & (feb_total == 0)
        mask_baseline_zero_feb_positive = (avg_baseline == 0) & (feb_total > 0)
        percent_diff = np.where(mask_both_zero, 0.0, np.where(mask_baseline_zero_feb_positive, np.nan, ((feb_total - avg_baseline) / avg_baseline) * 100.0))
        variance_ser = pd.Series(np.where(mask_both_zero, False, np.where(mask_baseline_zero_feb_positive, True, np.abs(percent_diff) > 25.0)), index=report_totals.index)
        increasing_ser = pd.Series(np.where(mask_both_zero, False, np.where(mask_baseline_zero_feb_positive, True, percent_diff >= 25.0)), index=report_totals.index)
        decreasing_ser = pd.Series(np.where(mask_both_zero, False, np.where(mask_baseline_zero_feb_positive, False, percent_diff <= -25.0)), index=report_totals.index)
        all_zero_ser = (feb_total == 0)
        sum_last_3_months = nov_totals + dec_totals + jan_totals
        out["Sum of last 3 months"] = out[doctor_col].map(lambda d: sum_last_3_months.get(d, 0))
        is_report = out[monthperiod_col] == report_jan
        out["All Zero"] = out[doctor_col].map(lambda d: all_zero_ser.get(d, True))
        out["Variance >25%"] = out[doctor_col].map(lambda d: variance_ser.get(d, False))
        out["Increasing"] = out[doctor_col].map(lambda d: increasing_ser.get(d, False))
        out["Decreasing"] = out[doctor_col].map(lambda d: decreasing_ser.get(d, False))
        out.loc[~is_report, "All Zero"] = False
        out.loc[~is_report, "Variance >25%"] = False
        out.loc[~is_report, "Increasing"] = False
        out.loc[~is_report, "Decreasing"] = False
    out = out.drop(columns=["_total_rx"], errors="ignore")
    return out


def validate_variance_flags(out: pd.DataFrame, product_cols: List[str]) -> None:
    if out.empty or "Variance >25%" not in out.columns or "Sum of last 3 months" not in out.columns:
        return
    prod = out[[c for c in product_cols if c in out.columns]].copy()
    for c in prod.columns:
        prod[c] = pd.to_numeric(prod[c], errors="coerce").fillna(0)
    feb_total = prod.sum(axis=1).to_numpy(dtype=float)
    avg_baseline = (out["Sum of last 3 months"].astype(float) / 3.0).to_numpy(dtype=float)
    mask_both_zero = (avg_baseline == 0) & (feb_total == 0)
    mask_baseline_zero_feb_positive = (avg_baseline == 0) & (feb_total > 0)
    expected_variance = np.where(
        mask_both_zero,
        False,
        np.where(
            mask_baseline_zero_feb_positive,
            True,
            np.abs(((feb_total - avg_baseline) / np.where(avg_baseline == 0, 1.0, avg_baseline)) * 100.0) > 25.0,
        ),
    )
    actual = out["Variance >25%"]
    actual_bool = actual.astype(str).str.strip().str.upper().isin(("TRUE", "1", "YES", "T")).to_numpy(dtype=bool) if actual.dtype != bool else actual.to_numpy(dtype=bool)
    if np.any(expected_variance != actual_bool):
        raise ValueError("Variance validation failed: recomputed variance flag does not match output.")


def load_input_df(
    input_path: str,
    encoding: str = "utf-8",
    low_memory: bool = False,
    chunksize: int = 0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load raw DCR from CSV or XLSX. Returns (df, columns list)."""
    if str(input_path).lower().endswith(".xlsx"):
        df = pd.read_excel(input_path, sheet_name=0)
        cols = clean_columns(list(df.columns))
        df.columns = cols
        return df, cols
    if chunksize and chunksize > 0:
        reader = pd.read_csv(
            input_path,
            encoding=encoding,
            low_memory=low_memory,
            chunksize=chunksize,
        )
        df = pd.concat([chunk for chunk in reader], ignore_index=True)
    else:
        df = pd.read_csv(input_path, encoding=encoding, low_memory=low_memory)
    cols = clean_columns(list(df.columns))
    df.columns = cols
    return df, cols


def build_output_in_template_format(
    input_csv: str,
    template_csv: str,
    output_csv: str,
    doctor_col: str = "Account: Customer Code",
    date_col: str = "Date",
    filed_date_col: str = "Filed Date",
    encoding: str = "utf-8",
    low_memory: bool = False,
    chunksize: int = 0,
    report_month: Optional[str] = None,
    hierarchy_csv: Optional[str] = None,
) -> None:
    # Read template schema; build division-specific output header (this division's ALLOWED_BRANDS between MONTH and Variance >25%)
    template_fields = read_template_header(template_csv, encoding=encoding)
    product_cols = ALLOWED_BRANDS
    output_fields = build_output_fields(template_fields, product_cols)
    # Insert "Sum of last 3 months" column before Variance >25% for validation
    i_var = next((i for i, f in enumerate(output_fields) if str(f).strip().lower() == "variance >25%"), -1)
    if i_var >= 0:
        output_fields = output_fields[:i_var] + ["Sum of last 3 months"] + output_fields[i_var:]

    # Flags that must exist (and be in template)
    flag_cols = ["Variance >25%", "Increasing", "Decreasing", "All Zero"]

    # Read input data (CSV or XLSX)
    df, cols = load_input_df(input_csv, encoding=encoding, low_memory=low_memory, chunksize=chunksize)
    slots, brand_map, rx_map = detect_slot_pairs(cols)
    if not slots:
        raise ValueError("No slot pairs detected. Expected BrandN and Rx/MonthN columns.")

    # Resolve important columns from input (case-insensitive)
    doctor_src = resolve_col(cols, doctor_col) or doctor_col
    date_src = resolve_col(cols, date_col) or date_col
    filed_src = resolve_col(cols, filed_date_col)  # optional

    if doctor_src not in df.columns:
        raise ValueError(f"Doctor column not found in input: {doctor_col}")
    if date_src not in df.columns:
        raise ValueError(f"Date column not found in input: {date_col}")

    # Normalize internal names: exact template names so output rows have correct column data
    rename_map = {}
    # doctor/date/filed
    rename_map[doctor_src.replace("\r", "").strip()] = doctor_col
    rename_map[date_src.replace("\r", "").strip()] = date_col
    if filed_src:
        rename_map[filed_src.replace("\r", "").strip()] = filed_date_col

    # Rename template-matched fields (case-insensitive)
    for c in template_fields:
        if c.strip() == "":
            continue
        src = resolve_col(df.columns, c)
        if src and src != c:
            rename_map[src] = c

    # Input columns that have different names in template (fill output correctly)
    for input_name, template_name in INPUT_TO_TEMPLATE_COLUMN_MAP.items():
        clean_input = input_name.replace("\r", "").strip()
        if clean_input in df.columns:
            rename_map[clean_input] = template_name

    df = df.rename(columns=rename_map)

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    if filed_date_col in df.columns:
        df[filed_date_col] = pd.to_datetime(df[filed_date_col], errors="coerce", dayfirst=False)

    # Build MonthPeriod + MONTH string
    df["_MonthPeriod"] = df[date_col].dt.to_period("M")
    df["MONTH"] = df[date_col].dt.strftime("%b").str.upper()  # JAN, FEB,...

    # Drop bad keys
    df = df.dropna(subset=[doctor_col, "_MonthPeriod"])

    # Determine report month (Feb) and baseline months (Nov, Dec, Jan)
    if report_month:
        report_jan = parse_report_month(report_month)
        if report_jan is None:
            raise ValueError(f"Invalid --report-month: {report_month}. Use e.g. 2026-02.")
        baseline_months = get_baseline_months(report_jan)
    else:
        report_jan = parse_report_month(DEFAULT_REPORT_MONTH)
        if report_jan is None:
            report_jan = df["_MonthPeriod"].max()
        baseline_months = get_baseline_months(report_jan)

    allowed_periods = set(baseline_months) | {report_jan}
    df = df[df["_MonthPeriod"].isin(allowed_periods)].copy()
    if df.empty:
        raise ValueError("No data left after filtering to the four months (Nov, Dec, Jan, Feb).")

    if report_jan not in df["_MonthPeriod"].values:
        raise ValueError(f"Report month {report_jan} has no data. Check --report-month and input dates.")

    # ---------------------------
    # SlotMAX extraction -> long table (Doctor, MonthPeriod, Brand, Rx)
    # Grain: unique per TBM (Territory Code IT*) per Account: Customer Code per month
    territory_code_col = resolve_col(cols, "Territory Code")
    if territory_code_col and territory_code_col in df.columns:
        df["_TBM"] = df[territory_code_col].map(_extract_tbm_from_territory_code)
    else:
        df["_TBM"] = ""
    tbm_col = "_TBM"
    group_cols = [tbm_col, doctor_col, "_MonthPeriod"]

    tie_cols = [date_col] + ([filed_date_col] if filed_date_col in df.columns else [])

    parts = []
    for i in slots:
        bcol = f"Brand{i}"
        rcol = f"Rx/Month{i}"
        if bcol not in df.columns or rcol not in df.columns:
            continue
        tmp = df[group_cols + tie_cols + [bcol, rcol]].copy()
        tmp["Rx"] = pd.to_numeric(tmp[rcol], errors="coerce")
        sort_by = group_cols + ["Rx"] + tie_cols
        ascending = [True, True, True] + [False] + [False] * len(tie_cols)
        tmp = tmp.sort_values(sort_by, ascending=ascending, kind="mergesort", na_position="last")
        best = tmp.drop_duplicates(subset=group_cols, keep="first").copy()
        best = best.assign(Brand=norm_brand(best[bcol]))
        best = best.dropna(subset=["Brand"])
        parts.append(best[group_cols + ["Brand", "Rx"]])

    long_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=group_cols + ["Brand", "Rx"])

    agg = (
        long_df.groupby([tbm_col, doctor_col, "_MonthPeriod", "Brand"], as_index=False)["Rx"]
        .max()
    )

    # Keep only allowed 4 brands (ignore all others)
    allowed_brands_upper = {str(b).upper().strip() for b in ALLOWED_BRANDS}
    agg["_BrandNorm"] = agg["Brand"].astype(str).str.upper().str.strip()
    agg = agg[agg["_BrandNorm"].isin(allowed_brands_upper)].copy()
    agg = agg.drop(columns=["_BrandNorm"])
    # Map Brand to canonical name (e.g. COLOSPA) for pivot column names
    norm_to_canonical = {b.upper().strip(): b for b in ALLOWED_BRANDS}
    agg["Brand"] = agg["Brand"].astype(str).str.upper().str.strip().map(
        lambda x: norm_to_canonical.get(x, x)
    )

    pivot = (
        agg.pivot_table(
            index=[tbm_col, doctor_col, "_MonthPeriod"],
            columns="Brand",
            values="Rx",
            aggfunc="max",
            fill_value=np.nan
        )
        .reset_index()
    )
    for p in product_cols:
        if p not in pivot.columns:
            pivot[p] = np.nan
    pivot = pivot[[tbm_col, doctor_col, "_MonthPeriod"] + product_cols]

    agg_numeric = agg.dropna(subset=["Rx"])
    tbm_doctor_brands = agg_numeric.groupby([tbm_col, doctor_col])["Brand"].apply(lambda x: set(x.unique()))
    valid_pairs = tbm_doctor_brands[tbm_doctor_brands.map(lambda b: set(product_cols).issubset(b))].index

    base_sort_cols = [tbm_col, doctor_col, "_MonthPeriod", date_col] + ([filed_date_col] if filed_date_col in df.columns else [])
    base_sorted = df.sort_values(
        base_sort_cols,
        ascending=[True, True, True, False] + ([False] if filed_date_col in df.columns else []),
        kind="mergesort"
    )
    base = base_sorted.drop_duplicates(subset=[tbm_col, doctor_col, "_MonthPeriod"], keep="first").copy()

    out = base.merge(pivot, on=[tbm_col, doctor_col, "_MonthPeriod"], how="left")

    for p in product_cols:
        out[p] = pd.to_numeric(out[p], errors="coerce")

    out = add_variance_flags_octdec_vs_jan(
        out,
        doctor_col=doctor_col,
        monthperiod_col="_MonthPeriod",
        report_jan=report_jan,
        baseline_months=baseline_months,
        product_cols=product_cols,
        tbm_col=tbm_col,
    )

    # Ensure flags exist
    for f in flag_cols:
        if f not in out.columns:
            out[f] = False

    # Ensure MONTH is present as string
    if "MONTH" not in out.columns:
        out["MONTH"] = out[date_col].dt.strftime("%b").str.upper()

    # Format Date like template (DD-MM-YY)
    out[date_col] = out[date_col].dt.strftime("%d-%m-%y")

    # Format booleans like template
    for f in flag_cols:
        out[f] = out[f].apply(format_bool_excel_style)

    valid_set = set(valid_pairs)
    out["_tbm_doc"] = list(zip(out[tbm_col], out[doctor_col]))
    out = out[(out["_MonthPeriod"] == report_jan) & (out["_tbm_doc"].map(lambda x: x in valid_set))].copy()
    out = out.drop(columns=["_MonthPeriod", "_tbm_doc"], errors="ignore")
    if "Territory Code" in out.columns:
        out["Territory Code"] = out[tbm_col]

    validate_variance_flags(out, product_cols)

    # Fill empty columns from hierarchy (ZBM -> ABM -> TBM; key by Division + User: Alias)
    # NOTE: For business handover, hierarchy must come from the division's raw DCR only.
    # The shared hierarchy.csv in this repo is division-specific and can contaminate other divisions,
    # so we explicitly disable hierarchy enrichment here.
    out = fill_hierarchy_from_raw_territory_codes(out, df)
    hierarchy_csv = None
    alias_to_division = {}
    if hierarchy_csv:
        hierarchy_lookup = load_hierarchy_lookup(hierarchy_csv, encoding=encoding)
        if hierarchy_lookup is not None and not hierarchy_lookup.empty:
            out = fill_empty_from_hierarchy(out, hierarchy_lookup, output_fields)
            if "Division" in out.columns and "User: Alias" in out.columns:
                h_div = hierarchy_lookup.drop_duplicates("User: Alias", keep="first").copy()
                h_div["User: Alias"] = h_div["User: Alias"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                for idx in out.index:
                    if _is_empty_val(out.at[idx, "Division"]) and "Division" in h_div.columns:
                        alias = _norm_alias(out.at[idx, "User: Alias"])
                        match = h_div[h_div["User: Alias"] == alias] if alias else pd.DataFrame()
                        if len(match) > 0:
                            out.at[idx, "Division"] = match["Division"].iloc[0]
            if "Division" in out.columns:
                out["Division"] = out["Division"].astype(str).replace("nan", "").str.strip()
            for _, r in hierarchy_lookup.drop_duplicates("User: Alias", keep="first").iterrows():
                alias_to_division[_norm_alias(r["User: Alias"])] = r["Division"]
        # Fill ABM/ZBM by Account: Customer Code from hierarchy (same or separate file)
        customer_code_lookup = load_customer_code_abm_zbm_lookup(hierarchy_csv, encoding=encoding)
        if customer_code_lookup is not None:
            out = fill_abm_zbm_by_customer_code(out, customer_code_lookup, customer_code_col="Account: Customer Code")

    # ---------------------------
    # Write CSV in division-specific schema (output_fields = leading + ALLOWED_BRANDS + trailing)
    # Blank hierarchy columns -> #N/A for business clarity
    # ---------------------------
    HIERARCHY_COLUMNS_NA = {"Territory Code", "ABM Code", "ABM Name", "ZBM Code", "ZBM Name", "User: Designation"}
    field_count = len(output_fields)

    with open(output_csv, "w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)

        writer.writerow(output_fields)

        for _, r in out.iterrows():
            row = [""] * field_count
            for idx, field in enumerate(output_fields):
                if field.strip() == "":
                    row[idx] = ""
                    continue
                if field in out.columns:
                    val = r[field]
                else:
                    val = None
                if field == "Division" and _is_empty_val(val) and alias_to_division:
                    alias_key = _norm_alias(r.get("User: Alias", ""))
                    val = alias_to_division.get(alias_key, val) if alias_key else val
                if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                    row[idx] = "#N/A" if field in HIERARCHY_COLUMNS_NA else ""
                else:
                    row[idx] = str(val)
            writer.writerow(row)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SlotMAX Division 42 -> out_jan_div42.csv")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Input raw DCR (CSV or XLSX)")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Template CSV path")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    p.add_argument("--doctor-col", default="Account: Customer Code", help="Doctor identifier column")
    p.add_argument("--date-col", default="Date", help="Date column")
    p.add_argument("--filed-date-col", default="Filed Date", help="Filed Date column (optional)")
    p.add_argument("--encoding", default="utf-8", help="File encoding (default utf-8)")
    p.add_argument("--chunksize", type=int, default=0, help="Read chunksize (0 = full read)")
    p.add_argument("--low-memory", action="store_true", help="Enable pandas low_memory mode")
    p.add_argument(
        "--report-month",
        default=str(DEFAULT_REPORT_MONTH),
        help="Report month (e.g. 2026-02). Baseline = Nov, Dec, Jan. If omitted, defaults to 2026-02.",
    )
    p.add_argument("--hierarchy", default=str(DEFAULT_HIERARCHY) if DEFAULT_HIERARCHY.exists() else None, help="Hierarchy CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    build_output_in_template_format(
        input_csv=args.input,
        template_csv=args.template,
        output_csv=args.output,
        doctor_col=args.doctor_col,
        date_col=args.date_col,
        filed_date_col=args.filed_date_col,
        encoding=args.encoding,
        low_memory=args.low_memory,
        chunksize=args.chunksize,
        report_month=args.report_month,
        hierarchy_csv=args.hierarchy,
    )
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
