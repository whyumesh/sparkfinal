import pandas as pd
import re
import os

# -------------------- INPUT FILES (YOU HAVE THESE TWO) --------------------
ABM_PATH = "ABM_MSL_report_with_CustomerCode.csv"
HIER_PATH = "Hierarchy_Div75.csv"

# -------------------- OPTIONAL DCR / VISITS FILE --------------------
# If you DON'T have DCR, keep this as empty string "" (default).
# If later you get DCR, set it like: DCR_PATH = "Visits.csv"
DCR_PATH = "DCR_Junction.csv"  # <-- no file = no crash

# Column names in DCR file (only used if DCR_PATH exists)
DCR_CODE_COL = "Account: Customer Code"
DCR_DATE_COL = "Date"

# Month/Year (only used if DCR file exists)
YEAR = 2026
MONTH = 1  # 1..12

# -------------------- OUTPUT FILES --------------------
STAMP = f"{YEAR:04d}-{MONTH:02d}"
OUT_ABM = f"ABM_with_MSL_Visits_Coverage_{STAMP}.csv"
OUT_MSL_LIST = "msl_list.csv"
OUT_VISITED_LIST = f"msl_visited_{STAMP}.csv"
OUT_SUMMARY = f"msl_coverage_summary_{STAMP}.csv"

# -------------------- NORMALIZATION HELPERS --------------------
DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
NBSP = "\u00A0"
dash_pattern = "[" + re.escape(DASHES) + "]"

def norm(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    s = s.str.replace(NBSP, " ", regex=False)              # NBSP -> space
    s = s.str.replace(dash_pattern, "-", regex=True)       # fancy dashes -> "-"
    s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return s

# -------------------- LOAD ABM + HIERARCHY --------------------
abm = pd.read_csv(ABM_PATH, dtype=str, keep_default_na=False)
hier = pd.read_csv(HIER_PATH, dtype=str, keep_default_na=False)

REQ_CODE_COL = "Account: Customer Code"  # used in your earlier scripts [2](https://abbott-my.sharepoint.com/personal/umesh_pawar_abbott_com/_layouts/15/Doc.aspx?sourcedoc=%7BBD42790A-56EC-48CF-BB53-1EBC3F2C7564%7D&file=ABM_MSL_report_with_CustomerCode.csv&action=default&mobileredirect=true)[1](https://abbott-my.sharepoint.com/personal/umesh_pawar_abbott_com/_layouts/15/Doc.aspx?sourcedoc=%7BA6395213-77EF-49B5-A842-9E1D10543676%7D&file=Hierarchy_Division33.csv&action=default&mobileredirect=true)

if REQ_CODE_COL not in abm.columns:
    raise KeyError(f"'{REQ_CODE_COL}' not found in ABM file. Columns: {list(abm.columns)}")
if REQ_CODE_COL not in hier.columns:
    raise KeyError(f"'{REQ_CODE_COL}' not found in Hierarchy file. Columns: {list(hier.columns)}")

abm["_code"] = norm(abm[REQ_CODE_COL])
hier["_code"] = norm(hier[REQ_CODE_COL])

# -------------------- 1) MSL DOCTORS (UNIVERSE) --------------------
hier_codes = set(hier["_code"])
abm["MSL Dr"] = abm["_code"].isin(hier_codes)

msl_universe_codes = sorted(set(abm.loc[abm["MSL Dr"], "_code"]) - {""})
msl_count = len(msl_universe_codes)

# Save MSL doctors list
pd.DataFrame({REQ_CODE_COL: msl_universe_codes}).to_csv(OUT_MSL_LIST, index=False)

# -------------------- 2) VISITS / DCR (OPTIONAL) --------------------
visited_codes = set()
visited_count = 0

if DCR_PATH and os.path.exists(DCR_PATH):
    
    visits = pd.read_csv(DCR_PATH, dtype=str, keep_default_na=False, encoding="cp1252")


    if DCR_CODE_COL not in visits.columns:
        raise KeyError(f"'{DCR_CODE_COL}' not found in {DCR_PATH}. Columns: {list(visits.columns)}")
    if DCR_DATE_COL not in visits.columns:
        raise KeyError(f"'{DCR_DATE_COL}' not found in {DCR_PATH}. Columns: {list(visits.columns)}")

    visits["_code"] = norm(visits[DCR_CODE_COL])
    visits["_visit_dt"] = pd.to_datetime(visits[DCR_DATE_COL], errors="coerce", dayfirst=True)

    visits = visits[visits["_visit_dt"].notna()].copy()
    visits["_year"] = visits["_visit_dt"].dt.year
    visits["_month"] = visits["_visit_dt"].dt.month

    visits_period = visits[(visits["_year"] == YEAR) & (visits["_month"] == MONTH)].copy()

    visited_codes = set(visits_period["_code"]) & set(msl_universe_codes)
    visited_count = len(visited_codes)

    # Save visited list
    pd.DataFrame({REQ_CODE_COL: sorted(visited_codes)}).to_csv(OUT_VISITED_LIST, index=False)

else:
    # No DCR file → create an empty visited list file (so your outputs still exist)
    pd.DataFrame({REQ_CODE_COL: []}).to_csv(OUT_VISITED_LIST, index=False)
    print("⚠️ DCR/Visits file not provided or not found. Visited count set to 0 and Coverage% set to 0.")

# -------------------- 3) COVERAGE % --------------------
coverage = (visited_count / msl_count * 100) if msl_count else 0.0
coverage = round(coverage, 2)

# -------------------- 4) ADD YOUR REQUIRED NEW COLUMNS --------------------
# These are summary fields repeated on each row (since they are month-level KPIs)
abm["Visited This Month"] = abm["_code"].isin(visited_codes)
abm["Doctors list (MSL doctors)"] = msl_count
abm["Doctors visit in that month (DCR report)"] = visited_count
abm["Coverage %"] = coverage

# Cleanup helper column
abm.drop(columns=["_code"], inplace=True)

# Save final ABM output
abm.to_csv(OUT_ABM, index=False)

# Save summary KPI file (one row)
summary = pd.DataFrame([{
    "Month": STAMP,
    "Doctors list (MSL doctors)": msl_count,
    "Doctors visit in that month (DCR report)": visited_count,
    "Coverage %": coverage
}])
summary.to_csv(OUT_SUMMARY, index=False)

# -------------------- DONE --------------------
print("\n✅ Done")
print(f"MSL Doctors (universe): {msl_count}")
print(f"Visited MSL Doctors in {STAMP}: {visited_count}")
print(f"Coverage %: {coverage}%")
print("\nOutputs created:")
print(f" - {OUT_ABM}")
print(f" - {OUT_MSL_LIST}")
print(f" - {OUT_VISITED_LIST}")
print(f" - {OUT_SUMMARY}")