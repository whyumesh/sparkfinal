#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIG
# ============================================================
KPI_FILE = "KPI - Feb 2026.xlsx"
KPI_SHEET = "final_KPI_TBM"

DCR_FILE = "DCR_RAW_STANDARDIZED_4div_2026-02-01_2026-02-28_4Div.xlsx"

COMEX_FILE = "Comex_AIL.xlsx"
COMEX_SHEET= "AIL"

OUTPUT_FILE = "FINAL_OUTPUT.xlsx"

ALLOWED_DIVISIONS = {"28", "30", "35", "42"}

# Division → list of required brands to consider a doctor valid
DIVISION_BRAND_MAP = {
    "30": ["UDILIV", "COLOSPA", "FLORACHAMP", "EZYBIXY"],
    "42": ["GANATON", "GANATON TOTAL", "DUPHALAC", "ELDICET"],
    "35": ["CREMAFFIN PLUS", "DIGERAFT PLUS", "CREMAFFIN", "LIBRAX"],
    "28": ["CREON", "HEPTRAL SAME", "VONEFI", "ROWASA"]
}

FINAL_COLS = [
    "Division",
    "ZBM CODE", "ZBM NAME",
    "ABM CODE", "ABM NAME",
    "Employee Code", "Full Name",
    "Territory Headquarter", "Abbott Designation",
    "DOJ", "Territory",
    "Last Submitted DCR Date",
    "Status",
    "Total Dr Total", "Total DR Visited",
    "Total Coverage",
    "Number of doctors with Rx entered",
    "RCPA Coverage"
]

# ============================================================
# HELPERS
# ============================================================
def clean_cols(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df

def norm_code(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).upper().strip())

def norm_brand(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())

def compute_total_coverage(df):
    visited = pd.to_numeric(df["Total DR Visited"], errors="coerce").fillna(0)
    total = pd.to_numeric(df.get("Total DR Total", df.get("Total Dr Total", 0)), errors="coerce").fillna(0)
    return np.where(total > 0, (visited / total) * 100, 0).round(2)

# --- Minimal addition: normalize doctor id to avoid hidden-duplicate IDs ---
ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff"
def norm_doc_id(x):
    """
    Normalize doctor id (Assignment):
      - remove zero-widths, normalize NBSP, trim, uppercase,
      - remove all internal whitespace to prevent duplicate-like variants.
    """
    s = "" if pd.isna(x) else str(x)
    s = s.translate({ord(ch): None for ch in ZERO_WIDTH_CHARS})
    s = s.replace("\xa0", " ")  # NBSP -> space
    s = s.strip().upper()
    s = re.sub(r"\s+", "", s)
    return s
# ---------------------------------------------------------------------------

# ============================================================
# READ KPI (MASTER)
# ============================================================
print("[INFO] Reading KPI (MASTER)...")
kpi = clean_cols(pd.read_excel(KPI_FILE, sheet_name=KPI_SHEET, engine="openpyxl"))
kpi["Division"] = kpi["Division"].astype(str).str.strip()
kpi = kpi[kpi["Division"].isin(ALLOWED_DIVISIONS)].copy()
kpi["Employee Code"] = kpi["Employee Code"].map(norm_code)
kpi["_emp_key"] = kpi["Employee Code"]

kpi["Total DR Visited"] = pd.to_numeric(kpi["Total DR Visited"], errors="coerce").fillna(0)
kpi["Total DR Total"] = pd.to_numeric(kpi.get("Total DR Total", kpi.get("Total Dr Total", 0)), errors="coerce").fillna(0)
kpi["Total Coverage"] = compute_total_coverage(kpi)

# ============================================================
# READ COMEX AIL (HIERARCHY via EHIER_CD / PAR_EHIER_CD)
# ============================================================
print("[INFO] Reading COMEX AIL for hierarchy (EHIER_CD based)...")
comex = clean_cols(pd.read_excel(COMEX_FILE, sheet_name=COMEX_SHEET, engine="openpyxl"))

# Normalize keys we will use
for c in ["DIVISION", "EMPLOYEE_CODE", "EHIER_CD", "PAR_EHIER_CD",
          "EMPLOYEE_NAME", "PAR_EMPLOYEE_NAME"]:
    if c not in comex.columns:
        comex[c] = ""

comex["DIVISION"] = comex["DIVISION"].astype(str).str.strip()
comex["EMPLOYEE_CODE"] = comex["EMPLOYEE_CODE"].map(norm_code)
comex["EHIER_CD"] = comex["EHIER_CD"].map(norm_code)
comex["PAR_EHIER_CD"] = comex["PAR_EHIER_CD"].map(norm_code)
comex["EMPLOYEE_NAME"] = comex["EMPLOYEE_NAME"].astype(str).str.strip()
comex["PAR_EMPLOYEE_NAME"] = comex["PAR_EMPLOYEE_NAME"].astype(str).str.strip()

# Keep only divisions of interest
comex = comex[comex["DIVISION"].isin(ALLOWED_DIVISIONS)].copy()

# Build fast lookups on the EHIER tree:
# - node_by_ehier: EHIER_CD -> row (Division, EMPLOYEE_CODE, EMPLOYEE_NAME, PAR_EHIER_CD)
node_by_ehier = (
    comex
    .drop_duplicates(subset=["DIVISION", "EHIER_CD"])
    .set_index("EHIER_CD")[["DIVISION", "EMPLOYEE_CODE", "EMPLOYEE_NAME", "PAR_EHIER_CD"]]
    .to_dict(orient="index")
)

# Map EMPLOYEE_CODE -> EHIER_CD (the node representing that employee, if present)
emp_to_ehier = {}
tmp_map = comex.dropna(subset=["EMPLOYEE_CODE", "EHIER_CD"])
for _, r in tmp_map.iterrows():
    if r["EMPLOYEE_CODE"]:
        emp_to_ehier[(str(r["DIVISION"]).strip(), r["EMPLOYEE_CODE"])] = r["EHIER_CD"]

def climb_to(role_prefix, start_ehier):
    """
    From a given EHIER code, climb parent links until a node whose EHIER_CD starts
    with the given prefix is found. Return (EHIER_CD, EMPLOYEE_NAME).
    If not found, return ("", "").
    """
    seen = set()
    cur = start_ehier
    while cur and cur in node_by_ehier and cur not in seen:
        seen.add(cur)
        row = node_by_ehier[cur]
        if cur.startswith(role_prefix):
            # Found the requested role
            return cur, row.get("EMPLOYEE_NAME", "")
        # Move to parent
        cur = row.get("PAR_EHIER_CD", "")
    return "", ""

# Build final hierarchy table keyed by (Division, _emp_key)
records_h = []
for _, r in kpi[["Division", "_emp_key"]].drop_duplicates().iterrows():
    div = str(r["Division"]).strip()
    emp = r["_emp_key"]

    zbm_code, zbm_name = "", ""
    abm_code, abm_name = "", ""

    # Find the employee's EHIER node first
    start_ehier = emp_to_ehier.get((div, emp), "")
    if start_ehier:
        # ABM = nearest IA..., ZBM = nearest RG...
        abm_code, abm_name = climb_to("IA", start_ehier)
        zbm_code, zbm_name = climb_to("RG", start_ehier)
    else:
        # Fallback: if EMPLOYEE_CODE not mapped to an EHIER node in this division,
        # try to locate via COMEX rows that share the same EMPLOYEE_CODE (rare)
        cand = comex[(comex["DIVISION"] == div) & (comex["EMPLOYEE_CODE"] == emp)]
        if not cand.empty:
            start_ehier = cand.iloc[0]["EHIER_CD"]
            abm_code, abm_name = climb_to("IA", start_ehier)
            zbm_code, zbm_name = climb_to("RG", start_ehier)
        # else: remain blanks

    records_h.append({
        "Division": div,
        "_emp_key": emp,
        "ABM CODE": abm_code,
        "ABM NAME": abm_name,
        "ZBM CODE": zbm_code,
        "ZBM NAME": zbm_name,
    })

hierarchy = pd.DataFrame.from_records(records_h)

# ============================================================
# READ DCR (DOCTOR-LEVEL BRAND AGGREGATION)
# ============================================================
print("[INFO] Reading DCR SlotMAX...")
dcr = clean_cols(pd.read_excel(DCR_FILE, sheet_name="4Div", engine="openpyxl"))
dcr["Division"] = dcr["Division"].astype(str).str.strip()
dcr = dcr[dcr["Division"].isin(ALLOWED_DIVISIONS)].copy()
dcr["_emp_key"] = dcr["User: Alias"].map(norm_code)

brand_cols = [c for c in dcr.columns if c.startswith("Brand")]
rx_cols = [c for c in dcr.columns if c.startswith("Rx/Month")]

# Normalize
dcr[brand_cols] = dcr[brand_cols].fillna("").astype(str)
dcr[rx_cols] = dcr[rx_cols].apply(pd.to_numeric, errors="coerce")

# >>> Minimal fix: normalize 'Assignment' so variants collapse to one doctor <<<
DOCTOR_ID_COL = "Assignment"
dcr[DOCTOR_ID_COL] = dcr[DOCTOR_ID_COL].apply(norm_doc_id)
# >>> End minimal fix <<<

# Doctor key parts (account/customer code) for identification
ACCOUNT_CUSTOMER_COL = None
for c in ["Account: Customer Code", "Customer Code", "Account"]:
    if c in dcr.columns:
        ACCOUNT_CUSTOMER_COL = c
        break
if ACCOUNT_CUSTOMER_COL is not None:
    dcr["_acc_code"] = dcr[ACCOUNT_CUSTOMER_COL].map(lambda x: norm_code(x) if pd.notna(x) else "")
else:
    dcr["_acc_code"] = ""

# Stable doctor identity:
# Prefer Account ID_18 (unique account id), else Account: Customer Code, else Assignment.
ACCOUNT_ID_COL = "Account ID_18" if "Account ID_18" in dcr.columns else None
if ACCOUNT_ID_COL is not None:
    dcr["_acc_id"] = dcr[ACCOUNT_ID_COL].map(lambda x: norm_code(x) if pd.notna(x) else "")
else:
    dcr["_acc_id"] = ""

DOCTOR_KEY_COL = "_doctor_key"
dcr[DOCTOR_KEY_COL] = np.where(
    dcr["_acc_id"] != "",
    dcr["_acc_id"],
    np.where(dcr["_acc_code"] != "", dcr["_acc_code"], dcr[DOCTOR_ID_COL]),
)

def _row_has_all_required_brands(row, brand_cols, rx_cols, required_norm):
    """True if this row has every required brand with non-NaN, >= 0 Rx."""
    for rb_norm in required_norm:
        found = False
        for b_col, r_col in zip(brand_cols, rx_cols):
            if norm_brand(row[b_col]) == rb_norm:
                v = row[r_col]
                if not pd.isna(v) and v >= 0:
                    found = True
                break
        if not found:
            return False
    return True

# ============================================================
# DOCTOR-LEVEL CONSOLIDATION
# - A doctor counts for a TBM ONLY IF all required division-brands
#   appear with Rx not NaN and not negative (zero is allowed).
# - A doctor with multiple rows is counted ONCE if, across rows,
#   all required brands meet the rule.
# - Division 42 only: at least one row must have all 4 required brands
#   with valid Rx (no cross-row aggregation-only validity).
# ============================================================
records = []
for (div, emp, doc_key), grp in dcr.groupby(["Division", "_emp_key", DOCTOR_KEY_COL]):
    brand_rx = {}
    # Accumulate brand→rx across ALL rows for that doctor; prefer first non-NaN
    for _, row in grp.iterrows():
        for b_col, r_col in zip(brand_cols, rx_cols):
            brand = norm_brand(row[b_col])
            if not brand:
                continue
            rx = row[r_col]
            if brand not in brand_rx or (pd.isna(brand_rx[brand]) and not pd.isna(rx)):
                brand_rx[brand] = rx

    required = DIVISION_BRAND_MAP.get(div, [])
    required_norm = [norm_brand(rb) for rb in required]
    valid = True
    for rb in required:
        rb_norm = norm_brand(rb)
        if rb_norm not in brand_rx:
            valid = False
            break
        v = brand_rx[rb_norm]
        if pd.isna(v) or (v < 0):
            valid = False
            break

    # Division 42: require at least one row to have all 4 required brands (avoids
    # counting doctors who only get all 4 when aggregating across rows)
    if valid and div == "42":
        any_row_ok = False
        for _, row in grp.iterrows():
            if _row_has_all_required_brands(row, brand_cols, rx_cols, required_norm):
                any_row_ok = True
                break
        if not any_row_ok:
            valid = False

    if valid:
        acc_code = grp["_acc_code"].iloc[0] if "_acc_code" in grp.columns else ""
        records.append({
            "Division": div,
            "_emp_key": emp,
            DOCTOR_KEY_COL: doc_key,
            "Account: Customer Code": acc_code,
        })

valid_doctors = pd.DataFrame(records)
# Defensive dedup (no logic change; protects against accidental duplicates)
if not valid_doctors.empty:
    valid_doctors = valid_doctors.drop_duplicates(["Division", "_emp_key", DOCTOR_KEY_COL])

# Unique count of doctors per TBM
rx_cnt = (
    valid_doctors
    .groupby(["Division", "_emp_key"])[DOCTOR_KEY_COL]
    .nunique()
    .reset_index(name="Number of doctors with Rx entered")
)

# ============================================================
# MERGE ON KPI
# ============================================================
out = kpi.merge(hierarchy, on=["Division", "_emp_key"], how="left")
out = out.merge(rx_cnt, on=["Division", "_emp_key"], how="left")
out["Number of doctors with Rx entered"] = (
    out["Number of doctors with Rx entered"].fillna(0).astype(int)
)

# Preserve your existing rule: if no visits, force Rx-entered count to 0
out.loc[out["Total DR Visited"] == 0, "Number of doctors with Rx entered"] = 0

# ============================================================
# RCPA COVERAGE
# ============================================================
out["RCPA Coverage"] = np.clip(
    np.where(
        out["Total DR Visited"] > 0,
        (out["Number of doctors with Rx entered"] / out["Total DR Visited"]) * 100,
        0
    ).round(2),
    0, 100
)

# ============================================================
# FINAL OUTPUT
# ============================================================
final = out.copy()
final["Total Dr Total"] = final["Total DR Total"]

for c in FINAL_COLS:
    if c not in final.columns:
        final[c] = ""

final = final[FINAL_COLS]
final.to_excel(OUTPUT_FILE, index=False)

print("\n[OK] FINAL_OUTPUT.xlsx created successfully")
print("[OK] EHIER_CD-based hierarchy applied (IA* -> ABM, RG* -> ZBM)")
print("[OK] Doctor-level brand-wise (>=0) logic preserved; Assignment normalized; Div42 row-level check applied")
print("[OK] All other logic preserved")
