import pandas as pd

# -------------------- INPUT FILES --------------------
ABM_PATH = "ABM_MSL_report_with_CustomerCode.csv"
HIER_PATH = "Hierarchy_Div75.csv"

# -------------------- REQUIRED COLUMNS --------------------
DATE_COL = "ABM MSL: Created Date"
CODE_COL = "Account: Customer Code"
ABM_COL = "ABM MSL: Owner Name"   # Column B (Manager)

# -------------------- LOAD FILES --------------------
abm = pd.read_csv(ABM_PATH, dtype=str)
hier = pd.read_csv(HIER_PATH, dtype=str)

# -------------------- CLEANING --------------------
def norm(x):
    return str(x).strip().lower()

abm["_code"] = abm[CODE_COL].apply(norm)
hier["_code"] = hier[CODE_COL].apply(norm)

# -------------------- MSL DOCTOR UNIVERSE --------------------
msl_codes = set(hier["_code"])
abm["MSL Dr"] = abm["_code"].isin(msl_codes)

# -------------------- DATE PROCESSING --------------------
abm["_date"] = pd.to_datetime(abm[DATE_COL], errors="coerce", dayfirst=True)
abm = abm[abm["_date"].notna()]

abm["_year"] = abm["_date"].dt.year
abm["_month"] = abm["_date"].dt.month

# -------------------- FILTER FOR REQUIRED MONTHS --------------------
TARGET_YEAR = 2026
TARGET_MONTHS = [2, 3]

filtered = abm[
    (abm["_year"] == TARGET_YEAR) &
    (abm["_month"].isin(TARGET_MONTHS))
]

# -------------------- FALLBACK (VERY IMPORTANT) --------------------
if filtered.empty:
    print("⚠️ No data for Feb/March 2026. Using latest available month instead.")

    latest_year = abm["_year"].max()
    latest_month = abm[abm["_year"] == latest_year]["_month"].max()

    filtered = abm[
        (abm["_year"] == latest_year) &
        (abm["_month"] == latest_month)
    ]

# -------------------- ONLY MSL DOCTORS --------------------
filtered_msl = filtered[filtered["MSL Dr"]]

# -------------------- UNIQUE VISITS --------------------
visited = filtered_msl.drop_duplicates(subset=["_code", "_month"])

# -------------------- ABM SUMMARY --------------------
summary = []

for (abm_name, month), group in visited.groupby([ABM_COL, "_month"]):

    # Total doctors under ABM (full universe)
    abm_all = abm[abm[ABM_COL] == abm_name]
    abm_msl_all = abm_all[abm_all["MSL Dr"]]

    total_doctors = abm_msl_all["_code"].nunique()
    visited_doctors = group["_code"].nunique()

    coverage = round((visited_doctors / total_doctors * 100), 2) if total_doctors else 0

    summary.append({
        "ABM Name": abm_name,
        "Month": month,
        "Total MSL Doctors": total_doctors,
        "Visited Doctors": visited_doctors,
        "Coverage %": coverage
    })

summary_df = pd.DataFrame(summary)

# -------------------- SAVE OUTPUT --------------------
summary_df.to_csv("final_abm_coverage_report.csv", index=False)

print("✅ Report Generated: final_abm_coverage_report.csv")
