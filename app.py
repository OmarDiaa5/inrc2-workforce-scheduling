import json
import pandas as pd
import streamlit as st
import plotly.express as px

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

st.set_page_config(page_title="Nurse Rostering GA Dashboard", layout="wide")

st.title("Nurse Rostering (Genetic Algorithm) — Weekly Dashboard")
st.caption("Timetable + coverage diagnostics + KPIs (single-week INRC-II instance)")

# ---------- Sidebar: file loading ----------
st.sidebar.header("Data input")

mode = st.sidebar.radio("Load data from:", ["Local data/ folder", "Upload files"], index=0)

def load_csv(path):
    return pd.read_csv(path)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

df_long = None
df_pivot = None
df_diag = None
metrics = None

if mode == "Local data/ folder":
    try:
        df_long  = load_csv("data/best_schedule_long_inrc2.csv")
        df_pivot = load_csv("data/best_schedule_pivot_inrc2.csv", index_col=0)
        df_diag  = load_csv("data/coverage_diag.csv")
        metrics  = load_json("data/best_metrics.json")
        st.sidebar.success("Loaded from data/ ✅")
    except Exception as e:
        st.sidebar.error("Could not load from data/. Switch to Upload files or check filenames.")
        st.sidebar.code(str(e))

else:
    up_long  = st.sidebar.file_uploader("Upload best_schedule_long_inrc2.csv", type=["csv"])
    up_pivot = st.sidebar.file_uploader("Upload best_schedule_pivot_inrc2.csv", type=["csv"])
    up_diag  = st.sidebar.file_uploader("Upload coverage_diag.csv", type=["csv"])
    up_met   = st.sidebar.file_uploader("Upload best_metrics.json (optional)", type=["json"])

    if up_long and up_pivot and up_diag:
        df_long = pd.read_csv(up_long)
        df_pivot = pd.read_csv(up_pivot, index_col=0)
        df_diag = pd.read_csv(up_diag)
        if up_met:
            metrics = json.load(up_met)
        st.sidebar.success("Uploaded ✅")
    else:
        st.sidebar.info("Upload the 3 CSVs to enable the dashboard.")

# ---------- Validate ----------
if df_long is None or df_pivot is None or df_diag is None:
    st.stop()

required_long_cols = {"nurse","day","shiftType","skill"}
if not required_long_cols.issubset(df_long.columns):
    st.error(f"df_long must contain columns: {sorted(required_long_cols)}")
    st.stop()

# ---------- KPIs ----------
st.subheader("Key metrics")

c1, c2, c3, c4, c5 = st.columns(5)

total_assignments = len(df_long)
missing_opt_total = int(df_diag["missing_opt"].sum()) if "missing_opt" in df_diag.columns else None
under_min_total   = int(df_diag["under_min"].sum()) if "under_min" in df_diag.columns else None

c1.metric("Total assignments", f"{total_assignments}")

if under_min_total is not None:
    c2.metric("Total under_min (should be 0)", f"{under_min_total}")
else:
    c2.metric("Total under_min", "N/A")

if missing_opt_total is not None:
    c3.metric("Total missing_opt", f"{missing_opt_total}")
else:
    c3.metric("Total missing_opt", "N/A")

if metrics and "pref_viol" in metrics:
    c4.metric("Preference violations", f"{metrics['pref_viol']}")
else:
    c4.metric("Preference violations", "N/A")

if metrics and "cost" in metrics:
    c5.metric("Total cost", f"{metrics['cost']}")
else:
    c5.metric("Total cost", "N/A")

with st.expander("Show GA breakdown (best_metrics.json)"):
    if metrics:
        st.json(metrics)
    else:
        st.info("Upload best_metrics.json to show the GA breakdown.")

st.divider()

# ---------- Layout ----------
left, right = st.columns([1.15, 0.85])

# ========== Left: Timetable + Filters ==========
with left:
    st.subheader("Weekly timetable (nurse × day)")

    # Make sure the pivot has the expected day columns (some CSV exports may reorder)
    present_days = [d for d in DAYS if d in df_pivot.columns]
    if present_days:
        st.dataframe(df_pivot[present_days], use_container_width=True)
    else:
        st.dataframe(df_pivot, use_container_width=True)

    st.subheader("Filter assignments")
    nurse_sel = st.selectbox("Nurse", ["All"] + sorted(df_long["nurse"].unique().tolist()))
    day_sel   = st.selectbox("Day", ["All"] + DAYS)
    shift_sel = st.selectbox("Shift type", ["All"] + sorted(df_long["shiftType"].unique().tolist()))
    skill_sel = st.selectbox("Skill", ["All"] + sorted(df_long["skill"].unique().tolist()))

    f = df_long.copy()
    if nurse_sel != "All": f = f[f["nurse"] == nurse_sel]
    if day_sel != "All":   f = f[f["day"] == day_sel]
    if shift_sel != "All": f = f[f["shiftType"] == shift_sel]
    if skill_sel != "All": f = f[f["skill"] == skill_sel]

    st.dataframe(f.sort_values(["day","shiftType","skill","nurse"]), use_container_width=True)

# ========== Right: Coverage dashboard ==========
with right:
    st.subheader("Coverage vs Demand (Diagnostics)")

    # show biggest problems first
    if "under_min" in df_diag.columns and "missing_opt" in df_diag.columns:
        df_diag_sorted = df_diag.sort_values(["under_min","missing_opt"], ascending=False)
    else:
        df_diag_sorted = df_diag.copy()

    st.dataframe(df_diag_sorted.head(20), use_container_width=True)

    st.subheader("Where we miss optimal coverage")
    if "missing_opt" in df_diag.columns:
        miss = df_diag[df_diag["missing_opt"] > 0].copy()
        if len(miss) == 0:
            st.success("No missing optimal coverage 🎉")
        else:
            fig = px.bar(
                miss,
                x="day",
                y="missing_opt",
                color="shiftType" if "shiftType" in miss.columns else None,
                hover_data=[c for c in ["skill","cov","min","opt"] if c in miss.columns],
                category_orders={"day": DAYS},
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("missing_opt column not found in coverage_diag.csv")

    st.subheader("Hard constraint check (under_min)")
    if "under_min" in df_diag.columns:
        u = df_diag[df_diag["under_min"] > 0].copy()
        if len(u) == 0:
            st.success("All minimum requirements satisfied ✅")
        else:
            st.error("Some minimum requirements are NOT satisfied ❌")
            st.dataframe(u.sort_values("under_min", ascending=False), use_container_width=True)
    else:
        st.info("under_min column not found in coverage_diag.csv")

st.caption("Tip: For portfolio, include screenshots of the timetable + the missing_opt chart + KPIs.")
