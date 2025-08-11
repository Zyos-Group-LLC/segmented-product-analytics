
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Segmented Product Analytics — Pro", layout="wide")
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_data(base: Path):
    fact = pd.read_csv(base / "fact_orders_monthly.csv")
    # type coercion
    for col in ["Year","Month"]:
        if col in fact.columns:
            fact[col] = pd.to_numeric(fact[col], errors="coerce").astype("Int64")
    for col in ["Revenue","Orders"]:
        if col in fact.columns:
            fact[col] = pd.to_numeric(fact[col], errors="coerce").fillna(0.0)
    # create Year-Month string
    if set(["Year","Month"]).issubset(fact.columns):
        fact["Year-Month"] = fact["Year"].astype(str) + "-" + fact["Month"].astype(int).astype(str).str.zfill(2)
    # synthetic date for recency/quarters
    if set(["Year","Month"]).issubset(fact.columns):
        fact["_date"] = pd.to_datetime(fact["Year"].astype(str) + "-" + fact["Month"].astype(int).astype(str) + "-01", errors="coerce")
    return fact

def compute_first_purchase(df: pd.DataFrame) -> pd.Series:
    if not set(["Account","_date"]).issubset(df.columns): 
        return pd.Series(index=df.index, dtype="datetime64[ns]")
    first = df.groupby("Account")["_date"].transform("min")
    return first

def add_new_flag(df: pd.DataFrame, window_min=None, window_max=None, mode="lifetime"):
    # mode: "lifetime" uses first purchase overall; "window" uses first purchase within the selected window
    if not set(["Account","_date"]).issubset(df.columns):
        df["Cust_Type"] = "Unknown"
        return df
    d = df.copy()
    if mode == "window" and window_min is not None and window_max is not None:
        # first purchase within the window per account
        mask = (d["_date"] >= window_min) & (d["_date"] <= window_max)
        first_in_win = d[mask].groupby("Account")["_date"].transform("min")
        d["Cust_Type"] = np.where(mask & (d["_date"] == first_in_win), "New (Window)", "Returning/Outside")
    else:
        first = d.groupby("Account")["_date"].transform("min")
        d["Cust_Type"] = np.where(d["_date"] == first, "New (Lifetime)", "Returning")
    return d

def add_recency_bucket(df: pd.DataFrame):
    if "_date" not in df.columns: 
        df["Recency_Bucket"] = "Unknown"
        return df
    last_per_acct = df.groupby("Account")["_date"].transform("max")
    max_date = df["_date"].max()
    days = (max_date - last_per_acct).dt.days
    bins = [-1, 30, 90, 180, 365, 999999]
    labels = ["0-30d","31-90d","91-180d","181-365d",">365d"]
    df["Recency_Bucket"] = pd.cut(days, bins=bins, labels=labels)
    return df

def add_abc(df: pd.DataFrame, scope="Global: Account & Bundle"):
    # scope options:
    # - Global: Account & Bundle (same as before, separate ABC for Account and for Bundle globally)
    # - Account within Bundle (per Bundle, rank Accounts by revenue in that bundle)
    # - Bundle within Account (per Account, rank Bundles)
    d = df.copy()
    if scope == "Global: Account & Bundle":
        if "Account" in d.columns and "Revenue" in d.columns:
            d = _abc_for_dim(d, "Account")
        if "Bundle" in d.columns and "Revenue" in d.columns:
            d = _abc_for_dim(d, "Bundle")
    elif scope == "Account within Bundle":
        if set(["Account","Bundle","Revenue"]).issubset(d.columns):
            d = _abc_nested(d, outer="Bundle", inner="Account", abc_col="AcctInBundle_ABC", pareto_col="AcctInBundle_P80")
    elif scope == "Bundle within Account":
        if set(["Account","Bundle","Revenue"]).issubset(d.columns):
            d = _abc_nested(d, outer="Account", inner="Bundle", abc_col="BundleInAcct_ABC", pareto_col="BundleInAcct_P80")
    return d

def _abc_for_dim(d: pd.DataFrame, dim: str) -> pd.DataFrame:
    agg = d.groupby(dim, as_index=False).agg(Revenue=("Revenue","sum"))
    agg = agg.sort_values("Revenue", ascending=False)
    total = agg["Revenue"].sum() or 1.0
    agg["CumShare"] = agg["Revenue"].cumsum() / total
    def abc(x):
        if x <= 0.80: return "A"
        if x <= 0.95: return "B"
        return "C"
    agg["ABC"] = agg["CumShare"].apply(abc)
    agg["Pareto80"] = np.where(agg["CumShare"] <= 0.80, "Yes", "No")
    d = d.merge(agg[[dim,"ABC","Pareto80"]], on=dim, how="left")
    d = d.rename(columns={"ABC": f"{dim}_ABC", "Pareto80": f"{dim}_Pareto80"})
    return d

def _abc_nested(d: pd.DataFrame, outer: str, inner: str, abc_col: str, pareto_col: str) -> pd.DataFrame:
    agg = d.groupby([outer, inner], as_index=False).agg(Revenue=("Revenue","sum"))
    agg = agg.sort_values([outer, "Revenue"], ascending=[True, False])
    # compute cumulative share within each outer
    agg["CumShare"] = agg.groupby(outer)["Revenue"].cumsum() / agg.groupby(outer)["Revenue"].transform("sum").replace(0, np.nan)
    def abc(x):
        if pd.isna(x): return "C"
        if x <= 0.80: return "A"
        if x <= 0.95: return "B"
        return "C"
    agg[abc_col] = agg["CumShare"].apply(abc)
    agg[pareto_col] = np.where(agg["CumShare"] <= 0.80, "Yes", "No")
    d = d.merge(agg[[outer, inner, abc_col, pareto_col]], on=[outer, inner], how="left")
    return d

def quarter_of_date(d: pd.Series) -> pd.Series:
    return "Q" + ((d.dt.month.sub(1)//3)+1).astype(str) + "-" + d.dt.year.astype(str)

def build_cohort_heatmap(df: pd.DataFrame):
    # Cohort based on first purchase quarter per account; activity quarter on axes
    if not set(["Account","_date","Revenue"]).issubset(df.columns):
        return None
    d = df.copy()
    first_date = d.groupby("Account")["_date"].transform("min")
    d["Cohort"] = quarter_of_date(first_date)
    d["ActivityQ"] = quarter_of_date(d["_date"])
    pivot = d.pivot_table(index="Cohort", columns="ActivityQ", values="Revenue", aggfunc="sum").fillna(0.0)
    if pivot.empty:
        return None
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", title="Revenue Heatmap by Cohort (first-purchase quarter) vs Activity Quarter")
    return fig

# Load
fact = load_data(DATA_DIR)

# Sidebar controls
with st.sidebar:
    st.header("Filters")
    years = sorted(fact["Year"].dropna().unique().tolist()) if "Year" in fact.columns else []
    year_sel = st.multiselect("Year", years, default=years)
    work = fact[fact["Year"].isin(year_sel)] if year_sel else fact.copy()
    months = sorted(work["Month"].dropna().unique().tolist()) if "Month" in work.columns else []
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_sel = st.multiselect("Month", months, default=months, format_func=lambda x: month_names.get(int(x), str(x)))
    accounts = sorted(work["Account"].dropna().unique().tolist()) if "Account" in work.columns else []
    bundles = sorted(work["Bundle"].dropna().unique().tolist()) if "Bundle" in work.columns else []
    acct_sel = st.multiselect("Account", accounts, default=accounts[:25] if len(accounts)>25 else accounts)
    bund_sel = st.multiselect("Bundle", bundles, default=bundles[:25] if len(bundles)>25 else bundles)
    st.divider()
    st.subheader("Segmentation")
    new_mode = st.radio("New customer definition", ["Lifetime", "Within current filters"], horizontal=True)
    abc_scope = st.selectbox("ABC scope", ["Global: Account & Bundle","Account within Bundle","Bundle within Account"])
    group_opts = [c for c in ["Year","Month","Year-Month","Account","Bundle","Cust_Type","Recency_Bucket","Account_ABC","Bundle_ABC","Account_Pareto80","Bundle_Pareto80","AcctInBundle_ABC","BundleInAcct_ABC","AcctInBundle_P80","BundleInAcct_P80"] if c in fact.columns]
    group_by = st.multiselect("Group by", group_opts, default=["Bundle","Cust_Type"] if "Bundle" in group_opts else group_opts[:2])
    export = st.checkbox("Enable CSV export", value=True)

# Apply basic filters
df = fact.copy()
if year_sel: df = df[df["Year"].isin(year_sel)]
if month_sel: df = df[df["Month"].isin(month_sel)]
if acct_sel: df = df[df["Account"].isin(acct_sel)]
if bund_sel: df = df[df["Bundle"].isin(bund_sel)]

# Window for "New" definition
win_min = df["_date"].min() if not df.empty else None
win_max = df["_date"].max() if not df.empty else None
df = add_new_flag(df, window_min=win_min, window_max=win_max, mode=("window" if new_mode == "Within current filters" else "lifetime"))

# Recency
df = add_recency_bucket(df)

# ABC scope application
df = add_abc(df, scope=abc_scope)

# Header & KPIs
st.title("Segmented Product Analytics — Pro")
st.caption("Zyos Group — ABC tiers, Pareto 80/20, New vs Returning (lifetime or within filter window), Recency, and Cohort heatmaps.")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Revenue", f"${df['Revenue'].sum():,.0f}" if "Revenue" in df.columns else "n/a")
c2.metric("Orders", int(df["Orders"].sum()) if "Orders" in df.columns else 0)
c3.metric("Accounts", df["Account"].nunique() if "Account" in df.columns else 0)
c4.metric("Bundles", df["Bundle"].nunique() if "Bundle" in df.columns else 0)

st.divider()
left, right = st.columns([1.25,1])

# Revenue by grouping
if group_by:
    agg = df.groupby(group_by, as_index=False).agg(Revenue=("Revenue","sum"), Orders=("Orders","sum"))
    if not agg.empty:
        top = agg.sort_values("Revenue", ascending=False).head(30)
        xdim = group_by[0]
        color = group_by[1] if len(group_by) > 1 else None
        title = "Revenue by " + " / ".join(group_by)
        fig = px.bar(top, x=xdim, y="Revenue", color=color, barmode="group", text_auto=".2s", title=title)
        fig.update_layout(xaxis_tickangle=-30, height=460, legend_title=None)
        left.plotly_chart(fig, use_container_width=True)
    else:
        left.info("No data for the selected grouping.")
else:
    left.info("Pick at least one grouping.")

# Trend
if set(["Year","Month","Revenue"]).issubset(df.columns):
    tr = df.groupby(["Year","Month"], as_index=False).agg(Revenue=("Revenue","sum"))
    tr["Year-Month"] = tr["Year"].astype(str) + "-" + tr["Month"].astype(int).astype(str).str.zfill(2)
    figt = px.line(tr.sort_values("Year-Month"), x="Year-Month", y="Revenue", markers=True, title="Revenue Trend")
    figt.update_layout(height=460)
    right.plotly_chart(figt, use_container_width=True)
else:
    right.info("Trend requires Year, Month, Revenue.")

st.divider()
st.subheader("Cohort Heatmap (First Purchase Quarter x Activity Quarter)")
heat = build_cohort_heatmap(df)
if heat:
    st.plotly_chart(heat, use_container_width=True, height=520)
else:
    st.info("Not enough data to build cohort heatmap.")

st.subheader("Detailed Breakdown")
show_cols = [c for c in ["Year","Month","Year-Month","Account","Bundle","Cust_Type","Recency_Bucket",
                         "Account_ABC","Bundle_ABC","Account_Pareto80","Bundle_Pareto80",
                         "AcctInBundle_ABC","BundleInAcct_ABC","AcctInBundle_P80","BundleInAcct_P80",
                         "Orders","Revenue"] if c in df.columns]
st.dataframe(df[show_cols].sort_values(["Year","Month"] if set(["Year","Month"]).issubset(df.columns) else show_cols))

st.subheader("Summary by Segment")
if group_by:
    summary = df.groupby(group_by, as_index=False).agg(
        Revenue=("Revenue","sum"),
        Orders=("Orders","sum"),
        Accounts=("Account","nunique") if "Account" in df.columns else ("Orders","sum")
    ).sort_values("Revenue", ascending=False)
    st.dataframe(summary)

if export:
    st.download_button("Download filtered CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="filtered.csv", mime="text/csv")
