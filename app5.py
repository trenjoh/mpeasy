import streamlit as st
import pandas as pd
import numpy as np
import re
# import matplotlib.pyplot as plt
from io import BytesIO
import pdfplumber

st.title("M-Pesa Statement Analyzer & Transactions Search Engine")
st.markdown("""
This application allows you to upload your M-Pesa statement pdf. It processes/Cleans the data, allows you search and trace transactions and provides quick insights and visualizations of your past spendings.
- Dowload your M-Pesa statement from Mpesa app and Upload it here. """)

def safe_text(x):
    """Ensure text is safe and decodable"""
    try:
        if isinstance(x, str):
            return x.encode("utf-8", "ignore").decode("utf-8", "ignore")
        return x
    except Exception:
        return ""

def parse_mpesa_pdf(file_like) -> pd.DataFrame:
    expected_headers = {"completion time","details","paid in","withdrawn","balance","transaction status","receipt no"}
    frames = []
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for tbl_i,tbl in enumerate(tables):
                raw = pd.DataFrame(tbl)
                if raw.empty: 
                    continue
                                # skip table 1, pg 1
                if not frames and tbl_i == 0:
                    continue  
                # find best header row
                best_idx, best_hits = 0, -1
                for i, row in raw.iterrows():
                    vals = [str(x).strip().lower() for x in row.values]
                    hits = sum(v in expected_headers for v in vals)
                    if hits > best_hits:
                        best_idx, best_hits = i, hits
                header = raw.iloc[best_idx].astype(str).str.strip().tolist()
                body = raw.iloc[best_idx+1:].copy()
                body.columns = header
                body = body.applymap(safe_text)   
                frames.append(body)

    if not frames:
        raise ValueError("No tables detected in PDF. Make sure it’s the original M-Pesa PDF.")

    df = pd.concat(frames, ignore_index=True)

    # clean up
    df = df.dropna(how="all").copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    for c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace({"nan": np.nan})
        )
    return df

def clean_mpesa_df(df: pd.DataFrame) -> pd.DataFrame:
    original_cols = df.columns.tolist()

    # header renaming
    rename_map = {}
    for c in original_cols:
        c_norm = re.sub(r"\s+", " ", str(c)).strip().lower()
        if c_norm in ["completion time","date","time","completion_time"]:
            rename_map[c] = "Completion Time"
        elif c_norm in ["details","narration","description"]:
            rename_map[c] = "Details"
        elif c_norm in ["paid in","paid_in","credit","in"]:
            rename_map[c] = "Paid in"
        elif c_norm in ["withdrawn","debit","out"]:
            rename_map[c] = "Withdrawn"
        elif c_norm in ["balance","running balance","acct balance"]:
            rename_map[c] = "Balance"
        elif c_norm in ["transaction status","status"]:
            rename_map[c] = "Transaction Status"
        elif c_norm in ["receipt no","receipt_no","receipt"]:
            rename_map[c] = "Receipt No"

    if rename_map:
        df = df.rename(columns=rename_map)

    # ensure required cols
    for req in ["Completion Time","Details","Paid in","Withdrawn","Balance","Transaction Status","Receipt No"]:
        if req not in df.columns:
            df[req] = np.nan
    # split completion time properly
    if "Completion Time" in df.columns:
        dt = pd.to_datetime(df["Completion Time"], errors="coerce", infer_datetime_format=True)
        df["completion_date"] = dt.dt.date.astype(str)   # only date
        df["completion_time"] = dt.dt.time.astype(str)   # only time
        
    df= df.drop(columns=["Completion Time"], errors="ignore")   

    # details clean
    df["Details_clean"] = df["Details"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    pattern = r"^(.*?)\s(?:to|from)\s(.*?)\s*-\s*(.*)$"
    extracted = df["Details_clean"].str.extract(pattern, flags=re.IGNORECASE)
    extracted.columns = ["transaction_type","account_number","entity_name"]
    df[["transaction_type","account_number","entity_name"]] = extracted

    # keep only first word of entity_name//./.
    df["entity_name"] = df["entity_name"].apply(
        lambda x: str(x).split()[0] if pd.notna(x) and str(x).strip() else np.nan
    )

    # Airtime fill
    df.loc[
        (df["entity_name"].isna()) &
        (df["Details_clean"].str.contains("Airtime purchase", case=False, na=False)),
        "entity_name"
    ] = "Airtime"

    # Special fills
    mask_transfer_fee = df["Details_clean"].str.contains("Customer Transfer of Funds Charge", case=False, na=False)
    df.loc[mask_transfer_fee, ["transaction_type","account_number","entity_name"]] = ["Customer Transfer","100001","MPesa"]

    mask_paybill_fee = df["Details_clean"].str.contains("Pay Bill Charge", case=False, na=False)
    df.loc[mask_paybill_fee, ["transaction_type","account_number","entity_name"]] = ["Customer Transfer","100001","MPesa"]

    #pythonic way
    if "Receipt No" in df.columns:
        col = df["Receipt No"]
        df["Receipt_No"] = col.iloc[:,0] if isinstance(col, pd.DataFrame) else col

    if "Transaction Status" in df.columns:
        col = df["Transaction Status"]
        df["Transaction_Status"] = col.iloc[:,0] if isinstance(col, pd.DataFrame) else col

    if "Paid in" in df.columns:
        col = df["Paid in"]
        df["Paid_in"] = col.iloc[:,0] if isinstance(col, pd.DataFrame) else col


    # snake_case pythonically
    df.columns = (
        pd.Series(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # reorder
    cols = list(df.columns)
    for col in ["completion_date","completion_time"]:
        if col in cols:
            cols.remove(col)
    if "receipt_no" in cols:
        insert_pos = cols.index("receipt_no") + 1
        cols[insert_pos:insert_pos] = ["completion_date","completion_time"]
    else:
        cols = ["completion_date","completion_time"] + cols
    cols = [c for i, c in enumerate(cols) if c not in cols[:i]]  # dedupe
        # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    df = df.reindex(columns=cols)

    return df

uploaded = st.file_uploader("Upload M-Pesa statement (PDF)", type=["pdf","csv"])

if uploaded:
    if uploaded.name.lower().endswith(".pdf"):
        df_raw = parse_mpesa_pdf(uploaded)
    else:
        df_raw = pd.read_csv(uploaded, on_bad_lines="skip")  # avoid parser errors

    df_clean = clean_mpesa_df(df_raw)
    
    for col in ["paid_in", "withdraw_n", "balance"]:
        df_clean[col] = (
            df_clean[col]
            .astype(str)                               # ensure string
            .str.replace(",", "", regex=False)         # remove commas
            .str.replace("—", "0", regex=False)        # replace dashes with 0
            .str.replace(" ", "", regex=False)         # remove spaces
        )
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean= df_clean.drop(columns = ['withdrawn'])
    
    # Clean the 'paid_in' column
    df_clean["paid_in"] = (
        df_clean["paid_in"]
        .astype(str)                        # make sure it's string
        .str.replace(",", "", regex=False)  # remove commas
        .str.extract(r"([\d\.]+)")          # extract numeric part
        .astype(float)                      # convert to float
        .fillna(0)                          # replace NaN with 0
    )
    
    # Clean the 'paid_in' column
    df_clean["withdraw_n"] = (
        df_clean["withdraw_n"]
        .astype(str)                        # make sure it's string
        .str.replace(",", "", regex=False)  # remove commas
        .str.extract(r"([\d\.]+)")          # extract numeric part
        .astype(float)                      # convert to float
        .fillna(0)                          # replace NaN with 0
    )
    df_clean["completion_time"] = (
        df_clean["completion_time"]
        .astype(str)                                 
        .str.strip()                                  
        .str.replace(r"[^\w\s:/-]", "", regex=True)
    )

    df_clean["completion_time"] = pd.to_datetime(
        df_clean["completion_date"].astype(str) + " " + df_clean["completion_time"].astype(str),
        errors="coerce"
    )


    st.subheader("...Search Transaction by Receipt No.")

    # search box
    search_id = st.text_input("Enter Receipt/Transaction Number:").upper().strip()

    if search_id:
        result = df_clean[df_clean["receipt_no"].astype(str).str.contains(search_id, case=False, na=False)]

        if not result.empty:
            st.write(" Transaction Found✅:")
            st.dataframe(result, use_container_width=True)
        else:
            st.warning("⚠️ No transaction found with that Receipt No.")

    st.subheader("...Search name of the sender/ reciever.")

    # search box2
    search_name = st.text_input("Enter Name:").upper().strip()

    if search_name:
        result = df_clean[df_clean["details"].astype(str).str.contains(search_name, case=False, na=False)]

        if not result.empty:
            st.write("Person/entity Found✅ :")
            st.dataframe(result, use_container_width=True)
        else:
            st.warning("⚠️ No person/entity with such a name.")


    st.subheader("Cleaned Preview : Analyst Mode : Data")

    # Initialize toggle state
    if "show_df" not in st.session_state:
        st.session_state.show_df = False

    # Buttons only set the state
    if not st.session_state.show_df:
        if st.button("show/hide dataframe"):
            st.session_state.show_df = True
    else:
        if st.button("show/hide dataframe"):
            st.session_state.show_df = False

    # Display dataframe if toggled
    if st.session_state.show_df:
        st.dataframe(df_clean, use_container_width=True)

    
    buf = BytesIO()
    df_clean.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        label="Download Cleaned CSV",
        data=buf,
        file_name="mpesa_cleaned.csv",
        mime="text/csv"
    )
    
    st.subheader("1. Transaction Exploration")
    # st.subheader("Visualizations")
    print(df_clean.columns)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots



    # datetime
    df_clean["completion_time"] = pd.to_datetime(df_clean["completion_time"])
    df_clean = df_clean.sort_values("completion_time")

    # Create subplot grid (1 row, 3 cols)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Paid In", "Withdrawn", "Balance"))

    # Paid In
    fig.add_trace(
        go.Scatter(
            x=df_clean["completion_time"], 
            y=df_clean["paid_in"],
            mode="lines",
            line=dict(color="lime", width=1),
            name="Paid In"
        ),
        row=1, col=1
    )
    # Withdrawn
    fig.add_trace(
        go.Scatter(
            x=df_clean["completion_time"], 
            y=df_clean["withdraw_n"],
            mode="lines",
            line=dict(color="red", width=1),
            name="Withdrawn"
        ),
        row=1, col=2
    )

    # Balance
    fig.add_trace(
        go.Scatter(
            x=df_clean["completion_time"], 
            y=df_clean["balance"],
            mode="lines",
            line=dict(color="cyan", width=1),
            name="Balance"
        ),
        row=2, col=1
    )

    # Layout settings (dark theme + linear ticks)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        width=900,
        title=" General M-Pesa Transactions Overview",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Force evenly spaced y ticks
    fig.update_yaxes(nticks=6, row=1, col=1)
    fig.update_yaxes(nticks=6, row=1, col=2)
    fig.update_yaxes(nticks=6, row=1, col=3)
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    st.write("- Top 5 Transaction types")
    import plotly.express as px
 

    # Top 5 transaction types only
    tx_counts = df_clean["details"].value_counts().nlargest(5).reset_index()
    tx_counts.columns = ["transaction_type", "count"]

    # Pie Chart
    fig_pie = px.pie(
        tx_counts,
        names="transaction_type",
        values="count",
        title="- Pie Chart",
        color_discrete_sequence=px.colors.qualitative.Set3,
        width  = 600,
        height  = 600
    )

    # Donut Chart
    fig_donut = px.pie(
        tx_counts,
        names="transaction_type",
        values="count",
        title="- Donut Chart",
        hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Set3,
        width  = 600,
        height  = 600
    )
    
    # Display side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_pie, width=False)
    with col2:
        st.plotly_chart(fig_donut, width=True)
        
    st.write("Top 5 Entities by Paid In")
    summary = (
        df_clean.groupby("entity_name")["paid_in"]
        .sum()
        .reset_index()
        .sort_values(by="paid_in", ascending=False)
        .head(5)
    )
    st.dataframe(summary, width = 'stretch')

    # Heat map
    df_clean["day_of_week"] = df_clean["completion_time"].dt.day_name()
    df_clean["hour"] = df_clean["completion_time"].dt.hour

    df_heatmap = (
        df_clean.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="count")
    )

    fig_heatmap = px.density_heatmap(
        df_heatmap,
        x="hour",
        y="day_of_week",
        z="count",
        color_continuous_scale="Viridis",
        title="Transaction Frequency (Day vs Hour)"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

#   Guage // fUNNEL
    st.subheader("2. Money Flow")
    st.write("Inflow vs Outflow")

    # calculate totals
    total_in = df_clean["paid_in"].sum()
    total_out = df_clean["withdraw_n"].sum()
    final_balance = df_clean["balance"].iloc[-1]  # last balance

    # Funnel chart (Inflow vs Outflow)
    funnel_fig = go.Figure(
        go.Funnel(
            y=["Total Inflow", "Total Outflow"],
            x=[total_in, total_out],
            textinfo="value+percent initial"
        )
    )
    funnel_fig.update_layout(title="Inflow vs Outflow Funnel")
    st.plotly_chart(funnel_fig, use_container_width=True)

    # Gauge chart (Balance retained % of inflows)
    retention_pct = (final_balance / total_in * 100) if total_in > 0 else 0
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=retention_pct,
            delta={"reference": 100, "relative": False},
            title={"text": "Retention % (Balance vs Inflow)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green"},
                "steps": [
                    {"range": [0, 30], "color": "red"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "lightgreen"},
                ],
            },
        )
    )
    st.plotly_chart(gauge_fig, use_container_width=True)


    st.subheader("3. Monthly Statement Summary")
    # Make sure completion_date is datetime
    df_clean["completion_date"] = pd.to_datetime(df_clean["completion_date"], errors="coerce")

    # Create a 'month' column (Year-Month format)
    df_clean["month"] = df_clean["completion_date"].dt.to_period("M")

    # Calculate inflow, outflow, closing balance
    monthly_summary = (
        df_clean.groupby("month")
        .agg(
            inflow=("paid_in", "sum"),
            outflow=("withdraw_n", "sum"),
            closing_balance=("balance", "last")  # last balance of the month
        )
        .reset_index()
    )

    st.dataframe(monthly_summary, use_container_width=True)

         