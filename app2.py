import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
import pdfplumber

st.title("M-Pesa Statement Cleaner")

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
                                # skip the very first table only
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

    # split completion time
    if "Completion Time" in df.columns:
        dt = pd.to_datetime(df["Completion Time"], errors="coerce", infer_datetime_format=True)
        df["completion_date"] = dt.dt.date.astype("string")
        df["completion_time"] = dt.dt.time.astype("string")

    # details clean
    df["Details_clean"] = df["Details"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    pattern = r"^(.*?)\s(?:to|from)\s(.*?)\s*-\s*(.*)$"
    extracted = df["Details_clean"].str.extract(pattern, flags=re.IGNORECASE)
    extracted.columns = ["transaction_type","account_number","entity_name"]
    df[["transaction_type","account_number","entity_name"]] = extracted

    # keep only first word of entity_name
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

    # optional pythonic aliases
    if "Receipt No" in df.columns:
        col = df["Receipt No"]
        df["Receipt_No"] = col.iloc[:,0] if isinstance(col, pd.DataFrame) else col

    if "Transaction Status" in df.columns:
        col = df["Transaction Status"]
        df["Transaction_Status"] = col.iloc[:,0] if isinstance(col, pd.DataFrame) else col

    if "Paid in" in df.columns:
        col = df["Paid in"]
        df["Paid_in"] = col.iloc[:,0] if isinstance(col, pd.DataFrame) else col


    # snake_case everything
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

uploaded = st.file_uploader("Upload M-Pesa statement (PDF or CSV)", type=["pdf","csv"])

if uploaded:
    if uploaded.name.lower().endswith(".pdf"):
        df_raw = parse_mpesa_pdf(uploaded)
    else:
        df_raw = pd.read_csv(uploaded, on_bad_lines="skip")  # avoid parser errors

    df_clean = clean_mpesa_df(df_raw)

    st.subheader("Cleaned Preview")
    st.dataframe(df_clean.head(20))

    buf = BytesIO()
    df_clean.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        label="Download Cleaned CSV",
        data=buf,
        file_name="mpesa_cleaned.csv",
        mime="text/csv"
    )
