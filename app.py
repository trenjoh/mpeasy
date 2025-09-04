# The line `!pip intapp streamlit` is not correct.
# If you want to install the `streamlit` package, use:
# !pip install streamlit 
    # --- Cleaning steps ---
import streamlit as st
import pandas as pd
import numpy as np
import regex
from io import BytesIO
from datetime import datetime
    
st.set_page_config(page_title="Data Cleaning App", layout="wide")
st.title("Data Cleaning App")
    
uploaded_file = st.file_uploader("upload your Mpesa statement", type=["pdf"])
if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Details_clean'] = df['Details'].str.replace(r'\s+', ' ', regex=True).str.strip()
        pattern = r'^(.*?)\s(?:to|from)\s(.*?)\s-\s(.*)$'
        df[['transaction_type', 'account_number', 'entity_name']] = df['Details_clean'].str.extract(pattern)

        df["entity_name"] = df["entity_name"].str.split().str[0]

        df.loc[
            (df["entity_name"].isna() | (df["entity_name"].str.strip() == "")) &
            (df["Details_clean"].str.contains("Airtime purchase", case=False, na=False)),
            "entity_name"
        ] = "Airtime"

        df.loc[
            df["Details_clean"].str.contains("Customer Transfer of Funds Charge", case=False, na=False),
            ["transaction_type", "account_number", "entity_name"]
        ] = ["Customer Transfer", "100001", "MPesa"]

        df.loc[
            df["Details_clean"].str.contains("Pay Bill Charge", case=False, na=False),
            ["transaction_type", "account_number", "entity_name"]
        ] = ["Customer Transfer", "100001", "MPesa"]

        df['Receipt_No'] = df['Receipt No']
        df['Transaction_Status'] = df['Transaction Status']
        df['Paid_in'] = df['Paid in']

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # preview)_cleaned_data
        st.subheader("Preview of Cleaned Data")
        st.dataframe(df.head(10))

        buffer = BytesIO()
        df.to_csv(buffer, index = False)
        buffer.seek(0)
        
        st.download_button(
            label = "Dowload Cleaned CSV",
            data = buffer,
            file_name = f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime = 'text/csv'
        )
        # def to_excel(df):
        #     output = BytesIO()
        #     writer = pd.ExcelWriter(output, engine='xlsxwriter')
        #     df.to_excel(writer, index=False, sheet_name='Sheet1')
        #     writer.save()
        #     processed_data = output.getvalue()
        #     return processed_data
    
        # excel_data = to_excel(df)
    
        # st.download_button(
        #     label="Download Cleaned Data as Excel",
        #     data=excel_data,
        #     file_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
        #     mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        