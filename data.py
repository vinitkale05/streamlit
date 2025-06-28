import streamlit as st
import pandas as pd
import requests
import time
import io

st.set_page_config(page_title="Vehicle Specs Cleaner", layout="wide")

st.title("📄 Vehicle Specs Cleaner and CSV Downloader")
api_url = "http://internal-apis.intangles.com/specs/listV2"

@st.cache_data(show_spinner=True)
def fetch_and_clean_specs():
    session = requests.Session()
    retries = 5
    timeout = 60
    specs_list = []

    for i in range(retries):
        try:
            res = session.get(api_url, timeout=timeout)
            res.raise_for_status()
            data = res.json()
            specs_list = data.get("specs", [])
            break
        except Exception as e:
            time.sleep(2 ** i)

    if not specs_list:
        st.error("❌ Failed to fetch specs data or data is empty.")
        return None

    df = pd.DataFrame(specs_list)
    original_count = len(df)

    # Drop rows where all columns except specs_id are NaN or empty
    if 'specs_id' in df.columns:
        non_id_cols = [col for col in df.columns if col != 'specs_id']
        df = df.dropna(subset=non_id_cols, how='all')

    # Remove rows where specs_id is missing or empty
    df = df[df['specs_id'].notna() & (df['specs_id'].astype(str).str.strip() != "")]

    # Drop completely duplicate rows (all columns match)
    before_dups = len(df)
    df = df.drop_duplicates(keep='first')
    after_dups = len(df)

    removed = original_count - len(df)
    return df, removed

# Fetch and clean on button click
if st.button("🚀 Fetch and Clean Specs"):
    with st.spinner("Fetching and cleaning data..."):
        result = fetch_and_clean_specs()

    if result:
        df_cleaned, rows_removed = result

        st.success(f"✅ Cleaned data loaded. {rows_removed} duplicate/empty rows removed.")
        st.write(f"🧾 Total Cleaned Rows: {len(df_cleaned)}")
        st.dataframe(df_cleaned)

        # Download cleaned CSV
        csv_buffer = io.StringIO()
        df_cleaned.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_bytes,
            file_name="vehicle_specs_cleaned.csv",
            mime="text/csv"
        )
