import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd


# Import recommendation logic directly
from api.main import recommend, RecommendRequest

# Page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="centered"
)

# UI
st.title("SHL Assessment Recommender")
st.write(
    "Paste a job description, query, or URL to get recommended SHL assessments."
)

query = st.text_area(
    "Job Description / Query / URL",
    height=180
)

# Button action
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Finding best assessments..."):
            try:
                # Create request object
                req = RecommendRequest(query=query)

                # Call recommendation logic directly
                result = recommend(req)

                recs = result.get("recommendations", [])

                if not recs:
                    st.info("No recommendations found.")
                else:
                    df = pd.DataFrame(recs)
                    df.columns = ["Assessment Name", "Assessment URL"]

                    st.success(f"Found {len(df)} assessments")
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Error occurred: {e}")

