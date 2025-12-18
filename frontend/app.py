import sys
import os

# Allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

# Import recommendation logic
from api.main import recommend, RecommendRequest

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="wide"
)

# ---------------- BASIC STYLING ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("SHL Assessment Recommender")
st.markdown(
    "Provide a job description, hiring query, or job URL to receive relevant SHL assessment recommendations."
)
st.markdown("---")

# ---------------- INPUT SECTION ----------------
query = st.text_area(
    "Job Description / Hiring Query / URL",
    placeholder=(
        "Example:\n"
        "Hiring an accountant with strong numerical ability, attention to detail, "
        "and compliance knowledge."
    ),
    height=180
)

# ---------------- ACTION ----------------
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a valid job description or query.")
    else:
        with st.spinner("Analyzing requirements and matching assessments..."):
            try:
                # Create request object
                req = RecommendRequest(query=query)

                # Call recommendation logic
                result = recommend(req)
                recs = result.get("recommendations", [])

                st.markdown("---")
                st.markdown("### Recommended SHL Assessments")

                if not recs:
                    st.info("No relevant assessments found.")
                else:
                    # Create clean DataFrame
                    df = pd.DataFrame(recs)

                    # Rename columns for professional display
                    df = df.rename(columns={
                        "assessment_name": "Assessment Name",
                        "assessment_url": "Assessment URL"
                    })

                    # Keep only required columns
                    df = df[["Assessment Name", "Assessment URL"]]

                    # Make URLs clickable
                    df["Assessment URL"] = df["Assessment URL"].apply(
                        lambda x: f"[View Assessment]({x})"
                    )

                    st.info(f"{len(df)} relevant assessments identified")

                    # Display table
                    st.markdown(
                        df.to_markdown(index=False),
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "SHL Assessment Recommendation System | Semantic retrieval over SHL public assessment catalog"
)
