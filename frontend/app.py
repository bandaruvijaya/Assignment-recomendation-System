import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

# Recommendation logic will be imported lazily inside the action handler to avoid errors on import when
# GEMINI_API_KEY or embedding artifacts are missing in the environment.
# We'll import `recommend` and `RecommendRequest` inside the button click handler.
# Page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="centered"
)

# Inject glassy iOS-like styles and font
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #f0f4ff 0%, #f8fff5 100%); font-family: 'Inter', sans-serif;}
    .app-header { display:flex; align-items:center; gap:12px; padding:12px 0; }
    .logo { width:48px; height:48px; border-radius:12px; background: linear-gradient(135deg,#7dd3fc,#60a5fa); display:inline-flex; align-items:center; justify-content:center; color:white; font-weight:700; box-shadow: 0 6px 18px rgba(59,130,246,0.25); }
    .gloss { color: #0f172a; font-weight:600; font-size:18px; }

    .glass-card { background: rgba(255,255,255,0.55); backdrop-filter: blur(10px) saturate(120%); border-radius: 16px; padding:18px; box-shadow: 0 8px 24px rgba(16,24,40,0.12); border:1px solid rgba(255,255,255,0.6); }
    .result-card { background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.55)); padding:14px; border-radius:12px; margin-bottom:12px; box-shadow: 0 6px 14px rgba(16,24,40,0.06); border:1px solid rgba(0,0,0,0.04); }
    .btn-primary { background: linear-gradient(90deg,#4f46e5,#06b6d4); color:white; padding:8px 14px; border-radius:10px; text-decoration:none; font-weight:600; }
    .small-muted { color: #6b7280; font-size:13px; }
    .result-title { font-size:16px; margin:0 0 6px 0; color:#0f172a; font-weight:700; }
    .result-url { font-size:13px; color:#2563eb; }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    "<div class='app-header'><div class='logo'>SR</div><div><div class='gloss'>SHL Assessment Recommender</div><div class='small-muted'>Paste a job description, query, or URL to get recommended assessments.</div></div></div>",
    unsafe_allow_html=True
)

# Input
query = st.text_area(
    "Job Description / Query / URL",
    height=180
)

# Helper: render a result card as HTML with a link that opens in a new tab
def render_result_card(name: str, url: str, category: str | None = None, test_type: str | None = None) -> str:
    cat_html = f"<div class='small-muted'>Category: {category}</div>" if category else ""
    tt_html = f"<div class='small-muted'>Type: {test_type}</div>" if test_type else ""
    return f"""
    <div class='result-card'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div style='flex:1'>
          <div class='result-title'>{name}</div>
          <div class='result-url'>{url}</div>
          {cat_html}
          {tt_html}
        </div>
        <div style='margin-left:12px'>
          <a class='btn-primary' href='{url}' target='_blank' rel='noopener noreferrer'>Open</a>
        </div>
      </div>
    </div>
    """

# Action
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Finding best assessments..."):
            try:
                req = RecommendRequest(query=query)
                result = recommend(req)
                recs = result.get("recommendations", [])

                if not recs:
                    st.info("No recommendations found.")
                else:
                    st.success(f"Found {len(recs)} assessments")

                    # Display as glassy cards
                    container = st.container()
                    for r in recs:
                        name = r.get("assessment_name")
                        url = r.get("assessment_url")
                        # Try to show category/test_type if available in metadata
                        # The API returns only name and assessment_url; if you want more, update API to include them.
                        html = render_result_card(name, url)
                        container.markdown(html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error occurred: {e}")

