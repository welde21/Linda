import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util

from streamlit_option_menu import option_menu

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Linda-Family Harmonization System",
    page_icon="🏥",
    layout="wide"
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()


# =====================================================
# HEADER (CLEAN WEBSITE STYLE)
# =====================================================

col1, col2 = st.columns([1, 5])

with col1:
    st.image("assets/who_logo.png", width=100)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #0e76a8, #1f4e79);
        padding:18px;
        border-radius:12px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.25);
    ">

    <h2 style="color:white; text-align:center; margin:0;">
        Linda-Family Project Data Harmonization
    </h2>

    <h4 style="color:#d9edf7; text-align:center; margin-top:6px;">
        National Level Data Elements vs WHO Data Standards
    </h4>

    </div>
    """, unsafe_allow_html=True)


# =====================================================
# NAVIGATION (CLEAN)
# =====================================================

selected = option_menu(
    menu_title=None,
    options=["Home", "About"],
    icons=["house", "info-circle"],
    orientation="horizontal",
    styles={
        "container": {"padding": "5px", "background-color": "#f5f7fa"},
        "nav-link-selected": {"background-color": "#0e76a8", "color": "white"},
    }
)


# =====================================================
# ABOUT PAGE
# =====================================================

if selected == "About":
    st.subheader("About This System")

    st.write("""
    This system is designed for:

    - Health data harmonization
    - WHO standard comparison
    - Semantic similarity matching using AI
    - National health data standardization

    Built for learning and research purposes.
    """)


# =====================================================
# HOME PAGE (MAIN APP)
# =====================================================

if selected == "Home":

    # ---------------- SIDEBAR ----------------
    st.sidebar.header("⚙️ Settings")

    match_threshold = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.80)
    review_threshold = st.sidebar.slider("Review Threshold", 0.0, 1.0, 0.50)

    # ---------------- FILE UPLOAD ----------------
    st.subheader("📂 Upload Data")

    col1, col2 = st.columns(2)

    with col1:
        local_file = st.file_uploader("Upload National Data", type=["csv", "xlsx"])

    with col2:
        ref_file = st.file_uploader("Upload WHO Reference Data", type=["csv", "xlsx"])

    # ---------------- PROCESS ----------------
    if local_file and ref_file:

        # Read files
        local_df = pd.read_csv(local_file) if local_file.name.endswith(".csv") else pd.read_excel(local_file)
        ref_df = pd.read_csv(ref_file) if ref_file.name.endswith(".csv") else pd.read_excel(ref_file)

        local_vars = local_df.iloc[:, 0].dropna().astype(str).tolist()
        ref_vars = ref_df.iloc[:, 0].dropna().astype(str).tolist()

        # Encode
        with st.spinner("Running AI similarity analysis..."):
            local_emb = model.encode(local_vars, convert_to_tensor=True)
            ref_emb = model.encode(ref_vars, convert_to_tensor=True)

        results = []
        used_refs = set()

        for i, lv in enumerate(local_vars):

            sims = util.cos_sim(local_emb[i], ref_emb)[0].cpu().numpy()

            best_idx = sims.argmax()
            best_score = sims[best_idx]
            best_match = ref_vars[best_idx]

            used_refs.add(best_match)

            if best_score >= match_threshold:
                status = "Match"
            elif best_score >= review_threshold:
                status = "Review"
            else:
                status = "Mismatch"

            results.append({
                "Local Variable": lv,
                "WHO Variable": best_match,
                "Similarity Score": round(float(best_score), 4),
                "Status": status
            })

        results_df = pd.DataFrame(results).sort_values("Similarity Score", ascending=False)

        # ---------------- SEARCH + FILTER ----------------
        st.subheader("🔍 Search & Filter")

        search = st.text_input("Search variable")

        status_filter = st.selectbox("Filter status", ["All", "Match", "Review", "Mismatch"])

        filtered = results_df.copy()

        if search:
            filtered = filtered[filtered["Local Variable"].str.contains(search, case=False)]

        if status_filter != "All":
            filtered = filtered[filtered["Status"] == status_filter]

        # ---------------- METRICS ----------------
        st.subheader("📊 Dashboard")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total", len(local_vars))
        c2.metric("Matched", (results_df["Status"] == "Match").sum())
        c3.metric("Reviewed", (results_df["Status"] == "Review").sum())

        # ---------------- TABLE ----------------
        st.subheader("📋 Results")

        st.dataframe(filtered, use_container_width=True)

        # ---------------- CHART ----------------
        st.subheader("📈 Summary Chart")

        summary = results_df["Status"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(summary.index, summary.values)
        st.pyplot(fig)

        # ---------------- SUMMARY TABLE ----------------
        summary_df = pd.DataFrame({
            "Metric": ["Total", "Match", "Review", "Mismatch"],
            "Count": [
                len(local_vars),
                (results_df["Status"] == "Match").sum(),
                (results_df["Status"] == "Review").sum(),
                (results_df["Status"] == "Mismatch").sum()
            ]
        })

        st.subheader("📑 Summary")

        st.table(summary_df)

        # ---------------- EXCEL EXPORT ----------------
        excel_file = "report.xlsx"

        with pd.ExcelWriter(excel_file) as writer:
            results_df.to_excel(writer, index=False, sheet_name="Results")
            summary_df.to_excel(writer, index=False, sheet_name="Summary")

        with open(excel_file, "rb") as f:
            st.download_button("⬇️ Download Excel", f, file_name="report.xlsx")

        # ---------------- PDF EXPORT ----------------
        def create_pdf(df):
            file = "report.pdf"
            doc = SimpleDocTemplate(file)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Harmonization Report", styles["Title"]))
            elements.append(Spacer(1, 12))

            for i in range(len(df)):
                text = f"{df.iloc[i]['Metric']}: {df.iloc[i]['Count']}"
                elements.append(Paragraph(text, styles["BodyText"]))

            doc.build(elements)
            return file

        pdf_file = create_pdf(summary_df)

        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download PDF", f, file_name="report.pdf")

        st.success("Analysis Completed Successfully")


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<hr>

<div style="
    text-align:center;
    padding:12px;
    font-size:13px;
    color:gray;
    background-color:#f0f2f6;
    border-radius:10px;
">

© 2026 All Rights Reserved <br>
<b>Linda-Family Data Harmonization System</b><br>
Customized by: Weldemariam Bahre - EPHI

</div>
""", unsafe_allow_html=True)