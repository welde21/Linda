import streamlit as st
import pandas as pd

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
    model_path = r"D:\Python Training\myproject\all_mpnet_base_v2_local"
    return SentenceTransformer(model_path)


model = load_model()


# =====================================================
# HEADER
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
# NAVIGATION
# =====================================================

selected = option_menu(
    menu_title=None,
    options=["Home", "About"],
    icons=["house", "info-circle"],
    orientation="horizontal"
)


# =====================================================
# ABOUT PAGE
# =====================================================

if selected == "About":
    st.subheader("About This System")
    st.write("""
    - Health data harmonization  
    - WHO standard comparison  
    - AI semantic matching  
    """)


# =====================================================
# HOME PAGE
# =====================================================

if selected == "Home":

    st.sidebar.header("⚙️ Settings")

    match_threshold = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.80)
    review_threshold = st.sidebar.slider("Review Threshold", 0.0, 1.0, 0.50)

    st.subheader("📂 Upload Data")

    col1, col2 = st.columns(2)

    with col1:
        local_file = st.file_uploader("Upload National Data", type=["csv", "xlsx"])

    with col2:
        ref_file = st.file_uploader("Upload WHO Reference Data", type=["csv", "xlsx"])

    if local_file and ref_file:

        local_df = pd.read_csv(local_file) if local_file.name.endswith(".csv") else pd.read_excel(local_file)
        ref_df = pd.read_csv(ref_file) if ref_file.name.endswith(".csv") else pd.read_excel(ref_file)

        local_vars = local_df.iloc[:, 0].dropna().astype(str).tolist()
        ref_vars = ref_df.iloc[:, 0].dropna().astype(str).tolist()

        with st.spinner("Running AI similarity analysis..."):
            local_emb = model.encode(local_vars, convert_to_tensor=True)
            ref_emb = model.encode(ref_vars, convert_to_tensor=True)

        results = []

        for i, lv in enumerate(local_vars):

            sims = util.cos_sim(local_emb[i], ref_emb)[0].cpu().numpy()

            best_idx = sims.argmax()
            best_score = sims[best_idx]
            best_match = ref_vars[best_idx]

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

        results_df = pd.DataFrame(results)

        st.subheader("📋 Results")
        st.dataframe(results_df, use_container_width=True)


        # ---------------- EXPORT EXCEL ----------------
        excel_file = "report.xlsx"

        with pd.ExcelWriter(excel_file) as writer:
            results_df.to_excel(writer, index=False)

        with open(excel_file, "rb") as f:
            st.download_button("⬇️ Download Excel", f, file_name="report.xlsx")


        # ---------------- EXPORT PDF ----------------
        def create_pdf(df):
            file = "report.pdf"
            doc = SimpleDocTemplate(file)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Harmonization Report", styles["Title"]))
            elements.append(Spacer(1, 12))

            for i in range(len(df)):
                text = f"{df.iloc[i]['Local Variable']} → {df.iloc[i]['WHO Variable']} ({df.iloc[i]['Similarity Score']})"
                elements.append(Paragraph(text, styles["BodyText"]))

            doc.build(elements)
            return file


        pdf_file = create_pdf(results_df)

        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download PDF", f, file_name="report.pdf")

        st.success("Analysis Completed Successfully")


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<hr>
<div style="text-align:center; color:gray;">
© 2026 Linda-Family Data Harmonization System
</div>
""", unsafe_allow_html=True)