import base64
import os
from datetime import UTC, datetime

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Vehicle Damage Check", page_icon="🚘", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .block-container { max-width: 1120px; padding-top: 2rem; }
    .hero { padding: 1.6rem 1.8rem; border-radius: 18px; color: #f8fafc;
        background: linear-gradient(115deg, #102a43, #1f5b75); margin-bottom: 1.3rem; }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.1rem; }
    .hero p { margin: 0; color: #d9e8f0; }
    .result-card { background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1rem 1.2rem; margin: .5rem 0; }
    .muted { color: #526477; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Vehicle damage check</h1>
      <p>Upload a clear photo for a demo classification and visual explanation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.info(
        "Portfolio demo: the model recognizes seven damage categories. "
        "It may not recognize other damage or undamaged vehicles. Use a professional inspection for decisions."
    )

analyze_tab, monitor_tab = st.tabs(["Analyze a photo", "Usage dashboard"])

with analyze_tab:
    left, right = st.columns([1, 1], gap="large")
    with left:
        uploaded = st.file_uploader(
            "Choose a vehicle photo", type=["jpg", "jpeg", "png", "webp"],
            help="A well-lit close-up with the damaged area in focus usually works best.",
        )
        if uploaded:
            st.image(uploaded, caption="Photo to analyze", use_container_width=True)
        analyze = st.button("Analyze photo", type="primary", disabled=uploaded is None, use_container_width=True)

    if analyze and uploaded:
        with st.spinner("Analyzing the photo…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=90,
                )
                if not response.ok:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"Analysis failed: {detail}")
                    result = None
                else:
                    result = response.json()
            except requests.RequestException as exc:
                st.error(f"Could not connect to the analysis service. Check that the API is running. ({exc})")
                result = None

        if result:
            assessment = result.get("confidence_assessment", {})
            with right:
                st.subheader("Result")
                st.metric(
                    "Most likely category",
                    result["predicted_class"].replace("_", " ").title(),
                    f"{result['confidence']:.0%} model confidence",
                )
                if assessment.get("review_recommended", False):
                    st.warning("Needs human review. Try a clearer photo; don’t rely on this prediction by itself.")
                else:
                    st.success("Passed the demo confidence checks. This still does not guarantee the prediction is correct.")

                top_k = result.get("top_k", [])
                if top_k:
                    st.caption("Other likely categories")
                    rank_df = pd.DataFrame(top_k)
                    rank_df["class"] = rank_df["class"].str.replace("_", " ").str.title()
                    rank_df = rank_df.rename(columns={"class": "Category", "probability": "Model score"})
                    st.dataframe(rank_df, hide_index=True, use_container_width=True)

            gradcam_b64 = result.get("gradcam_overlay_base64")
            if gradcam_b64:
                st.subheader("What the model focused on")
                st.image(base64.b64decode(gradcam_b64), use_container_width=True)
                st.caption("Grad-CAM is a visual aid. Highlighted areas do not prove what caused the prediction.")

            severity = result.get("severity", {})
            cost = result.get("repair_cost", {})
            with st.expander("Demo estimates and technical details"):
                st.warning("Severity and repair-cost values are illustrative heuristics, not inspection measurements or repair quotes.")
                metric_cols = st.columns(3)
                metric_cols[0].metric("Heuristic severity", f"{severity.get('level', '—').title()} · {severity.get('score', '—')}/100")
                if cost:
                    metric_cols[1].metric("Illustrative cost range", f"₹{cost.get('min', 0):,}–₹{cost.get('max', 0):,}")
                metric_cols[2].metric("Model version", result.get("model_version", "—"))
                st.caption(f"Request ID: {result.get('request_id', '—')} · Inference: {result.get('latency_ms', '—')} ms")

    st.caption("Evaluation reference: 78.3% accuracy on a 143-image held-out test set. Real-world performance may differ.")

with monitor_tab:
    st.caption("Aggregate prediction activity recorded by the API. This dashboard does not measure correctness without labeled outcomes.")
    bucket_choice = st.selectbox("Time period", ["1 hour", "1 day"], index=0)
    bucket_seconds = 3600 if bucket_choice == "1 hour" else 86400
    if st.button("Refresh dashboard"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/monitoring/summary", params={"bucket_seconds": bucket_seconds}, timeout=30)
        resp.raise_for_status()
        summary = resp.json()
    except requests.RequestException as exc:
        summary = None
        st.error(f"Could not connect to the monitoring service. Check that the API is running. ({exc})")

    if summary is not None:
        st.metric("Total analyzed photos", summary["total_predictions"])
        if summary["buckets"]:
            df = pd.DataFrame(summary["buckets"])
            df["time"] = df["bucket_start"].map(lambda ts: datetime.fromtimestamp(ts, tz=UTC))
            df = df.set_index("time")
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.subheader("Analysis volume")
                st.bar_chart(df["count"])
            with chart_cols[1]:
                st.subheader("Average model confidence")
                st.line_chart(df["avg_confidence"])
        else:
            st.info("No photos analyzed yet. Try the Analyze a photo tab first.")
        if summary["class_counts"]:
            st.subheader("Predictions by category")
            class_df = pd.DataFrame(
                sorted(summary["class_counts"].items(), key=lambda item: item[1], reverse=True),
                columns=["Category", "Count"],
            ).set_index("Category")
            st.bar_chart(class_df)
