import base64
import os
from datetime import UTC, datetime

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Vehicle Damage Detection", page_icon="🚗", layout="centered")
st.title("🚗 Vehicle Damage Detection")
st.caption("Thin client: inference and prediction logging are performed by the FastAPI service.")

predict_tab, monitor_tab = st.tabs(["Analyze", "Monitoring"])

with predict_tab:
    uploaded = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        st.image(uploaded, caption="Input image", use_container_width=True)
        if st.button("Analyze damage", type="primary"):
            try:
                response = requests.post(f"{API_URL}/predict", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}, timeout=90)
                if not response.ok:
                    st.error(response.json().get("detail", response.text))
                else:
                    result = response.json()
                    st.success(f"{result['predicted_class']} — {result['confidence']:.1%} confidence")
                    st.json({k: v for k, v in result.items() if k != "gradcam_overlay_base64"})
                    overlay = base64.b64decode(result["gradcam_overlay_base64"])
                    st.image(overlay, caption="Grad-CAM explanation", use_container_width=True)
            except requests.RequestException as exc:
                st.error(f"Could not reach API: {exc}")

with monitor_tab:
    st.caption("Live view of everything the /predict endpoint has logged, pulled from GET /monitoring/summary.")
    bucket_choice = st.selectbox("Bucket size", ["1 hour", "1 day"], index=0)
    bucket_seconds = 3600 if bucket_choice == "1 hour" else 86400

    if st.button("Refresh"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/monitoring/summary", params={"bucket_seconds": bucket_seconds}, timeout=30)
        resp.raise_for_status()
        summary = resp.json()
    except requests.RequestException as exc:
        summary = None
        st.error(f"Could not reach API: {exc}")

    if summary is not None:
        st.metric("Total logged predictions", summary["total_predictions"])

        if summary["buckets"]:
            df = pd.DataFrame(summary["buckets"])
            df["time"] = df["bucket_start"].map(lambda ts: datetime.fromtimestamp(ts, tz=UTC))
            df = df.set_index("time")

            st.subheader("Prediction volume over time")
            st.bar_chart(df["count"])

            st.subheader("Average confidence over time")
            st.line_chart(df["avg_confidence"])
        else:
            st.info("No predictions logged yet -- run a few through the Analyze tab first.")

        if summary["class_counts"]:
            st.subheader("Predictions by class")
            class_df = pd.DataFrame(
                sorted(summary["class_counts"].items(), key=lambda kv: kv[1], reverse=True),
                columns=["class", "count"],
            ).set_index("class")
            st.bar_chart(class_df)
