
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="BridgeCare",
    page_icon="🏗️",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "best.pt"

# Previously reported validation metrics
METRICS = {
    "Precision": 56.4,
    "Recall": 67.9,
    "Mask mAP@50": 66.6,
    "Mask mAP@50-95": 26.6
}

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏗️ BRIDGECARE")
st.subheader("Bridge Inspection & Maintenance")

st.write(
    "A bridge inspection support system for "
    "identifying and reviewing visible crack patterns."
)

st.divider()

# --------------------------------------------------
# MODEL PERFORMANCE DASHBOARD
# --------------------------------------------------

st.header("Model Performance Dashboard")

st.caption(
    "Validation performance measured on the model's "
    "evaluation dataset. These are not image-specific accuracy scores."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Precision", "56.4%")
c2.metric("Recall", "67.9%")
c3.metric("Mask mAP@50", "66.6%")
c4.metric("Mask mAP@50–95", "26.6%")

st.subheader("Performance Comparison")

st.bar_chart(
    {
        "Metric": [
            "Precision",
            "Recall",
            "Mask mAP@50",
            "Mask mAP@50–95"
        ],
        "Score (%)": [56.4, 67.9, 66.6, 26.6]
    },
    x="Metric",
    y="Score (%)"
)

st.divider()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.header("Bridge Inspection")

uploaded_file = st.file_uploader(
    "Upload Inspection Photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    left, right = st.columns(2)

    with left:
        st.subheader("Original Photograph")
        st.image(image, use_container_width=True)

    if st.button("Analyse Image", type="primary"):

        with st.spinner("Inspecting photograph..."):

            try:
                model = load_model()

                result = model.predict(
                    image_array,
                    imgsz=640,
                    conf=0.25,
                    verbose=False
                )[0]

                output = image_array.copy()
                crack_count = 0

                # Draw segmentation boundaries only
                if result.masks is not None:

                    polygons = result.masks.xy
                    crack_count = len(polygons)

                    for polygon in polygons:

                        points = np.round(
                            polygon
                        ).astype(np.int32)

                        if len(points) >= 2:
                            cv2.polylines(
                                output,
                                [points],
                                isClosed=True,
                                color=(255, 255, 0),
                                thickness=2,
                                lineType=cv2.LINE_AA
                            )

                # ----------------------------------
                # DETECTION OUTPUT
                # ----------------------------------

                with right:
                    st.subheader("Crack Boundary Map")
                    st.image(
                        output,
                        use_container_width=True
                    )

                st.divider()

                st.header("Inspection Report")

                r1, r2 = st.columns(2)

                r1.metric(
                    "Cracks Identified",
                    crack_count
                )

                if crack_count > 0:

                    r2.warning("Review Required")

                    st.info(
                        "Highlighted areas require review. "
                        "Arrange a detailed physical inspection "
                        "to assess the condition and severity."
                    )

                else:

                    r2.success("No Visible Cracks Detected")

                    st.info(
                        "No visible crack boundaries were identified "
                        "in this photograph. This does not guarantee "
                        "that the structure is crack-free."
                    )

                # ----------------------------------
                # DOWNLOAD RESULT
                # ----------------------------------

                success, encoded_image = cv2.imencode(
                    ".png",
                    cv2.cvtColor(
                        output,
                        cv2.COLOR_RGB2BGR
                    )
                )

                if success:
                    st.download_button(
                        label="Download Inspection Result",
                        data=encoded_image.tobytes(),
                        file_name="BridgeCare_Inspection.png",
                        mime="image/png"
                    )

            except Exception as e:

                st.error(
                    "Inspection could not be completed. "
                    "Please try again."
                )

                st.exception(e)

st.divider()

st.caption(
    "BridgeCare | Inspection observations are preliminary. "
    "Final structural assessment must be carried out by "
    "qualified professionals."
)
