
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="BridgeCare",
    page_icon="🏗️",
    layout="wide"
)

# Load model only once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Application heading
st.title("🏗️ BRIDGECARE")
st.subheader("Bridge Inspection & Maintenance")

st.write(
    "Upload a bridge photograph to identify "
    "and review potential crack regions."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Inspection Photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.image(
        image_array,
        caption="Original Inspection Photograph",
        use_container_width=True
    )

    if st.button("Analyse Image", type="primary"):

        with st.spinner("Analysing inspection photograph..."):

            results = model.predict(
                image_array,
                imgsz=640,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            output = image_array.copy()
            crack_count = 0

            if result.masks is not None:

                for polygon in result.masks.xy:

                    points = np.array(
                        polygon,
                        dtype=np.int32
                    ).reshape((-1, 1, 2))

                    cv2.polylines(
                        output,
                        [points],
                        True,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    crack_count += 1

        st.divider()
        st.subheader("Inspection Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image_array,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                output,
                caption="Crack Boundary Map",
                use_container_width=True
            )

        st.subheader("Inspection Report")

        if crack_count > 0:
            st.warning("Review Required")

            st.metric(
                "Cracks Identified",
                crack_count
            )

            st.write(
                "Potential crack regions were identified. "
                "Please review the highlighted areas."
            )

            st.write(
                "Recommendation: Arrange a detailed "
                "physical inspection."
            )

        else:
            st.success("No Visible Cracks Detected")

            st.metric("Cracks Identified", 0)

            st.write(
                "No crack regions were identified "
                "in this photograph."
            )

        st.download_button(
            "Download Inspection Result",
            data=Image.fromarray(output).tobytes()
            if False else cv2.imencode(
                ".png",
                cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            )[1].tobytes(),
            file_name="BridgeCare_Inspection.png",
            mime="image/png"
        )

st.divider()

st.caption(
    "BridgeCare | Structural Inspection Support | Version 1.0"
)

st.caption(
    "Results are preliminary observations and must be "
    "verified by a qualified structural inspection professional."
)
