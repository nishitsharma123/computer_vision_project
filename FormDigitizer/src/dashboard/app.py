
import sys
import os
import json
from pathlib import Path

import streamlit as st
import cv2
import numpy as np

from PIL import Image

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

UPLOAD_DIR = PROJECT_ROOT / "dataset" / "forms"
SEGMENTATION_DIR = PROJECT_ROOT / "outputs" / "segmentation"
RECOGNITION_DIR = PROJECT_ROOT / "outputs" / "recognition"
WORD_CROP_DIR = RECOGNITION_DIR / "word_crops"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_hindi_crnn_full.pth"
)

FIELD_NAMES = ["name", "date", "address", "amount"]

for directory in [
    UPLOAD_DIR,
    SEGMENTATION_DIR,
    RECOGNITION_DIR,
    WORD_CROP_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# IMPORT EXISTING PIPELINE
# --------------------------------------------------

from src.segmentation.field_detector import detect_fields

from src.pipeline.form_ocr_pipeline import (
    load_recognition_model,
    recognize_field
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FormDigitizer AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f7f8fc;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

.dashboard-header {
    padding: 24px;
    border-radius: 14px;
    background: linear-gradient(120deg, #101c3a, #263f78);
    color: white;
    margin-bottom: 25px;
}

.dashboard-header h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 5px;
}

.dashboard-header p {
    color: #d6def2;
    margin-bottom: 0;
}

.metric-card {
    background: white;
    padding: 18px;
    border: 1px solid #e7eaf0;
    border-radius: 12px;
}

.metric-label {
    color: #6b7280;
    font-size: 13px;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #18243b;
}

.field-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 12px;
}

.field-title {
    font-weight: 700;
    font-size: 17px;
    color: #1e293b;
}

.prediction {
    background: #f0f5ff;
    border: 1px solid #dbe7ff;
    border-radius: 8px;
    padding: 12px;
    font-size: 19px;
    color: #183b78;
    min-height: 48px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------

@st.cache_resource
def get_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    return load_recognition_model()


# --------------------------------------------------
# PIPELINE EXECUTION
# --------------------------------------------------

def execute_pipeline(uploaded_file):

    progress = st.progress(0)

    status = st.empty()

    # ----------------------------------------------
    # STEP 1: SAVE UPLOADED IMAGE
    # ----------------------------------------------

    status.info("Step 1/4: Reading uploaded form...")

    image = Image.open(uploaded_file).convert("RGB")

    upload_path = UPLOAD_DIR / "dashboard_upload.png"

    image.save(upload_path, format="PNG")

    progress.progress(10)

    # ----------------------------------------------
    # STEP 2: FIELD DETECTION
    # ----------------------------------------------

    status.info("Step 2/4: Detecting handwritten fields...")

    detect_fields(upload_path)

    annotations_path = (
        SEGMENTATION_DIR / "field_annotations.json"
    )

    if not annotations_path.exists():
        raise FileNotFoundError(
            "Field annotation file was not generated."
        )

    with open(
        annotations_path,
        "r",
        encoding="utf-8"
    ) as file:

        annotations = json.load(file)

    detected_names = {
        item["field"] for item in annotations
    }

    if detected_names != set(FIELD_NAMES):
        raise ValueError(
            f"Unexpected detected fields: {detected_names}"
        )

    progress.progress(35)

    # ----------------------------------------------
    # STEP 3: LOAD MODEL
    # ----------------------------------------------

    status.info("Step 3/4: Loading Hindi CRNN model...")

    model, vocabulary = get_model()

    progress.progress(50)

    # ----------------------------------------------
    # STEP 4: RECOGNIZE EACH FIELD
    # ----------------------------------------------

    status.info("Step 4/4: Recognizing handwritten text...")

    results = {}

    for index, field_name in enumerate(FIELD_NAMES):

        result = recognize_field(
            model,
            vocabulary,
            field_name
        )

        results[field_name] = result

        current_progress = 50 + int(
            ((index + 1) / len(FIELD_NAMES)) * 45
        )

        progress.progress(current_progress)

    # ----------------------------------------------
    # SAVE JSON
    # ----------------------------------------------

    output_file = RECOGNITION_DIR / "recognized_form.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    progress.progress(100)

    status.success("Pipeline execution completed.")

    return {
        "original": upload_path,
        "detected": SEGMENTATION_DIR / "detected_fields.png",
        "annotations": annotations,
        "results": results,
        "json_path": output_file
    }


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("📄 FormDigitizer AI")

    st.caption("Handwritten Document Intelligence")

    st.divider()

    st.markdown("### Pipeline modules")

    st.markdown("""
    - Image Upload
    - Field Detection
    - Image Segmentation
    - Hindi CRNN Recognition
    - JSON Generation
    """)

    st.divider()

    st.markdown("### Model information")

    st.write("Architecture: CNN + BiLSTM + CTC")
    st.write("Input height: 64 pixels")
    st.write("Character classes: 100")
    st.write("CTC blank index: 0")
    st.write("Device: CPU / CUDA")

    st.divider()

    st.caption("FormDigitizer AI | Academic Project")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="dashboard-header">
    <h1>FormDigitizer AI Dashboard</h1>
    <p>
        Intelligent handwritten Hindi form processing
        using computer vision and deep learning.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Upload your handwritten form</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a handwritten form image",
    type=["png", "jpg", "jpeg"],
    help="Upload the form containing Name, Date, Address and Amount fields."
)

if uploaded_file is not None:

    left, right = st.columns([1.5, 1])

    with left:

        st.markdown("#### Uploaded document")

        st.image(
            uploaded_file,
            use_container_width=True
        )

    with right:

        st.markdown("#### Document information")

        preview = Image.open(uploaded_file)

        st.write("Filename:", uploaded_file.name)
        st.write("Width:", preview.width, "pixels")
        st.write("Height:", preview.height, "pixels")
        st.write("Format:", preview.format)

    st.divider()

    if st.button(
        "▶ Run OCR Pipeline",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Executing FormDigitizer pipeline..."):

                output = execute_pipeline(uploaded_file)

                st.session_state["pipeline_output"] = output

            st.success("Form processed successfully.")

        except Exception as error:

            st.error("Pipeline execution failed.")

            st.exception(error)


# --------------------------------------------------
# RESULTS DASHBOARD
# --------------------------------------------------

if "pipeline_output" in st.session_state:

    output = st.session_state["pipeline_output"]

    results = output["results"]

    st.divider()

    st.markdown(
        '<div class="section-title">Pipeline overview</div>',
        unsafe_allow_html=True
    )

    total_words = sum(
        item["word_count"] for item in results.values()
    )

    detected_fields = len(output["annotations"])

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "Fields detected", detected_fields),
        (c2, "Fields recognized", len(results)),
        (c3, "Word crops", total_words),
        (c4, "Model", "CRNN")
    ]

    for column, label, value in metrics:

        with column:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ----------------------------------------------
    # DETECTION VISUALIZATION
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">1. Field detection</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Green rectangles indicate the fields detected by OpenCV."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Original form**")

        st.image(
            str(output["original"]),
            use_container_width=True
        )

    with col2:

        st.markdown("**Detected field boundaries**")

        st.image(
            str(output["detected"]),
            use_container_width=True
        )

    # ----------------------------------------------
    # FIELD CROPS
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">2. Extracted field images</div>',
        unsafe_allow_html=True
    )

    field_columns = st.columns(2)

    for index, field_name in enumerate(FIELD_NAMES):

        with field_columns[index % 2]:

            crop_path = SEGMENTATION_DIR / f"{field_name}.png"

            st.markdown(
                f'<div class="field-card">'
                f'<div class="field-title">{field_name.upper()}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.image(
                str(crop_path),
                use_container_width=True
            )

            annotation = next(
                item for item in output["annotations"]
                if item["field"] == field_name
            )

            st.caption(
                f"Bounding box: {annotation['bbox']}"
            )

    # ----------------------------------------------
    # OCR RESULTS
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">3. Hindi CRNN recognition results</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "These are raw model predictions. No confidence score "
        "or correction has been applied."
    )

    for field_name in FIELD_NAMES:

        result = results[field_name]

        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-title">{field_name.upper()}</div>
                <div class="prediction">
                    {result["recognized_text"] or "No text recognized"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            f"View {field_name} recognition details"
        ):

            st.write("Word count:", result["word_count"])

            st.write("Individual predictions:")

            for word_index, prediction in enumerate(
                result["word_predictions"]
            ):

                st.write(
                    f"Word {word_index + 1}: {prediction}"
                )

                crop_path = (
                    WORD_CROP_DIR
                    / f"{field_name}_word_{word_index + 1}.png"
                )

                if crop_path.exists():

                    st.image(
                        str(crop_path),
                        width=400
                    )

    # ----------------------------------------------
    # JSON OUTPUT
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">4. Structured JSON output</div>',
        unsafe_allow_html=True
    )

    json_data = json.dumps(
        results,
        ensure_ascii=False,
        indent=4
    )

    st.code(json_data, language="json")

    st.download_button(
        label="⬇ Download recognized_form.json",
        data=json_data.encode("utf-8"),
        file_name="recognized_form.json",
        mime="application/json",
        use_container_width=True
    )

    # ----------------------------------------------
    # RESET
    # ----------------------------------------------

    st.divider()

    if st.button("Clear dashboard results"):

        del st.session_state["pipeline_output"]

        st.rerun()

else:

    st.info(
        "Upload a handwritten form and click "
        "'Run OCR Pipeline' to view the processing results."
    )
