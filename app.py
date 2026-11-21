import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Page setup
st.set_page_config(
    page_title="Tomato Leaf Health",
    page_icon="🍅",
    layout="wide",
)

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "tomato_leaf_disease_model.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"
IMAGE_SIZE = (180, 180)

# Styling and animations
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

        :root {
            --forest: #153f35;
            --paper: #f5f7f2;
            --ink: #1b2b23;
            --muted: #64776b;
        }

        .stApp {
            background:
                radial-gradient(ellipse at 10% 5%, rgba(208, 232, 209, .52), transparent 32%),
                var(--paper);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        @keyframes floatDecor {
            0%, 100% { transform: translateY(0) rotate(-8deg); }
            50% { transform: translateY(-12px) rotate(8deg); }
        }

        @keyframes appear {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes breathe {
            0%, 100% { box-shadow: 0 0 0 rgba(170, 224, 177, 0); }
            50% { box-shadow: 0 0 22px rgba(170, 224, 177, .28); }
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.5rem;
            border-radius: 26px;
            color: white;
            background: linear-gradient(118deg, #143d34 0%, #24664c 58%, #43815b 100%);
            box-shadow: 0 16px 38px rgba(25, 70, 51, .16);
            animation: appear .65s ease-out both;
        }

        .hero-decor {
            position: absolute;
            right: 7%;
            top: 12%;
            display: flex;
            gap: 1rem;
            align-items: center;
            font-size: 3rem;
            opacity: .9;
            pointer-events: none;
        }

        .hero-decor span {
            display: inline-block;
            animation: floatDecor 4s ease-in-out infinite;
        }

        .hero-decor span:nth-child(2) {
            margin-top: 2rem;
            font-size: 2.5rem;
            animation-delay: .7s;
        }

        .hero-decor span:nth-child(3) {
            margin-top: -.8rem;
            font-size: 2rem;
            animation-delay: 1.2s;
        }

        .hero-kicker {
            position: relative;
            z-index: 1;
            margin-bottom: .55rem;
            color: #c5e6ce;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            color: #fff;
            font-family: 'Manrope', sans-serif;
            font-size: clamp(2rem, 5vw, 3.05rem);
            font-weight: 800;
            letter-spacing: -.045em;
        }

        .hero p {
            position: relative;
            z-index: 1;
            max-width: 650px;
            margin: .65rem 0 0;
            color: #e1f0e5;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .section-heading {
            margin: .2rem 0 .35rem;
            color: var(--forest);
            font-family: 'Manrope', sans-serif;
            font-size: 1.15rem;
            font-weight: 750;
        }

        .section-hint {
            margin: 0 0 1rem;
            color: var(--muted);
            font-size: .92rem;
        }

        div[data-testid="stFileUploader"] {
            padding: 1rem;
            border: 1px dashed #9bb9a2;
            border-radius: 18px;
            background: rgba(255, 255, 255, .8);
            animation: breathe 4s ease-in-out infinite;
        }

        div[data-testid="stFileUploader"] section {
            background: transparent;
        }

        .result-card {
            padding: 1.5rem;
            border: 1px solid #e0e9df;
            border-radius: 20px;
            background: rgba(255, 255, 255, .92);
            box-shadow: 0 10px 28px rgba(34, 67, 45, .08);
            animation: appear .55s ease-out both;
        }

        .result-kicker {
            color: #648073;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .result-name {
            margin: .45rem 0 1rem;
            color: var(--forest);
            font-family: 'Manrope', sans-serif;
            font-size: 1.4rem;
            font-weight: 800;
            line-height: 1.35;
            text-transform: capitalize;
        }

        .status-pill {
            display: inline-block;
            margin-bottom: .8rem;
            padding: .38rem .72rem;
            border-radius: 999px;
            color: #24563d;
            background: #e4f3e6;
            font-size: .8rem;
            font-weight: 700;
        }

        .empty-state {
            padding: 2rem 1.4rem;
            border: 1px solid #e2e9e0;
            border-radius: 20px;
            background: rgba(255, 255, 255, .72);
            color: #66786c;
            text-align: center;
        }

        .empty-icon {
            margin-bottom: .45rem;
            font-size: 2.3rem;
        }

        .footer {
            margin-top: 2.5rem;
            color: #718076;
            font-size: .82rem;
            text-align: center;
        }

        @media (max-width: 650px) {
            .hero {
                padding: 1.7rem;
            }

            .hero-decor {
                right: 5%;
                top: 8%;
                gap: .3rem;
                font-size: 2rem;
                opacity: .55;
            }

            .hero-decor span:nth-child(2) {
                font-size: 1.8rem;
            }

            .hero-decor span:nth-child(3) {
                font-size: 1.5rem;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Check model files
if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error(
        "Model files are missing. Put `tomato_leaf_disease_model.keras` "
        "and `class_names.json` inside the project's `models` folder."
    )
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero-decor" aria-hidden="true">
            <span>🍅</span>
            <span>🍃</span>
            <span>🌿</span>
        </div>
        <div class="hero-kicker">AI plant health studio</div>
        <h1>Tomato Leaf Health</h1>
        <p>
            A computer vision demo that classifies tomato leaf images across
            10 categories. Add a clear photo to see the model's prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main interface
left_column, right_column = st.columns([1.05, 0.95], gap="large")

with left_column:
    st.markdown(
        '<div class="section-heading">01 &nbsp; Add a leaf photo</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-hint">For best results, use a clear, well-lit image.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded leaf image", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Try another JPG or PNG file.")

with right_column:
    st.markdown(
        '<div class="section-heading">02 &nbsp; Model result</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-hint">Prediction and confidence for this image.</div>',
        unsafe_allow_html=True,
    )

    if image is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">🌿</div>
                <strong>Your result will appear here</strong><br>
                Upload a leaf photo to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        resized_image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(resized_image, dtype=np.float32)
        image_batch = np.expand_dims(image_array, axis=0)

        probabilities = model.predict(image_batch, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        prediction = class_names[best_index].replace("_", " ")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="status-pill">● &nbsp; Analysis complete</div>
                <div class="result-kicker">Most likely class</div>
                <div class="result-name">{prediction}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(confidence, text=f"Model confidence: {confidence:.1%}")

        if confidence < 0.60:
            st.warning(
                "The model is uncertain. Try a sharper photo with the leaf "
                "well lit and clearly visible."
            )
        else:
            st.caption(
                "Confidence is the model's score, not a guarantee that its "
                "classification is correct."
            )

# Project limitations
st.markdown("---")

with st.expander("About this project and its limitations"):
    st.write(
        "This educational prototype was trained on public tomato leaf images. "
        "Its performance on the validation set may not reflect performance on "
        "farm photos, different cameras, lighting, or unseen conditions. "
        "It is not a confirmed plant diagnosis or treatment recommendation."
    )

st.markdown(
    '<div class="footer">Built with TensorFlow / Keras and Streamlit · Educational portfolio project</div>',
    unsafe_allow_html=True,
)