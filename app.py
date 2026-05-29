import streamlit as st
import joblib
import gdown

import torch
import torch.nn as nn

import timm
import numpy as np
import os

import zipfile

from PIL import Image

from torchvision import transforms

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="DR Classification",

    page_icon="👁️",

    layout="wide",

    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL WHITE UI
# =========================================================

st.markdown("""

<style>

/* ================================
MAIN BACKGROUND
================================ */

.stApp {
    background-color: #FFFFFF;
}

/* ================================
MAIN BACKGROUND
================================ */

.stApp {
    background-color: #FFFFFF;
}

/* ================================
GENERAL TEXT
================================ */

body {
    font-family: 'Segoe UI', sans-serif;
}

/* ================================
HEADINGS
================================ */

h1, h2, h3 {

    color: #111111 !important;
}

/* ================================
PARAGRAPH
================================ */

p {

    color: #333333 !important;
}

/* ================================
TABS
================================ */

.stTabs [data-baseweb="tab"] {

    color: #111111 !important;
}

/* ================================
SIDEBAR
================================ */

[data-testid="stSidebar"] {

    background-color: #F8F9FA;
}


/* ================================
MAIN CONTAINER
================================ */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* ================================
TITLE
================================ */

h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #111111;
}

/* ================================
SUBTITLE
================================ */

h2, h3 {
    color: #222222;
}

/* ================================
TABS
================================ */

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}

.stTabs [data-baseweb="tab"] {

    height: 50px;

    white-space: pre-wrap;

    background-color: #F7F7F7;

    border-radius: 10px;

    color: #111111;

    font-size: 16px;

    font-weight: 600;

    padding-left: 20px;

    padding-right: 20px;
}

/* ================================
ACTIVE TAB
================================ */

.stTabs [aria-selected="true"] {

    background-color: #EAEAEA;

    color: #000000;
}

/* ================================
FILE UPLOADER
================================ */

.stFileUploader {

    border: 2px dashed #CCCCCC;

    border-radius: 12px;

    padding: 1rem;
}

/* ================================
BUTTONS
================================ */

.stButton>button {

    border-radius: 10px;

    height: 3em;

    font-weight: 600;

    border: none;

    background-color: #111111;

    color: white;
}

/* ================================
SIDEBAR
================================ */

[data-testid="stSidebar"] {

    background-color: #FAFAFA;
}

/* ================================
METRIC BOX
================================ */

[data-testid="stMetric"] {

    background-color: #F8F9FA;

    border-radius: 12px;

    padding: 15px;
}

/* ================================
IMAGE
================================ */

img {

    border-radius: 12px;
}

</style>

""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("👁️ Diabetic Retinopathy Detection System")

st.write("""
Ensemble Deep Learning for Retinal Fundus Classification
""")

main_container = st.container()

with main_container:

    tab1, tab2 = st.tabs([

        "🔍 Prediction",

        "📊 Performance"
    ])


# =========================================================
# DEVICE
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD ENSEMBLE CONFIG
# =========================================================

ensemble_config = torch.load(

    "ensemble_config.pth",

    map_location=DEVICE,

    weights_only=False
)
calibrator = joblib.load(
    "calibrator.pkl"
)

MODEL_NAMES = ensemble_config["model_names"]

MODEL_PATHS = ensemble_config["model_paths"]

POWER_K = ensemble_config["power_k"]

ALL_WEIGHTS = ensemble_config["weights"]

BASE_MODEL_WEIGHTS = ensemble_config["base_model_weights"]

CLASS_NAMES = ensemble_config["class_names"]

# =========================================================
# MODEL CLASS
# =========================================================

class Model(nn.Module):

    def __init__(self, name):

        super().__init__()

        self.model = timm.create_model(

            name,

            pretrained=False
        )

        in_features = self.model.get_classifier().in_features

        self.model.reset_classifier(0)

        self.head = nn.Sequential(

            nn.Linear(in_features, 512),

            nn.BatchNorm1d(512),

            nn.SiLU(),

            nn.Dropout(0.4),

            nn.Linear(512, 128),

            nn.BatchNorm1d(128),

            nn.SiLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.model(x)

        x = self.head(x)

        return x

# =====================================================
# DOWNLOAD MODELS
# =====================================================

if not os.path.exists("models"):

    os.makedirs("models", exist_ok=True)

    file_id = "1VRdtUqnoNRoRU5uPKwHGEt83SahClFHp"

    url = f"https://drive.google.com/uc?id={file_id}"

    output = "models.zip"

    with st.spinner("Downloading models..."):

        gdown.download(

            url,

            output,

            quiet=False
        )

        with zipfile.ZipFile(

            "models.zip",

            'r'
        ) as zip_ref:

            zip_ref.extractall("models")

    st.success("Models downloaded successfully!")

# =========================================================
# LOAD ALL MODELS
# =========================================================

@st.cache_resource

def load_models():

    ensemble_models = {}

    for model_name in MODEL_NAMES:

        fold_models = []

        for path in MODEL_PATHS[model_name]:

            model = Model(model_name)

            model.load_state_dict(

                torch.load(

                    path,

                    map_location=DEVICE,

                    weights_only=False
                )
            )

            model.to(DEVICE)

            model.eval()

            fold_models.append(model)

        ensemble_models[model_name] = fold_models

    return ensemble_models

ensemble_models = load_models()

# =========================================================
# PREPROCESSING
# =========================================================

def crop_retina(img):

    img = np.array(img)

    gray = np.mean(img, axis=2)

    mask = gray > 20

    if mask.sum() == 0:

        return Image.fromarray(img)

    coords = np.argwhere(mask)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    img = img[y0:y1, x0:x1]

    return Image.fromarray(img)

# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Lambda(crop_retina),

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]
    )
])

# =========================================================
# ENSEMBLE PREDICTION
# =========================================================

def predict_ensemble(image):

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    base_preds = []

    # =====================================================
    # EACH BASE LEARNER
    # =====================================================

    with torch.no_grad():

        for model_name, weights in zip(

            MODEL_NAMES,

            ALL_WEIGHTS
        ):

            fold_preds = []

            for model in ensemble_models[model_name]:

                prob = torch.sigmoid(

                    model(image).squeeze(1)

                ).item()

                fold_preds.append(prob)

            fold_preds = np.array(fold_preds)

            # =============================================
            # POWER VOTING
            # =============================================

            fold_preds = fold_preds ** POWER_K

            # =============================================
            # WEIGHTED FOLD ENSEMBLE
            # =============================================

            base_prob = np.average(

                fold_preds,

                weights=weights
            )

            # =============================================
            # SAVE BASE PRED
            # =============================================

            base_preds.append(base_prob)


    # =====================================================
    # FINAL ENSEMBLE
    # =====================================================

    final_prob = np.average(

        base_preds,

        weights=BASE_MODEL_WEIGHTS
    )

    final_prob = calibrator.predict_proba(

        np.array([[final_prob]])

    )[0][1]

    return final_prob

# =========================================================
# FILE UPLOADER
# =========================================================

with tab1:

    uploaded_file = st.file_uploader(

        "Upload Fundus Image",

        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:


# =========================================================
# PREDICTION
# =========================================================


        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(

            image,

            caption="Uploaded Image",

            width=300
        )

        with st.spinner("Predicting..."):

            prob = predict_ensemble(image)

            pred = 1 if prob > 0.5 else 0

        st.subheader("Prediction Result")

        # =====================================================
        # RESULT
        # =====================================================

        if pred == 1:

            st.error(
                f"{CLASS_NAMES[1]} Detected"
            )

        else:

            st.success(
                f"{CLASS_NAMES[0]} Detected"
            )

        # =====================================================
        # PROBABILITY
        # =====================================================

        st.write(
            f"Probability: {prob:.4f}"
        )

        st.progress(float(prob))

        # =====================================================
        # CONFIDENCE
        # =====================================================

        if prob >= 0.90:

            st.info("High Confidence")

        elif prob >= 0.70:

            st.info("Medium Confidence")

        else:

            st.info("Low Confidence")

with tab2:

    st.header("Model Performance")

    # =====================================================
# ENSEMBLE CM
# =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(

            "performance/ensemble_cm.png",

            caption="Ensemble Confusion Matrix",

            width="stretch"
        )

    with col2:

        st.image(

            "performance/roc_curve.png",

            caption="ROC Curve",

            width="stretch"
        )

    with col3:

        st.image(

            "performance/calibration_curve.png",

            caption="Calibration Curve",

            width="stretch"
        )

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Model Information")

st.sidebar.markdown("""

### Base Learners

- EfficientNet-B3
- ConvNeXt-Tiny
- Swin-Tiny

---

### Ensemble Methods

- Stratified 5-Fold Cross Validation
- Weighted Soft Voting
- Power Voting
- Weighted Ensemble Learning
- Probability Calibration (Platt Scaling)

---

### Input Specification

- JPG
- JPEG
- PNG

---

### Image Preprocessing

- Retina Cropping
- Resize 224 × 224
- Image Normalization

---

### Classification

- No_DR
- DR

""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Diabetic Retinopathy Classification using Ensemble Deep Learning"
)
