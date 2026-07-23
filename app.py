
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage import exposure
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ContrastAI",
    page_icon="◈",
    layout="wide"
)


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background:#04050a;
    color:#e8e6f0;
    font-family:Segoe UI;
}

.block-container {
    max-width:1450px;
    padding-top:1.5rem;
    padding-bottom:2rem;
}

#MainMenu, footer, header {
    visibility:hidden;
}

/* HEADER */

.main-title {
    font-size:42px;
    font-weight:800;
    margin-bottom:0;
    color:white;
}

.sub {
    color:#64748b;
    font-size:15px;
    margin-bottom:20px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background:#07080f;
    border-right:1px solid #111827;
}

/* BUTTON */

.stButton>button {
    width:100%;
    height:50px;
    border:none;
    border-radius:12px;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:white;
    font-weight:700;
    font-size:16px;
}

/* METRIC CARDS */

.metric-card {
    background:#0b0d15;
    padding:18px;
    border-radius:16px;
    border:1px solid #111827;
    text-align:center;
    min-height:105px;
}

.metric-name {
    font-size:11px;
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:0.08em;
}

.metric-value {
    font-size:24px;
    font-weight:800;
    color:white;
    margin-top:8px;
}

/* GLOW RESULT */

.glow-box {
    background:#0b0d15;
    border:1px solid #1d4ed8;
    padding:22px;
    border-radius:18px;
    box-shadow:0 0 25px rgba(59,130,246,0.25);
    animation:pulse 2s infinite;
    min-height:170px;
}

@keyframes pulse {

    0% {
        box-shadow:0 0 10px rgba(59,130,246,0.20);
    }

    50% {
        box-shadow:0 0 30px rgba(59,130,246,0.45);
    }

    100% {
        box-shadow:0 0 10px rgba(59,130,246,0.20);
    }
}

/* CONFIDENCE */

.conf-box {
    background:#0b0d15;
    border:1px solid #111827;
    padding:22px;
    border-radius:18px;
    text-align:center;
    min-height:170px;
}

.conf-big {
    font-size:58px;
    font-weight:900;
    color:#6366f1;
    line-height:1.1;
    margin-top:10px;
    margin-bottom:10px;
}

.conf-small {
    font-size:11px;
    color:#64748b;
    letter-spacing:0.08em;
    text-transform:uppercase;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED CATBOOST PIPELINE
# ============================================================

MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Histogram_Contrast_Enhancement_Project/"
    "models/"
    "catboost_enhancement_pipeline.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:

    st.error(
        "The trained CatBoost model could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CONTRAST STRETCHING
# ============================================================

def contrast_stretch(image):

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        return image.copy()

    stretched = (
        image.astype(np.float32) - min_val
    ) * (
        255.0 / (max_val - min_val)
    )

    return np.clip(
        stretched,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# HISTOGRAM EQUALIZATION
# ============================================================

def apply_he(image):

    return cv2.equalizeHist(image)


# ============================================================
# ADAPTIVE HISTOGRAM EQUALIZATION
# ============================================================

def apply_ahe(image):

    enhanced = exposure.equalize_adapthist(
        image
    )

    return (
        enhanced * 255
    ).astype(np.uint8)


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(image):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(image)


# ============================================================
# BI-HISTOGRAM EQUALIZATION
# ============================================================

def apply_bhe(image):

    mean_val = int(
        np.mean(image)
    )

    lower_mask = image <= mean_val
    upper_mask = image > mean_val

    lower_part = image.copy()
    upper_part = image.copy()

    lower_part[
        ~lower_mask
    ] = mean_val

    upper_part[
        ~upper_mask
    ] = mean_val

    lower_eq = cv2.equalizeHist(
        lower_part.astype(np.uint8)
    )

    upper_eq = cv2.equalizeHist(
        upper_part.astype(np.uint8)
    )

    bhe_img = image.copy()

    bhe_img[
        lower_mask
    ] = lower_eq[
        lower_mask
    ]

    bhe_img[
        upper_mask
    ] = upper_eq[
        upper_mask
    ]

    return bhe_img


# ============================================================
# FEATURE EXTRACTION
# EXACT SAME 6 FEATURES USED FOR CATBOOST TRAINING
# ============================================================

def extract_features(original, enhanced):

    # Mean
    mean_val = np.mean(
        enhanced
    )

    # Standard deviation
    std_val = np.std(
        enhanced
    )

    # Entropy
    entropy_val = shannon_entropy(
        enhanced
    )

    # PSNR
    if np.array_equal(
        original,
        enhanced
    ):

        psnr_val = np.nan

    else:

        psnr_val = psnr(
            original,
            enhanced
        )

    # Contrast
    contrast_val = (
        float(np.max(enhanced))
        -
        float(np.min(enhanced))
    )

    # Laplacian variance / sharpness
    laplacian_val = cv2.Laplacian(
        enhanced,
        cv2.CV_64F
    ).var()

    return (
        mean_val,
        std_val,
        entropy_val,
        psnr_val,
        contrast_val,
        laplacian_val
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙ Controls")

    file = st.file_uploader(
        "Upload X-ray Image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    method = st.selectbox(
        "Enhancement Method",
        [
            "Contrast Stretching",
            "HE - Histogram Equalization",
            "AHE - Adaptive Histogram Equalization",
            "CLAHE - Contrast Limited Adaptive Histogram Equalization",
            "BHE - Bi-Histogram Equalization"
        ]
    )

    run = st.button(
    "🚀 Run Enhancement"
    )

    if run:
      st.session_state["enhancement_run"] = True


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<div class='main-title'>◈ ContrastAI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub'>Histogram-Based Contrast Enhancement with ML-Based Quality Evaluation</div>",
    unsafe_allow_html=True
)


# ============================================================
# EMPTY STATE
# ============================================================

if not file:

    st.info(
        "Upload an X-ray image from the sidebar to begin."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

elif file and st.session_state.get("enhancement_run", False):

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        file
    ).convert("L")

    img = np.array(
        image
    )

    # Exact preprocessing size used during training
    img = cv2.resize(
        img,
        (256, 256)
    )


    # --------------------------------------------------------
    # APPLY SELECTED ENHANCEMENT
    # --------------------------------------------------------

    if method == "Contrast Stretching":

        out = contrast_stretch(
            img
        )

        chosen = "Contrast Stretching"


    elif method.startswith("HE"):

        out = apply_he(
            img
        )

        chosen = "HE"


    elif method.startswith("AHE"):

        out = apply_ahe(
            img
        )

        chosen = "AHE"


    elif method.startswith("CLAHE"):

        out = apply_clahe(
            img
        )

        chosen = "CLAHE"


    else:

        out = apply_bhe(
            img
        )

        chosen = "BHE"


    # ========================================================
    # IMAGE COMPARISON
    # ========================================================

    st.subheader(
        "🔍 Compare Images"
    )

    alpha = st.slider(
        "Blend Compare",
        min_value=0,
        max_value=100,
        value=50
    )

    blend = cv2.addWeighted(
        img,
        1 - alpha / 100,
        out,
        alpha / 100,
        0
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.image(
            img,
            caption="Original",
            use_container_width=True
        )

    with c2:

        st.image(
            blend,
            caption="Slider Compare",
            use_container_width=True
        )

    with c3:

        st.image(
            out,
            caption=f"Enhanced - {chosen}",
            use_container_width=True
        )


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    (
        mean,
        std,
        ent,
        p,
        contrast,
        lap
    ) = extract_features(
        img,
        out
    )


    # ========================================================
    # FEATURE CARDS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Extracted Features"
    )

    cols = st.columns(6)

    psnr_display = (
        "N/A"
        if np.isnan(p)
        else f"{p:.2f}"
    )

    values = [
        ("Mean", f"{mean:.2f}"),
        ("Std Dev", f"{std:.2f}"),
        ("Entropy", f"{ent:.2f}"),
        ("PSNR", psnr_display),
        ("Contrast", f"{contrast:.2f}"),
        ("Sharpness", f"{lap:.2f}")
    ]

    for col, (name, value) in zip(
        cols,
        values
    ):

        with col:

            card_html = (
                f"<div class='metric-card'>"
                f"<div class='metric-name'>{name}</div>"
                f"<div class='metric-value'>{value}</div>"
                f"</div>"
            )

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )


    # ========================================================
    # HISTOGRAM COMPARISON
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📈 Histogram Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.hist(
        img.ravel(),
        bins=256,
        alpha=0.55,
        label="Original"
    )

    ax.hist(
        out.ravel(),
        bins=256,
        alpha=0.55,
        label="Enhanced"
    )

    ax.legend()

    ax.set_xlabel(
        "Pixel Intensity"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_facecolor(
        "#04050a"
    )

    fig.patch.set_facecolor(
        "#04050a"
    )

    ax.tick_params(
        colors="white"
    )

    ax.xaxis.label.set_color(
        "white"
    )

    ax.yaxis.label.set_color(
        "white"
    )

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


    # ========================================================
    # PREPARE FEATURES FOR CATBOOST
    # ========================================================

    feature_data = pd.DataFrame(
        [[
            mean,
            std,
            ent,
            p,
            contrast,
            lap
        ]],
        columns=[
            "Mean",
            "StdDev",
            "Entropy",
            "PSNR",
            "Contrast",
            "Laplacian"
        ]
    )

    feature_data = feature_data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    # ========================================================
    # REAL CATBOOST PREDICTION
    # ========================================================

    prediction = int(
        model.predict(
            feature_data
        )[0]
    )

    probabilities = model.predict_proba(
        feature_data
    )[0]

    confidence = float(
        probabilities[
            prediction
        ] * 100
    )


    # ========================================================
    # ML EVALUATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🤖 ML Evaluation"
    )

    left, right = st.columns(
        [2, 1]
    )


    # --------------------------------------------------------
    # VERDICT CARD
    # --------------------------------------------------------

    with left:

        if prediction == 1:

            verdict_html = (
                "<div class='glow-box'>"
                "<h2 style='color:#22c55e;'>"
                "✔ Good Enhancement"
                "</h2>"
                "<p>"
                "The trained CatBoost model classified the "
                f"<strong>{chosen}</strong> enhancement as "
                "<strong>GOOD</strong>."
                "</p>"
                "<p style='color:#94a3b8;'>"
                "The extracted image-quality features indicate "
                "an effective enhancement according to the "
                "trained model."
                "</p>"
                "</div>"
            )

        else:

            verdict_html = (
                "<div class='glow-box'>"
                "<h2 style='color:#f59e0b;'>"
                "⚠ Poor Enhancement"
                "</h2>"
                "<p>"
                "The trained CatBoost model classified the "
                f"<strong>{chosen}</strong> enhancement as "
                "<strong>POOR</strong>."
                "</p>"
                "<p style='color:#94a3b8;'>"
                "Try another enhancement method and compare "
                "the resulting image quality."
                "</p>"
                "</div>"
            )

        st.markdown(
            verdict_html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # CONFIDENCE CARD
    # --------------------------------------------------------

    with right:

        confidence_html = (
            "<div class='conf-box'>"
            "<div class='conf-small'>"
            "AI Confidence"
            "</div>"
            f"<div class='conf-big'>{confidence:.1f}%</div>"
            "<div class='conf-small'>"
            "CatBoost Prediction"
            "</div>"
            "</div>"
        )

        st.markdown(
            confidence_html,
            unsafe_allow_html=True
        )


# ============================================================
# IMAGE UPLOADED BUT BUTTON NOT CLICKED
# ============================================================

elif file:

    st.info(
        "Image uploaded. Click 'Run Enhancement'."
    )
