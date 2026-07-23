# AI-Based Histogram Contrast Enhancement Evaluation

An intelligent image-processing system for evaluating histogram-based contrast enhancement techniques on X-ray images using **Machine Learning, Deep Learning, and Transformer models**.

The project applies multiple contrast enhancement techniques, extracts quantitative image-quality features, compares different classification approaches, and integrates the best-performing model into an interactive **Streamlit application – ContrastAI**.

---

## Project Overview

Contrast enhancement plays an important role in improving the visual quality of medical images. This project evaluates different histogram-based enhancement techniques and uses machine learning to classify the resulting enhancement quality.

Five contrast enhancement techniques are implemented:

- Contrast Stretching
- Histogram Equalization (HE)
- Adaptive Histogram Equalization (AHE)
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Bi-Histogram Equalization (BHE)

Image-quality features are extracted from the enhanced images and used to classify enhancement quality as **GOOD** or **POOR**.

---

## Extracted Image Quality Features

The following quantitative features are used for enhancement-quality evaluation:

- Mean Intensity
- Standard Deviation
- Entropy
- Peak Signal-to-Noise Ratio (PSNR)
- Contrast
- Laplacian Variance / Sharpness

---

## Model Experiments

Three categories of models were experimentally evaluated.

### Machine Learning

The following Machine Learning models were evaluated:

- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Gradient Boosting
- Extra Trees
- K-Nearest Neighbours (KNN)
- CatBoost

**Best ML Model: CatBoost – 94.33% Accuracy**

### Deep Learning

The following Deep Learning architectures were evaluated:

- Custom CNN
- MobileNetV2
- ResNet50
- EfficientNetB0
- DenseNet121

**Best DL Model: ResNet50 – 88.67% Accuracy**

### Transformer Models

The following Transformer-based vision models were evaluated:

- Vision Transformer (ViT)
- Swin Transformer
- DeiT
- MobileViT

**Best Transformer Model: DeiT – 87.67% Accuracy**

### Overall Best Model

**CatBoost achieved the highest overall accuracy of 94.33%** and was selected for integration into the final ContrastAI application.

---

## Model Performance Summary

| Model Category | Best Model | Accuracy |
|---|---|---:|
| Machine Learning | CatBoost | 94.33% |
| Deep Learning | ResNet50 | 88.67% |
| Transformer | DeiT | 87.67% |

**Overall Best Model: CatBoost – 94.33% Accuracy**

---

## ContrastAI – Streamlit Application

**ContrastAI** is an interactive Streamlit-based interface developed to demonstrate the complete contrast enhancement and ML evaluation pipeline.

The application allows users to:

- Upload an X-ray image
- Select one of five contrast enhancement techniques
- View the original and enhanced images
- Use blend comparison to visually inspect the enhancement
- View extracted image-quality features
- Compare original and enhanced image histograms
- Obtain an ML-based **GOOD / POOR** enhancement-quality classification
- View the CatBoost model's prediction confidence

The final application uses the trained **CatBoost classification pipeline** for enhancement-quality evaluation.

---

## Dataset

The X-ray dataset used in this project is approximately **2.29 GB** and is therefore not included directly in this repository.

The dataset download link and setup information are available in:

`dataset/README.md`

To reduce data leakage, the original X-ray images are split into training and testing sets **before generating their enhanced versions**. This prevents enhanced variants of the same original image from appearing in both the training and testing datasets.

---

## Project Structure

```text
AI-Based-Histogram-Contrast-Enhancement-Evaluation/
│
├── dataset/
│   └── README.md
│
├── models/
│   ├── catboost_enhancement_pipeline.pkl
│   └── model_metadata.json
│
├── notebooks/
│   ├── histogram_contrast_enhancement.ipynb
│   ├── ContrastAI_Streamlit_App.ipynb
│   └── README.md
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Repository Contents

### `notebooks/histogram_contrast_enhancement.ipynb`

Contains the main experimental pipeline, including:

- Image preprocessing
- Five contrast enhancement techniques
- Image-quality feature extraction
- Dataset generation
- Machine Learning experiments
- Deep Learning experiments
- Transformer model experiments
- Model evaluation and comparison

### `notebooks/ContrastAI_Streamlit_App.ipynb`

Contains the development and testing of the **ContrastAI Streamlit application**, including integration of the trained CatBoost model with the frontend.

### `app.py`

The standalone Streamlit application used to run the ContrastAI interface.

### `models/`

Contains the final trained CatBoost pipeline and associated model metadata used by the application.

### `dataset/`

Contains dataset information and instructions for obtaining and setting up the X-ray dataset. The full dataset is not stored in the repository because of its large size.

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-Based-Histogram-Contrast-Enhancement-Evaluation
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the ContrastAI Application

Ensure that the trained model is available at:

```text
models/catboost_enhancement_pipeline.pkl
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The ContrastAI application will then open in your web browser.

---

## Technologies Used

### Image Processing

- OpenCV
- Scikit-image
- NumPy

### Machine Learning

- Scikit-learn
- CatBoost
- XGBoost

### Deep Learning

- TensorFlow / Keras
- PyTorch

### Transformer Models

- Hugging Face Transformers
- Vision Transformer (ViT)
- Swin Transformer
- DeiT
- MobileViT

### Data Analysis and Visualisation

- Pandas
- Matplotlib

### Application Development

- Streamlit

---

## Workflow

```text
X-ray Image
      │
      ▼
Image Preprocessing
      │
      ▼
Contrast Enhancement
      │
      ├── Contrast Stretching
      ├── HE
      ├── AHE
      ├── CLAHE
      └── BHE
      │
      ▼
Image Quality Feature Extraction
      │
      ├── Mean
      ├── Standard Deviation
      ├── Entropy
      ├── PSNR
      ├── Contrast
      └── Laplacian Variance
      │
      ▼
Machine Learning Evaluation
      │
      ▼
CatBoost Classification
      │
      ▼
GOOD / POOR Enhancement
      +
Prediction Confidence
```

---

## Key Features

- Multiple histogram-based contrast enhancement techniques
- Quantitative image-quality feature extraction
- Machine Learning model comparison
- Deep Learning model comparison
- Transformer-based vision model experimentation
- Train/test separation at the original-image level
- CatBoost-based enhancement-quality classification
- Interactive Streamlit frontend
- Original vs enhanced image comparison
- Histogram visualisation
- Prediction confidence display

---

## Future Scope

The project can be further extended through:

- Evaluation on additional medical imaging datasets
- Automated selection of the optimal enhancement technique for each image
- Additional objective and perceptual image-quality metrics
- Explainable AI techniques for interpreting model predictions
- Larger-scale model training and external validation
- Deployment of ContrastAI as a publicly accessible web application
