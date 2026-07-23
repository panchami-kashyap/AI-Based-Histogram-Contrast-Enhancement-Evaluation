# Project Notebooks

## histogram_contrast_enhancement.ipynb 

This notebook evaluates multiple Machine Learning, Deep Learning, and Transformer models for classifying X-ray contrast enhancement quality.

- **Machine Learning:** Random Forest, SVM, XGBoost, Gradient Boosting, Extra Trees, KNN, and CatBoost were evaluated. **CatBoost achieved the best performance with 94.33% accuracy**, followed closely by Random Forest with 93.33%.

- **Deep Learning:** Custom CNN, MobileNetV2, ResNet50, EfficientNetB0, and DenseNet121 were evaluated. **ResNet50 performed best with 88.67% accuracy**.

- **Transformer Models:** ViT, Swin Transformer, DeiT, and MobileViT were evaluated. **DeiT achieved the best Transformer performance with 87.67% accuracy**.

Overall, **CatBoost was selected as the best-performing model with 94.33% accuracy**.

## ContrastAI_Streamlit_App.ipynb

This notebook implements the Streamlit-based ContrastAI frontend integrated with the trained CatBoost model. It supports five contrast enhancement techniques, image comparison, feature and histogram visualisation, and ML-based GOOD/POOR enhancement quality prediction with confidence scores.
