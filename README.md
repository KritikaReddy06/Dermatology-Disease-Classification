#  Dermatology Disease Classification using EfficientNet-B0

This project presents a **Streamlit-based web application** for multi-class dermatological image classification using a **fine-tuned EfficientNet-B0** model.

The system classifies skin images into **22 disease categories**, enabling real-time inference through a simple and interactive interface.

---

##  Project Overview

- **Dataset**: ~14,000 dermatology images
- **Classes**: 22 skin disease categories
- **Model**: EfficientNet-B0 (transfer learning + fine-tuning)
- **Frameworks**: TensorFlow, Streamlit
- **Accuracy**: ~50% (realistic for high inter-class visual similarity)

A baseline CNN was first trained, followed by transfer learning using EfficientNet-B0. Model-specific preprocessing and fine-tuning were applied to achieve improved performance.

---


> ⚠️ Trained model files are not included in the repository due to size constraints.

---

##  Model Files (Required)

Download the trained model separately and place it in the project root:

- `efficientnet_dermatology_final.keras` **or**
- `efficientnet_dermatology_final.h5`

Update `app.py` if required to match the model filename.


