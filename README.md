# Deep Learning Assignment

This repository contains three deep learning models built using TensorFlow and Keras, each focusing on different machine learning tasks:

1. **Classification** - Fashion MNIST Dataset  
2. **Regression** - California Housing Dataset  
3. **Image Classification** - CIFAR-10 Dataset  

Each model includes comprehensive metrics, visualizations, and integration with Weights and Biases (W&B) for experiment tracking.

---

## 📂 Repository Structure

```
├── Classification_on_Fashion_MNIST.ipynb      # Classification model notebook
├── Regression_on_California_Housing_Dataset.ipynb  # Regression model notebook
├── Image_Classification_on_cifar10.ipynb     # Image classification model notebook
└── README.md                                  # Project documentation
```

---

## 🚀 Models Overview

### 1. 🧥 Classification - Fashion MNIST
- **Problem:** Classify images of clothing items into 10 categories.
- **Metrics:** 
  - Accuracy, Precision, Recall, F1-score (per class and overall)
  - ROC and PR curves
  - Per-class examples and error analysis
- **Artifacts:** Saved in `artifacts/classification/`
- **Notebook:** [Classification_on_Fashion_MNIST.ipynb](Classification_on_Fashion_MNIST.ipynb)

### 2. 🏠 Regression - California Housing Dataset
- **Problem:** Predict median house values using features from the California housing dataset.
- **Metrics:** 
  - Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score
  - Residual plots and error distribution
- **Artifacts:** Saved in `artifacts/regression/`
- **Notebook:** [Regression_on_California_Housing_Dataset.ipynb](Regression_on_California_Housing_Dataset.ipynb)

### 3. 🖼️ Image Classification - CIFAR-10
- **Problem:** Classify 32x32 images into 10 object categories.
- **Metrics:** 
  - Accuracy, Precision, Recall, F1-score (per class and overall)
  - ROC and PR curves
  - Per-class examples and error analysis
- **Artifacts:** Saved in `artifacts/image_classification/`
- **Notebook:** [Image_Classification_on_cifar10.ipynb](Image_Classification_on_cifar10.ipynb)

---

## 📊 Metrics and Artifacts
Each notebook includes:
- Model diagrams (visualized using `plot_model` from Keras)
- Integration with **Weights and Biases (W&B)** for experiment tracking
- Confusion matrices, ROC and PR curves, and error analysis
- Per-class metric breakdowns and example predictions

---

## 🛠️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/deep-learning-assignment.git
   cd deep-learning-assignment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks:**
   Open any `.ipynb` file using Google Colab or Jupyter Notebook.

---

## 📝 Walkthrough Video 🎥
A detailed code walkthrough video explaining model architectures, metrics, and error analysis is available [here](https://youtu.be/QLe8wnZ1dNs). 
---

## 📦 Requirements
- Python 3.8+
- TensorFlow
- Keras
- Weights & Biases (wandb)
- Scikit-learn
- Matplotlib, Seaborn


---
