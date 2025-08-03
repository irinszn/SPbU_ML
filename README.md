![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

# 🧠 Machine Learning Course — Homework Projects

This repository contains a collection of hands-on assignments completed as part of the Machine Learning course at Saint Petersburg State University (SPbU).

The goal of the course was to develop a solid understanding of ML algorithms through implementation from scratch and practical use of real-world tools. Each task involved working with real or synthetic datasets and focused on solving core ML problems such as regression, classification, clustering, dimensionality reduction, and text analysis.

Assignments combine algorithmic work with data preprocessing, visualization, metric analysis, and application of modern Python tools such as `scikit-learn`, `CatBoost`, `cvxopt`, `nltk`, `NumPy`, and others.

---

## 📂 Homework List

| # | Title | Summary | Links |
|--:|-------|---------|-------|
| 2 | KD-Tree | Manual implementation of KD-Tree for fast k-NN search in multidimensional space. | [📁 homework_2](homeworks/homework_2) · [PR #2](https://github.com/irinszn/SPbU_ML/pull/2) |
| 3 | Linear Regression | Predict house prices from the Kaggle dataset. Includes pipelines, preprocessing, regularization (L1/L2), feature and target transformations. | [📁 homework_3](homeworks/homework_3_linreg) |
| 4 | Gradient Descent | Custom implementation of gradient descent for linear regression with different loss functions (MSE, MAE). Convergence analysis. | [📁 homework_4](homeworks/homework_4_gradient_descent) |
| 5 | Support Vector Machine | Solving SVM using `cvxopt`: both linear and kernelized versions. Tested on synthetic data, kernel influence analysis. | [📁 homework_5](homeworks/homework_5_svm) |
| 6 | Ensembles: Random Forest & CatBoost | Random Forest implementation + CatBoost usage on real VK social network data. Task: predict user gender and age. | [📁 homework_6](homeworks/homework_6_ensembles) |
| 7 | Clustering | Manual implementation of KMeans, DBScan, Agglomerative Clustering. Includes image color quantization with clustering. | [📁 homework_7](homeworks/homework_7_clustering) |
| 8 | Text Classification | Spam classifier using Bag of Words, TF-IDF, Snowball stemmer, Naive Bayes. End-to-end NLP pipeline. | [📁 homework_8](homeworks/homework_8_texts) |

---

## 🚀 Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/irinszn/SPbU_ML.git
   ```

2. Navigate to the homework folder you want to explore:
   ```bash
   cd SPbU_ML/src/homeworks/homework_3
   ```
3. Launch the Jupyter Notebook interface:
   ```bash
   jupyter notebook
   ```
4. Open the corresponding `.ipynb` file (e.g., `linreg.ipynb`) in your browser and run the code interactively.

