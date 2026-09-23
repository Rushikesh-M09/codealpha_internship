# CodeAlpha Machine Learning Internship

## Internship Details

| | |
|---|---|
| **Intern** | Rushikesh Mahajan |
| **Organization** | CodeAlpha |
| **Role** | Machine Learning Intern |
| **Duration** | 10th September 2026 – 10th October 2026 |
| **Mode** | Virtual (Remote) |

CodeAlpha offered this virtual Machine Learning Internship to provide hands-on experience through practical, project-based assignments. The internship required active participation, timely completion of assigned tasks, and professional conduct throughout, with a performance-based stipend tied to contribution and completion of the assigned work.

This repository contains the projects completed as part of the internship.

## Projects

### 1. Creditworthiness — Credit Risk Classification
A binary classification project that predicts the probability of a borrower defaulting on credit. A synthetic dataset of borrower financial attributes (income, debt payments, credit utilization, delinquencies, etc.) is used to engineer credit-risk features like **Debt-to-Income ratio** and **Credit Utilization rate**, then a **Logistic Regression** baseline is compared against an **XGBoost** challenger model. Both models are evaluated with classification reports and ROC-AUC, with class imbalance handled via balanced class weights / `scale_pos_weight`.

📄 File: `Creditworthiness.ipynb` — see `README.md` for full details.

### 2. HCR — Handwritten Character Recognition
A progression of handwriting-recognition experiments, moving from a simple baseline to a more complete pipeline:
- A **CNN** trained on **MNIST** to recognize handwritten digits (0–9)
- A deeper **CNN** trained on **EMNIST Letters** to recognize handwritten letters (a–z)
- A **video/webcam OCR pipeline** (OpenCV) that detects handwritten characters in frames and reconstructs lines of text using the trained letter model
- Supporting scripts for single-image inference and exploring the EMNIST Balanced (47-class) dataset

📄 File: `HCR.ipynb` — see `HCR_README.md` for full details.

## Repository Structure

```
.
├── Creditworthiness.ipynb        # Project 1: credit default risk model
├── README.md                     # Detailed README for the Creditworthiness project
├── HCR.ipynb                     # Project 2: handwritten character recognition
├── HCR_README.md                 # Detailed README for the HCR project
├── CodeAlpha_Offer_Letter.pdf    # Internship offer letter
└── Internship_README.md          # This file — internship overview
```

## Acknowledgment

Thanks to **CodeAlpha** and **Swati Srivastava (Co-Founder & CEO)** for the opportunity to work on these projects as part of the Machine Learning Internship.
