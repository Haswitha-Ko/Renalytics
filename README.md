# Renalytics

Machine learning driven prediction of chronic nephrology related diseases

## Overview
Renalytics is an AI-powered tool for predicting the likelihood of chronic kidney disease in patients. The project leverages machine learning algorithms and transformer models to analyze both numerical and text data, providing a more holistic approach to CKD risk assessment.

## Features
- Predicts chronic kidney disease using clinical data and patient notes.
- Integrates XGBoost for structured data and transformers for text feature extraction.
- Handles missing values and categorical encoding automatically.
- Provides data visualization, including heatmaps and count plots, for insights into feature correlations and distributions.
- Evaluates model performance using accuracy metrics.

## Installation
To run this project, you'll need to install the required libraries:

```bash
pip install pandas scikit-learn xgboost transformers matplotlib seaborn numpy```

## Usage
Clone this repository:
```bash
git clone https://github.com/yourusername/Renalytics.git
Navigate to the project directory:
bash``

cd Renalytics
Run the main script:
```bash
python main.py```
