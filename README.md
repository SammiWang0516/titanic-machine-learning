# Titanic Survival Prediction – Machine Learning Solution

## Overview
This project presents a comprehensive machine learning solution for the classic **Kaggle Titanic competition**, aimed at predicting passenger survival based on demographic and travel features. Using advanced data preprocessing, exploratory data analysis (EDA), feature engineering, and state-of-the-art classification models, this solution demonstrates practical skills in data science and predictive modeling.

## Key Features

- **Robust Data Cleaning & Feature Engineering**: Handling missing values, extracting titles from names, creating family size and alone indicators, and transforming ticket and cabin information.
- **Exploratory Data Analysis (EDA)**: Insightful visualization and statistical testing to reveal data patterns and feature relationships with the survival outcome.
- **Statistical Significance Testing**: Application of chi-squared tests and t-tests to validate feature importance, ensuring interpretable and data-driven feature selection.
- **Pipeline & Model Optimization**: Leveraging `scikit-learn` pipelines and GridSearchCV for clean, repeatable preprocessing and hyperparameter tuning.
- **Ensemble Learning Models**: Utilizing Logistic Regression and Random Forest Classifier to achieve strong and balanced predictive performance.
- **Evaluation Metrics & Visualization**: Thorough model validation with accuracy, ROC-AUC, precision-recall curves, and detailed classification reports.

## Technologies & Tools

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn (preprocessing, model selection, pipelines)
- Statistical libraries (scipy, statsmodels)
- Jupyter Notebook (interactive development and visualization)

## How to Use

1. Clone the repository:
git clone https://github.com/SammiWang0516/titanic-machine-learning.git

2. Install dependencies:
pip install -r requirements.txt

3. Place the training data file (`train.csv`) inside the `data/` directory.

4. Run the pipeline script to train and tune the Random Forest model:
scripts/Kaggle_Titanic_Pipeline.py

5. After successful run, the best model and grid search results will be saved automatically inside the `models/` folder.

6. You can explore, modify, or extend the pipeline in the `Kaggle_Titanic_Pipeline.py` script or examine detailed analysis and further experiments in the Jupyter notebook `Kaggle_Titanic_RandomForest.ipynb`.

7. To experiment interactively, launch Jupyter Notebook:
notebooks/Kaggle_Titanic_RandomForest.ipynb

This streamlined workflow ensures easy reproduction of model training, tuning, and evaluation with modular and maintainable code.

## Project Structure

- `Kaggle_Titanic_RandomForest.ipynb` — An in-depth Jupyter notebook presenting the full modeling workflow including data import, exploratory data analysis, feature engineering, model building, hyperparameter tuning, and evaluation.
- `Kaggle_Titanic_Pipeline.py` — Python script implementing a modular pipeline for preprocessing and training a Random Forest model with hyperparameter tuning using GridSearchCV.
- `data/` — Directory designated for storing the Titanic dataset files, such as training and test CSV files.
- `models/` — Folder for storing serialized models and GridSearchCV results (e.g., pickle files) generated after training.

## Results

- Achieved strong and balanced predictive performance using both Logistic Regression and Random Forest models, validated through cross-validation and hold-out validation sets.
- Statistically validated key features influencing survival prediction using chi-squared and t-tests, enabling interpretable and data-driven feature selection.
- Developed a clear, modular, and maintainable codebase incorporating preprocessing pipelines and hyperparameter tuning for reproducible model training and evaluation.
- Generated comprehensive evaluation metrics including accuracy scores, classification reports, and ROC/AUC curves to rigorously assess model performance.
- Produced serialized model files and grid search results facilitating easy deployment and further experimentation.

---

🚀 **Dive into the notebook and see how each step is crafted with best practices in data science and machine learning!**

For questions or collaborations, feel free to open an issue or contact me directly through GitHub.

---

© 2025 by Sammi Wang | Data Science Enthusiast | Kaggle Competitor 
