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
cd titanic-machine-learning

2. Install dependencies:
pip install -r requirements.txt

3. Run the notebook `Kaggle_Titanic_RandomForest.ipynb` to explore the dataset, understand preprocessing steps, and train the models.

4. Modify and experiment with different models or feature engineering strategies to improve accuracy!

## Project Structure

- `Kaggle_Titanic_RandomForest.ipynb` — Detailed modeling notebook including data import, cleaning, visualization, feature engineering, modeling, tuning, and evaluation.
- `data/` — Directory intended for training and test datasets.
- `models/` — Saves model checkpoints and serialized objects.
- `.gitignore` — Recommended files to ignore for clean version control.

## Results

- Achieved balanced performance using Logistic Regression and Random Forest.
- Statistically validated features driving survival prediction.
- Clear, maintainable codebase suitable for extension and further research.

---

🚀 **Dive into the notebook and see how each step is crafted with best practices in data science and machine learning!**

For questions or collaborations, feel free to open an issue or contact me directly through GitHub.

---

© 2025 by Sammi Wang | Data Science Enthusiast | Kaggle Competitor 
