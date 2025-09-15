# import basic libraries
import pandas as pd
import numpy as np

# hide warnings if max iteration time does not converge
import warnings

# file location
import os

# data visualization (along with EDA stage)
import matplotlib.pyplot as plt
import seaborn as sns

# data preprocessing (along with EDA stage)
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder

# data splitting
from sklearn.model_selection import train_test_split

# for evaluation of statistical significance of categorical feature vs categorical target
from scipy.stats import chi2_contingency

# for evaluation of statistical significance of numeric feature vs categorical target
from scipy.stats import ttest_ind

# pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Inputing missing value
from sklearn.impute import SimpleImputer

# GridSearchCV
from sklearn.model_selection import GridSearchCV

# modeling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# model evaluation
from sklearn.metrics import accuracy_score, classification_report

# for saving model as pickle file
import joblib

# loading data to Pandas DataFrame
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

# feature engineering before data splitting and preprocessing
def feature_engineering(df):
    # combine SibSp & Parch: FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # check FamilySize and see if the passenger is alone: IsAlone
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 0, 'IsAlone'] = 1

    # helper function to change th
    def change_prefix(title):
        common_prefix = ['Mr', 'Miss', 'Mrs', 'Master']
        if title in common_prefix:
            return title
        else:
            return 'Others'

    # extract prefix from Name: Title (into 5 categories)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand = False)
    df['Title'] = df['Title'].apply(change_prefix)

    # estimate Ticket length: Ticket_length
    df['Ticket_length'] = df['Ticket'].apply(lambda x: len(str(x)))

    return df

# pipeline include preprocessing and model
def build_pipeline():
    power_features = ['Age', 'Fare']                                        # impute median missing value, fix skewness, and scale
    scale_features = ['FamilySize', 'Ticket_length']                        # scale only
    onehot_features = ['Pclass', 'Sex', 'Title', 'Embarked']                # impute most-frequent missing value, and onehot encoder
    binary_feature = ['IsAlone']

    preprocessor = ColumnTransformer (
        transformers = [
            ('power_scaler', Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'median')),
                ('power', PowerTransformer())                               # standardize = True, method = 'yeo-johnson' by default
            ]), power_features),

            ('scaler', StandardScaler(), scale_features),

            ('onehot_encoder', Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
            ]), onehot_features),

            ('binary', 'passthrough', binary_feature)
        ],
        remainder = 'drop'
    )
    # create the final model pipeline - random forest classifier
    random_forest_model = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state = 42))
    ])

    return random_forest_model

# GridSearchCV
def train_and_tune(model, x_train, y_train):
    param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
    }

    random_forest_grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    cv = 5,
    scoring = 'accuracy',
    n_jobs = 1,
    verbose = 1
    )

    random_forest_grid_search.fit(x_train, y_train)

    return {
        'best_estimator': random_forest_grid_search.best_estimator_,
        'best_score': random_forest_grid_search.best_score_,
        'best_params': random_forest_grid_search.best_params_,
        'cv_results': random_forest_grid_search.cv_results_,
        'grid_search_result': random_forest_grid_search
    }

# save models (both grid search result and model with best parameters)
def save_models():
    current_script = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_script, '..', 'models')
    return os.path.abspath(models_dir)

def main():

    # hide warnings
    warnings.filterwarnings('ignore')

    # load data
    current_script = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(current_script, '..', 'data', 'train.csv')
    df = load_data(train_file_path)

    # make a copy of df
    df_clean = df.copy()

    # feature engineering before data splitting
    df_clean = feature_engineering(df_clean)

    # data splitting (validation set = 20% training set)
    x_train, x_val, y_train, y_val = train_test_split(df_clean.drop(columns = ['Survived']), 
        df_clean['Survived'], 
        test_size = 0.2,
        random_state = 42, 
        stratify = df_clean['Survived']
    )

    # model after pipeline (preprocessor + random forest classifier)
    model = build_pipeline()
    
    # tuning hyperparameters using training set only
    tuning_result = train_and_tune(model, x_train, y_train)

    # print the crucial result
    print(f'Best CV score during tuning: {round(tuning_result['best_score'], 4)}')
    print(f'Best Parameters: {tuning_result['best_params']}')

    # predict the validatin set and see the accuracy score
    y_pred = tuning_result['best_estimator'].predict(x_val)
    print(f'Accuracy Score tested on validation dataset: {round(accuracy_score(y_val, y_pred), 2)}')

    # save model
    models_dir = save_models()
    joblib.dump(tuning_result['best_estimator'], os.path.join(models_dir, 'best_model_titanic.pkl'))

    # save entire grid search result
    joblib.dump(tuning_result['grid_search_result'], os.path.join(models_dir, 'grid_search_result.pkl'))

    print(f'Models saved to: {models_dir}')

if __name__ == '__main__':
    main()