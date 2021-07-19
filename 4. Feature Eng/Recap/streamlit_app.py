import streamlit as st 

import time
import math

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import impute
from sklearn import compose
from sklearn import metrics
from sklearn import set_config

from PIL import Image

set_config(display='diagram')


# Data Enhancement Function
#Based on Gender


def income_RNG_on_gender(df):
    #Copying to a dummy dataframe
    gen_data = df.copy() 
    
    #Localizing standard deviation based on gender
    for gender in gen_data['Genre'].unique():
        gen_std = gen_data[gen_data['Genre']==gender]
        income_std = gen_std ['Annual Income (k$)'].std()

        #Altering the data based on std
        for i in range (gen_data[gen_data['Genre']==gender].shape[0]):
            if np.random.randint(2)==1:
                gen_data['Annual Income (k$)'].values[i] += income_std/10
            else:
                gen_data['Annual Income (k$)'].values[i] -= income_std/10

    return gen_data

st.title("Malls customer data analysis")

df = pd.read_csv("data/Mall_Customers.csv")



# Setting up the column variables for pipeline
st.write(df.head())

df['SpendingLabel'] = df['Spending Score (1-100)'].apply(lambda row: 1 if row>50 else 0)
df.drop("Spending Score (1-100)", axis=1, inplace=True)

st.write(df.head())

cat_vars = ['Genre']
num_vars = ['Age', 'Annual Income (k$)']

st.write(f"Categorical columns: {cat_vars}")
st.write(f"Numerical columns: {num_vars}")
# Preprocessing steps

# Numerical columns
num_preproc1 = pipeline.Pipeline(steps=[('imputer', impute.SimpleImputer(strategy='most_frequent')),
                                        ('scaler', preprocessing.StandardScaler()),
                                        ('normalizer', preprocessing.QuantileTransformer(n_quantiles=100))])

# Categorical columns
cat_preproc1 = pipeline.Pipeline(steps=[('imputer', impute.SimpleImputer(strategy='constant', fill_value=-1)),
                                        ('encoder', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1))])

# Putting the two together
tree_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_preproc1, num_vars),
    ('cat', cat_preproc1, cat_vars)
    ], remainder = 'drop')

image = Image.open('./pipeline.png')
st.image(image, caption='Pipeline', use_column_width='always')

# Import and create dictionary of all models
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier
# from sklearn.svm           import SVC

tree_classifiers = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":   ExtraTreesClassifier(n_estimators=100),
  "Random Forest": RandomForestClassifier(n_estimators=100),
  "AdaBoost":      AdaBoostClassifier(n_estimators=100),
  "Skl GBM":       GradientBoostingClassifier(n_estimators=100),
  "Skl HistGBM":   HistGradientBoostingClassifier(max_iter=100),
  "XGBoost":       XGBClassifier(n_estimators=100),
  "LightGBM":      LGBMClassifier(n_estimators=100),
  "CatBoost":      CatBoostClassifier(n_estimators=100),
#   "SVM":           SVC(kernel='linear')
}

tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}



X = df.drop('SpendingLabel', axis=1)
y = df['SpendingLabel']

page_names = ['Original Data', '30\% Enhanced Data', 'Feature Generation']
page = st.radio('Select data', page_names)

# Generate the Generated Data
generated_data = income_RNG_on_gender(df)

# Take 30% of the information
extra_sample = generated_data.sample(math.floor(generated_data.shape[0] * 30 / 100))

if page == '30\% Enhanced Data':

    ## ADDITION OF DATA ENHANCEMENT

    ## COMMENT THIS OUT IF YOU WANT TO ADD THE ENHANCED DATA

    ## Concatenate train dataset with extra_sample from generated data

    if 'Income_spending_cat' in cat_vars:
        cat_vars.remove('Income_spending_cat')
    X = pd.concat([X, extra_sample.drop(['SpendingLabel'],axis=1)])
    y = pd.concat([y, extra_sample['SpendingLabel']])

    X_train, x_test, Y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    stratify = y,   # ALWAYS RECOMMENDED FOR BETTER VALIDATION
    random_state=42  # Recommended for reproducibility
)


elif page == 'Original Data':
    if 'Income_spending_cat' in cat_vars:
        cat_vars.remove('Income_spending_cat')

    X_train, x_test, Y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    stratify = y,   # ALWAYS RECOMMENDED FOR BETTER VALIDATION
    random_state=42  # Recommended for reproducibility
)

 

elif page == 'Feature Generation':
    # Feature Generation : SpendingLabel (Our Target)
    # Will add the label column based on Spending Score: If SS>50=1 else 0


    Income_spending_cat =[]
    for income in df['Annual Income (k$)']:
        
        if income < 40:
            Income_spending_cat.append(0) 

        elif 40 < income < 60:
            Income_spending_cat.append(1)
            
        else:
            Income_spending_cat.append(2)

    df['Income_spending_cat'] = Income_spending_cat

    cat_vars.extend('Income_spending_cat')

    X_train, x_test, Y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    stratify = y,   # ALWAYS RECOMMENDED FOR BETTER VALIDATION
    random_state=42  # Recommended for reproducibility
)

# Second Train_Test_Split as an Alternative for One-Time Cross Validation

x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train,
    test_size=0.2,
    stratify = Y_train,   # ALWAYS RECOMMENDED FOR BETTER VALIDATION
    random_state=42  # Recommended for reproducibility
)

results = pd.DataFrame({'Model': [], 'Recall.': [], 'Precision': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

for model_name, model in tree_classifiers.items():

    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_val)
    
    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                              "Recall.": metrics.recall_score(y_val,pred)*100,
                              "Precision": metrics.precision_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)

results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

st.write(results_ord)

best_model = tree_classifiers[results_ord.iloc[0].Model]
best_model.fit(X_train,Y_train)

test_pred = best_model.predict(x_test)


st.write("Best model recall:", metrics.recall_score(y_test, test_pred))
st.write("Best model precision:", metrics.precision_score(y_test, test_pred))
st.write("Best model accuracy:", metrics.accuracy_score(y_test, test_pred))