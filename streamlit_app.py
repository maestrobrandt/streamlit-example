import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv('Case - Study Dataset.xlsx - Sheet1.csv')

from sklearn.model_selection import train_test_split, cross_val_score
features = ['GRE', 'TOPNOTCH', 'GPA']

X = df[features]
y = df['ADMIT'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_train_resampled, y_train_resampled = ros.fit_sample(X_train, y_train)

X_train_resampled_scaled = scaler.transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100, min_samples_split = 5, min_samples_leaf = 8, max_features ='sqrt',
 max_depth = 60)

forest.fit(X_train_resampled_scaled, y_train_resampled)

st.title('Admission Predictor')
st.subheader('GPA:')
gpa_slider = st.slider("Choose to the nearest 0.1", min_value = 0.0, max_value = 4.0, step = 0.1)
st.write('GPA:', gpa_slider)
gpa = gpa_slider
st.subheader('GRE Score:')
gre_slider = st.slider("Choose to the nearest 1", min_value = 0.0, max_value = 800.0, step = 1.0)
st.write('GRE Score:', gre_slider)
gre = gre_slider
tn_box = st.checkbox('Attended Top Notch')
tn = 0
if not tn_box:
    st.write('Did not attend top notch school')
else:
    st.write('Attended top notch school')
    tn = 1

st.header('Chance of Admission:')
button = st.button('PREDICT')
if button:
    st.write(round(((forest.predict_proba([[gre, tn, gpa]])[:,1])[0])*100, 2), "%")
