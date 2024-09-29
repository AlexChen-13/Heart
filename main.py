import streamlit as st
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
import sklearn
sklearn.set_config(transform_output="pandas")
import joblib

# model = CatBoostClassifier()
# model.load_model('/home/aleksey/DS_bootcamp/ds-phase-1/06-supervised/model.cbm')
ml_pipeline = joblib.load('ml_pipeline.pkl')

st.title('Предсказание сердечных заболеваний на предобученной модели')

feature1 = st.slider('Введите возраст:', min_value=0.0, max_value=120.0, value=0.0, step=1.0)
feature2 = st.selectbox('Введите пол:', options=['F', 'M'])
feature3 = st.selectbox('Введите ChestPainType:', options=['ATA', 'NAP', 'ASY', 'TA'])
feature4 = st.slider('Введите RestingBP:', min_value=0, max_value=200, value=0, step=1)
feature5 = st.slider('Введите Cholesterol:', min_value=0, max_value=700, value=0, step=1)
feature6 = st.selectbox('Введите FastingBS:', options=[0, 1])
feature7 = st.selectbox('Введите RestingECG:', options=['Normal', 'ST', 'LVH'])
feature8 = st.slider('Введите MaxHR:', min_value=30, max_value=300, value=0, step=1)
feature9 = st.selectbox('Введите ExerciseAngina:', options=['N', 'Y'])
feature10 = st.slider('Введите Oldpeak:', min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
feature11 = st.selectbox('Введите ST_Slope:', options=['Up', 'Flat', 'Down'])

input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, 
                             feature6, feature7, feature8, feature9, feature10, 
                             feature11]], 
                           columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 
                                    'Cholesterol', 'FastingBS', 'RestingECG', 
                                    'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])


if st.button('Сделать прогноз'):
    try:
        prediction = ml_pipeline.predict(input_data)
        if prediction[0] == 0: 
            st.write('Вам пора к врачу')
        else:
            st.write('Поздравляем, вы здоровы!')
    except Exception as e:
        st.write("Ошибка при выполнении предсказания:", e)

