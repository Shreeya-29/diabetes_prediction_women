#pip install streamlit
#pip install pandas
#pip install sklearn
#pip install scikit-learn
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Set the page config to wide layout
st.set_page_config(layout="wide") 

#reading dataset
df=pd.read_csv(r'C:\Prediction of diabetes in women\diabetes.csv')

# headings
st.title('Women\'s Diabetes Predictor')
st.sidebar.header('Patient Data')
st.subheader('Data Stats: Key Insights from the Training Dataset')
st.write(df.describe())

#visulization
st.subheader('Visualizing Diabetes Cases: Incidence Chart')
chart_data = df['Outcome'].value_counts()
fig = go.Figure(data=[go.Pie(labels=chart_data.index, values=chart_data.values, hole=0.5)])
st.plotly_chart(fig, use_container_width=True)

# x and y data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

#splitting dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()

st.subheader('Personal Health Data: Enter Your Information')
st.write(user_data)

# MODEL
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
user_data_scaled = scaler.transform(user_data)

rf = RandomForestClassifier()
rf.fit(x_train_scaled, y_train)

user_result = rf.predict(user_data_scaled)

# OUTPUT
st.subheader('Health Diagnosis: Your Diabetes Report')

if user_result[0] == 1:
    st.markdown(f'<h1 style="color:#ff0000;font-size:24px;">{"You are Diabetic"}</h1>', unsafe_allow_html=True)
else:
    st.markdown(f'<h1 style="color:#00f900;font-size:24px;">{"You are not Diabetic"}</h1>', unsafe_allow_html=True)

st.subheader('Model Evaluation: Accuracy Assessment')
st.write(str(accuracy_score(y_test, rf.predict(x_test_scaled)) * 100) + '%')
