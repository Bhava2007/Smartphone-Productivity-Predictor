import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Productivity Predictor", page_icon="📱")

st.title("📱 Smartphone Usage Productivity Predictor")

@st.cache_data
def load_data():
    return pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000.csv")

df = load_data()

target = df.columns[-1]
X = df.drop(target, axis=1)
y = df[target]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    return model

model = train_model()

st.sidebar.header("Enter Usage Details")

user_input = {}

for col in X.columns:
    user_input[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([user_input])

if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Productivity Level: {prediction[0]}")
