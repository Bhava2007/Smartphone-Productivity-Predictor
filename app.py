import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Productivity Predictor", page_icon="📱")

st.title("📱 Smartphone Usage Productivity Predictor")

df = pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000.csv")

st.subheader("Dataset Preview")
st.write(df.head())

target = df.columns[-1]
X = df.drop(target, axis=1)
y = df[target]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.sidebar.header("Enter Usage Details")

user_input = {}

for col in X.columns:
    user_input[col] = st.sidebar.number_input(col, 0.0)

input_df = pd.DataFrame([user_input])

prediction = model.predict(input_df)

if st.sidebar.button("Predict"):
    st.success(f"Predicted Productivity Level: {prediction[0]}")
