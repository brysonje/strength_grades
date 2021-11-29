
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import sklearn

# basic title
st.title("Strength Prediction")
st.caption("email: bryson_je@hotmail.com")
st.caption("twitter: @bryson_je")

# basic instruction
st.write("1. Start by select your **input** values from left side panel")
st.write("2. Select below the **prediction** algorithm")

# uploading pickle models
lnr_model = pickle.load(open("linear_regressor_prediction_trained_model.pkl", "rb"))
dtr_model = pickle.load(open("decision_tree_regressor_prediction_trained_model.pkl", "rb"))

# user input by using slider
st.sidebar.header("Use the slider for your input")
X1 = st.sidebar.slider("X1", 1, 12, 6)
X2 = st.sidebar.slider("X2", 2, 3)
X3 = st.sidebar.slider("X3", 0.00, 3.75, 1.85)
X4 = st.sidebar.slider("X4", 0.05, 4.45, 2.25)
X5 = st.sidebar.slider("X5", 325, 500, 412)
X6 = st.sidebar.slider("X6", 1.45, 6.00, 3.70)
X7 = st.sidebar.slider("X7", 0.80, 3.30, 2.05)
X8 = st.sidebar.slider("X8", 0.00, 3.00, 1.50)
X9 = st.sidebar.slider("X9", 17.00, 23.00, 20.00)
X10 = st.sidebar.slider("X10", 0.80, 5.50, 3.20)
X11 = st.sidebar.slider("X11", 59.00, 65.00, 62.00)
X12 = st.sidebar.slider("X12", 2.90, 5.50, 4.20)
X13 = st.sidebar.slider("X13", 0.00, 0.20, 0.10)
X14 = st.sidebar.slider("X14", 0.03, 9.10, 4.60)

# select the algorithm
radio_text = ""
radio_options = ["linear", "decision tree"]
prediction_model = st.radio(radio_text, radio_options)
if prediction_model == "linear":
    clf, RMSE, model = lnr_model, 2.08, "Linear Regression"
else:
    clf, RMSE, model = dtr_model, 1.92, "Decision Tree Regression"

# start prediction
st.write("3. Click the button below when **ready**")
if st.button("ready"):
    X_input = np.array([X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14])
    # X_array = np.array(user_input_values)
    y_pred = clf.predict(X_input.reshape(-1, 14))
    st.metric("Strength", value = float(y_pred))
    st.write("The RMSE for", model, "is around:", str(RMSE))
