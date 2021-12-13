
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import sklearn
import matplotlib.pyplot as plt

# this is to set a restart button
restart_button = 0

# basic title
st.title("Strength Prediction")
col1, col2 = st.columns(2)
with col1:
    st.caption("bryson_je@hotmail.com")
with col2:
    st.write("**Basic Data Visualization** [link](https://basicdatavisualization.herokuapp.com)")

# basic instruction
st.write("1. Start by select your **input** values from left side panel")

# uploading pickle models
lnr_model = pickle.load(open("linear_regressor_prediction_trained_model.pkl", "rb"))
dtr_model = pickle.load(open("decision_tree_regressor_prediction_trained_model.pkl", "rb"))
df = pd.read_csv("Y1.csv")

# user input by using slider
st.sidebar.header("Use the slider for your input")
X1 = st.sidebar.slider("X1", 1, 12, 6)
X2 = st.sidebar.slider("X2", 2, 3)
X3 = st.sidebar.slider("X3", 0.00, 3.75, 1.86)
X4 = st.sidebar.slider("X4", 0.05, 4.45, 2.25)
X5 = st.sidebar.slider("X5", 330, 500, 415)
X6 = st.sidebar.slider("X6", 1.45, 6.00, 3.73)
X7 = st.sidebar.slider("X7", 0.80, 3.30, 2.05)
X8 = st.sidebar.slider("X8", 0.00, 3.00, 1.50)
X9 = st.sidebar.slider("X9", 17.80, 23.00, 20.40)
X10 = st.sidebar.slider("X10", 0.80, 5.50, 3.15)
X11 = st.sidebar.slider("X11", 59.00, 65.00, 62.00)
X12 = st.sidebar.slider("X12", 2.90, 5.10, 4.00)
X13 = st.sidebar.slider("X13", 0.00, 0.20, 0.10)
X14 = st.sidebar.slider("X14", 0.03, 9.10, 4.55)

# select the algorithm
clf1, RMSE1, model1 = lnr_model, 2.08, "**Linear Regression**"
clf2, RMSE2, model2 = dtr_model, 1.92, "**Decision Tree Regression**"
RMSE3, model3 = 1.52, "**Ensemble**"
lnr_coef, dtr_coef = 0.215, 0.785

# start prediction
st.write("2. Click the button below when **ready**")
if st.button("ready"):
    fig, ax = plt.subplots(figsize = (6, 2))
    ax.hist(df["Y1"], bins = 40)
    st.pyplot(fig)
    X_input = np.array([X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14])
    col1, col2, col3 = st.columns(3)
    with col1:
        y_pred1 = clf1.predict(X_input.reshape(-1, 14))
        st.write(model1)
        st.metric("Strength Prediction", value = float("%.2f" % y_pred1))
        st.write("RMSE(cv) is:", str(RMSE1))
    with col2:
        y_pred2 = clf2.predict(X_input.reshape(-1, 14))
        st.write(model2)
        st.metric("Strength Prediction", value = float("%.2f" % y_pred2))
        st.write("RMSE(cv) is:", str(RMSE2))
    with col3:
        y_pred3 = (y_pred1 * lnr_coef + y_pred2 * dtr_coef)
        st.write(model3)
        st.metric("Strength Prediction", value = float("%.2f" % y_pred3))
        st.write("RMSE(cv) is :", str(RMSE3))
    if st.button("Click to restart"):
        if restart_button == 0:
            restart_button == 0
