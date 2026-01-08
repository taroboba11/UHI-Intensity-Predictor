#set up
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
df = pd.read_excel('data.xlsx')
columns = ["Urban_name","NDVI_urb_CT_act","DelNDVI_annual","DelDEM","UHI_annual_day"]
df = df[columns]
df = df.dropna()
st.title("Urban Heat Island Prediction Modeling Framework")
st.write(
"""
This software predicts and compares UHI intensity using a formula-based model and a ML-based linear regression model along with user-provided variables such as the vegetation, albedo, and building density.
"""
)

#select + display city data
city = st.selectbox(
    "Select and Urban Area:",
    sorted(df["Urban_name"].unique())
)
city_data = df[df["Urban_name"]== city]
st.subheader(f"City Data")
st.dataframe(city_data)

ndvi = city_data["NDVI_urb_CT_act"].mean()
del_ndvi = city_data["DelNDVI_annual"].mean()
del_dem = city_data["DelDEM"].mean()
del_dem_scale = del_dem/100
uhi_actual = city_data["UHI_annual_day"].mean()

#sliders
st.subheader("Simulate Changes")
ndvi_input = st.slider("Urban NDVI",0.0,1.0, float(ndvi),0.01)
del_ndvi_input = st.slider("ΔNDVI (Urban-Rural))",-0.5,0.5, float(del_ndvi),0.01)
del_dem_input = st.slider("ΔDEM (Elevation Difference in m))",-50.0,50.0, float(del_dem),0.1)
albedo_input = st.slider("Albedo (0-1)",0.1,0.9,0.3,0.01)
urban_frac_input = st.slider("Urban Fraction (0-1)",0.0,1.0,0.5,0.01)
del_dem_input_scale = del_dem_input/100
#formula model
X_phys = pd.DataFrame({
    "inv_ndvi": 1 - df["NDVI_urb_CT_act"],
    "del_ndvi": df["DelNDVI_annual"],
    "del_dem": df["DelDEM"] / 100
})

y_phys = df["UHI_annual_day"]

phys_model = LinearRegression()
phys_model.fit(X_phys, y_phys)

ALPHA, BETA, GAMMA = phys_model.coef_
INTERCEPT = phys_model.intercept_

uhi_formula_input = (INTERCEPT +
    ALPHA * (1 - ndvi_input) +
    BETA * del_ndvi_input +
    GAMMA * del_dem_input_scale)
st.subheader("UHI Calculation using Formula-based Model")

st.write(f"**Predicted UHI (Formula Model):** {uhi_formula_input:.2f} °C")
st.write(f"**Actual UHI:** {uhi_actual:.2f} °C")
#formula sensitivity analysis
ndvi_range = np.arange(0,1,0.05)
predicted_uhi_sensitivity = []
for test in ndvi_range:
    predicted_uhi_sensitivity.append(INTERCEPT +
    ALPHA * (1 - test) +
    BETA * del_ndvi_input +
    GAMMA * del_dem_input_scale)
sensitivity_table = pd.DataFrame({
    "Simulated NDVI": ndvi_range,
    "Predicted UHI °C": predicted_uhi_sensitivity
})
st.subheader("Sensitivity Analysis: Impact of Vegetation on UHI Intensity")
st.dataframe(sensitivity_table)

fig = px.line(sensitivity_table, x = "Simulated NDVI", y="Predicted UHI °C", title="Predicted UHI vs NDVI",labels = {"Simulated NDVI":"NDVI","Predicted UHI °C":"UHI (°C)"})
st.plotly_chart(fig)
comparison_df = pd.DataFrame({
    "Type": ["Formula Prediction", "Actual"],
    "UHI (°C)": [uhi_formula_input, uhi_actual]
})
fig2 = px.bar(comparison_df, x="Type", y="UHI (°C)", text = "UHI (°C)",title="Actual vs Formula Model UHI", color="Type")
st.plotly_chart(fig2)

#ML model
train_df = (
    df[df["Urban_name"] != city]
    .groupby("Urban_name")
    .median()
    .reset_index()
)

test_df = (
    df[df["Urban_name"] == city]
    .groupby("Urban_name")
    .median()
    .reset_index()
)

x_train = train_df[["NDVI_urb_CT_act", "DelNDVI_annual", "DelDEM"]]
y_train = train_df["UHI_annual_day"]

x_test = test_df[["NDVI_urb_CT_act", "DelNDVI_annual", "DelDEM"]]
y_test = test_df["UHI_annual_day"]

x_train["DelDEM"] = x_train["DelDEM"] / 100
x_test["DelDEM"] = x_test["DelDEM"] / 100

ml_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,   
    reg_lambda=1.0,   
    objective="reg:squarederror",
    random_state=42
)

ml_model.fit(
    x_train,
    y_train,
    eval_set=[(x_test, y_test)],
    verbose=False
)

y_test_pred = ml_model.predict(x_test)
uhi_ml_city = y_test_pred.mean()


ml_input = pd.DataFrame([{
    "NDVI_urb_CT_act": ndvi_input,
    "DelNDVI_annual": del_ndvi_input,
    "DelDEM": del_dem_input / 100
}])
uhi_ml_input = ml_model.predict(ml_input)[0]

mae_ml = mean_absolute_error(y_test, y_test_pred)
st.subheader("UHI Calculation using ML-based Model")

st.write(f"**Predicted UHI (ML Model):** {uhi_ml_input:.2f} °C")
st.write(f"**Actual UHI:** {uhi_actual:.2f} °C")
#ML sensitivity analysis
predicted_ml_sensitivity = [
    ml_model.predict(pd.DataFrame([{
        "NDVI_urb_CT_act": ndvi_val,
        "DelNDVI_annual": del_ndvi_input,
        "DelDEM": del_dem_input / 100
    }]))[0]
    for ndvi_val in ndvi_range
]
ml_sensitivity_table = pd.DataFrame({
    "Simulated NDVI": ndvi_range,
    "Predicted UHI °C": predicted_ml_sensitivity
}) 
st.subheader("Sensitivity Analysis: Impact of Vegetation on UHI Intensity (ML Model)")
st.dataframe(ml_sensitivity_table)
fig4 = px.line(ml_sensitivity_table, x = "Simulated NDVI", y="Predicted UHI °C", title="Predicted UHI vs NDVI (ML Model)",labels = {"Simulated NDVI":"NDVI","Predicted UHI °C":"UHI (°C)"})
st.plotly_chart(fig4)
#comparison plot
st.subheader("Comparison of Actual UHI, ML Model Prediction, and Formula Model Prediction")
comparison_df = pd.DataFrame({
    "Type": ["Actual", "ML Prediction", "Formula Prediction"],
    "UHI (°C)": [uhi_actual, uhi_ml_input, uhi_formula_input]
})
fig3 = px.bar(comparison_df, x="Type", y="UHI (°C)", text = "UHI (°C)",title="Actual vs ML Model vs Formula Model Predicted UHI", color="Type")
st.plotly_chart(fig3)

#model accuracy evaluation
y_pred_ml = ml_model.predict(x_test)

def formula_model(row):
    return (
        INTERCEPT +
        ALPHA * (1 - row["NDVI_urb_CT_act"]) +
        BETA * row["DelNDVI_annual"] +
        GAMMA * (row["DelDEM"] / 100)
    )

y_pred_formula = x_test.apply(formula_model, axis=1)

mae_ml = mean_absolute_error(y_test, y_pred_ml)
mae_formula = mean_absolute_error(y_test, y_pred_formula)

st.subheader("Model Accuracy Evaluation")
st.write(f"**ML Model MAE:** {mae_ml:.2f} °C")
st.write(f"**Formula Model MAE:** {mae_formula:.2f} °C")

error_df = pd.DataFrame({
    "Model": ["Formula Model", "ML Model"],
    "Mean Absolute Error (°C)": [mae_formula, mae_ml]
})

fig_error = px.bar(
    error_df,
    x="Model",
    y="Mean Absolute Error (°C)",
    title="Average Prediction Error Comparison",
    text="Mean Absolute Error (°C)"
)
st.plotly_chart(fig_error)

#testing result