import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

data_url="https://raw.githubusercontent.com/cakirogl/unloading_hours/refs/heads/main/inliers0.15.csv"
df = pd.read_csv(data_url, header=0)

# Encode categorical variable
#le = LabelEncoder()
#df['Fiber Type'] = le.fit_transform(df['Fiber Type'])

# Split features and target
x = df.iloc[:,:-1].values  # Convert to numpy array
y = df.iloc[:,-1].values   # Convert to numpy array
scaler=StandardScaler()
x=scaler.fit_transform(x)

#et_url="https://raw.githubusercontent.com/cakirogl/splitting_tensile_composite/main/et_model.pkl"
#lgbm_url="https://raw.githubusercontent.com/cakirogl/splitting_tensile_composite/main/lgbm_model.pkl"
#xgb_url="https://raw.githubusercontent.com/cakirogl/splitting_tensile_composite/main/xgb_model.pkl"
#response = requests.get(et_url)
#et_model = pickle.loads(response.content)
lgbm_model = LGBMRegressor()
lgbm_model.fit(x,y)

ic=st.container()
ic1,ic2 = ic.columns(2)
with ic1:
    gross_weight = st.number_input("**Truck gross weight [$kg$]**", min_value=2900.0, max_value=21500.0, step=300.0, value=5000.0);
    leg_distance = st.number_input("**Leg distance [km]**", min_value=0.03, max_value=250.0, step=2.0, value=50.0)

with ic2:
    load_of_leg = st.number_input("**Load of leg [$kg$]**", min_value=7.0, max_value=13770.0, step=200.0, value=4000.0)
   
oc=st.container()
new_sample = np.array([[gross_weight, leg_distance, load_of_leg]], dtype=object)
with ic2:
    st.write(f":blue[**Unloading hours = **{lgbm_model.predict(new_sample)[0]:.2f}** [h]**]")