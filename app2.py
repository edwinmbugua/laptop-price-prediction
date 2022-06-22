import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

#load the model and dataframe
df = pd.read_csv('laptop_cleaned.csv')
pipe= pickle.load(open("pipe.pkl","rb"))

st.title("Laptop Price Predictor")

#Now we will take user input one by one as per our dataframe

#Brand
Company = st.selectbox('Brand', df['Company'].unique())

#Type of laptop
lap_type = st.selectbox("Laptop Type", df['TypeName'].unique())

#Ram
ram = st.selectbox("Ram(in GB)", [2,4,6,8,12,16,24,32,64])
#Operating system
os = st.selectbox('Operating System',df['OpSys'].unique())
#weight
weight = st.number_input("Weight of the Laptop(Kg)")
#Touch screen
touchscreen = st.selectbox("TouchScreen", ['No', 'Yes'])
#IPS
ips = st.selectbox("IPS(“In-Plane Switching” monitor for better color accuracy)", ['No', 'Yes'])
#Cpu speed
Cpu_Speed=st.selectbox('CPU_Speed (GHz)',df['Cpu_Speed'].unique())
#screen size
screen_size = st.number_input('Screen Size (Inches)')
# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu_type=st.selectbox('CPU_type',df['Cpu_Type'].unique())
gpu_vender = st.selectbox('GPU',df['Gpu_vender'].unique())
hdd= st.selectbox('Hard Drive(HDD) -GB)-traditional storage devices',[0,128,256,512,1024,2048])
ssd = st.selectbox('Solid State Drive(SSD) -GB)-use newer technology ',[0,128,256,512,1024,2048])
#Prediction
if st.button('Predict Price'):
    ppi=None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0
        
    if ips == "Yes":
        ips = 1
    else:
        ips = 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res**2)) ** 0.5 / screen_size
    query=np.array([Company,lap_type,ram,os,weight,touchscreen,ips,Cpu_Speed,cpu_type,gpu_vender,hdd,ssd,ppi])
    query = query.reshape(1,13)
    prediction = str(int(np.exp(pipe.predict(query)[0])))
    st.title("Based on the selected specs laptop's predicted price is $" + prediction)   