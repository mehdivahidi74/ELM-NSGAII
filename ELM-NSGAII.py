import streamlit as st 
import os
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from termcolor import colored
from colorama import Fore
## Get ELM-NSGAII Weights
EXCt = pd.read_excel('Weight-Ct-EX.xlsx')
EXCr = pd.read_excel('Weight-Cr-EX.xlsx')
st.subheader("Extreme Learning Machine Optimized Based On NSGA-II ")
Breakwater = Image.open('Breakwater.PNG')
st.image(Breakwater)
# """RealCt = pd.read_excel('Weight-Ct-Real.xlsx')
# RealCr = pd.read_excel('Weight-Cr-Real.xlsx') """
P = 5
M = 20
Model = st.selectbox('Select Output' , ['Transmission Coefficient' , 'Reflection Coefficient'])
if Model == 'Transmission Coefficient':
    def user_inputs_EX():
        st.subheader('Select Variable (cm)')
        B = st.sidebar.slider('Chamber Width (B)',14 , 224 , 40)
        IH = st.sidebar.slider('Impermeable Height (IH)',4 , 16 , 12)
        H = st.sidebar.slider('Incident Wave Height (H)',4 ,12 , 8)
        L = st.sidebar.slider('Wavelength (L)',56.0 , 454.0 , 283.59 , step = 0.1)
        h = st.sidebar.slider('Water depth (h)',10 , 100 , 40 , step = 10)
        # def func(y , T):
        #     eq = [(981/(2*np.pi)) * T**2 * np.tanh((2*np.pi/(y[0])) * 40 )- y[0]]
        #     return eq
        # L = st.sidebar.slider('Wavelength (L)', (fsolve(func , [1] , T).tolist())[0])
        return B , IH , H , L ,h
    B , IH , H , L , h = user_inputs_EX()
    MaxEX = [('B / h' , 5.605) , ('IH / h' , 0.4) , ('H / L' , 0.071184) , ('H / h' , 0.3)]
    MinEX = [('B / h' , 0.35) , ('IH / h' , 0.1) , ('H / L' , 0.022066) , ('H / h' , 0.1) ]
    st.subheader('Non-dimensional Inputs Variables')
    st.write('B / h: ' ,"{:.2f}".format(B/h) )
    st.write('IH / h: ' ,"{:.2f}".format(IH/h) )
    st.write('H / L: ' ,"{:.2f}".format(H/L) )
    st.write('H / h: ' ,"{:.2f}".format(H/h) )
   
    X = np.array([[B/h,IH/h,H/L,H/h]])
    for i in range (np.shape(X)[0]):
        for j in range (np.shape(X)[1]):
            X[i][j] = (X[i][j] - MinEX[j][1])/ (MaxEX[j][1] - MinEX[j][1])
    #Evaluate Ct in Experimental Scale
    WEXCt = EXCt.iloc[0:20 , 0:5]
    WEXCt = np.transpose(WEXCt)
    WEXCt = np.array(WEXCt)
    BetaEXCt = EXCt.iloc[0:21 , 5:6]
    BetaEXCt = np.array(BetaEXCt)
    one = np.ones((np.shape(X)[0] , 1))*-1
    X = np.append(one,X , axis = 1)
    H = np.matmul(X , WEXCt)
    H = np.tanh(H)
    one = np.ones((np.shape(X)[0] , 1))*-1
    H = np.append(one,H , axis = 1)
    PredCt = np.matmul(H , BetaEXCt)
    st.subheader('Transmission Coefficient Prediction (Ct)')
    st.subheader("{:.2f}".format(PredCt.tolist()[0][0])
    #Evaluate Cr in Experimental Scale
# """     WEXCr = EXCr.iloc[0:20 , 0:6]
#     WEXCr = np.transpose(WEXCr)
#     WEXCr = np.array(WEXCr)
#     BetaEXCr = EXCr.iloc[0:21 , 6:7]
#     BetaEXCr = np.array(BetaEXCr)
#     H = np.matmul(X , WEXCr)
#     H = np.tanh(H)
#     one = np.ones((np.shape(X)[0] , 1))*-1
#     H = np.append(one,H , axis = 1)
#     PredCr = np.matmul(H , BetaEXCr)
#     st.subheader('Reflection Coefficient Prediction (Cr)')
#     st.write(PredCr) """
if Model == 'Reflection Coefficient':
    def user_inputs_EX():
        st.subheader('Select Variable (cm)')
        B = st.sidebar.slider('Chamber Width (B)',14 , 224 , 40)
        IH = st.sidebar.slider('Impermeable Height (IH)',4 , 16 , 12)
        H = st.sidebar.slider('Incident Wave Height (H)',4 ,12 , 8)
        L = st.sidebar.slider('Wavelength (L)',56.0 , 454.0 , 283.59 , step = 0.1)
        h = st.sidebar.slider('Water depth (h)',10 , 100 , 40 , step = 10)
        # def func(y , T):
        #     eq = [(981/(2*np.pi)) * T**2 * np.tanh((2*np.pi/(y[0])) * 40 )- y[0]]
        #     return eq
        # L = st.sidebar.slider('Wavelength (L)', (fsolve(func , [1] , T).tolist())[0])
        return B , IH , H , L , h
    B , IH , H , L , h = user_inputs_EX()
    MaxEX = [('B / h' , 5.605) , ('IH / h' , 0.4) , ('H / L' , 0.071184) , ('H / h' , 0.3)]
    MinEX = [('B / h' , 0.35) , ('IH / h' , 0.1) , ('H / L' , 0.022066) , ('H / h' , 0.1) ]
    st.subheader('Non-dimensional Inputs Variables')
    st.write('B / h: ' ,"{:.2f}".format(B/h) )
    st.write('IH / h: ' ,"{:.2f}".format(IH/h) )
    st.write('H / L: ' ,"{:.2f}".format(H/L) )
    st.write('H / h: ' ,"{:.2f}".format(H/h) )
    X = np.array([[B/h,IH/h,H/L,H/h]])
    for i in range (np.shape(X)[0]):
        for j in range (np.shape(X)[1]):
            X[i][j] = (X[i][j] - MinEX[j][1])/ (MaxEX[j][1] - MinEX[j][1])
    #Evaluate Ct in Experimental Scale
    one = np.ones((np.shape(X)[0] , 1))*-1
    X = np.append(one,X , axis = 1)
    WEXCr = EXCr.iloc[0:20 , 0:5]
    WEXCr = np.transpose(WEXCr)
    WEXCr = np.array(WEXCr)
    BetaEXCr = EXCr.iloc[0:21 , 5:6]
    BetaEXCr = np.array(BetaEXCr)
    H = np.matmul(X , WEXCr)
    H = np.tanh(H)
    one = np.ones((np.shape(X)[0] , 1))*-1
    H = np.append(one,H , axis = 1)
    PredCr = np.matmul(H , BetaEXCr)
    st.subheader('Reflection Coefficient Prediction (Cr)')
    st.subheader("{:.2f}".format(PredCr.tolist()[0][0])
# """     def user_inputs_Real():
#         B = st.sidebar.slider('Chamber Width (B)',3.5 , 56.0 , 10.0 , step = 0.1)
#         IH = st.sidebar.slider('Impermeable Height (IH)',1 , 4 , 3)
#         H = st.sidebar.slider('Incident Wave Height (H)',1.0 , 3.0 , 1.3 , step = 0.1)
#         L = st.sidebar.slider('Wavelength (L)',6.0 , 113.2 ,30.5 , step = 0.1)
#         T = st.sidebar.slider('Wave Period (T)',12.0 , 3.0 , 4.5 , step = 0.1)
        
#         return B , IH , H , L , T
#     B , IH , H , L , T = user_inputs_Real()
#     MaxReal = [('B' , 56) , ('IH' , 4) , ('H' , 3) , ('L' , 113.25) , ('T' , 12)]
#     MinReal = [('B' , 3.5) , ('IH' , 1) , ('H' , 1) , ('L' , 14.047) , ('T' , 3)]
#     X = np.array([[B,IH,H,L,T]])
#     for i in range (np.shape(X)[0]):
#         for j in range (np.shape(X)[1]):
#             X[i][j] = (X[i][j] - MinReal[j][1])/ (MaxReal[j][1] - MinReal[j][1])

#     #Evaluate Ct in Real Scale'
#     one = np.ones((np.shape(X)[0] , 1))*-1
#     X = np.append(one,X , axis = 1)
#     WRealCt = RealCt.iloc[0:20 , 0:6]
#     WRealCt = np.transpose(WRealCt)
#     WRealCt = np.array(WRealCt)
#     BetaRealCt = RealCt.iloc[0:21 , 6:7]
#     BetaRealCt = np.array(BetaRealCt)
#     H = np.matmul(X , WRealCt)
#     H = np.tanh(H)
#     one = np.ones((np.shape(X)[0] , 1))*-1
#     H = np.append(one,H , axis = 1)
#     PredCt = np.matmul(H , BetaRealCt)
#     st.subheader('Transmission Coefficient Prediction (Ct)')
#     st.write(PredCt)
#     #Evaluate Cr in Real Scale
#     WRealCr = RealCr.iloc[0:20 , 0:6]
#     WRealCr = np.transpose(WRealCr)
#     WRealCr = np.array(WRealCr)
#     BetaRealCr = RealCr.iloc[0:21 , 6:7]
#     BetaRealCr = np.array(BetaRealCr)
#     H = np.matmul(X , WRealCr)
#     H = np.tanh(H)
#     one = np.ones((np.shape(X)[0] , 1))*-1
#     H = np.append(one,H , axis = 1)
#     PredCr = np.matmul(H , BetaRealCr)
#     st.subheader('Reflection Coefficient Prediction (Cr)')
#     st.write(PredCr) """
