# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:10:24 2017

@author: Mark
"""
import numpy as np
import matplotlib.pyplot as plt

#Define functions
def model(state_0,A,B):
    state_1 = A*state_0 +  np.random.normal(0,B)
    return state_1

state_null=np.random.normal(0,0.4)

def observation_function(state,R):
    obs=state+np.random.normal(0,R)
    return obs

def forecast(state_0,cov_0,A,B):
    state_1=A*state_0
    cov_1=A*cov_0*A+B
    return state_1,cov_1

def analysis_formulas(state_1_hat,cov_1_hat,K,H,obs_0):
    state_1 = state_1_hat - K*(H*state_1_hat - obs_0)
    cov_1 = cov_1_hat - K*H*cov_1_hat
    return state_1, cov_1

def kalman_gain(cov_1_hat,H,R):
    K = cov_1_hat*H*(R+H*cov_1_hat*H)**(-1)
    return K

#Initialize model parameters
A = 0.5
H = 1
B = 0.5
R = 0.1
lev = 100 

#Sythetic Model
STATE_real = np.zeros(lev)
OBS_real = np.zeros(lev)
STATE_real[0] = np.random.normal(5,0.1)
OBS_real[0] = observation_function(STATE_real[0],R)
for i in range (1,lev-1):
    STATE_real[i] = model(STATE_real[i-1],0.4,0.01)
    OBS_real[i] =  observation_function(STATE_real[i],R)


#Kalman-filter
STATE = np.zeros(lev)
COV = np.zeros(lev)
STATE[0] = state_null
COV[0] = B

for i in range (1,lev-1):
    (state_hat,cov_hat) = forecast(STATE[i-1],COV[i-1],A,B)
    K = kalman_gain(cov_hat,H,R)
    (STATE[i],COV[i]) = analysis_formulas(state_hat,cov_hat,K,H,OBS_real[i])

plt.plot(STATE)
plt.plot(STATE_real)
