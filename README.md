<big>A computational model of the mouse primary visual area (V1) with 4 types of Generalised Leaky Integrate and Fire neurons.</big>

Recreating the mouse primary visual area (V1) using biophysically detailed neuron models. 

The glif neuron parameters can be found in glif.py as follows: 



```
'''
VARIABLE DESCRITPIONS:

V(t) = Membrane Potential
I_j(t) =   After-spike currents
Theta_s(t) = Spike-dependent threshold component
Theta_v(t) = Voltage-dependent threshold component

PARAMTER DESCRIPTIONS:                                                  FIT FROM:  

C = Capacitance                                                 <-      Sub-threshold nosie
R = Membrane Resistance                                         <-      Sub-threshold nosie
E_L = Resting Potential                                         <-      Resting V before noise
Theta_inf = Instantaneous Threshold                             <-      Short Square input
delta t = Spike cut length                                      <-      All noise spikes
f_v = Voltage Fraction following spike                          <-      All noise spike
delta V = Voltage addition following spike                      <-      All noise spike
b_s = Spike-induced threshold time constant                     <-      Triple short square
delta Theta_s = Threshold addition following spike              <-      Triple short square
delta I_j = After-spike current amplitudes                      <-      Supra-threshold noise
k_j = After-spike current time constants                        <-      Supra-threshold noise
f_v = Current fraction following spike                          <-      Set to 1
a_v = Adaptation index of threshold                             <-      Prespike V supra-thr. noise
b_v = Voltage-induced threshold time constant                   <-      Prespike V supra-thr. noise
'''

```
