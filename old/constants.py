#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from trm import dnl, subs, sla
from scipy.optimize import curve_fit
from sys import argv, exit
import mgutils as mg, stars, runs
import subprocess, shlex


""" A load of constants
"""
#obvious ones
c = 299792458.
G = 6.67408e-11
h = 6.626e-34   #planck
planck = h
H0 = 2.25e-18   # Hubble (in s^-1)
hubble = H0
H0Mpc = 67.6 # Hubble constant in km/s/Mpc
#subatomic
e = 1.602e-19
eV = e
me = 9.109e-31
mp = 1.673e-27
#thermodynamics
kB = 1.3806e-23     #boltzmann
stefanBoltzmann = 5.67e-8
#standard conditions
g = 9.80665
atm = 101325


#time
secondsPerDay = 86400.
secondsPerYear = 86400. * 365.25
minutesPerDay = 86400./60
tH = 4.55e17    #Hubble time in seconds


#distance
ly = 9.461e15
pc = 3.086e16
au = 1.496e11

#energy
mJy = 1e-29

##properties of objects
#mass
mEarth = 5.972e24
mJup = 1.898e27
mNep = 1.024e26
mSun = 1.98855e30
mSol = mSun
msol = mSun
msun = mSun
#radius
rEarth = 6.371e6
rSun = 695.7e6
rSol = rSun
rsol = rSun
rsun = rSun
#others
lSun = 3.828e26
lSol = lSun


