#!/usr/bin/env python

import numpy as np
import mgutils as mg

class Star:
	def __init__(self, name):
		self.name = name
	
	def coloursFromSDSS(self,string):
		split = string.split()
		
		self.r = float(split[0])
		self.g = float(split[2]) + self.r
		self.u = float(split[1]) + self.g
		self.i = self.r - float(split[3])
		self.z = self.i - float(split[4])
		self.rerr = float(split[5])
		self.gerr = np.sqrt( float(split[7])**2 + self.rerr**2)
		self.uerr = np.sqrt( float(split[6])**2 + self.gerr**2)
		self.ierr = np.sqrt( float(split[8])**2 + self.rerr**2)
		self.zerr = np.sqrt( float(split[9])**2 + self.ierr**2)
	
	def magsFromSDSSNavi(self, string):
		split = string.split()
		
		self.u = float(split[0])
		self.g = float(split[1])
		self.r = float(split[2])
		self.i = float(split[3])
		self.uerr = float(split[4])
		self.gerr = float(split[5])
		self.rerr = float(split[6])
		self.ierr = float(split[7])
	
	
	def mag(self,band):
		if band == 'u' or band == 'b':
			return self.u
		if band == 'g':
			return self.g
		if band == 'r':
			return self.r
		if band == 'i':
			return self.i
		if band == 'z':
			return self.z
	
	def magerr(self,band):
		if band == 'u' or band == 'b':
			return self.uerr
		if band == 'g':
			return self.gerr
		if band == 'r':
			return self.rerr
		if band == 'i':
			return self.ierr
		if band == 'z':
			return self.zerr




gaia = Star("Gaia14aae")
gaia.coords = '16:11:33.97 +63:08:31.8'

feige34 = Star("Feige 34")
feige34.coloursFromSDSS("11.423     -0.509     -0.508     -0.347     -0.265         0.002      0.004      0.003      0.002      0.003")
feige34.coords = "10:39:36.73      +43:06:09.2"

bd35 = Star("BD+35 3659")
bd35.coloursFromSDSS("10.089      0.925      0.333      0.117      0.022         0.001      0.002      0.001      0.001      0.001")
bd35.coords = "19:31:09.22      +36:09:10.1"

feige22 = Star("Feige 22")
feige22.coloursFromSDSS("13.024      0.050     -0.333     -0.303     -0.273         0.001      0.004      0.002      0.002      0.004")
feige22.coords = "02:30:16.62      +05:15:50.6"



comp2 = Star("Comp2")
comp2.coords = gaia.coords
comp2.magsFromSDSSNavi("17.89 	16.19 	15.62 	15.46	0.01 	0.00 	0.00 	0.00")

comp3 = Star("Comp3")
comp3.coords = gaia.coords
comp3.magsFromSDSSNavi("17.52 	15.45 	14.62 	14.34	0.01 	0.00 	0.00 	0.00")

comp5 = Star("Comp5")
comp5.coords = gaia.coords
comp5.magsFromSDSSNavi("16.18 	14.32 	13.65 	13.66	0.01 	0.00 	0.01 	0.00")

comp7 = Star("Comp7")
comp7.coords = gaia.coords
comp7.magsFromSDSSNavi("17.74 	16.21 	15.61 	15.38	0.01 	0.00 	0.00 	0.00")













