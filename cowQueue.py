#!/usr/bin/env python
	
import numpy as np
import matplotlib.pyplot as plt
from astropy import time
from astropy.io import fits
from trm import dnl, subs, sla
from scipy.optimize import curve_fit
from sys import argv, exit
import subprocess, shlex, os
import mgutils as mg, stars, runs, constants as co


""" Various things for interacting with the cow queue
"""

def readCowQueue():

	username = "phulbz"
	tempname = "/home/astro/phulbz/cow.tmp"
	
	p = subprocess.Popen(shlex.split("qstat -u %s"%(username)), stdout=subprocess.PIPE)
	
	stdout = p.communicate()[0]
	
	if stdout == '':
		return None, None, None, None
	
	with open(tempname, 'w') as f:
		f.write(stdout)
	
	idstr, script, running, uptime = np.loadtxt(tempname, unpack=True, skiprows=5, dtype=str, usecols=(0,3,9,10))
	
	os.remove(tempname)
	
	idno = []
	for string in idstr:
		num = string.split('.')[0]
		idno += [num]
		
	return np.array(idno), script, running, uptime

def submitToQueue(script):
	print "submitting", script
	p = subprocess.Popen(shlex.split("qsub %s"%(script)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdout, stderr = p.communicate()
	if stderr:
		raise SystemError(stderr)
	return stdout, stderr

def killFromQueue(idno):
	print "Killing %s"%(idno)
	p = subprocess.Popen(shlex.split("qdel %s"%(idno)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdout, stderr = p.communicate()
	if stderr:
		raise SystemError(stderr)
	return stdout, stderr

def addToFrontQueue(script):
	idno, scripts, running, uptime = readCowQueue()
	stdout0, stderr0 = submitToQueue(script)
	
	if scripts is not None:
		for j,scr in enumerate(scripts):
			if running[j] == 'Q':
				stdout, stderr = submitToQueue(scr)
				stdout, stderr = killFromQueue(idno[j])
	return stdout0, stderr0





if __name__ == "__main__":
	idno, scripts, running, uptime = readCowQueue()
	for num in idno:
		if float(num) >= 1619914:
			killFromQueue(num)
	
