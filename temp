#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from trm import dnl, subs, sla
from scipy.optimize import curve_fit
from sys import argv, exit
import stars, runs

coords = '16:11:33.97 +63:08:31.8'
obsname = 'WHT'						# This is the setup for the object I'm currently working on
period = 0.034519576212191858
perr = 8.413697e-07
t0 = 57166.116142215891
t0err = 1.2137219e-06

detrendFlag = True		# True if need to detrend - dependant on data set


bandDict = {'u':3, 'b':3, 'g':2, 'r':1, 'i':1, 'z':1}




##############  Shapes and functions for fitting ##########


def flat(x,c):
	return np.zeros(len(x)) + c

def line(x,a,c):
	return a*(x) + c

def x2(x,a,b,d,c):
	return a*(d*(x-b))**2 + c

def x3(x,a,b,d,c):
	return a*(d*(x-b))**3 + c

def x3plusx2(x,a,b,d,e,f,c):
	return a*(d*(x-b))**3 + e*(f*(x-b))**2 + c

def x2plusx(x,a,b,d,e,c):
	return a*(d*(x-b))**2 + e*(x-b) + c

def sinx(x,a,b,d,c):
	return a*np.sin(d*(x-b)) + c	
	
def gauss(x, a, m, s):
	return a*np.exp(- (x-m)**2 / (2 * s**2))


def circle(r, n=200):
	""" Returns coordinates of a circle of radius r
	"""
	angle = np.linspace(0,2*np.pi,n)
	
	x = r * np.sin(angle)
	y = r * np.cos(angle)
	
	return x, y



##########  Stats and manipulation  ################


def weightedAv(x, err):
	av = np.average(x, weights=1/err**2)
	averr = np.sqrt(1 / (np.sum(1/err**2)))		# Equation from Tom H's book p50
	return av, averr

def chisq(model, data, err):
	return np.sum( (data-model)**2 / err**2 )


def iterativeFit(funct, x, y, yerr, numMask, maskLim, p0=None, inmask=None):
	""" Fits iteratively over a data set, each iteration masking one data point and refitting.
		Takes function and associated x and y values, maximum number of points to mask, and sigma limit
		below which it will not mask (masks until either of those criteria is met). 
		Returns final fit parameters and the final mask used.
	"""
	if inmask is None:
		mask = np.ones(len(x),dtype=bool)
	else:
		mask = inmask
	assert len(mask) == len(x)
	assert len(mask) == len(y)
	
	x = np.array(x)	# in case passed in as non-numpy arrays
	y = np.array(y)
	
	popt, pcov = curve_fit(funct, x[mask], y[mask], p0=p0)
	i=0
	while (len(mask[mask==False]) < numMask):
		popt, pcov = curve_fit(funct, x[mask], y[mask], p0=p0)
		fit = funct(x,*popt)
		diffs = np.abs((y - fit) / yerr)
		if diffs[mask].max() > maskLim:
			mask[diffs == diffs[mask].max()] = False
			i+=1
			#print "masked",i,"sig",diffs[mask].max()##
		else:
			break
	return popt, pcov, mask


def overlap(bx, rx):
	""" Returns masks for which of array 1 are in array 2 and v.v.
	"""
	okb = np.zeros(len(bx),dtype=bool)			# all this to mask points where magnitude data not available for both
	for i in xrange(len(bx)):
		if bx[i] in rx:
			okb[i] = True
	okr = np.zeros(len(rx),dtype=bool)
	for i in xrange(len(rx)):
		if rx[i] in bx:
			okr[i] = True
	return okb, okr


############  Science things  		###########


def blackbody(temp, freq='default', wave='default', units='SI', useFreq=True, useWave=False, n=1000):
	""" Plot the blackbody curve for a given temperature. useFreq determines whether x-axis will be frequency or wavelength.
	"""
	if useWave == True:
		useFreq = False	# obviously can't use both. Will only check usefreq from this point on.

	## Constants
	c = 299792458 
	kB = 1.38064852e-23
	h = 6.626070040e-34

	## Handle units, work out factor to convert to SI
	if units in ['SI']:
		fac = 1
	elif units in ['Hz', 'hz']:
		fac = 1
		useFreq = True
	elif units in ['m']:
		fac = 1
		useFreq = False
	elif units in ['GHz', 'ghz']:
		fac = 1e9
		useFreq = True
	elif units in ['nm']:
		fac = 1e-9
		useFreq = False
	elif units in ['A', 'Angstroms']:
		fac = 1e-10
		useFreq = False
	else:
		print "!! Units not recognised !!"
		return False

	## Generate default x-axis values if desired, else convert supplied x-values to SI units
	if useFreq and freq == 'default':
		freq = np.logspace(14,16)
	if not useFreq and wave == 'default':
		wave = np.logspace(-8,-6)
	else:
		freq = freq * fac
		wave = wave * fac

	## Do the actual calculation
	if useFreq:
		bspec = (2 * h * freq**3 / c**2) * (1 / (np.exp(h * freq / kB / temp) - 1))
		return freq / fac, bspec / bspec.max()	#convert x-axis back

	else:
		bspec = (2 * h * c**2 / wave**5) * (1 / (np.exp(h * c / wave / kB / temp) -1))
		return wave / fac, bspec / bspec.max()





#########  Graph Plotting 		###################################

def formatGraph(suptitle=None,xlabel=None,ylabel=None):
	""" General formatting things like style sheet, axes etc. Title and axis labels are applied to the current figure and axes.
	"""
	plt.style.use('mgplot')
	if xlabel is not None:
		plt.gca().set_xlabel(xlabel)
	if ylabel is not None:
		plt.gca().set_ylabel(ylabel)
	if suptitle is not None:
		plt.suptitle(suptitle)
	return True

def formatSubplots(subshape, xlabel=None, ylabel=None, sharex=False, sharey=False, figsize=(11,8), **kwargs):
	""" Create a figure containing a set of subplots. Unrecognised keywords are passed into plt.subplots, thence into plt.figure
	"""
	formatGraph()
	fig, axs = plt.subplots(subshape[0], subshape[1], sharex=sharex, sharey=sharey, figsize=figsize, **kwargs)
	if xlabel is not None:
		for ax in axs:
			ax.set_xlabel(xlabel)
	if ylabel is not None:
		for ax in axs:
			ax.set_ylabel(ylabel)
	return fig, axs

def escapeForLatex(string):
	""" Escapes characters in a string which LaTeX will otherwise interpret as special characters, such as underscores.
	    This prevents the error messages that the pyplot latex format otherwise gives.
	"""
	b = string.replace('_','\_')
	b = b.replace('^','\^')

	return b

def adjLimits(x,margins=0.1):
	""" For a given array and fraction, returns min minus fraction of range and max + fraction of range. 
	    Intended for setting display limits that are data dependant.
	"""
	rangex = x.max() - x.min()
	xlow = x.min() - rangex * margins
	xhigh = x.max() + rangex * margins
	return xlow, xhigh

def lightcurve(time, y, yerr=None, fmt='g.', fignum=1,title=None,	\
				xax=None, phase=False, btdb=False, yax=None, mj=False, mags=False, colour=False, \
				xtoggle = True, ytoggle = True, \
				xleft=None, xright=None, yhigh=None, ylow=None, round=1, \
				sub=None):
	""" Plots a lightcurve and labels the axes accordingly.
		'colour' can be a string describing which bands used, or simply True for multiple colours
	"""
	formatGraph()
	
	plt.figure(fignum)
		
	if btdb:
		toff = np.around(time.min(),round)
		time = time - toff			#tidying up time display
	
	if sub is None:
		plt.errorbar(time, y, yerr, fmt=fmt, capsize=0)
	else:
		sub.errorbar(time, y, yerr, fmt=fmt, capsize=0)
	if title is not None:
		plt.suptitle(title)
	if xax is not None:
		plt.xlabel(xax)
	elif phase:
		plt.xlabel("Phase")
	elif btdb:
		plt.xlabel("BMJD (TDB) - %g"%(toff))
	else:
		plt.xlabel("MJD")
	if yax is not None:
		plt.ylabel(yax)
	elif mj:
		plt.ylabel("Flux density (mJy)")
	elif mags:
		plt.ylabel("Magnitude")
		plt.gca().invert_yaxis()
	elif colour == True:
		plt.ylabel("Colour")
		plt.gca().invert_yaxis()
	elif colour:
		plt.ylabel("%s colour"%(colour))
		plt.gca().invert_yaxis()
		
	if xleft is not None:
		plt.xlim(xmin=xleft)
	if xright is not None:
		plt.xlim(xmax=xright)
	if ylow is not None:
		plt.ylim(ymin=ylow)
	if yhigh is not None:
		plt.ylim(ymax=yhigh)
		
	if xtoggle is False:
		plt.xlabel('')
		if sub is not None:
			sub.set_xticklabels([])
	if ytoggle is False:
		plt.ylabel('')
		if sub is not None:
			sub.set_yticklabels([])
			
	return plt.figure(fignum)
	
def saveAsTemp(fignum=None,ax=None,fig=None):
	""" Saves as a temporary file (temp.pdf and temp.png) in my home directory
	"""
	if ax is not None:
		plt.sca(ax)
	elif fig is not None:
		plt.figure(fig.number)
	elif fignum is not None:
		plt.figure(fignum)
	plt.savefig("/home/astro/phulbz/temp.pdf",dpi='figure',bbox_inches='tight')
	plt.savefig("/home/astro/phulbz/temp.png",dpi='figure',bbox_inches='tight')
	return True




######## Phase Folding 		############################


def phaseFold(btdb, terr, data, t0, t0err, period, perr, bins, errors=None):
	""" Phase folds and averages the data, outputting phase, 'continuous phase', and means, errors, and counts for each bin 
		"Continuous phase" is simply how many periods you are away from t0.
	"""
	phaseCont = ((btdb-t0) / period)
	phase = phaseCont % 1
	phase[phase>0.5] = phase[phase>0.5] - 1	
	if terr is not None:
		pherr = np.sqrt( (np.sqrt(t0err**2 + terr**2) / (btdb-t0))**2 + (perr/period)**2 ) * phase
	else:
		pherr = np.sqrt( (np.sqrt(t0err**2) / (btdb-t0))**2 + (perr/period)**2 ) * phase
	mids = (bins[1:] + bins[:-1]) / 2	#midpoint of each bin
	numBins = len(bins) - 1
	means = np.zeros(numBins)
	errs = np.zeros(numBins)
	counts = np.zeros(numBins)
	miderrs = np.zeros(numBins)
	
	for i in xrange(numBins):
		ran = (phase>bins[i]) & (phase<bins[i+1])
		if len(data[ran]) == 0:
			continue
		means[i] = np.average(data[ran], weights=(1/errors[ran]**2))
		errs[i] = np.sqrt(1 / (np.sum(1/errors[ran]**2)))		# Equation from Tom H's book p50
		counts[i] = len(data[ran])
		miderrs[i] = np.average(pherr[ran])			#don't know the proper way to work this out, this is just an estimate
	
	return phase, phaseCont, pherr, mids, miderrs, means, errs, counts

def phaseFromTimes(btdb, terr, t0, t0err, period, perr):
	""" Phase folds times only.
	"""
	phaseCont = ((btdb-t0) / period)
	phase = phaseCont % 1
	phase[phase>0.5] = phase[phase>0.5] - 1	
	if terr is not None:
		pherr = np.sqrt( (np.sqrt(t0err**2 + terr**2) / (btdb-t0))**2 + (perr/period)**2 ) * phase
	else:
		pherr = np.sqrt( (np.sqrt(t0err**2) / (btdb-t0))**2 + (perr/period)**2 ) * phase

	return phase, phaseCont, pherr
		

######### Magnitudes and colours 		#########################



	
def airmass(utc, coords, obsname, wave=0.55):
	""" Returns an array of airmasses corresponding to input utc times.
		Input utc, coords of star, and name of observatory (latter both as strings).
		Optionally include wavelength, in microns.
	"""
	tel,obs,longitude,latitude,height = \
			subs.observatory(obsname)
	
	ra,dec,syst = subs.str2radec(coords)
	
	airmass, alt, az, ha, pa, delz = \
			sla.amass(utc,longitude,latitude,height,ra,dec,wave)
	
	return airmass


def extinction(l, coords, obsname, show=False, fignum=False, mask=None):
	""" Finds and returns the extinction coefficient of an object over one dataset. 
		Takes the object data, l, as a ulog-type object.
		Returns extinction coefficient and its error as taken from the covarience matrix
	"""
	ok = (l.y.data != 0)
	amass = airmass(l.x.data[ok], coords, obsname)
	amerr = airmass(l.x.data[ok]+l.x.errors[ok], coords, obsname) - amass
	
	fakemags = -2.5 * np.log10(l.y.data[ok])			#magnitude + offset
	fmerr = np.abs(-2.5 * np.log10(l.y.data[ok]+l.y.errors[ok]) - fakemags)
	
	numMask = len(amass)/100
	maskLim = 3
	if mask is not None:
		assert len(mask) == len(amass)	#check starting mask fits data
	popt, pcov, mask = iterativeFit(line, amass, fakemags, fmerr, numMask, maskLim, inmask=mask)
	
	ext = popt[0]
	exterr = np.sqrt(pcov[0,0])
	
	if fignum or (fignum==0):
		plt.figure(fignum)
	if fignum or (fignum==0) or show:
		plt.errorbar(amass, fakemags, fmerr, amerr, fmt='b.')
		plt.plot(amass, line(amass, *popt), 'k')
		plt.plot(amass[mask==False], fakemags[mask==False], 'rD')
		plt.figtext(0.2,0.8, "Extinction = %f +/- %f"%(ext, exterr))
	else:
		plt.close()
	if show:
		plt.show()
	
	return ext, exterr

def magsFromComp(l, sl, star, std, run, band):
	""" Function to calculate magnitudes from ultracam counts based on a comparison star, in the same field, with known magnitude.
	"""
	ok = (l.y.data != 0) & (sl.y.data != 0)
	
	amass = airmass(l.x.data[ok], star.coords, run.obsname)
	amerr = airmass(l.x.data[ok]+l.x.errors[ok], star.coords, run.obsname) - amass
	samass = airmass(sl.x.data, std.coords, run.obsname)
	samerr = airmass(sl.x.data+sl.x.errors, std.coords, run.obsname) - samass
	
	smag = std.mag(band)
	smerr = std.magerr(band)
	
	
	mag = smag - 2.5*np.log10((l.y.data[ok]) / (sl.y.data[ok]))
	
	
	a1 = (smag+smerr - 2.5*np.log10((l.y.data[ok]) / (sl.y.data[ok])) ) - mag
	a2 = smag - 2.5*np.log10(((l.y.data[ok]+l.y.errors[ok])) / (sl.y.data[ok]))  - mag
	a3 = smag - 2.5*np.log10((l.y.data[ok]) / ((sl.y.data[ok]+sl.y.errors[ok])))  - mag
	
	magerr = np.sqrt(a1**2 + a2**2 + a3**2)
	
	
	return l.x.data[ok], l.x.errors[ok], mag, magerr
	



def magnitudes(l, sl, star, std, run, srun, band):
	""" Converts ultracam counts to magnitudes, based on a single standard.
		Returns the magnitudes and their errors
		Takes ulg of star, aperture number of star, ulg of std, star obj, std obj, run obj, std run obj, band - one of 'ugriz'
	"""
	
	ok = (l.y.data != 0)
	#print l.y.data[ok]
	
	amass = airmass(l.x.data[ok], star.coords, run.obsname)
	amerr = airmass(l.x.data[ok]+l.x.errors[ok], star.coords, run.obsname) - amass
	samass = airmass(sl.x.data, std.coords, srun.obsname)
	samerr = airmass(sl.x.data+sl.x.errors, std.coords, srun.obsname) - samass
	
	smag = std.mag(band)
	smerr = std.magerr(band)
	
	popt, pcov, mask = iterativeFit(flat, sl.x.data, sl.y.data, sl.y.errors, numMask=1, maskLim=3)
	
	sy = popt[0]#np.average(sl.y.data, weights=1/sl.y.errors**2)
	syerr = np.sqrt(1 / (np.sum(1/sl.y.errors[mask]**2)))		# Equation from Tom H's book p50
	sa = np.average(samass[mask], weights=1/samerr[mask]**2)
	saerr = np.sqrt(1 / (np.sum(1/samerr[mask]**2)))
	
	
	
	mag = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*run.ext(band) + sa*srun.ext(band)
	
	
	a1 = (smag+smerr - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*run.ext(band) + sa*srun.ext(band)) - mag
	a2 = smag - 2.5*np.log10(((l.y.data[ok]+l.y.errors[ok])) / (sy)) - amass*run.ext(band) + sa*srun.ext(band) - mag
	a3 = smag - 2.5*np.log10((l.y.data[ok]) / ((sy+syerr))) - amass*run.ext(band) + sa*srun.ext(band) - mag
	a4 = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - (amass+amerr)*run.ext(band) + sa*srun.ext(band) - mag
	a5 = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*(run.ext(band)+run.exterr(band)) + sa*srun.ext(band) - mag
	a6 = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*run.ext(band) + (sa+saerr)*srun.ext(band) - mag
	a7 = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*run.ext(band) + sa*(srun.ext(band)+srun.exterr(band)) - mag
	
	magerr = np.sqrt(a1**2 + a2**2 + a3**2 + a4**2 + a5**2 + a6**2 + a7**2)
	
	#print "errors:\n", magerr, '\n', a1, '\n', a2, '\n', a3, '\n', a4, '\n', a5, '\n', a6, '\n', a7			#useful for finding dominant errors
	
	return l.x.data[ok], l.x.errors[ok], mag, magerr
	
	
def magnitudes3(r, g, b, sr, sg, sb, star, std, run, srun):
	""" Quite simply returns magnitudes in 3 bands at once
	"""
	rx, rxerr, rm, rerr = magnitudes(r, sr, star, std, run, srun, 'r')
	gx, gxerr, gm, gerr = magnitudes(g, sg, star, std, run, srun, 'g')
	bx, bxerr, bm, berr = magnitudes(b, sb, star, std, run, srun, 'b')
	return rx, gx, bx, rxerr, gxerr, bxerr, rm, gm, bm, rerr, gerr, berr



def colour(bx, b, berr, rx, r, rerr, blue=False):
	""" Finds colours over time from an array of magnitudes.
	Input: bluer band first
	NB the x-arrays given here must be of same length as the y-arrays 
	-- ie if one has had zeros removed the other must have had corresponding elements removed too
	"""
	assert len(b) == len(bx) and len(r) == len(rx)
	
	#if blue:
		#r2 = np.zeros(np.floor(len(r)/3)*3)
		#m = np.ones(len(r), dtype=bool)
		#if (len(r) % 3) == 1:
			#m[-1] = False
		#if (len(r) % 3) == 2:
			#m[-1] = False
			#m[-2] = False
		##print len(r2), len(r), len(r2[::3]),len(r[m][::3])
		#r2[::3] = r[m][::3] + r[m][1::3] + r[m][2::3]
		#r2[1::3] = r[m][::3] + r[m][1::3] + r[m][2::3]
		#r2[2::3] = r[m][::3] + r[m][1::3] + r[m][2::3]
		
	#r = r*3
	
	nzb = b != 0
	nzr = r != 0
	okb = np.zeros(len(bx[nzb]),dtype=bool)			# all this to mask points where magnitude data not available for both
	for i in xrange(len(bx[nzb])):
		if bx[nzb][i] in rx[nzr]:
			okb[i] = True
	okr = np.zeros(len(rx[nzr]),dtype=bool)
	for i in xrange(len(rx[nzr])):
		if rx[nzr][i] in bx[nzb]:
			okr[i] = True
	
	col = b[okb] - r[okr]
	
	cerr = np.sqrt(berr[nzb][okb]**2 + rerr[nzr][okr]**2)
	
	return bx[okb], col, cerr


def mJyFromMags(mags, magerr):
	""" Converts AB magnitudes to milliJanskys, finds errors by funct approach
	"""
	mj = 10 ** ((23.9-mags)/2.5) / 1000			#from Wikipedia
	mjerr = np.abs(10 ** ((23.9-mags+magerr)/2.5) / 1000 - mj)
	return mj, mjerr


def mJyFromCounts(l, sl, star, std, run, srun, band):
	""" Converts counts to AB magnitudes and thence to milliJanskys. 
	"""
	x, xerr, m, merr = magnitudes(l, sl, star, std, run, srun, band)
	mj, mjerr = mJyFromMags(m, merr)
	return x, xerr, mj, mjerr

def mJyFromCountsWithComp(l, sl, star, std, run, band):
	""" Converts counts to AB magnitudes and thence to milliJanskys. Uses comparison with known magnitude.
	"""
	x, xerr, m, merr = magsFromComp(l, sl, star, std, run, band)
	mj, mjerr = mJyFromMags(m, merr)
	return x, xerr, mj, mjerr


def magsFromMJy(mj, mjerr):
	""" Converts milliJanskys to AB magnitudes
	"""
	mags = 23.9 - 2.5*np.log10(1000*mj)
	magerr = np.abs(23.9 - 2.5*np.log10(1000*(mj+mjerr)) - mags)
	return mags, magerr



###### Ultracam associated functions 		####################

## File and input handling 

def readulog(address,oi=1,comp=2):
	""" Reads a .log file, as created by the Ultracam pipeline, 
	divides by a comparison star, and returns r,g,b objects containing 
	times, counts and errors.
	Assumes that object of interest is aperture 1 and comparison is aperture 2
	Comparison can be set to None to avoid dividing
	Usage: readulog(address)"""
	ulg = dnl.ulog.Ulog(address)
	
	if comp is not None:				#deprecated -- better to use ulg2mjys or ulg2mjysWithComp for comparisonning
		"""Divide by comparison star"""
		rc = ulg.tseries(1,comp) 
		r = ulg.tseries(1,oi) / rc
		gc = ulg.tseries(2,comp)
		g = ulg.tseries(2,oi) / gc
		bc = ulg.tseries(3,comp)
		b = ulg.tseries(3,oi) / bc
		# the error propogation here is contained within tom's objects
		
		### Then multiply by average of comparison data to restore counts to right order
		r.y.data *= np.average(rc.y.data)
		r.y.errors *= np.average(rc.y.data)
		g.y.data *= np.average(gc.y.data)
		g.y.errors *= np.average(gc.y.data)
		b.y.data *= np.average(bc.y.data)
		b.y.errors *= np.average(bc.y.data)
	else:
		r = ulg.tseries(1,oi)			# the error propogation here is contained within tom's objects
		g = ulg.tseries(2,oi)
		b = ulg.tseries(3,oi)
	return r,g,b


def readdat(address):
	""" Reads a .dat file, ie data in the format useful for lcurve"""
	data = np.loadtxt(address)
	times = data[:,0]
	terr = data[:,1]
	y = data[:,2]
	yerr = data[:,3]
	
	return times, terr, y, yerr


def readalldat(addressg):
	""" Reads 3 lcurve-style .dat files at once, returning data for all 3 bands
	"""
	gt, gterr, g, gerr = readdat(addressg)
	rt, rterr, r, rerr = readdat(addressg[:-4]+"r.dat")
	bt, bterr, b, berr = readdat(addressg[:-4]+"b.dat")
	
	return rt, rterr, r, rerr, gt, gterr, g, gerr, bt, bterr, b, berr


def dat2tseries(addressg):
	""" Reads 3 lcurve-style .dat files at once, returns 3 Dseries objects
	"""
	r,g,b = readulog("may.log")	#as far as I can tell this is the only way to initialise a ulg object?
	g.x.data, g.x.errors, g.y.data, g.y.errors = readdat(addressg)
	r.x.data, r.x.errors, r.y.data, r.y.errors = readdat(addressg[:-4]+"r.dat")
	b.x.data, b.x.errors, b.y.data, b.y.errors = readdat(addressg[:-4]+"b.dat")
	return r, g, b


def ulg2mjys(address, star, std, run, srun, oi=1, comp=2):
	""" Reads a ulog file and finds magnitudes. 
	"""
	r,g,b = readulog(address, oi=oi, comp=None)
	cr, cg, cb = readulog(address, oi=comp, comp=None)
	sr, sg, sb = readulog(srun.locn+".log", comp=None)
	
	rx, rxerr, ry, ryerr = mJyFromCounts(r, sr, star, std, run, srun, band='r')
	gx, gxerr, gy, gyerr = mJyFromCounts(g, sg, star, std, run, srun, band='g')
	bx, bxerr, by, byerr = mJyFromCounts(b, sb, star, std, run, srun, band='b')
	
	crx, crxerr, cry, cryerr = mJyFromCounts(cr, sr, star, std, run, srun, band='r')
	cgx, cgxerr, cgy, cgyerr = mJyFromCounts(cg, sg, star, std, run, srun, band='g')
	cbx, cbxerr, cby, cbyerr = mJyFromCounts(cb, sb, star, std, run, srun, band='b')
	
	okr, okcr = overlap(rx,crx)
	okg, okcg = overlap(gx, cgx)
	okb, okcb = overlap(bx, cbx)	
	
	r = ry[okr] / cry[okcr] * np.average(cry[okcr])
	g = gy[okg] / cgy[okcg] * np.average(cgy[okcg])
	b = by[okb] / cby[okcb] * np.average(cby[okcb])
	
	rerr = np.sqrt((ryerr[okr]/ry[okr])**2 + (cryerr[okcr]/cry[okcr])**2) * r
	gerr = np.sqrt((gyerr[okg]/gy[okg])**2 + (cgyerr[okcg]/cgy[okcg])**2) * g
	berr = np.sqrt((byerr[okb]/by[okb])**2 + (cbyerr[okcb]/cby[okcb])**2) * b
	
	return rx[okr], rxerr[okr], r, rerr, gx[okg], gxerr[okg], g, gerr, bx[okb], bxerr[okb], b, berr


def ulg2mjysWithComp(address, star, std, run, oi=1, comp=2):
	""" Reads a ulog file and finds magnitudes, using a comparison of known magnitude.
	"""
	r,g,b = readulog(address, oi=oi, comp=None)
	cr, cg, cb = readulog(address, oi=comp, comp=None)
	
	rx, rxerr, ry, ryerr = mJyFromCountsWithComp(r, cr, star, std, run, band='r')
	gx, gxerr, gy, gyerr = mJyFromCountsWithComp(g, cg, star, std, run, band='g')
	bx, bxerr, by, byerr = mJyFromCountsWithComp(b, cb, star, std, run, band='b')	
	
	return rx, rxerr, ry, ryerr, gx, gxerr, gy, gyerr, bx, bxerr, by, byerr	
	
	

def correctTimes(mjd, coords, obsname):
	""" Applies barycentric corrections to a set of times in MJD, returning 
	BMJD(TDB)."""
	
	"""finds position of observatory"""
	tel,obs,longitude,latitude,height = \
		subs.observatory(obsname)
	
	"""applies time correction"""
	ra,dec,syst = subs.str2radec(coords)
	tt,tdb,btdb,hutc,htdb,vhel,vbar = \
		sla.utc2tdb(mjd,longitude,latitude,height,ra,dec)
	
	return btdb



def readUcmFits(fname):
	""" Reads a fits file, as created by ucm2fits, and returns a 2d numpy array 
	in a nice format"""
	hdulist = fits.open(fname)
	data = np.hstack((np.array(hdulist[1].data),np.array(hdulist[2].data))).T
				# transpose so x and y coords match up with image
	return data

def readLcurve(fname):
	"""Reads a file of the format output by lroche"""
	
	data = np.loadtxt(fname)
	
	x = data[:,0]
	y = data[:,2]
	return x, y


def detrend(funct,p0,x,y,yerr,subtract=True,fmt='g.'):
	""" Removes a trend from the data, after fitting the supplied function to the 
	data by least squares fit"""
	yax = "Counts (arbitrary units)"
	"""plot starting values"""
	lightcurve(x, y, yerr, fmt=fmt, fignum=0, title="Starting Values", yax=yax)
	lightcurve(x, funct(x,*p0), fmt='k', fignum=0)
	#plt.show(block=True)##
	
	"""do the fit"""
	numMask = len(x) / 15			#max number of points to mask
	maskLim = 5						#significance to mask to
	
	popt, pcov, mask = iterativeFit(funct, x, y, yerr, numMask, maskLim, p0)
	
	"""plot fit"""
	lightcurve(x, y, yerr, fmt=fmt, fignum=1, title="Fitted trend", yax=yax)
	lightcurve(x, funct(x,*popt), fmt='k', fignum=1)
	lightcurve(x[mask==False], y[mask==False], fmt='rD', fignum=1)
	
	"""detrend"""
	if subtract == True:
		ymean = y.mean()
		detrended = y - funct(x,*popt) + ymean
		detErr = yerr
	else:
		detrended = y / funct(x,*popt)
		detmean = detrended.mean()
		detrended = detrended/detmean
		detErr = yerr / funct(x,*popt)/detmean
	
	yax = "Counts / Mean Counts"
	"""plot detrended data"""
	lightcurve(x, detrended, detErr, fmt=fmt, fignum=2, title="Detrended data", yax=yax)
	
	"""show trend for inspection by eye -- user can close these windows to accept the fit"""
	plt.show()
	return detrended, detErr


def ulogToLcurve(fname,detrendFlag=False,funct=None,p0=None,yax=None):
	""" Reads an ultracam log file, detrends if required,
	and outputs in a format useful for 'lcurve'"""
	
	"""read file and apply time corrections"""
	r, g, b = readulog(fname)
	btdb = correctTimes(g.x.data, coords, obsname)
	
	"""apply detrend if necessary"""
	if detrendFlag:
		y,yerr = detrend(funct,p0,btdb,g.y.data,g.y.errors)
	else:
		y,yerr = g.y.data,g.y.errors
	
	"""output in desired format"""
	return outputForLcurve(btdb,g.x.errors,y,yerr)


#########  Lcurve, lroche and binary modelling associated functions

##  Handling data files (.dat)


def outputForLcurve(times,terr,y,yerr,weights=None):
	""" Outputs data in a format that can be saved and used for 'lcurve'"""
	if weights is None:
		weights = np.ones(len(times))
	for t,te,f,fe,w in zip(times,terr,y,yerr,weights):
		print '{0:15.9f} {1:9.4g} {2:7.5f} {3:7.5f} {4} 1'.format(t,2*te,f,fe,w)
	return zip(times,terr,y,yerr)


def saveForLcurve(times,terr,y,yerr,sname,weights=None):
	if weights is None:
		weights = np.ones(len(times))
	otherThing = np.ones(len(times))
	all = np.dstack((times,terr,y,yerr,weights,otherThing))[0]
	np.savetxt(sname, all, fmt='%15.9f %9.4g %7.5f %7.5f %f %d')


##  Handling model files (.mod)

def readMod(fname):
	""" Read in an lroche model file (.mod). Returns columns, as well as the extra stuff at the bottom.
	"""
	with open(fname) as lis:
		numlines = sum(1 for line in lis)
	labels, vals, a, b, c, d = np.genfromtxt(fname, usecols=(0,2,3,4,5,6),dtype=str,unpack=True,skip_footer=34)
	footer = np.loadtxt(fname, skiprows=(numlines-34),dtype=str)
	vals = vals.astype(float)

	return labels, vals, a, b, c, d, footer


def writeMod(fname,labels, vals, a, b, c, d, footer):
	""" Write a set of columns to a model file. Could take direct output from readMod if so desired.
	"""
	eqs = np.array(['=']*len(labels))
	top = np.column_stack((labels, eqs, vals, a, b, c, d))

	np.savetxt(fname, top, fmt=['%s','%s','%s','%s','%s','%s','%s'])
	with open(fname,'a') as f:
		for line in footer:
			line = line[0]+' '+line[1]+' '+line[2]+'\n'
			f.write(line)
	return True

##  Handling mcmc output

def readMcmcLog(fname, par):
	""" For a specified parameter, read data points from mcmc .log file and return the data for the relevant parameter
	"""
	# find params that varied
	for line in open(fname):
		if line.startswith('##'):
			labels = line.split()[1:]
			labels = np.array(labels, dtype=str)
			break
	#read data and select the desired parameter
	data = np.loadtxt(fname)
	oi = data[:, (labels==par)]
	if len(oi) == 0:
		print "!!ERROR finding that parameter in log file!!"
	return oi

def mcmcHist(data, plotHists=True, label='', chisq=None, cburn=None):
	"""	Returns the mean and std of a dataset, as well correlation with a Gaussian distribution. Can optionally plot a histogram.
	    If an array of chisq passed and cburn specified, will ignore points with chisq > cburn.
	"""
	## Allow for some high chisq points to be ignored
	if cburn is not None and chisq is not None:
		ok = chisq < cburn
		if not ok.any():
			print "No points above cburn"
		else:
			print "%s points above cburn"%(np.sum(ok))
	else:
		ok = np.ones(len(data),dtype=bool)

	## Get histogram of results
	if plotHists:
		# This one plots histogram and returns values
		n, bins, patches = plt.hist(data[ok], bins=20)
	else:
		# This one just returns values, without plotting
		n, bins = np.histogram(data[ok], bins=20)
	
	## Fit with Gaussian
	try:
		mids = (bins[1:] + bins[:-1]) / 2
		a0 = data[ok].max()
		m0 = data[ok].mean()
		s0 = data[ok].std()
		popt, pcov = curve_fit(mg.gauss, mids, n, p0=[a0,m0,s0])
		
		a, m, s = popt
		
		rms = np.sqrt( ((n - mg.gauss(mids, *popt))**2).mean() )
		
		print label, "RMS from Gaussian distribution", rms
		
		if plotHists:
			# Don't bother plotting gaussian, saves a bit of time
			plt.plot(mids, mg.gauss(mids, *popt), 'r')
	except RuntimeError:
		print label, "did not converge"
	
	if plotHists:
		if label:
			plt.suptitle(label)
		mg.formatGraph()
		plt.show()
	return data[ok].mean(), data[ok].std(), rms


##  To do with model itself


def findChisq(modfile, datfile):
	""" Goodness of fit of model. Read the chisq of an lroche model to a data file, by calling lroche 
	"""
	## Run lroche to get output using subprocess module
	cmd = "/storage/astro1/phsaap/software/bin/lcurve/lroche %s %s device=null scale=yes nfile=0 \\\\"%(modfile, datfile)
	process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)		# runs lroche
	output, err = process.communicate()		# output is the stdout of lroche
	exit_code = process.wait()		# exit code, in case needed

	## find chisq in output with regex
	reg = re.compile(r"Weighted chi\*\*2\s\=\s.+")
	line = reg.search(output)
	if line is not None:
		chisq = float(line.group(0).split()[3][:-1])	#convert to string, split line, take number, trim comma, convert to float
	else:
		chisq = -1

	return chisq


def toingeg(x,y):
	""" Converts x,y coords to ingress, egress phases. Wrapper around trm.roche.ineg that does the iteration for you
	"""
	ings,egs = [],[]

	for j in xrange(len(x)):
		ing, eg = roche.ineg(q, iangle, x[j], y[j])
		ings += [ing]
		egs += [eg]
		
	return ings, egs


def iestream(q, iangle, step=1./1000, n=200):
	""" Returns ingress, egress, x and y positions for stream. A wrapper around trm.roche.stream
	"""
	xstream, ystream = roche.stream(q, step, n)
	ings, egs = toingeg(xstream,ystream)
	
	return ings, egs, xstream, ystream
	

def spotProfile(x,blen,bexn,bexm):
	""" Returns the bright spot profile as a function of distance along spot (x).
		x should be in units of angular separation with x=0 at the peak of the profile (x=0 at radius_spot)
	"""
	xmax = (bexn / bexm) ** (1 / bexm) * blen	# the distance from flux=0 to flux=max
	profile = (((x+xmax)/blen)**bexn)*np.exp(-((x+xmax)/blen)**bexm)
	return profile / profile.max()


############################################################



""" If run as script, apply some defaults (suitable for May data)"""
if (__name__ == "__main__"):
	if len(argv) <= 1:
		print "Please specify file"
		exit()
	fname = argv[1]
	
	detrendFlag = True
	funct=x3plusx2
	p0=[-1000,57166.,0.15,1000,0.15,0]
	
	ulogToLcurve(fname,detrendFlag,funct,p0)
