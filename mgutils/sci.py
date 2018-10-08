#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
from astropy import time
import re, datetime
from scipy.optimize import curve_fit
from scipy import ndimage
from sys import argv, exit
import mgutils as mg, mgutils.constants as co


####### Science things for specific applications




def blackbody(temp, freq='default', wave='default', units='SI', useFreq=True, useWave=False, n=1000):
    """ Plot the blackbody curve for a given temperature. useFreq determines whether x-axis will be frequency or wavelength.
    """
    if useWave == True:
        useFreq = False # obviously can't use both. Will only check usefreq from this point on.

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
        print ("!! Units not recognised !!")
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
        return freq / fac, bspec / bspec.max()  #convert x-axis back

    else:
        bspec = (2 * h * c**2 / wave**5) * (1 / (np.exp(h * c / wave / kB / temp) -1))
        return wave / fac, bspec / bspec.max()


def mass_from_log_g(logg, rad):
    """ Takes log g (assumed to be in cgs units) and radius in solar radii. Returns mass in solar masses.
    """
    return 10**(logg-2) * (rad*co.rSun)**2 / co.G / co.mSun
    
    
def orbital_separation(m1, m2, period):
    """
    Returns orbital semi-major axis in solar radii given masses in solar masses
    and an orbital period in days, i.e. Kepler's third law. Works with scalar
    or numpy array inputs.  ##Taken from Tom's subs code
    """

    return (co.G*co.mSun*(m1+m2)*(co.secondsPerDay*period/(2.*np.pi))**2)**(1./3.)/co.rSun

def rad_from_log_g(logg, mass):
    return  np.sqrt((mass*co.mSun) * co.G / 10**(logg-2)) / co.rSun

def log_g_from_mass_rad(mass, rad):
    return np.log10( (mass*co.mSun) * co.G / (rad*co.rSun)**2 ) + 2


def findCalcTime(obs, t0, porb):
    """ Finds expected eclipse times from a given set of observed times and an ephemeris
    """
    e = (np.array(obs) - t0) / porb
    calc = np.around(e) * porb + t0
    print (obs, calc, obs-calc)
    return calc


def findEps(q):
    """ Empirical eps(q) relation from Patterson 2005
    """
    return 0.18*q + 0.29*q**2


def findQ(eps):
    """ Inverse of Patterson 2005 eqn
    """
    return quadraticFormula(0.29,0.18,-eps)[0]

def findEpsKato(q):
    """ Empirical relation from Kato 20??
    """
    epsAsterisk = 0.00027 + 0.402*q - 0.467*q**2 + 0.297*q**3
    eps = epsAsterisk / (1 - epsAsterisk)
    return eps

functApp = mg.functApp

def findQKnigge(eps,epserr):
    """ Knigge 2006's eps-q relation (with errors)
    """
    def calculate(a,b,c,eps):
        return a + b*(eps-c)
    return functApp(calculate,0.114,0.005,3.97,0.41,0.025,0,eps,epserr,)

def findQMcAllisterB(eps,epserr):
    """ McAllister 2018's update to Knigge's eps-q relation (with errors) for phase B superhumps
    """
    def calculate(a,b,c,eps):
        return a + b*(eps-c)
    return functApp(calculate,0.118,0.003,4.45,0.28,0.025,0,eps,epserr,)
    
def findQMcAllisterC(eps,epserr):
    """ McAllister 2018's update to Knigge's eps-q relation (with errors) for phase C superhumps
    """
    def calculate(a,b,c,eps):
        return a + b*(eps-c)
    return functApp(calculate,0.135,0.004,5.0,0.7,0.025,0,eps,epserr,)


######### Magnitudes and colours        #########################


def vMagFromSDSSgr(g,r):
    V = g - 0.59*(g-r) - 0.01   #From Jester et al 2005 (from http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php)
    return V
    
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


def extinction(l, coords, obsname, show=False, fignum=False, mask=None, ax=None):
    """ Finds and returns the extinction coefficient of an object over one dataset. 
        Takes the object data, l, as a ulog-type object.
        Returns extinction coefficient and its error as taken from the covarience matrix
    """
    ok = (l.y.data != 0)
    amass = airmass(l.x.data[ok], coords, obsname)
    amerr = airmass(l.x.data[ok]+l.x.errors[ok], coords, obsname) - amass
    
    fakemags = -2.5 * np.log10(l.y.data[ok])            #magnitude + offset
    fmerr = np.abs(-2.5 * np.log10(l.y.data[ok]+l.y.errors[ok]) - fakemags)
    
    numMask = len(amass)/100
    maskLim = 3
    if mask is not None:
        assert len(mask) == len(amass)  #check starting mask fits data
    popt, pcov, mask = iterativeFit(line, amass, fakemags, fmerr, numMask, maskLim, inmask=mask)
    
    ext = popt[0]
    exterr = np.sqrt(pcov[0,0])
    
    if fignum or (fignum==0):
        plt.figure(fignum)
    if fignum or (fignum==0) or show:
        if ax is None:
            ax = plt.gca()
        lightcurve(amass, fakemags, fmerr, sub=ax, xax='Airmass', yax='Mag + c', fmt='b.')
        #ax.errorbar(amass, fakemags, fmerr, amerr, fmt='b.')
        ax.plot(amass, line(amass, *popt), 'k')
        ax.plot(amass[mask==False], fakemags[mask==False], 'rD')
        ax.annotate("Extinction = %f +/- %f"%(ext, exterr),(0.2,0.8), xycoords='axes fraction', fontsize=12)
        ax.set_xlim(adjLimits(amass[mask==True]))
        ax.set_ylim(adjLimits(fakemags[mask==True], margins=0.5))
    else:
        plt.close()
    if show:
        plt.show()
    
    return ext, exterr

def magsFromComp(l, sl, star, std, run, band):
    """ Function to calculate magnitudes from ultracam counts based on a comparison star, in the same field, with known magnitude.
    """
    
    ok = (l.y.data != 0) & (sl.y.data != 0)
    
    smag = std.mag(band)
    smerr = std.magerr(band)
    
    print ("Standard mag", smag, "in band", band)
    
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
    sok = (sl.y.data != 0)
    #print l.y.data[ok]
    
    amass = airmass(l.x.data[ok], star.coords, run.obsname)
    amerr = airmass(l.x.data[ok]+l.x.errors[ok], star.coords, run.obsname) - amass
    samass = airmass(sl.x.data, std.coords, srun.obsname)
    samerr = airmass(sl.x.data+sl.x.errors, std.coords, srun.obsname) - samass
    
    smag = std.mag(band)
    smerr = std.magerr(band)
    
    popt, pcov, mask = iterativeFit(flat, sl.x.data[sok], sl.y.data[sok], sl.y.errors[sok], numMask=1, maskLim=3, show=True)
    
    sy = popt[0]#np.average(sl.y.data, weights=1/sl.y.errors**2)
    syerr = np.sqrt(1 / (np.sum(1/sl.y.errors[sok][mask]**2)))      # Equation from Tom H's book p50
    sa = np.average(samass[sok][mask], weights=1/samerr[sok][mask]**2)
    saerr = np.sqrt(1 / (np.sum(1/samerr[sok][mask]**2)))
    
    mag = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*run.ext(band) + sa*srun.ext(band)
    
    
    a1 = (smag+smerr - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*run.ext(band) + sa*srun.ext(band)) - mag
    a2 = smag - 2.5*np.log10(((l.y.data[ok]+l.y.errors[ok])) / (sy)) - amass*run.ext(band) + sa*srun.ext(band) - mag
    a3 = smag - 2.5*np.log10((l.y.data[ok]) / ((sy+syerr))) - amass*run.ext(band) + sa*srun.ext(band) - mag
    a4 = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - (amass+amerr)*run.ext(band) + sa*srun.ext(band) - mag
    a5 = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*(run.ext(band)+run.exterr(band)) + sa*srun.ext(band) - mag
    a6 = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*run.ext(band) + (sa+saerr)*srun.ext(band) - mag
    a7 = smag - 2.5*np.log10((l.y.data[ok]) / (sy)) - amass*run.ext(band) + sa*(srun.ext(band)+srun.exterr(band)) - mag
    
    magerr = np.sqrt(a1**2 + a2**2 + a3**2 + a4**2 + a5**2 + a6**2 + a7**2)
    
    #print "errors:\n", magerr, '\n', a1, '\n', a2, '\n', a3, '\n', a4, '\n', a5, '\n', a6, '\n', a7            #useful for finding dominant errors
    
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
    okb = np.zeros(len(bx[nzb]),dtype=bool)         # all this to mask points where magnitude data not available for both
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
    mj = 10 ** ((23.9-mags)/2.5) / 1000         #from Wikipedia
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



######## Phase Folding      ############################


def phaseFold(btdb, terr, data, t0, t0err, period, perr, bins, errors=None, returnStds=False, useste=False):
    """ Phase folds and averages the data, outputting phase, 'continuous phase', and means, errors, and counts for each bin 
        "Continuous phase" is simply how many periods you are away from t0.
        Can return bins and their y values, or can just return the phase-folded times
        If you don't care about the y data, send None to this or to bins
        terr can be None, but y errors must be given if you want bin values to be calculated
    """
    if terr is not None:
        assert len(terr) == len(btdb)
    phaseCont = ((btdb-t0) / period)
    phase = phaseCont % 1
    phase[phase>0.5] = phase[phase>0.5] - 1 
    if terr is not None:
        pherr = np.sqrt( (np.sqrt(t0err**2 + terr**2) / (btdb-t0))**2) * phaseCont
        pherr = np.sqrt( (pherr/phase)**2 + (perr/period)**2) * phase
    else:
        pherr = np.sqrt( (np.sqrt(t0err**2) / (btdb-t0))**2 + (perr/period)**2 ) * phaseCont
    
    if type(bins) == int:
        bins = np.linspace(-0.5,0.5,bins+1)
    
    if bins is not None and data is not None:
        mids = (bins[1:] + bins[:-1]) / 2   #midpoint of each bin
        numBins = len(bins) - 1
        means = np.zeros(numBins)
        errs = np.zeros(numBins)
        counts = np.zeros(numBins)
        miderrs = np.zeros(numBins)
        stds = np.zeros(numBins)
        
        for i in xrange(numBins):
            ran = (phase>bins[i]) & (phase<bins[i+1])
            if len(data[ran]) == 0:
                continue
            
            means[i] = np.average(data[ran], weights=(1/errors[ran]**2))
            if useste:
                errs[i] = data[ran].std()/np.sqrt(len(data[ran]))
            else:
                errs[i] = np.sqrt(1 / (np.sum(1/errors[ran]**2)))       # Equation from Tom H's book p50
            counts[i] = len(data[ran])
            miderrs[i] = np.average(pherr[ran])         #don't know the proper way to work this out, this is just an estimate
            stds[i] = chisq(data[ran].mean(), data[ran], errors[ran])
            
        if returnStds:
            return phase, phaseCont, pherr, mids, miderrs, means, errs, counts, stds
        else:
            return phase, phaseCont, pherr, mids, miderrs, means, errs, counts
    else:   #If y data or bin sizes given as None
        return phase, phaseCont, pherr

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
        
