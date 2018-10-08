#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
from astropy import time
from trm import dnl, subs, sla, roche
import re, datetime
from scipy.optimize import curve_fit
from scipy import ndimage
from sys import argv, exit
import stars, runs, constants as co

coords = '16:11:33.97 +63:08:31.8'
obsname = 'WHT'                     # This is the setup for the object I'm currently working on
t0 = 57153.6890966
t0err = 4.02767506355e-07
period = 0.0345195708393
perr = 7.85384131425e-11

detrendFlag = True      # True if need to detrend - dependant on data set


bandDict = {'u':3, 'b':3, 'g':2, 'r':1, 'i':1, 'z':1}

avphi = 0.037134275925356952

##############  General stuff   ###########################

def grep(fname, string):
    """ Greps through a file using re, returns 0th match
    """
    
    with open(fname) as origin_file:
        for line in origin_file:
            line = re.findall(string, line)
            if line:
                return line[0]
    return None #if nothing found

def find_nearest(arr, val, arg=False):
    """ Given an array and a value, returns the array component closest to that value. If arg=True, returns argument of that component (the index)
    """
    idx = (np.abs(arr - val)).argmin()
    if arg:
        return idx
    else:
        return arr[idx]



def loopOver(loop, funct, numOutputs=1, *args, **kwargs):
    """ Loops over a given list of inputs, performing the same function using those inputs, and returns an array or arrays of the outputs. 
        If the function returns more than one output you must set the number expected.
        Probably not the most efficient way to do anything but might save on typing
    """
    outss = [np.array([])]*numOutputs
    for j, l in enumerate(loop):
        out = funct(l, *args, **kwargs)
        if hasattr(out, '__iter__'):
            for k in range(numOutputs):
                outs = outss[k]
                outss[k] = np.append(outs, out[k])
        else:
                outss = np.append(outss, out)
    return outss
    


def loadFromCsv(loadname, title=None, usecols=None, delimiter=';', skiprows=0, lastline=None, dtype=str):
    """ Load a whole table from a csv file. Should be fairly general
    """
    with open(loadname) as f:
        lines = f.read().splitlines()[skiprows:]
    if lastline is None:
        lastline = len(lines) + skiprows
    
    
    table = {"Title":title, "Length":len(lines)-1} 
    
    
    headers = lines[0].split(delimiter)
    
    for j, header in enumerate(headers):
        # Find values in table and make array
        if usecols is not None and not j in usecols:
            continue
        arr = np.array([],dtype=dtype)
        for line in lines[1:lastline-skiprows]:
            vals = line.split(delimiter)
            arr = np.append(arr, vals[j])
        # Add array to dictionary
        table[header] = arr
    return table
    

##############  Shapes and functions for fitting ##########


def flat(x,c):
    try:
        return np.zeros(len(x)) + c
    except TypeError:
        return c

def line(x,a,c):
    return a*(x) + c
slope = line

def x2(x,a,b,d,c):
    return a*(d*(x-b))**2 + c

def x3(x,a,b,d,c):
    return a*(d*(x-b))**3 + c

def x3plusx2(x,a,b,d,e,f,c):
    return a*(d*(x-b))**3 + e*(f*(x-b))**2 + c

def x2plusx(x,a,b,d,e,c):
    return a*(d*(x-b))**2 + e*(x-b) + c

def sinx(x,a,x0,f,c):
    return a*np.sin(2*np.pi*f*(x-x0)) + c    
sin = sinx

def sinf(f):
    def sin(x,a,x0,c):
        return a*np.sin(2*np.pi*f*(x-x0)) + c    
    return sin

def sinNoFloor(x,a,x0,f):
    return a*np.sin(2*np.pi*f*(x-x0))

def sinPlusSlope(x,a,x0,f,c,d):
    return a*np.sin(2*np.pi*f*(x-x0)) + c + d * (x-x0)

def gauss(x, a, m, s):
    return a*np.exp(- (x-m)**2 / (2 * s**2))

def doubleGauss(x, a1, m1, s1, a2, m2, s2, c, d):
    return a1*np.exp(- (x-m1)**2 / (2 * s1**2)) + a2*np.exp(- (x-m2)**2 / (2 * s2**2)) + d*x + c
   
def gauss2d(xy, a, mx, sx, my, sy, c):
    """ Returns a 2d gaussian reshaped to be 1d for easier fitting
    """
    x,y = xy
    return (a * np.exp(- (x-mx)**2 / (2 * sx**2)) * np.exp(- (y-my)**2 / (2 * sy**2)) + c).ravel()   
    
def gauss2dplusSlope(xy, a, mx, sx, dx, my, sy, dy, c):
    """ Returns a 2d gaussian reshaped to be 1d for easier fitting
    """
    x,y = xy
    return (a * np.exp(- (x-mx)**2 / (2 * sx**2)) * np.exp(- (y-my)**2 / (2 * sy**2)) + dx*(x-mx) + dy*(y-my) + c).ravel()


def circle(r, xc=0, yc=0, n=200):
    """ Returns coordinates of a circle of radius r
    """
    angle = np.linspace(0,2*np.pi,n)
    
    x = r * np.sin(angle) + xc
    y = r * np.cos(angle) + yc
    
    return x, y



##########  Stats and manipulation  ################

def near(a, b, prec=0.01):
    return np.abs(a - b) < prec


def weightedAv(x, err):
    if type(x) == list:
        x = np.array(x)
    if type(err) == list:
        err = np.array(err)
    av = np.average(x, weights=1/err**2)
    averr = np.sqrt(1 / (np.sum(1/err**2)))     # Equation from Tom H's book p50
    return av, averr

def weightedAvErr(err):
    averr = np.sqrt(1 / (np.sum(1/err**2)))     # Equation from Tom H's book p50
    return averr

def chisq(model, data, err):
    return np.sum( (data-model)**2 / err**2 )

def rms(data):
    return np.sqrt( np.mean( (data - data.mean())**2 ) )

def quad(*args):
    """ Combine a list of arguments in quadrature
    """
    args = np.array(args)
    return np.sqrt(np.sum(args**2))

def functApp(funct, *args):
    """ Functional approach on an arbitrary function. Usage: functApp(funct, a, aerr, b, berr ...). 
        BEWARE: Basically no (programming) error handling as of yet
    """
    plus, minus = True, True
    verbose = True
    var, err = [], []
    if len(args) % 2 != 0:
        # If odd number of args passed -- ie if they've missed off an error or something
        print "Functional approach has been passed the wrong number of arguments (%d excluding the function pointer)"%(len(args))
        exit()
    #split variables and errors into 2 arrays
    for i, arg in enumerate(args):
        if i % 2 == 0:
            var.append(arg)
        else:
            err.append(arg)
    var = np.array(var)
    err = np.array(err)
    
    #Find the 'expected' result
    result = funct(*var)
    
    # For each error, propogate it's effect through
    # Let user choose whether to add or take away error, or do both and average
    if plus and minus:
        diffs = []
        for j, (v, e) in enumerate(zip(var, err)):
            toPass1 = np.copy(var)
            toPass2 = np.copy(var)
            toPass1[j] = v + e   
            toPass2[j] = v - e   
            d1 = funct(*toPass1) - result
            d2 = funct(*toPass2) - result
            
            # Avoid any nans that might have come up if only on one side
            if isinstance(d1,float) or isinstance(d1,int):
                if np.isnan(d1):
                    d1 = d2
                if np.isnan(d2):
                    d2 = d1
            else:
                print type(d1)
                d1[np.isnan(d1)] = d2[np.isnan(d1)]
                d2[np.isnan(d2)] = d1[np.isnan(d2)]
            
            
            diffs.append((np.abs(d1)+np.abs(d2))/2.)
        diffs = np.array(diffs)
    elif plus:
        diffs = []
        for j, (v, e) in enumerate(zip(var, err)):
            toPass = np.copy(var)
            toPass[j] = v + e   
            d1 = funct(*toPass) - result
            diffs.append(d1)
        diffs = np.array(diffs)
    elif minus:
        diffs = []
        for j, (v, e) in enumerate(zip(var, err)):
            toPass = np.copy(var)
            toPass[j] = v - e   
            d1 = funct(*toPass) - result
            diffs.append(d1)
        diffs = np.array(diffs)
    
    print "Error contributions:", diffs
    # combine error contributions in quadrature
    return result, quad(diffs)
    
def iterativeFit(funct, x, y, yerr=None, numMask=5, maskLim=3, p0=None, inmask=None, show=False, silent=False, verbose=False, bounds=None, chisqIters=0,\
                                *args, **kwargs):
    """ Fits iteratively over a data set, each iteration masking one data point and refitting.
        Takes function and associated x and y values, maximum number of points to mask, and sigma limit
        below which it will not mask (masks until either of those criteria is met). 
        Returns final fit parameters and the final mask used.
    """
    if silent:
        verbose = False
    if inmask is None:
        mask = np.ones(np.shape(x),dtype=bool)
    else:
        mask = inmask
    assert len(mask) == len(x)
    #assert len(mask) == len(y)
    if yerr == None or (np.array(yerr) == 0).all():
        yerr = np.zeros(len(y))
        sigma = None
    else:
        sigma = yerr[mask]
    
    x = np.array(x) # in case passed in as non-numpy arrays
    y = np.array(y)
    
    if bounds is not None:
        # I hate this -- there are better ways
        # Define a new function that acts like the old function within bounds but shoots off outside of bounds
        def applyBounds(funct, bounds):
            
            def newFunct(x,*p):
                pen = 0
                for thisp, lo, hi in zip(p, bounds[0], bounds[1]):
                    if thisp < lo:
                        pen += np.abs(lo - thisp)*1e20
                    elif thisp > hi:
                        pen += np.abs(hi - thisp)*1e20
                return funct(x,*p) + pen
            return newFunct
        funct = applyBounds(funct, bounds)
    
    
    
    ## Iterate over number of masked points requested
    popt, pcov = curve_fit(funct, x[mask], y[mask], sigma=sigma, p0=p0, *args, **kwargs)
    i=0
    while (len(mask[mask==False]) <= numMask):
        popt, pcov = curve_fit(funct, x[mask], y[mask], sigma=sigma, p0=p0)
        fit = funct(x,*popt)
        diffs = np.abs((y - fit) / yerr)
        if diffs[mask].max() > maskLim and len(mask[mask==False]) < numMask:
            mask[diffs == diffs[mask].max()] = False
            i+=1
            if verbose:
                print "Masking point", np.argwhere(diffs == diffs[mask].max()), "at sig", diffs[mask].max()
        else:
            if not silent:
                if i > 0:
                    print "iterativeFit masked",i,"sig",diffs[mask==False].min()
                else:
                    print "iterativeFit masked 0"
            break
    
    
    ## Iterate over number of chisq iterations
    scaled_yerr = yerr 
    for j in range(chisqIters):
        csq = chisq(funct(x[mask],*popt), y[mask], scaled_yerr[mask])
        wnok = np.sum(mask) - len(popt)
        scaled_yerr = scaled_yerr * np.sqrt(csq/wnok)
        popt, pcov = curve_fit(funct, x[mask], y[mask], sigma=scaled_yerr[mask], p0=p0)
        if verbose:
            print "Chisq iter", j, ":", csq, "->", chisq(funct(x[mask],*popt), y[mask], scaled_yerr[mask])
    
    
    if verbose and not silent:
        errs = np.sqrt(np.diag(pcov))
        print "Results:", popt
        print "Errors: ", errs
        print "Corr.ns:"
        for j in range(len(popt)):
            print "            "*j, pcov[j+1:,j] / errs[j] / errs[j+1:]
        
        if (yerr==0).all():
            print "Lsq", np.sum((funct(x[mask],*popt) - y[mask])**2), "wnok", len(x) - len(popt)
        else:
            print "Chisq", chisq(funct(x[mask],*popt), y[mask], scaled_yerr[mask]), "wnok", len(x) - len(popt)
        
    
    if show:
        plt.plot(x,y,'b.')
        plt.plot(x[mask==False], y[mask==False], 'rD')
        plt.plot(x, funct(x,*popt), 'k--')
        plt.show()
    return popt, pcov, mask

def quadraticFormula(a,b,c):
    return (-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b-np.sqrt(b**2 - 4*a*c))/(2*a)


def quadraticFormulaSingleValue(a,b,c):
    return (-b + np.sqrt(b**2 - 4*a*c))/(2*a)


def overlap(bx, rx):
    """ Returns masks for which of array 1 are in array 2 and v.v.
    """
    okb = np.zeros(len(bx),dtype=bool)          # all this to mask points where magnitude data not available for both
    for i in xrange(len(bx)):
        if bx[i] in rx:
            okb[i] = True
    okr = np.zeros(len(rx),dtype=bool)
    for i in xrange(len(rx)):
        if rx[i] in bx:
            okr[i] = True
    return okb, okr

def fitLine(x1,y1,x2,y2):
    """ Fits two points with an y = mx + c line.
    """
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c


def convert(x, x1, y1, x2, y2):
    """ Convert from xfig coordinates to real coordinates.
    """
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    
    return coords * m + c

############  Science things        ###########


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
        return freq / fac, bspec / bspec.max()  #convert x-axis back

    else:
        bspec = (2 * h * c**2 / wave**5) * (1 / (np.exp(h * c / wave / kB / temp) -1))
        return wave / fac, bspec / bspec.max()


def mass_from_log_g(logg, rad):
    """ Takes log g (assumed to be in cgs units which appears to be standard) and radius in solar radii. Returns mass in solar masses.
    """
    return 10**(logg-2) * (rad*co.rSun)**2 / co.G / co.mSun

def rad_from_log_g(logg, mass):
    return  np.sqrt((mass*co.mSun) * co.G / 10**(logg-2)) / co.rSun

def log_g_from_mass_rad(mass, rad):
    return np.log10( (mass*co.mSun) * co.G / (rad*co.rSun)**2 ) + 2


def findCalcTime(obs, t0, porb):
    """ Finds expected eclipse times from a given set of observed times and an ephemeris
    """
    e = (np.array(obs) - t0) / porb
    calc = np.around(e) * porb + t0
    print obs, calc, obs-calc
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


def convertRA(hr,min,sec):
    return (hr + (min + sec/60.)/60.) * 360. / 24.
    
def convertRAback(ra):
    hr = ra / 360. * 24. // 1
    min = (ra / 360. * 24. - hr) * 60 // 1
    sec = ((ra / 360. * 24. - hr) * 60 - min) * 60
    return hr, min, sec
    
def convertDecback(dec):
    deg = np.sign(dec) * (np.abs(dec) // 1)
    min = (np.abs(dec) - np.abs(deg)) * 60 // 1
    sec = ((np.abs(dec) - np.abs(deg)) * 60 - min) * 60
    return deg, min, sec

def convertDec(deg, min, sec):
    if np.size(deg) == 1:
        sign = +1 if deg==0 else np.sign(deg)
    else:
        sign = np.sign(deg)
        sign[deg==0] = +1
    return sign * (np.abs(deg) + (min + sec/60.)/60.)


def convertVel(lam,cenwave):
    return (lam-cenwave)/cenwave * 299792.458
    
def convertLam(vel,cenwave):
    return vel / 299792.458 * cenwave + cenwave


#########  Graph Plotting       ###################################

def formatGraph(fignum=1,suptitle=None,xlabel=None,ylabel=None,returnFig=False,poster=True,*args,**kwargs):
    """ General formatting things like style sheet, axes etc. Title and axis labels are applied to the current figure and axes.
    """
    fig = plt.figure(fignum, *args,**kwargs)
    if poster:
        plt.style.use('mgposter')
    else:        
        plt.style.use('mgplot')
    # Matplotlib 2 colours -- less bright and more colourblind friendly
    plt.rc('axes', prop_cycle=(cycler('color',['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])))
    if xlabel is not None:
        plt.gca().set_xlabel(xlabel)
    if ylabel is not None:
        plt.gca().set_ylabel(ylabel)
    if suptitle is not None:
        plt.suptitle(suptitle)
    
    if returnFig:
        return fig
    else:
        return plt.gca()

def formatSubplots(subshape, xlabel=None, ylabel=None, sharex=False, sharey=False, tidyformat=True, figsize=(8,11), **kwargs):
    """ Create a figure containing a set of subplots. Unrecognised keywords are passed into plt.subplots, thence into plt.figure
    """
    plt.style.use('mgplot')
    fig, axs = plt.subplots(subshape[0], subshape[1], sharex=sharex, sharey=sharey, figsize=figsize, **kwargs)
    if subshape == (1,1):
        axs = [axs]     # to avoid problems trying to iterate over a single set of axes -- makes it possible to use this function to generate a single "subplot" if you want to for some reason
    j = 0
    if xlabel is not None:
        if tidyformat:
            if subshape[1] == 1:
                axs[-1].set_xlabel(xlabel)
            else:
                for ax in axs.flatten()[-subshape[1]:]:
                    ax.set_xlabel(xlabel)
        else:
            for ax in axs:
                ax.set_xlabel(xlabel)
        
    if ylabel is not None:
        if tidyformat:
            if subshape[0] == 1:
                axs[0].set_ylabel(ylabel)
            else:
                for ax in axs.flatten()[::subshape[1]]:
                    ax.set_ylabel(ylabel)
        else:
            for ax in axs:
                ax.set_ylabel(ylabel)
        
    formatGraph(fig.number)
    return fig, axs

def addMinorTicks(ax,numx=5,numy=1):
    """ A wrapper around the functions that add minor ticks.
        Num = 1 + number of ticks required
    """
    # Add minor ticks
    minor_locator_x = AutoMinorLocator(numx)
    ax.xaxis.set_minor_locator(minor_locator_x)
    minor_locator_y = AutoMinorLocator(numy)
    ax.yaxis.set_minor_locator(minor_locator_y)


def escapeForLatex(string):
    """ Escapes characters in a string which LaTeX will otherwise interpret as special characters, such as underscores.
        This prevents the error messages that the pyplot latex format otherwise gives.
    """
    b = string.replace('_','\_')
    b = b.replace('^','\^')
    return b

def escapeForLatexArray(arr):
    """ Escapes characters in an array of strings which LaTeX will otherwise interpret as special characters, such as underscores.
    """
    a2 = np.array([],dtype=str)
    for st in arr:
        b = escapeForLatex(st)
        a2 = np.append(a2, [b])
    return a2

def adjLimits(x,xerr=None,margins=0.1):
    """ For a given array and fraction, returns min minus fraction of range and max + fraction of range. 
        Intended for setting display limits that are data dependant.
    """
    if xerr is not None:
        rangex = (x+xerr).max() - (x-xerr).min()
        xlow = (x-xerr).min() - rangex * margins
        xhigh = (x+xerr).max() + rangex * margins
    else:
        rangex = x.max() - x.min()
        xlow = x.min() - rangex * margins
        xhigh = x.max() + rangex * margins
    return xlow, xhigh

def lightcurve(time, y, yerr=None, fmt='g.', fignum=1,title=None,   \
                xax=None, phase=False, btdb=False, mjd=False, jd=False, yax=None, mj=False, mags=False, colour=False, \
                xtoggle = True, ytoggle = True, \
                xleft=None, xright=None, yhigh=None, ylow=None, round=1, \
                sub=None,\
                noformat=False,\
                *args, **kwargs):
    """ Plots a lightcurve and labels the axes accordingly.
        'colour' can be a string describing which bands used, or simply True for multiple colours
    """
    if not noformat:
        try:
            formatGraph(fignum=fignum,*args, **kwargs)
        except TypeError:
            formatGraph(fignum=fignum)
    
    plt.figure(fignum)
        
    if btdb or mjd:
        toff = np.around(time.min(),round)  # Round = number of decimal places to go to, ie 0 => nearest integer
        time = time - toff          #tidying up time display
    
    if sub is None:
        ax = plt.gca()
    else:
        ax = sub
    
    ax.errorbar(time, y, yerr, fmt=fmt, capsize=0, *args, **kwargs)
    if title is not None:
        plt.suptitle(title)
    if xax is not None:
        ax.set_xlabel(xax)
    elif phase:
        ax.set_xlabel("Phase")
    elif btdb:
        ax.set_xlabel("BMJD (TDB) - %g"%(toff))
    elif mjd:   # 'or True' so default x axis
        ax.set_xlabel("MJD - %g"%(toff))
    if yax is not None:
        ax.set_ylabel(yax)
    elif mj:
        ax.set_ylabel("Flux density (mJy)")
    elif mags:
        ax.set_ylabel("Magnitude")
        ax.invert_yaxis()
    elif colour == True:
        ax.set_ylabel("Colour")
        ax.invert_yaxis()
    elif colour:
        ax.set_ylabel("%s colour"%(colour))
        ax.invert_yaxis()
        
    if xleft is not None:
        ax.set_xlim(xmin=xleft)
    if xright is not None:
        ax.set_xlim(xmax=xright)
    if ylow is not None:
        ax.set_ylim(ymin=ylow)
    if yhigh is not None:
        ax.set_ylim(ymax=yhigh)
        
    if xtoggle is False:
        ax.set_xlabel('')
        if sub is not None:
            sub.set_xticklabels([])
    if ytoggle is False:
        ax.set_ylabel('')
        if sub is not None:
            sub.set_yticklabels([])
            
    return plt.figure(fignum)
    
def saveAsTemp(num=None,fignum=None,ax=None,fig=None):
    """ Saves as a temporary file (temp.pdf and temp.png) in my home directory
    """
    if ax is not None:
        plt.sca(ax)
    elif fig is not None:
        plt.figure(fig.number)
    elif fignum is not None:
        plt.figure(fignum)
    if num is None:
        plt.savefig("/home/astro/phulbz/temp.pdf",dpi='figure',bbox_inches='tight')
        plt.savefig("/home/astro/phulbz/temp.png",dpi='figure',bbox_inches='tight')
    else:
        plt.savefig("/home/astro/phulbz/temp%d.pdf"%(num),dpi='figure',bbox_inches='tight')
        plt.savefig("/home/astro/phulbz/temp%d.png"%(num),dpi='figure',bbox_inches='tight')
        
    return True




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
        print perr
        print period
        print pherr
        print phaseCont
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
    
    print "Standard mag", smag, "in band", band
    
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



###### Ultracam associated functions        ####################

## File and input handling 

def readulog(address,oi=1,comp=2, ultraspec=False, singleCCD=False):
    """ Reads a .log file, as created by the Ultracam pipeline, 
    divides by a comparison star, and returns r,g,b objects containing 
    times, counts and errors.
    Assumes that object of interest is aperture 1 and comparison is aperture 2
    Comparison can be set to None to avoid dividing
    Usage: readulog(address)"""
    ulg = dnl.ulog.Ulog(address)
    
    if ultraspec or singleCCD:
        if comp is None:
            l = ulg.tseries(1,oi)           # the error propogation here is contained within tom's objects
            return l
        else:
            lc = ulg.tseries(1,comp)
            l = ulg.tseries(1,oi) / lc
            #l.y.data = l.y.data * np.mean(lc.y.data)
            
            return l
    else:
        if comp is not None:                #use ulg2mjys or ulg2mjysWithComp if comp mag known
            """Divide by comparison star"""
            rc = ulg.tseries(1,comp) 
            r = ulg.tseries(1,oi) / rc
            gc = ulg.tseries(2,comp)
            g = ulg.tseries(2,oi) / gc
            bc = ulg.tseries(3,comp)
            b = ulg.tseries(3,oi) / bc
            # the error propogation here is contained within tom's objects
            
        else:
            r = ulg.tseries(1,oi)           # the error propogation here is contained within tom's objects
            g = ulg.tseries(2,oi)
            b = ulg.tseries(3,oi)
        return r,g,b


def plotFromUlog(address,oi=1,comp=2, show=True, star=None, std=None, run=None, mag=False):
    """ Reads from a .log file (using readulog()) and plots """
    if star is None or std is None or run is None:  
        r,g,b = readulog(address,oi,comp)
        lightcurve(r.x.data,r.y.data,r.y.errors, fmt='r.', yax="Counts/comparison")
        lightcurve(g.x.data,g.y.data,g.y.errors, fmt='g.')
        ok = b.y.data != 0
        lightcurve(b.x.data[ok],b.y.data[ok],b.y.errors[ok], fmt='b.')
    else:
        rx, rxerr, ry, ryerr, gx, gxerr, gy, gyerr, bx, bxerr, by, byerr = ulg2mjysWithComp(address, star, std, run, oi, comp)
        if mag:
            ry, ryerr = magsFromMJy(ry, ryerr)
            gy, gyerr = magsFromMJy(gy, gyerr)
            by, byerr = magsFromMJy(by, byerr)
        lightcurve(rx, ry, ryerr, fmt='r.', mj=True)
        lightcurve(gx, gy, gyerr, fmt='g.')
        lightcurve(bx, by, byerr, fmt='b.')
        print "Mean red", ry.mean(), ry.std()
        print "Mean green", gy.mean(), gy.std()
        print "Mean blue", by.mean(), by.std()
    
    if show:
        plt.show()


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
    r,g,b = readulog("may.log") #as far as I can tell this is the only way to initialise a ulg object?
    g.x.data, g.x.errors, g.y.data, g.y.errors = readdat(addressg)
    r.x.data, r.x.errors, r.y.data, r.y.errors = readdat(addressg[:-4]+"r.dat")
    b.x.data, b.x.errors, b.y.data, b.y.errors = readdat(addressg[:-4]+"b.dat")
    return r, g, b


def ulg2mjys(address, star, std, run, srun, oi=1, comp=2):
    """ Reads a ulog file and finds magnitudes. 
        Returns rx, rxerr, r, rerr, gx, gxerr, g, gerr, bx, bxerr, b, berr
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
    
    # Fit a flat line to the comparison and multiply by the outcome -- this rather than averaging to allow for masking of clouds
    # You might want to play around with numMask and maskLim (the number of points to mask and the sigma limit to mask above)
    popt, pcov, mask = iterativeFit(flat, crx[okcr], cry[okcr], cryerr[okcr], numMask=10, maskLim=4, show=True)
    r = ry[okr] / cry[okcr] * popt[0]
    popt, pcov, mask = iterativeFit(flat, cgx[okcg], cgy[okcg], cgyerr[okcg], numMask=10, maskLim=4, show=True)
    g = gy[okg] / cgy[okcg] * popt[0]
    popt, pcov, mask = iterativeFit(flat, cbx[okcb], cby[okcb], cbyerr[okcb], numMask=10, maskLim=4, show=True)
    b = by[okb] / cby[okcb] * popt[0]
    
    
    
    rerr = np.sqrt((ryerr[okr]/ry[okr])**2 + (cryerr[okcr]/cry[okcr])**2) * r
    gerr = np.sqrt((gyerr[okg]/gy[okg])**2 + (cgyerr[okcg]/cgy[okcg])**2) * g
    berr = np.sqrt((byerr[okb]/by[okb])**2 + (cbyerr[okcb]/cby[okcb])**2) * b
    
    return rx[okr], rxerr[okr], r, rerr, gx[okg], gxerr[okg], g, gerr, bx[okb], bxerr[okb], b, berr


def ulg2mjysWithComp(address, star, std, run, oi=1, comp=2, ultraspec=False, band='g'):
    """ Reads a ulog file and finds magnitudes, using a comparison of known magnitude.
    """
    if not ultraspec:
        r,g,b = readulog(address, oi=oi, comp=None)
        cr, cg, cb = readulog(address, oi=comp, comp=None)
        
        rx, rxerr, ry, ryerr = mJyFromCountsWithComp(r, cr, star, std, run, band='r')
        gx, gxerr, gy, gyerr = mJyFromCountsWithComp(g, cg, star, std, run, band='g')
        bx, bxerr, by, byerr = mJyFromCountsWithComp(b, cb, star, std, run, band='b')   
        
        return rx, rxerr, ry, ryerr, gx, gxerr, gy, gyerr, bx, bxerr, by, byerr 
        
    if ultraspec:
        l = readulog(address, oi=oi, comp=None, ultraspec=True)
        cl = readulog(address, oi=comp, comp=None, ultraspec=True)
        
        
        lx, lxerr, ly, lyerr = mJyFromCountsWithComp(l, cl, star, std, run, band=band)
        
        return lx, lxerr, ly, lyerr
    

def parseDates(string):
    """ A work in progress. Handle different formats of dates and return a datetime.date object
    """
    monDict = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"July":7,"Aug":8,"Sep":9,"Sept":9,"Oct":10,"Nov":11,"Dec":12,"Sept":9}
    
    if re.compile('[A-Za-z]{3,4}-[0-9]+-[0-9]{4}').match(string) is not None:
        monstr, day, year = string.split('-')[:3]
        if monstr not in monDict:
            raise KeyError("Month %s was not recognised"%(monstr))
        date = datetime.date(int(year), monDict[monstr], int(day))
    else:
        raise KeyError("Format of %s was not recognised"%(string))
    return date
        

def parseTimes(string):
    """ A work in progress. Handle different formats of dates and return a datetime.date object
    """
    if re.compile('.*-[0-9]+:[0-9]+:[0-9]+[\.]*[0-9]*').match(string) is not None:
        hr, min, sec = string.split('-')[-1].split(':')[-3:]
        microsec = int((float(sec)%1)*1e6)
        sec = int(np.floor(float(sec)))
        time = datetime.time(int(hr),int(min),sec,microsec)
    else:
        raise KeyError("Format of %s not recognised"%(string))
    return time
        


def isotDateTime(string):
    """ Work in progress. Parses a string containing a time and returns it in 'isot' format.
    """
    date = parseDates(string)
    time = parseTimes(string)
    return date.isoformat()+'T'+time.isoformat()

def temp(string):
    isot = isotDateTime(string)
    return correctTimesAll(time.Time(isot, format='isot', scale='utc').mjd, coords, '200-in')[3] + 2400000.5
    


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


def correctTimesAll(mjd, coords, obsname):
    """ Applies barycentric corrections to a set of times in MJD, returning 
    time in a variety of forms"""
    
    """finds position of observatory"""
    tel,obs,longitude,latitude,height = \
        subs.observatory(obsname)
    
    """applies time correction"""
    ra,dec,syst = subs.str2radec(coords)
    tt,tdb,btdb,hutc,htdb,vhel,vbar = \
        sla.utc2tdb(mjd,longitude,latitude,height,ra,dec)
    
    return tt,tdb,btdb,hutc,htdb


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
    numMask = len(x) / 15           #max number of points to mask
    maskLim = 5                     #significance to mask to
    
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


def saveForLcurve(times,terr,y,yerr,sname,weights=None, subdivisions=None, trimNans=True, fmt='%15.9f %9.4g %7.5f %7.5f %f %d'):
    if weights is None:
        weights = np.ones(len(times))
    if subdivisions is None:
        subdivisions = np.ones(len(times))
    all = np.dstack((times,terr,y,yerr,weights,subdivisions))[0]
    if trimNans:
        all = all[np.isnan(y)==False]
    np.savetxt(sname, all, fmt=fmt)


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
        elif line.startswith('# #'):
            labels = line.split()[2:]
            labels = np.array(labels, dtype=str)
            break
    else:
        raise IOError("Unable to find labels in file!")
    #read data and select the desired parameter
    if not par in labels:
        raise KeyError("!!ERROR finding that parameter in log file!!")
    data = np.loadtxt(fname)
    if len(np.shape(data)) == 2:    #ie if 2d
        oi = data[:, (labels==par)]
    elif len(np.shape(data)) == 1:  #if 1d -- happens when the data file has only one column
        oi = data
    return oi

def readMcmcLogAll(fname,skip=1,last=None, nchop=None):
    """ Read in mcmc log, returning data in columns and a list of labels for those columns. Includes chisq, lp and pprob.
        If last is set to N, reads only the final N trials (eg if last=1 and there are two hundred walkers, reads the last 200 lines
    """
    # find params that varied
    labels, nwalker = None, None
    for line in open(fname):
        if line.startswith('##'):
            labels = line.split()[1:]
            labels = np.array(labels, dtype=str)
        if "nwalker" in line:
            nwalker = int(line.split()[-1])
        if not (nwalker is None) and not (labels is None):
            break
    else:
        nwalker = 1
        #raise IOError("Either unable to find labels or unable to find nwalker")
    
    # read data
    if nchop is not None:
        data = np.loadtxt(fname)[nchop::skip,:]
    elif last is not None:
        data = np.loadtxt(fname)[:last*nwalker:skip,:]
    else:
        data = np.loadtxt(fname)[::skip,:]
    return labels, data
    

def mcmcHist(data, plotHists=True, label='', chisq=None, cburn=None):
    """ Returns the mean and std of a dataset, as well correlation with a Gaussian distribution. Can optionally plot a histogram.
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
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)        # runs lroche
    output, err = process.communicate()     # output is the stdout of lroche
    exit_code = process.wait()      # exit code, in case needed

    ## find chisq in output with regex
    reg = re.compile(r"Weighted chi\*\*2\s\=\s.+")
    line = reg.search(output)
    if line is not None:
        chisq = float(line.group(0).split()[3][:-1])    #convert to string, split line, take number, trim comma, convert to float
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
    xmax = (bexn / bexm) ** (1 / bexm) * blen   # the distance from flux=0 to flux=max
    profile = (((x+xmax)/blen)**bexn)*np.exp(-((x+xmax)/blen)**bexm)
    return profile / profile.max()


############################################################


def findi(q, deltai):
    """ Finds value of i from q and deltai, assuming an eclipse phase width given in mgutils. If iangle would be > 90, sets = 90.
    """
    # By MJG
    avphi = 0.037134275925356952
    iangle = roche.findi(q, avphi) + deltai
    if (iangle > 90.0).any():
        iangle = 90.0
        
    print iangle, "from", q, deltai, "(q would give", roche.findi(q, avphi), ")"
    with open("log.tmp",'a') as tmp:
        tmp.write(str(iangle) + "from" + str( q) + str(deltai) + "(q would give" + str(roche.findi(q, avphi)) + ")")
    return iangle


########################

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
