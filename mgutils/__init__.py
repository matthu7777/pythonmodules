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

#### Gaia14aae shortcuts

coords = '16:11:33.97 +63:08:31.8'
obsname = 'WHT'                 
t0 = 57153.6890966
t0err = 4.02767506355e-07
period = 0.0345195708393
perr = 7.85384131425e-11


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
    if type(arr)==list:
        arr = np.array(arr)
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
        Replicates some of the astropy tables functionality, but that is probably better
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

def x2(x,a,b,c):
    return a*((x-b))**2 + c

def x3(x,a,b,d,c):
    return a*(d*(x-b))**3 + c

def x3plusx2(x,a,b,d,e,f,c):
    return a*(d*(x-b))**3 + e*(f*(x-b))**2 + c

def x2plusx(x,a,b,e,c):
    return a*((x-b))**2 + e*(x-b) + c

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
        print ("Functional approach has been passed the wrong number of arguments (%d excluding the function pointer)".format(len(args)))
        exit()
    #split variables and errors into 2 arrays
    for i, arg in enumerate(args):
        if i % 2 == 0:
            var.append(float(arg))
        else:
            err.append(float(arg))
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
    
    print ("Error contributions:", diffs)
    # combine error contributions in quadrature
    return result, quad(diffs)
    
def iterativeFit(funct, x, y, yerr=None, numMask=0, maskLim=3, p0=None, inmask=None, show=False, silent=False, verbose=False, bounds=None, chisqIters=0,\
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
    if yerr is None or (np.array(yerr) == 0).all():
        yerr = np.zeros(len(y))
        sigma = None
    else:
        sigma = yerr[mask]
    
    x = np.array(x) # in case passed in as non-numpy arrays
    y = np.array(y)
    
    
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
                print ("Masking point", np.argwhere(diffs == diffs[mask].max()), "at sig", diffs[mask].max())
        else:
            if not silent:
                if i > 0:
                    print ("iterativeFit masked",i,"sig",diffs[mask==False].min())
                else:
                    print ("iterativeFit masked 0")
            break
    
    
    ## Iterate over number of chisq iterations
    scaled_yerr = yerr 
    for j in range(chisqIters):
        csq = chisq(funct(x[mask],*popt), y[mask], scaled_yerr[mask])
        wnok = np.sum(mask) - len(popt)
        scaled_yerr = scaled_yerr * np.sqrt(csq/wnok)
        popt, pcov = curve_fit(funct, x[mask], y[mask], sigma=scaled_yerr[mask], p0=p0)
        if verbose:
            print ("Chisq iter", j, ":", csq, "->", chisq(funct(x[mask],*popt), y[mask], scaled_yerr[mask]))
    
    
    if verbose and not silent:
        errs = np.sqrt(np.diag(pcov))
        print ("Results:", popt)
        print ("Errors: ", errs)
        print ("Corr.ns:")
        for j in range(len(popt)):
            print ("            "*j, pcov[j+1:,j] / errs[j] / errs[j+1:])
        
        if (yerr==0).all():
            print ("Lsq", np.sum((funct(x[mask],*popt) - y[mask])**2), "wnok", len(x) - len(popt))
        else:
            print ("Chisq", chisq(funct(x[mask],*popt), y[mask], scaled_yerr[mask]), "wnok", len(x) - len(popt))
        
    
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


def addNoise(y,e):
    """ Add Gaussian errors to a set of data
    """
    return y + np.random.normal(0,e,len(y))
    

############  Science things        ###########

#(more specific science things are found in mgutils.sci.py

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

def formatGraph(fignum=1,suptitle=None,xlabel=None,ylabel=None,returnFig=False,poster=False,*args,**kwargs):
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






if __name__ == "__main__":
    print ("Hello world")
