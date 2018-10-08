#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
from astropy import time
from trm import dnl, sla, roche
import re, datetime
from scipy.optimize import curve_fit
from scipy import ndimage
from sys import argv, exit



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
