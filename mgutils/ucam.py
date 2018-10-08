#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
from astropy import time
#from trm import dnl, subs, sla, roche
import re, datetime
from scipy.optimize import curve_fit
from scipy import ndimage
from sys import argv, exit

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
        print ("Mean red", ry.mean(), ry.std())
        print ("Mean green", gy.mean(), gy.std())
        print ("Mean blue", by.mean(), by.std())
    
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

