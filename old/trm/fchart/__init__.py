
"""
Module to make finding charts
"""

import os
import math as m
from trm import subs, sla
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
import astropy.io.fits as fits
import astLib.astWCS as astWCS
import urllib, urllib2

# Surveys to get from DSS
SURVEYS = {'POSS2RED'  : 'poss2ukstu_red',
           'POSS2BLUE' : 'poss2ukstu_blue',
           'POSS2IR'   : 'poss2ukstu_ir',
           'POSS1RED'  : 'poss1_red',
           'POSS1BLUE' : 'poss1_blue',
           'QUICKV'    : 'quickv'}
# colours
RED  = '#aa0000'
BLUE = '#0000aa'

def get_from_dss(ra, dec, size, fname=None, prec=1, survey='POSS2BLUE',
                 clobber=True):
    """
    Gets a FITS file from the DSS. Arguments::

    ra     : (string)
           ra (hh mm ss.ss)

    dec    : (string)
           dec (+/-dd mm ss.ss)

    fname  : (string)
           name of output (including .fits.gz). If not specified it will be
           constructed from the RA and Dec.

    prec   : (int)
           used to construct fname if not specified. See precision parameter of
           trm.subs.d2hms to understand this.

    size   : (float)
           field size (arcmin)

    Returns fname

    fname    -- the name of the FITS file created.
    """
    if not survey in SURVEYS:
        raise Exception('get_from_dss: survey name = ' +
                        survey + ' is invalid.')

    rad  = subs.hms2d(ra)
    decd = subs.hms2d(dec)
    if rad < 0. or rad >= 24 or decd < -90. or decd > +90.:
        raise Exception('get_from_dss: position out of range ra,dec = '
                        + ra + ' ' + dec)

    if fname is None:
        ras   = ra[:ra.find('.')].replace(' ','')
        decs  = dec[:dec.find('.')].replace(' ','')
        fname = ras + decs + '.fits.gz'
    elif not fname.endswith('.fits.gz'):
        raise Exception('get_from_dss: filename = ' + fname +
                        ' does not end with .fits.gz')

    if clobber or not os.path.exists(fname):

        # URL at STScI
        url = 'http://stdatu.stsci.edu/cgi-bin/dss_search'

        # encode the request string.
        data = urllib.urlencode({'v' : SURVEYS[survey], 'r': ra,
                                 'd' : dec, 'e' : 'J2000', \
                                 'h' : size, 'w' : size, 'f' : 'fits', \
                                 'c' : 'gz', 's' : 'on'})

        # get FITS data
        results = urllib2.urlopen(url, data)
        with open(fname, 'w') as f:
            f.write(results.read())

    return fname

def make_eso_chart(fname, cname, target, info, pid, pi, pos, field, scale,
                   stype='sec', source=None, angle=None, mark=True, s1=0.04,
                   s2=0.1, plo=50.0, phi=99.8, sloc=-0.43, pwidth=6,
                   fsize=12, aspect=0.6, pm=None, obsdate=None, 
                   lhead=0.02):
    """Make a chart suitable for VLT. Arguments::

    fname   : (string)
            fits file from DSS, e.g. 'PG1018-047.fits'

    cname   : (string)
            chart name (e.g. "name.pdf". The type of file is deduced 
            from the suffix. Either .pdf or .jpg are recognised)

    target  : (string)
            target name, e.g. 'PG1018-047'

    info    : (string or list of strings)
            extra information, e.g. 'Slit angle: parallactic'. 
            A list of strings will be printed line by line

    pid     : (string)
            programme ID, e.g. '088.D-0041(A)'

    pi      : (string)
            PI name, e.g. 'Marsh'

    pos     : (string)
            position, e.g. '10 21 10.6 -04 56 19.6' (assumed 2000)

    field   : (float)
            field width to display, arcmin, e.g. 1.0

    scale   : (float)
            size of scale to indicate in arcsec, e.g. 30

    stype   : (string)
            'sec' or 'min' to indicate how to mark the scale

    source  : (string)
            source of fits file, e.g. 'DSS blue'. Ignore for automatic version.

    angle   : (float)
            indicator / slit angle to display, degrees. None to ignore.

    mark    : (bool)
            True to indicate object

    s1      : (float)
            fraction of field for start of slit and object markers
            (displaced from object)

    s2      : (float)
            fraction of field for end of slit and object markers
            (displaced from object)

    plo     : (float)
            Low image display level, percentile

    phi     : (float)
            High image display level, percentile

    sloc    : (float) 
            vertical location of scale bar in terms of 'field'

    pwidth  : (float)
            (approx) plot width in inches

    fsize   : (int)
            fontsize, pt

    aspect  : (float)
            aspect (vertical/horizontal) [should be < 1]

    pm      : (tuple)
            proper motion in RA, and dec, arcsec/year in both
            coords. This is used to correct the predicted position of the
            target in the image by finding the date on which the image was
            taken.

    obsdate : (string)
            if you set pm, then if you also set date, an arrow
            arrow will be drawn from target to its expected position at
            obsdate and the chart will be centred on the head of the
            arrow. obsdate should be in YYYY-MM-DD format. 

    lhead   : (float)
            length of arrow head as fraction of field. If this turns out
            to be longer than the entire arrow, no arrow is drawn.

    Returns (ilo,ihi) display levels used
    """

    # ra, dec in degrees
    ra,dec,sys = subs.str2radec(pos)
    ra *= 15

    # Read data
    hdulist   = fits.open(fname)
    data      = hdulist[0].data
    head      = hdulist[0].header
    hdulist.close()

    arrow = False
    if pm:
        # try to correct position to supplied date or
        # date of chart

        if 'DATE-OBS' in head:
            dobs    = head['DATE-OBS']
            year    = int(dobs[:4])
            month   = int(dobs[5:7])
            day     = int(dobs[8:10])
            deltat  = (sla.cldj(year,month,day) - sla.cldj(2000,1,1))/365.25
            dra     = pm[0]*deltat/np.cos(np.radians(dec))/3600.
            ddec    = pm[1]*deltat/3600.
            ra  += dra
            dec += ddec
            if obsdate:
                yearp,monthp,dayp = obsdate.split('-')
                deltat  = (sla.cldj(int(yearp),int(monthp),int(dayp)) - \
                               sla.cldj(year,month,day))/365.25
                dra     = pm[0]*deltat/np.cos(np.radians(dec))/3600.
                ddec    = pm[1]*deltat/3600.
                ra += dra
                dec += ddec
                arrow   = True
        else:
            print 'WARNING: Could not find DATE-OBS in header'

    # Read WCS info
    wcs = astWCS.WCS(fname)
    dx  = 60.*wcs.getXPixelSizeDeg()
    dy  = 60.*wcs.getYPixelSizeDeg()
    rot = wcs.getRotationDeg()
    if rot > 180.:
        rot -= 360.
    if not wcs.coordsAreInImage(ra,dec):
        print 'WARNING: coordinates not inside image'

    # pixel position corresponding to desired ra and dec
    x,y = wcs.wcs2pix(ra, dec)
    if arrow:
        if not wcs.coordsAreInImage(ra-dra,dec-ddec):
            print 'WARNING: End of proper motion arrow is not inside image'
        xa,ya = wcs.wcs2pix(ra-dra, dec-ddec)

    ny, nx = data.shape

    # plot limits
    limits = (dx*(-0.5-x),dx*(nx-0.5-x),dy*(-0.5-y),dy*(ny-0.5-y))

    if source is None:
        if 'SURVEY' in head:
            source = head['SURVEY']
            if source == 'POSSII-F':
                source = 'POSS-II red'
            elif source == 'POSSII-J':
                source = 'POSS-II blue'
            elif source == 'POSSII-N':
                source = 'POSS-II ir'
            elif source == 'POSSI-E':
                source = 'POSS-I red'
            elif source == 'POSSI-O':
                source = 'POSS-I blue'
            elif source == 'SERC-I':
                source = 'SERC ir'
            elif source == 'SERC-J':
                source = 'SERC blue'
            elif source == 'AAO-SES':
                source = 'AAO red'
            elif source == 'AAO-GR':
                source = 'AAO red'
        else:
            source = 'UNKNOWN'

    # Start plotting
    fig  = plt.figure(figsize=(pwidth,pwidth))
    axes = fig.add_subplot(111,aspect='equal')
    mpl.rc('font', size=fsize)

    # Derive the plot range
    ilo  = stats.scoreatpercentile(data.flat, plo)
    ihi  = stats.scoreatpercentile(data.flat, phi)

    # Plot
    plt.imshow(data, vmin=ilo, vmax=ihi, cmap=cm.binary, extent=limits,
               interpolation='nearest', origin='lower')
    axes.autoscale(False)

    # draw slit angle
    if angle is not None:
        ca = m.cos(m.radians(angle+rot))
        sa = m.sin(m.radians(angle+rot))
        t1 = s1*field
        t2 = s2*field
        plt.plot([-sa*t1,-sa*t2],[+ca*t1,+ca*t2],BLUE,lw=3)
        plt.plot([+sa*t1,+sa*t2],[-ca*t1,-ca*t2],BLUE,lw=3)

    # Mark object
    if mark:
        t1 = s1*field
        t2 = s2*field
        plt.plot([t1,t2],[0,0],RED,lw=3)
        plt.plot([0,0],[t1,t2],RED,lw=3)

    # draw arrow
    if arrow:
        # Slightly complicated because matplotlib
        # arrow stupidly starts drawing the head of the
        # arrow at the end point so we need to shrink the
        # arrow. If it becomes negative, don't draw at all,
        # just issue a warning
        delx, dely = dx*(x-xa), dy*(y-ya)
        length = m.sqrt(delx**2+dely**2)
        hl = field*lhead
        lnew = length-hl
        if lnew > 0:
            shrink = lnew/length
            plt.arrow(-delx, -dely, shrink*delx, shrink*dely,
                       head_width=0.5*hl, head_length=hl)
        else:
            print 'WARNING: arrow head too long; arrow not drawn'

    # Draw scale bar
    if stype == 'sec' or stype == 'min':
        if stype == 'sec':
            plt.text(0, (sloc-0.05)*field, str(scale) + ' arcsec',
                     horizontalalignment='center',color=RED)
        elif stype == 'min':
            plt.text(0, (sloc-0.05)*field, str(scale/60) + ' arcmin',
                     horizontalalignment='center',color=RED)

        scale /= 60.
        plt.plot([-0.5*scale,+0.5*scale],[sloc*field,sloc*field],RED,lw=3)
        plt.plot([+0.5*scale,+0.5*scale],[(sloc+0.02)*field,(sloc-0.02)*field],
                 RED,lw=3)
        plt.plot([-0.5*scale,-0.5*scale],[(sloc+0.02)*field,(sloc-0.02)*field],
                 RED,lw=3)

    # North-East indicator. ev, nv = East / North vectors
    xc, yc = 0.4*field, -0.4*field
    rot   = m.radians(rot)
    rmat  = np.matrix(((m.cos(rot),-m.sin(rot)),(m.sin(rot),m.cos(rot))))

    nv = np.array((0.,0.2*field))
    nv = nv.reshape((2,1))
    nv = rmat*nv
    nv = np.array(nv).flatten()

    ev = np.array((-0.2*field,0.))
    ev = ev.reshape((2,1))
    ev = rmat*ev
    ev = np.array(ev).flatten()

    plt.plot([xc,xc+nv[0]],[yc,yc+nv[1]],RED,lw=3)
    plt.text(xc+1.1*nv[0], yc+1.1*nv[1], 'N', horizontalalignment='center',
             color=RED)
    plt.plot([xc,xc+ev[0]],[yc,yc+ev[1]],RED,lw=3)
    plt.text(xc+1.15*ev[0], yc+1.1*ev[1], 'E', verticalalignment='center',
             color=RED)

    # finally the textual info with a helper function
    def ptext(x, y, field, tstr):
        """
        Converts relative (0-1) x,y limits into data coords and plots
        a string.
        """
        xd = -field/2.+field*x
        yd = -field/2.+field*y
        plt.text(xd, yd, tstr, horizontalalignment='left',color=BLUE)

    xoff = 0.02
    dely = 0.035
    yoff = 0.96

    if pid != '':
        ptext(xoff, yoff, field, pid)
        yoff -= dely

    if pi != '':
        ptext(xoff, yoff, field, 'PI: ' + pi)
        yoff -= 1.5*dely

    if target != '':
        ptext(xoff, yoff, field, 'Target: ' + target)
        yoff -= 1.5*dely

    rah,ram,ras,decd,decm,decs = pos.split()
    ptext(xoff, yoff, field, 'RA (2000) : ' + rah + ' ' + ram + ' ' + ras)

    yoff -= dely
    ptext(xoff, yoff, field, 'Dec (2000): ' + decd + ' ' + decm + ' ' + decs)

    yoff -= 1.5*dely
    ptext(xoff, yoff, field, "Field: " + str(field) + "' x " + str(field) + "'")

    if 'DATE-OBS' in head:
        yoff -= dely
        ptext(xoff, yoff, field, 'Date: ' + head['DATE-OBS'][:10])

    if source != '':
        yoff -= dely
        ptext(xoff, yoff, field, 'Survey: ' + source)

    yoff -= 2.*dely
    if info is not None:
        if isinstance(info, list):
            for line in info:
                ptext(xoff, yoff, field, line)
                yoff -= dely
        else:
            ptext(xoff, yoff,  field, info)

    plt.xlim(-field/2.,field/2.)
    plt.ylim(-field/2.,field/2.)
    plt.savefig(cname,bbox_inches='tight')
    return (ilo,ihi)

