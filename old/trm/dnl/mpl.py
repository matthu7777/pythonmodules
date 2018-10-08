"""
Code to enable plotting of Dset objects with matplotlib commands
"""
from base import *
from core import *

def plabels(dset, axes, xoff=0., yoff=0., control='XY',
            xform='.5f', yform='.5f', **kwargs):
    """
    Adds axis labels to a plot using default labels and
    units contained in a Dset.

    Arguments::

    dset : the Dset providing the information

    axes : matplotlib axes instance

    xoff : offset in X applied when plotting Dset

    yoff : offset in Y applied when plotting Dset

    control : string to specify which out of the X axis label, Y axis
          label and title to plot.

    xform : format to use to show xoff

    yform : format to use to show yoff

    kwargs : to feed through to the label routines
    """
    ctrl = control.upper()

    if ctrl.find('X') > -1:
        xlabel = dset.x.label
        if xoff < 0.:
            xlabel += (' + {0:' + xform + '}').format(-xoff)
        elif xoff > 0.:
            xlabel += (' - {0:' + xform + '}').format(xoff)
        if dset.x.units is not None and dset.x.units != '':
            xlabel += ' [' + dset.x.units + ']'
        plt.xlabel(xlabel)

    if ctrl.find('Y') > -1:
        ylabel = dset.y.label
        if yoff < 0.:
            ylabel += (' + {0:' + yform + '}').format(-yoff)
        elif yoff > 0.:
            ylabel += (' - {0:' + yform + '}').format(yoff)
        if dset.y.units is not None and dset.y.units != '':
            ylabel += ' [' + dset.y.units + ']'
        plt.ylabel(ylabel)

def ppoints(dset, axes, symb='.', dcol='g', erry=True,
            errx=False, xoff=0., yoff=0., pbad=False, ecol=None,
            elwidth=None):
    """
    Plots data as points with (optionally) errors. By default Y
    errors will be plotted but X errors will not, but can be switched
    on.

    Arguments::

    dset : the Dset to plot

    axes : matplotlib axes instance

    symb : symbol

    dcol : data colour

    erry : plot errors in Y

    errx : plot errors in X

    xoff : offset to remove from X

    yoff : offset to remove from Y

    pbad : plots bad as opposed to good data

    ecol : colour for errors (None = same as data)

    elwidth : line width for error bars
    """

    nplot = dset.nbad if pbad else dset.ngood

    if nplot:
        ex = errx and dset.x.has_errors
        ey = erry and dset.y.has_errors

        # only plot anything if there are points to plot
        ok = dset.bad if pbad else dset.good
        if ey and ex:
            axes.errorbar(dset.x.data[ok]-xoff, dset.y.data[ok]-yoff,
                          dset.y.errors[ok], dset.x.errors[ok],
                          fmt=symb+dcol, ecolor=ecolor, elinewidth=elwidth,
                          capsize=0)
        elif ey:
            axes.errorbar(dset.x.data[ok]-xoff, dset.y.data[ok]-yoff,
                          dset.y.errors[ok],fmt=symb+dcol, ecolor=ecol,
                          elinewidth=elwidth, capsize=0)
        elif ex:
            axes.errorbar(dset.x.data[ok]-xoff, dset.y.data[ok]-yoff,
                          None, dset.x.errors[ok],fmt=symb+dcol, ecolor=ecol,
                          elinewidth=elwidth, capsize=0)
        else:
            axes.plot(dset.x.data[ok]-xoff, dset.y.data[ok]-yoff,
                      symb, color=dcol)



def pbin(dset, axes, dcol='g', erry=True, xoff=0., yoff=0.,
         pbad=False, ecol=None, dlwidth=1.0, elwidth=1.0):
    """
    Plot data as binned data with (optionally) errors. By default Y
    errors will be plotted but X errors will not, but can be switched
    on. Bad data are skipped.

    Arguments::

    dset : Dset
         the Dset to plot

    axes :
         matplotlib axes instance

    dcol :
         data colour

    erry : (bool)
         plot errors in Y

    xoff : (float)
         offset to remove from X

    yoff : (float)
         offset to remove from Y [NB it does not get removed
         from the errors]

    pbad : (bool)
         plots bad as opposed to good data

    ecol : 
         colour for errors (None = same as data)

    dlwidth : (float) 
            line width for data

    elwidth : (float)
            line width for error bars
    """

    nplot = dset.nbad if pbad else dset.ngood

    if nplot:
        # only plot anything if there are points to plot
        ok = dset.bad if pbad else dset.good

        my = np.ma.masked_array(dset.y.data,~ok)
        axes.step(dset.x.data-xoff, my-yoff, color=dcol, where='mid',
                  linewidth=dlwidth)

        if erry and dset.y.has_errors:
            my = np.ma.masked_array(dset.y.errors,~ok)
            axes.step(dset.x.data-xoff, my, color=dcol, where='mid',
                      linewidth=elwidth)

