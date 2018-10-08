"""
Sub-package to plot Dsets with pgplot
"""
from base import *
from core import *
from ppgplot import *

def ppoints(dset, symb=17, dcol=1, erry=True, errx=False,
            xoff=0., yoff=0., pbad=False, ecol=2, elwidth=1, dsize=1.):
        """
        Plots data as point with (optionally) errors. The current line width and
        colour index may be altered by this routine.

        Arguments::

          dset : the Dset to plot

          symb : plot symbol, -1 to 127 inclusive.

          dcol : colour index of data

          xoff  : offset in X to subtract

          yoff  : offset in Y to subtract

          pbad  : plot the bad (as opposed to good) data
        """
        nplot = dset.nbad if pbad else dset.ngood

        if nplot:
            ex = errx and dset.x.has_errors
            ey = erry and dset.y.has_errors

            # only plot anything if there are points to plot
            ok = dset.bad if pbad else dset.good
            pgsci(ecol)
            if ey and ex:
                pgslw(elwidth)
                pgerrx(dset.x.data[ok]-xoff-dset.x.errors[ok],
                       dset.x.data[ok]-xoff+dset.x.errors[ok],
                       dset.y.errors[ok], 0)
                pgerry(dset.x.data[ok], dset.y.data[ok]-xoff-dset.y.errors[ok],
                       dset.y.data[ok]-xoff+dset.y.errors[ok], 0)
            elif ey:
                pgerry(dset.x.data[ok], dset.y.data[ok]-xoff-dset.y.errors[ok],
                       dset.y.data[ok]-xoff+dset.y.errors[ok], 0)
            elif ex:
                pgerrx(dset.x.data[ok]-xoff-dset.x.errors[ok],
                       dset.x.data[ok]-xoff+dset.x.errors[ok],
                       dset.y.errors[ok], 0)

            pgsci(dcol)
            pgsch(dsize)
            pgpt(dset.x.data[ok]-xoff, dset.y.data[ok]-yoff, symb)


def pline(dset, dcol=1, xoff=0., yoff=0., pbad=False, dlwidth=1):
        """
        Plots data as a joined line. The current line width and
        colour index may be altered by this routine.

        Arguments::

          dset : the Dset to plot

          dcol : colour index of data

          xoff  : offset in X to subtract

          yoff  : offset in Y to subtract

          pbad  : plot the bad (as opposed to good) data

          dlwidth : line width to use
        """

        nplot = dset.nbad if pbad else dset.ngood

        if nplot:
            # only plot anything if there are points to plot
            ok = dset.bad if pbad else dset.good
            pgsci(dcol)
            pgslw(dlwidth)
            pgline(dset.x.data[ok]-xoff, dset.y.data[ok]-yoff)



