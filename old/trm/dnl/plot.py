"""
Code to enable plotting of Dsets
"""
from base import *
from core import *

class MplDset(Dset):
    """
    Dset with matplotlib functionality.
    """

    def __init__(self,  x, y, head=None, good=None, ptype=Dset.POINTS):
        """
        Creates a new MplDset.

        Arguments::

        x : Axis or ndarray
            The x-axis data and errors

        y : Axis or ndarray
            The y-axis data and errors

        head : astropy.fits.Header
               Ordered dictionary of header information

        good : 1D boolean ndarray or None
               Mask array to say which data are OK.

        ptype : [POINTS, LINE, BAR]
                default type to use when plotting.
        """
        Dset.__init__(self,x,y,head,good,ptype)

    @classmethod
    def fromDset(self, dset):
        """
        Given a Dset returns an MplDset
        """
        dset.__class__ = MplDset
        return dset

    def ppoints(self, axes, symb='.', dcol='g', mcol=None, erry=True, errx=False, 
                xoff=0., yoff=0., pbad=False, ecol=None, elwidth=None):
        """
        Plots data as points with (optionally) errors.

        Arguments::

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

        nplot = self.nbad if pbad else self.ngood

        if nplot:
            ex = errx and self.x.has_errors
            ey = erry and self.y.has_errors

            # only plot anything if there are points to plot
            ok = self.bad if pbad else self.good
            if ey and ex:
                axes.errorbar(self.x.data[ok]-xoff, self.y.data[ok]-yoff,
                              self.y.errors[ok], self.x.errors[ok],
                              fmt=symb+dcol, ecolor=ecolor, elinewidth=elwidth,
                              capsize=0)
            elif ey:
                axes.errorbar(self.x.data[ok]-xoff, self.y.data[ok]-yoff,
                              self.y.errors[ok],fmt=symb+dcol, ecolor=ecol, elinewidth=elwidth,
                              capsize=0)
            elif ex:
                axes.errorbar(self.x.data[ok]-xoff, self.y.data[ok]-yoff,
                              None, self.x.errors[ok],fmt=symb+dcol, ecolor=ecol, elinewidth=elwidth,
                              capsize=0)
            else:
                axes.plot(self.x.data[ok]-xoff, self.y.data[ok]-yoff, symb)

class PgDset(Dset):
    """
    Dset with PGPLOT functionality. NB You must have opened the device and set up axes
    appropriately to use the methods of this class, and if you want have modified the 
    colour indices to your taste.
    """

    def __init__(self,  x, y, head=None, good=None, ptype=Dset.POINTS):
        """
        Creates a new Dset.

        Arguments::

        x : Axis or ndarray
            The x-axis data and errors

        y : Axis or ndarray
            The y-axis data and errors

        head : astropy.fits.Header
               Ordered dictionary of header information

        good : 1D boolean ndarray or None
               Mask array to say which data are OK.

        ptype : [POINTS, LINE, BAR]
                default type to use when plotting.
        """
        Dset.__init__(self,x,y,head,good,ptype)

    @classmethod
    def fromDset(self, dset):
        """
        Given a Dset returns an MplDset
        """
        dset.__class__ = MplDset
        return dset


    def ppoints(self, axes, symb=17, dcol=1, erry=True, errx=False, xoff=0., yoff=0., pbad=False,
                ecol=2, elwidth=1, dsize=1.):
        """
        Plots data as point with (optionally) errors. The current line width and
        colour index may be altered by this routine.

        Arguments::


          symb : plot symbol, -1 to 127 inclusive.

          dcol : colour index of data

          xoff  : offset in X to subtract

          yoff  : offset in Y to subtract

          pbad  : plot the bad (as opposed to good) data
        """
        nplot = self.nbad if pbad else self.ngood

        if nplot:
            ex = errx and self.x.has_errors
            ey = erry and self.y.has_errors

            # only plot anything if there are points to plot
            ok = self.bad if pbad else self.good
            pgsci(ecol)
            if ey and ex:
                pgslw(elwidth)
                pgerrx(self.x.data[ok]-xoff-self.x.errors[ok], self.x.data[ok]-xoff+self.x.errors[ok],
                       self.y.errors[ok], 0)
                pgerry(self.x.data[ok], self.y.data[ok]-xoff-self.y.errors[ok],
                       self.y.data[ok]-xoff+self.y.errors[ok], 0)
            elif ey:
                pgerry(self.x.data[ok], self.y.data[ok]-xoff-self.y.errors[ok],
                       self.y.data[ok]-xoff+self.y.errors[ok], 0)
            elif ex:
                pgerrx(self.x.data[ok]-xoff-self.x.errors[ok], self.x.data[ok]-xoff+self.x.errors[ok],
                       self.y.errors[ok], 0)

            pgsci(dcol)
            pgsch(dsize)
            pgpt(self.x.data[ok]-xoff, self.y.data[ok]-yoff, symb)
