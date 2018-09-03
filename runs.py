#!/usr/bin/env python

import numpy as np
import mgutils as mg
import copy

runlist = ["jan14", "jan15", "jan16", "jan17", "may", "june"]

monthDict = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

class Run:
    def __init__(self, locn):
        self.locn = locn
    
    def ext(self, band):
        if band == 'u' or band == 'b':
            return self.extb
        if band == 'g':
            return self.extg
        if band == 'r' or band == 'i' or band == 'z':
            return self.extr
            
    def exterr(self, band):
        if band == 'u' or band == 'b':
            return self.extberr
        if band == 'g':
            return self.extgerr
        if band == 'r' or band == 'i' or band == 'z':
            return self.extrerr
    
    def dateMNRAS(self):
        y,m,d = self.date.split('-')
        mStr = monthDict[int(m)]
        return " ".join((d,mStr,y))
    
may = Run("/storage/astro2/phulbz/ucam/2015-05-23/run025")
may.name = "may"
may.obsname = 'WHT'
may.filters = "ugr"
may.date = "2015-05-23"
may.numec = 6
may.exp = 2.472819          #exposures in seconds
may.extr = 0.0841525
may.extg = 0.14845998       
may.extb = 0.4290130
may.extrerr = 0.0005506
may.extgerr = 0.000474460
may.extberr = 0.0025790

maystd1 = copy.deepcopy(may)
maystd1.locn = "/storage/astro2/phulbz/ucam/2015-05-23/run013"
maystd1.exp = 0.4

maystd2 = copy.deepcopy(may)
maystd2.locn = "/storage/astro2/phulbz/ucam/2015-05-23/run026"
maystd2.exp = 0.2


jan16 = Run("/storage/astro2/phulbz/ucam/2015-01-16/run018")
jan16.name = "jan16"
jan16.obsname = 'WHT'
jan16.filters = "ugr"
jan16.date = "2015-01-16"
jan16.numec = 5
jan16.extr = 0.0732118
jan16.extg = 0.146807
jan16.extb = 0.4832335
jan16.extrerr = 0.00021739
jan16.extgerr = 0.00021870
jan16.extberr = 0.00093947

jan16std1 = copy.deepcopy(jan16)
jan16std1.locn = "/storage/astro2/phulbz/ucam/2015-01-16/run008"


jan14 = Run("/storage/astro2/phulbz/ucam/2015-01-14/run018")
jan14.name = "jan14"
jan14.obsname = 'WHT'
jan14.filters = "ugr"
jan14.date = "2015-01-14"
jan14.numec = 3
jan14.extr = 0.08510260
jan14.extg = 0.1935477
jan14.extb = 0.52397243
jan14.extrerr = 0.000492657
jan14.extgerr = 0.00043945
jan14.extberr = 0.002270


jan15 = Run("/storage/astro2/phulbz/ucam/2015-01-15/run023")
jan15.name = "jan15"
jan15.obsname = 'WHT'
jan15.filters = "ugr"
jan15.date = "2015-01-15"
jan15.numec = 4
jan15.extr = 0.0911113
jan15.extg = 0.1531842
jan15.extb = 0.4893782
jan15.extrerr = 0.00029928
jan15.extgerr = 0.0002995
jan15.extberr = 0.0015138


jan17= Run("/storage/astro2/phulbz/ucam/2015-01-17/run015")
jan17.name = "jan17"
jan17.obsname = 'WHT'
jan17.filters = "ugr"
jan17.date = "2015-01-17"
jan17.numec = 3
jan17.extr = 0.085907
jan17.extg = 0.182898
jan17.extb = 0.591442
jan17.extrerr = 0.0006834
jan17.extgerr = 0.000619529
jan17.extberr = 0.00310757



june = Run("/storage/astro2/phulbz/ucam/2015-06-22/run024")
june.name = "june"
june.obsname = 'WHT'
june.filters = "ugr"
june.date = "2015-06-22"
june.numec = 4
june.extr = 0.068667
june.extg = 0.142672
june.extb = 0.4406133
june.extrerr = 0.00024594
june.extgerr = 0.0002086
june.extberr = 0.0012520


mar12 = Run("/storage/astro2/phulbz/uspec/2016-03-12/run028")
mar12.name = "mar12"
mar12.obsname = 'TNT'
mar12.filters = "KG5"
mar12.date = "2016-03-12"
mar12.numec = 1

mar13 = Run("/storage/astro2/phulbz/uspec/2016-03-13/run044")
mar13.name = "mar13"
mar13.obsname = 'TNT'
mar13.filters = "KG5"
mar13.date = "2016-03-13"
mar13.numec = 2

mar14 = Run("/storage/astro2/phulbz/uspec/2016-03-14/run055")
mar14.name = "mar14"
mar14.obsname = 'TNT'
mar14.filters = "KG5"
mar14.date = "2016-03-14"
mar14.numec = 3

mar15 = Run("/storage/astro2/phulbz/uspec/2016-03-15/run031")
mar15.name = "mar15"
mar15.obsname = 'TNT'
mar15.filters = "g"
mar15.date = "2016-03-15"
mar15.numec = 2

aug06 = Run("/storage/astro2/phulbz/gaia14aae/chimera")
aug06.name = "aug06"
aug06.obsname = '200-in'
aug06.filters = "gr"
aug06.date = "2016-08-06"
aug06.numec = 7

aug07 = Run("/storage/astro2/phulbz/gaia14aae/chimera")
aug07.name = "aug07"
aug07.obsname = '200-in'
aug07.filters = "gr"
aug07.date = "2016-08-07"
aug07.numec = 6

aug08 = Run("/storage/astro2/phulbz/gaia14aae/chimera")
aug08.name = "aug08"
aug08.obsname = '200-in'
aug08.filters = "gr"
aug08.date = "2016-08-08"
aug08.numec = 6

feb21 = Run("/storage/astro2/phulbz/uspec/2016-03-15/gaia14aae")
feb21.name = "feb21"
feb21.obsname = 'TNT'
feb21.filters = "KG5"
feb21.date = "2017-02-21"
feb21.numec = 1



eclnums = {\
"jan14-r-1" : 1, \
"jan14-r-2" : 2, \
"jan14-r-3" : 3, \
"jan15-r-1" : 4, \
"jan15-r-2" : 5, \
"jan15-r-3" : 6, \
"jan15-r-4" : 7, \
"jan16-r-1" : 8, \
"jan16-r-2" : 9, \
"jan16-r-3" : 10, \
"jan16-r-4" : 11, \
"jan16-r-5" : 12, \
"jan17-r-1" : 13, \
"jan17-r-2" : 14, \
"jan17-r-3" : 15, \
"may-r-1" : 16, \
"may-r-2" : 17, \
"may-r-3" : 18, \
"may-r-4" : 19, \
"may-r-5" : 20, \
"may-r-6" : 21, \
"june-r-1" : 22, \
"june-r-2" : 23, \
"june-r-3" : 24, \
"june-r-4" : 25, \
"jan14-b-1" : 1, \
"jan14-b-2" : 2, \
"jan14-b-3" : 3, \
"jan15-b-1" : 4, \
"jan15-b-2" : 5, \
"jan15-b-3" : 6, \
"jan15-b-4" : 7, \
"jan16-b-1" : 8, \
"jan16-b-2" : 9, \
"jan16-b-3" : 10, \
"jan16-b-4" : 11, \
"jan16-b-5" : 12, \
"jan17-b-1" : 13, \
"jan17-b-2" : 14, \
"jan17-b-3" : 15, \
"may-b-1" : 16, \
"may-b-2" : 17, \
"may-b-3" : 18, \
"may-b-4" : 19, \
"may-b-5" : 20, \
"may-b-6" : 21, \
"june-b-1" : 22, \
"june-b-2" : 23, \
"june-b-3" : 24, \
"june-b-4" : 25, \
"jan14-g-1" : 1, \
"jan14-g-2" : 2, \
"jan14-g-3" : 3, \
"jan15-g-1" : 4, \
"jan15-g-2" : 5, \
"jan15-g-3" : 6, \
"jan15-g-4" : 7, \
"jan16-g-1" : 8, \
"jan16-g-2" : 9, \
"jan16-g-3" : 10, \
"jan16-g-4" : 11, \
"jan16-g-5" : 12, \
"jan17-g-1" : 13, \
"jan17-g-2" : 14, \
"jan17-g-3" : 15, \
"may-g-1" : 16, \
"may-g-2" : 17, \
"may-g-3" : 18, \
"may-g-4" : 19, \
"may-g-5" : 20, \
"may-g-6" : 21, \
"june-g-1" : 22, \
"june-g-2" : 23, \
"june-g-3" : 24, \
"june-g-4" : 25, \
"jan14-1" : 1, \
"jan14-2" : 2, \
"jan14-3" : 3, \
"jan15-1" : 4, \
"jan15-2" : 5, \
"jan15-3" : 6, \
"jan15-4" : 7, \
"jan16-1" : 8, \
"jan16-2" : 9, \
"jan16-3" : 10, \
"jan16-4" : 11, \
"jan16-5" : 12, \
"jan17-1" : 13, \
"jan17-2" : 14, \
"jan17-3" : 15, \
"may-1" : 16, \
"may-2" : 17, \
"may-3" : 18, \
"may-4" : 19, \
"may-5" : 20, \
"may-6" : 21, \
"june-1" : 22, \
"june-2" : 23, \
"june-3" : 24, \
"june-4" : 25, \
"mar12-1" : 26, \
"mar13-1" : 27, \
"mar13-2" : 28, \
"mar14-1" : 29, \
"mar14-2" : 30, \
"mar14-3" : 31, \
"mar15-1" : 32, \
"mar15-2" : 33, \
"aug06-r-1" : 34, \
"aug06-r-2" : 35, \
"aug06-r-3" : 36, \
"aug06-r-4" : 37, \
"aug06-r-5" : 38, \
"aug06-r-6" : 39, \
"aug06-r-7" : 40, \
"aug06-g-1" : 34, \
"aug06-g-2" : 35, \
"aug06-g-3" : 36, \
"aug06-g-4" : 37, \
"aug06-g-5" : 38, \
"aug06-g-6" : 39, \
"aug06-g-7" : 40, \
"aug07-r-1" : 41, \
"aug07-r-2" : 42, \
"aug07-r-3" : 43, \
"aug07-r-4" : 44, \
"aug07-r-5" : 45, \
"aug07-r-6" : 46, \
"aug07-g-1" : 41, \
"aug07-g-2" : 42, \
"aug07-g-3" : 43, \
"aug07-g-4" : 44, \
"aug07-g-5" : 45, \
"aug07-g-6" : 46, \
"aug08-r-1" : 47, \
"aug08-r-2" : 48, \
"aug08-r-3" : 49, \
"aug08-r-4" : 50, \
"aug08-r-5" : 51, \
"aug08-r-6" : 52, \
"aug08-g-1" : 47, \
"aug08-g-2" : 48, \
"aug08-g-3" : 49, \
"aug08-g-4" : 50, \
"aug08-g-5" : 51, \
"aug08-g-6" : 52, \
"aug06-1" : 34, \
"aug06-2" : 35, \
"aug06-3" : 36, \
"aug06-4" : 37, \
"aug06-5" : 38, \
"aug06-6" : 39, \
"aug06-7" : 40, \
"aug07-1" : 41, \
"aug07-2" : 42, \
"aug07-3" : 43, \
"aug07-4" : 44, \
"aug07-5" : 45, \
"aug07-6" : 46, \
"aug08-1" : 47, \
"aug08-2" : 48, \
"aug08-3" : 49, \
"aug08-4" : 50, \
"aug08-5" : 51, \
"aug08-6" : 52, \
"feb21-1" : 53, \
}

def flattenEclnum(eclname):
    eclnum = eclnums[eclname]
    if eclnum >= 34:
        eclnum -= 33
    elif eclnum >= 26:
        eclnum -= 25
    return eclnum

runoblist = [jan14, jan15, jan16, jan17, may, june]
runoblistTotal = [jan14, jan15, jan16, jan17, may, june, mar12, mar13, mar14, mar15, aug06, aug07, aug08, feb21]

def findRun(eclname):
    """ Find run object given the eclipse name (or run name eg mar12)
    """
    for run in runoblistTotal:
        if run.name == eclname.split('-')[0]:
            return run
    raise IOError("Unable to find run %s"%(eclname))


def findEclname(eclnum):
    for k, v in eclnums.iteritems():
        if v == eclnum:
            return k
            


