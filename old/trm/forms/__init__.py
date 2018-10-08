#!/usr/bin/env python

from __future__ import print_function

"""
Some helper routines for processing form data sent by email
"""

import re
import trm.subs as subs

# keywords for searching forms
KEYWORDS = ( \
             'Programme ID','Your name','Your e-mail address','Telephone number',
             'Alternative telephone number', 'Number of nights awarded', 'Yes',
             'Your proposal (pdf or ps)', 'Comments', 'Target name', 'Priority',
             'RA (J2000)', 'Declination (J2000)', 'Magnitude', 'Filters (e.g. ugr)',
             'Desired exposure time (sec)', 'Maximum exposure time (sec)',
             'OK to use u-band coadds', 'Total time on target (mins)',
             'Minimum unbroken time on target (mins)', 'Must be photometric',
             'Seeing (worst case)', 'Lunar phase (worst case)',
             'Time critical (e.g. eclipses and satellites)', 'Filter',
             'Maximum u-band exposure time (sec)', 'Finding chart(s) (see below)',
             'Comments specific to target', 'Exposure time (sec) (S)',
             'Total time on target (mins) (S)', 'Minimum unbroken time on target (mins) (S)',
             'Standards', 'Slit angle', 'Slit', 'Grism', 'Filters (S)', 'Expected S-to-N',
             'Minimum S-to-N', 'Filters (I)', 'Exposure time (sec) (I)',
             'Total time on target (mins) (I)', 'Minimum unbroken time on target (mins) (I)',
             'Category','Minimum visits', 'Maximum visits', 'Photometric', 'Moon',
             'Binning','Timing', 'Ephemeris', 'Time scale',
             'Phase start 1', 'Phase end 1', 'Phase flag 1',
             'Phase start 2', 'Phase end 2', 'Phase flag 2',
             'Time start 1', 'Time end 1', 'Time flag 1',
             'Time start 2', 'Time end 2', 'Time flag 2',
             'Time start 3', 'Time end 3', 'Time flag 3',
             'Time start 4', 'Time end 4', 'Time flag 4',
             'Time start 5', 'Time end 5', 'Time flag 5',
             'Zero point (value)', 'Zero point (error)',
             'Period (value)', 'Period (error)',
             'Office phone', 'Home phone', 'Mobile phone',
             'Proposal ID', 'Name', 'Red filter', 'Red backup', 'Green filter', 'Blue filter',
             'Proposal attached', 'E-mail', 'Wavebands', 'Types', 'Publish',
             'Telescope', 'Instrument', 'Status',
             'MJD start 1', 'MJD end 1',
             'MJD start 2', 'MJD end 2',
             'MJD start 3', 'MJD end 3',
             'MJD start 4', 'MJD end 4',
             'MJD start 5', 'MJD end 5',
             'Day', 'Entry', 'Extra times', 'Type', 'Times',
             'Blue coadds', 'Max. blue coadds',
             'Tel RA (J2000)', 'Tel Declination (J2000)', 'Tel PA',
         )

def ttime(tstr):
    """
    ttime translates the time string signifying arrival time into a float
    so that the messages can be ordered by arrival time.
    """

    days = (0,31,59,90,120,151,181,212,243,273,304,334)
    try:
        wday,mday,month,year,utc = tstr.split()[:5]
        (hour,min,sec) = utc.split(':')
        iy = int(year)
        if iy % 4 == 0:
            lday = 1
        else:
            lday = 0
        im = subs.month2int(month) - 1

        day = float(365*(iy - 2000)) + float(days[im]) + float(mday) - 1 + \
              (int(hour) + int(min)/60. + int(sec)/3600.)/24.

        # correct for leap day of the current year
        if im > 1:
            day += 1.
        # correct for leap days of previous years
        for y in range(2000,iy):
            if y % 4 == 1:
                day += 1
    except:
        day = 0

    return day

def glue(str):
    """
    glue prevent HTML line breaking by sticking in non-breaking spaces
    """
    return '&nbsp;'.join(str.split())


def poskey(fields):
    """Returns a key out of the RA, Dec and target name for ordering targets.

    fields : a dictionary with keys 'RA (J2000)' and 'Declination (J2000)'
             which return strings giving the corresponding values and a key
             'Target name'.

    """
    ram = re.compile(r'^\s*(\d\d)[\s:](\d\d)[\s:](\d\d(?:\.\d*)?)\s*$').match(fields['RA (J2000)'])
    rapart  = '{0:02d}{1:02d}{2:05.2f}'.format(
        int(ram.group(1)),int(ram.group(2)),float(ram.group(3))
        )
    decm = re.compile(r'^\s*(-?|\+?)(\d\d)[\s:](\d\d)[\s:](\d\d(\.\d*)?)\s*$').match(fields['Declination (J2000)'])
    decpart = '{0:s}{1:02d}{2:02d}{3:04.1f}{4:s}'.format(
        decm.group(1),int(decm.group(2)),int(decm.group(3)),
        float(decm.group(4)),fields['Target name'])
    return rapart + decpart

def escape(text):
    """Makes entry more suited for web page display"""

    # let some stuff through
    text = text.replace('<p>','U??U')
    text = text.replace('</p>','V??V')
    text = text.replace('<strong>','W??W')
    text = text.replace('</strong>','X??X')

    # we first do the reverse of what we are
    # about to do to prevent doubling up of escapes
    text = text.replace('&amp;','&')
    text = text.replace('&lt;','<')
    text = text.replace('&gt;','>')
    text = text.replace('&quot;','"')
    text = text.replace('&#x27;',"'")
    text = text.replace('&#39;',"'")
    text = text.replace('&#x2F;','/')
    text = text.replace('&#47;','/')

    # now do what we wanted to do
    text = text.replace('&','&amp;')
    text = text.replace('<','&lt;')
    text = text.replace('>','&gt;')
    text = text.replace('"','&quot;')
    text = text.replace("'",'&#x27;')
    text = text.replace('/','&#x2F;')

    # replace OK stuff
    text = text.replace('U??U', '<p>')
    text = text.replace('V??V', '</p>')
    text = text.replace('W??W', '<strong>')
    text = text.replace('X??X', '</strong>')

    return text

def ultparse(formdata):
    """
    ultparse parses the text part of the e-mails from the phase II forms for
    ultracam and ultraspec returning a dictionary of the field values.
    """
    text = formdata.get_payload(decode=True)
    try:
        text = text.decode('ASCII')
    except:
        pass

    text = text.replace('\r\n ','')
    lines = text.splitlines()

    # Delete everything before start entry
    for i in range(len(lines)):
        if lines[i].startswith('Programme ID:') or lines[i].startswith('Category:') or \
           lines[i].startswith('Proposal ID:') or lines[i].startswith('Name:'):
            del lines[:i]
            break
    else:
        raise Exception('ultparse error: could not find Programme ID / Category / Proposal ID / Name')

    # delete everything after 'Click ..' or 'END OF FORM DATA'
    for i in range(len(lines)):
        if lines[i].startswith('END OF FORM DATA'):
            del lines[i-1:]
            break
    else:
        for i in range(len(lines)):
            if lines[i].startswith('Click'):
                del lines[i-1:]
                break

   # stick into a dictionary, cope with a couple of special cases.
    field = {}

    # extract comment strings
    for i in range(len(lines)):
        if lines[i].startswith('Comments:') or lines[i].startswith('Comments specific to target:'):
            comments = ''
            for j in range(i+1,len(lines)):
                if not lines[j].isspace() and lines[j] != '':
                    comments += ' ' + lines[j]
            del lines[i:]
            field['Comments'] = escape(comments)
            break

    # combine extra times
    for i in range(len(lines)):
        if lines[i].startswith('Extra times:'):
            extra_times = ''
            for j in range(i+1,len(lines)):
                if not lines[j].isspace() and lines[j] != '':
                    extra_times += lines[j]

            del lines[i:]
            field['Extra times'] = extra_times
            break

    j = 0
    for i in range(len(lines)):
        for key in KEYWORDS:
            if lines[j].startswith(key + ':'):
                if j < len(lines)-1:
                    if key == 'Your name':
                        field[key] = glue(escape(lines[j+1].strip()))
                    elif key == 'Total time on target (mins)' or \
                            key == 'Minimum unbroken time on target (mins)':
                        num = re.compile(r'(\d+(?:\.\d*)?)')
                        m = num.match(lines[j+1].strip())
                        if m:
                            field[key] = m.group(1)
                        else:
                            field[key] = '0'
                    elif key == 'Filters (e.g. ugr)' or key == 'Filter':
                        field[key] = escape(lines[j+1].strip()).replace(',',', ')
                    else:
                        # cludge here to get over weird problem caused when
                        # Mattew cut-n-pasted proposal IDs.
                        if len(lines[j]) > len(key)+1:
                            field[key] = escape(lines[j][len(key)+1:].strip() + lines[j+1].strip())
                        else:
                            field[key] = escape(lines[j+1].strip())
                    j += 1
                else:
                    field[key] = ''
                break
        j += 1
        if j >= len(lines):
            break

    # return
    return field

def sultparse(formdata):
    """
    sultparse parses the text part of the e-mails from the phase II forms for
    ultracam and ultraspec returning a dictionary of the field values with minimal
    alteration. e.g. unlike ultparse it does not replace blanks with &nbsp;
    """

    lines = formdata.get_payload(decode=True).splitlines()

    # Delete everything before Programme ID or Category
    for i in range(len(lines)):
        if lines[i].startswith('Programme ID:') or lines[i].startswith('Category:'):
            del lines[:i]
            break
    else:
        raise Exception('sultparse error: could not find Programme ID / Category')

    # delete everything after 'Click ..' or 'END OF FORM DATA'
    found_end = False
    for i in range(len(lines)):
        if lines[i].startswith('END OF FORM DATA'):
            del lines[i-1:]
            found_end = True
            break
    if not found_end:
        for i in range(len(lines)):
            if lines[i].startswith('Click'):
                del lines[i-1:]
                break

    # combine comment strings
    first = True
    for i in range(len(lines)):
        if lines[i].startswith('Comments:') or lines[i].startswith('Comments specific to target:'):
            clines = lines[i+1:len(lines)]
            break
    else:
        raise FormsError('ultparse error: could not find any comments')

    # stick into a dictionary
    field = {}
    j = 0
    for i in range(len(lines)):
        for key in KEYWORDS:
            if lines[j].startswith(key + ':'):
                if key.startswith('Comments'):
                    field[key] = clines
                else:
                    if j < len(lines)-1:
                        field[key] = lines[j+1].strip()
                        j += 1
                    else:
                        field[key] = ''
                break
        j += 1
        if j >= len(lines):
            break

    # return
    return field

def ultchk(fields):
    """
    checks various fields of the ultra-cam/-spec forms
    """
    progID = fields['Programme ID']
    rachk  = re.compile(r'^J?\s*\d{1,2}[^0-9]+?\d{1,2}[^0-9]+?\d{1,2}(\.\d*)?[^0-9]*$')
    decchk = re.compile(r'^\s*(\-|\+|\s)?\s*\d{1,2}[^0-9]+?\d{1,2}[^0-9]+?\d{1,2}(\.\d*)?[^0-9]*$')
    for (key,value) in fields.iteritems():
        if key == 'RA (J2000)' and not rachk.match(fields[key]):
            print('Programme = ' + progID + ', target = ' + fields['Target name'] + ', RA = ' + value + ' has invalid format')
            print('Will skip this entry.')
            return False
        if key == 'Declination (J2000)' and not decchk.match(fields[key]):
            print('Programme = ' + progID + ', target = ' + fields['Target name'] + ', Declination = ' + value + ' has invalid format')
            print('Will skip this entry.')
            return False
        if key == 'Total time on target (mins) (S)' or key == 'Total time on target (mins) (I)':
            try:
                f = float(value)
            except:
                # special case
                if value == '2 nights':
                    fields[key] = str(2.0*10.5*60)
                else:
                    print('Programme = ' + progID + ', target = ' + fields['Target name'] + ', could not interpret total time = ' + value)
                    fields[key] = str(0.0)
                    print('Will set = 0.')
    return True

def genparse(formdata, fields):
    """parses the text part of the e-mails returning a dictionary of the field
    values. This one is generic which means that it does not do much.

    formdata : the e-mail returned by the formsbuilder form

    fields : the fields to look for. The first one should be the first field
             of the form which will define the start of it. Do not include the
             ending ':'
    """

    lines = formdata.get_payload(decode=True).splitlines()

    # Top and tail the message:
    for i in range(len(lines)):
        if lines[i].startswith(fields[0] + ':'):
            del lines[:i]
            break
    else:
        raise FormError('genparse error: could not find ' + fields[0])

    for i in range(len(lines)):
        if lines[i].startswith('Click <http'):
            del lines[i-1:]
            break

    # Define a dictionary
    fdict = {}
    j = 0
    for i in range(len(lines)):
        for key in fields:
            if lines[j].startswith(key + ':'):
                if j < len(lines)-1:
                    fdict[key] = lines[j+1].strip()
                    j += 1
                elif j == len(lines)-1:
                    fdict[key] = ''
                break
        j += 1
        if j >= len(lines): break

    return fdict

# Exception class
class FormError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

