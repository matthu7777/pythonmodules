
"""
dynamically interprets formulae to enable more accurate fitting.

This package enables the interpretation and evaluation of algebraic
expressions.  The aim is that this can be useful in function fitting using the
Levenburg-Marquardt method where one ideally supplies analytic
derivatives. For a complex function these are often a pain to derive and so
one often falls back on numerical derivatives (often without knowing it if you
use Python scipy.optimize.leastsq), which can lead to poor results in
difficult cases, the usual symptom being that one or more parameters do not
move from the start values. This package defines a class 'Formula' which can
load an expression, parse it and compute derivatives, and 'Fit' which enables
general functions to be defined, along with initial values. If fitting is what
you want look at the help on 'Fit'.

Example:

form = Formula('a*(b+x)')
x    = numpy.linspace(1.,4.,10)
vd   = {'a' : 1.2, 'b' : 2.4, 'x' : x}
print form.value(vd)

will print out an array of values evaluating the expression shown, while

dx = form.deriv('x')
db = form.deriv('b')

will return Formula objects representing the derivatives of 'form' with
respect to 'x' and 'b'. The latter can then be fed through to
scipy.optimize.leastsq as an exact derivative for instance.  See the
documentation on trm.formula.Fit for a simple implementation. Note that arrays
can be sent (for one argument only) so that the Formula is evaluated at every
one of the points of the array, making for efficient computation.
"""

from scipy import sqrt, sin, power, sin, cos, exp, log, pi, empty
from scipy.optimize import leastsq
import copy

# functions and the number of arguments each requires.
FUNCTIONS = {'sqrt' : 1, 'sqr' : 1, 'cos' : 1, 'sin' : 1, 'exp' : 1, 'pow' : 2, 'ln' : 1, '**' : 2}

# keyword values
KEYWORDS = {'UNIT' : 1., 'ZERO' : 0., 'HALPHA' : 6562.76, 'HBETA' : 4861.327, \
                'HGAMMA' : 4340.465, 'PI' : pi, 'TWOPI' : 2.*pi, 'DAY' : 86400., 'MUNIT' : -1., \
                'VLIGHT' : 299792458. }

class Formula (object):

    """
    Represents a branch of a formula tree, and can also
    be the root of an entire tree.

    Recognised functions:

      */+-   : standard arithematic functions
      **     : exponentiation operator
      cos    : cosine function
      exp    : exponential
      ln     : natural log
      pow    : C-like power function as in pow(x,3)
      sin    : sine function
      sqr    : square function
      sqrt   : square root function

    Recognised keywords:

      DAY    : 86400. [secs in a day]
      HALPHA : 6562.76
      HBETA  : 4861.327 [Angstroms]
      HGAMMA : 4340.465 [Angstroms]
      MUNIT  : -1.
      PI     : Pi
      TWOPI  : 2.*Pi
      UNIT   : 1.
      VLIGHT : c [m/s]
      ZERO   : 0.

    """

    def __init__ (self, *args):
        """

        Constructor:

        Formula()                    : default.
        Formula(express)             : from an expression, number or keyword.
        Formula(op, arg1, args2, ..) : from an operation with arguments.

        Examples:

        form = Formula('PI')
        form = Formula('2*x')
        form = Formula('*', form1, form2)

        the last one combining a couple of Formula objects together with '*', 'deep' 
        copying each one into the new Formula.
        """

        self._ovn    = None
        self._args   = []

        if len(args) == 1:
            express = args[0]
            if isinstance(express, float) or express in KEYWORDS:
                self._ovn = express
            else:
                tnode = _parser(express)
                self._ovn  = tnode._ovn
                self._args = tnode._args

        elif len(args) > 1:
            self._ovn  = args[0]
            for arg in args[1:]:
                self._args.append(copy.deepcopy(arg))

    def value(self, vdict={}):
        """
        Evaluates the value of this node given a dictionary of
        variable / values. The routine branches to other nodes
        unless there are none available in which case it interprets
        the value either as a keyword (e.g. 'UNIT' translates to 1),
        a number or a variable to be looked up in the input dictionary,
        'vdict'.
        """
        if len(self._args) == 0:
            if isinstance(self._ovn,float):
                return self._ovn
            else:
                try:
                    return KEYWORDS[self._ovn]
                except KeyError:
                    return vdict[self._ovn]

        elif len(self._args) == 2:
            if self._ovn == '+':
                return self._args[0].value(vdict) + self._args[1].value(vdict)
            elif self._ovn == '-':
                return self._args[0].value(vdict) - self._args[1].value(vdict)
            elif self._ovn == '*':
                return self._args[0].value(vdict) * self._args[1].value(vdict)
            elif self._ovn == '/':
                return self._args[0].value(vdict) / self._args[1].value(vdict)
            elif self._ovn == 'pow' or self._ovn == '**':
#                return power(self._args[0].value(vdict) , self._args[1].value(vdict))
                return self._args[0].value(vdict)**self._args[1].value(vdict)

        elif len(self._args) == 1:
            if self._ovn == 'cos':
                return cos(self._args[0].value(vdict))
            elif self._ovn == 'exp':
                return exp(self._args[0].value(vdict))
            elif self._ovn == 'ln':
                return log(self._args[0].value(vdict))
            elif self._ovn == 'sin':
                return sin(self._args[0].value(vdict))
            elif self._ovn == 'sqr':
                return self._args[0].value(vdict)**2
            elif self._ovn == 'sqrt':
                return sqrt(self._args[0].value(vdict))
            elif self._ovn == 'u+':
                return self._args[0].value(vdict)
            elif self._ovn == 'u-':
                return -self._args[0].value(vdict)

    def __str__(self):
        level = 0
        ostr  = ''
        level, ostr = _lister(self, level, ostr)
        return ostr

    def __repr__(self):
        return self.__str__()

    def nnodes(self):
        """
        Returns the total number of nodes in a Formula
        """
        return _nnodes(self, 0)

    def deriv(self, var, trim=True):
        """
        Returns the derivative of the Formula with respect to the variable 'var'.
        'trim' indicates whether to try pruning the formula prior to returning it.
        """
        dform = _deriv(self, var)
        if trim:
            dform.prune()
        return dform

    def narg(self):
        """
        Returns number of arguments
        """
        return len(self._args)

    def arg(self, n):
        """
        Returns argument n (starting from 0)
        """
        return self._args[n]

    def is_op(self, top):
        """
        Tests whether operation equals the test operation top
        """
        return self._ovn == top

    def is_const(self):
        """
        Tests whether the Formula is a fixed constant. It does not recurse down 
        any branches and is designed to establish whether end nodes are constants
        """

        if self.narg() or (not isinstance(self._ovn,float) and self._ovn not in KEYWORDS):
            return False
        else:
            return True

    def is_unit(self):
        """
        Tests whether the Formula == 1.0. It does not recurse down any branches
        and is designed to establish whether end nodes are constants
        """
        if self._ovn == 'UNIT' or \
                (self.narg() == 0 and isinstance(self._ovn,float) and self._ovn == 1.0):
            return True
        else:
            return False

    def is_munit(self):
        """
        Tests whether the Formula == -1.0. It does not recurse down any branches
        and is designed to establish whether end nodes are constants
        """
        if self._ovn == 'MUNIT' or \
                (self.narg() == 0 and isinstance(self._ovn,float) and self._ovn == -1.0):
            return True
        else:
            return False

    def is_zero(self):
        """
        Tests whether the Formula == 0.0. It does not recurse down any branches
        and is designed to establish whether end nodes are constants
        """
        if self._ovn == 'ZERO' or \
                (self.narg() == 0 and isinstance(self._ovn,float) and self._ovn == 0.0):
            return True
        else:
            return False

    def prune(self):
        """
        Try to simplify a formula as far as possible
        """

        trimmed, again = _pruner(self)
        while again:
            trimmed, again = _pruner(trimmed)
        self._ovn  = trimmed._ovn
        self._args = trimmed._args

    def vnames(self):
        """
        Returns the set of variable names needed to define the Formula
        """
        if self.narg() == 0:
            if not isinstance(self._ovn,float) and self._ovn not in KEYWORDS:
                s = set([self._ovn])
            else:
                s = set()
        else:
            s = set()
            for n in range(self.narg()):
                s = s.union(self.arg(n).vnames())
        return s

def _parser(form):
    """
    Parses a string version of a formula so that it is broken down into
    operations, numbers and variables that can be represented by a tree
    of Formula nodes.

    Arguments:

      form   : string expression representing the formula

    Returns a Formula representing 'form'
    """

    # strip redundant brackets and space
    form = _stripper(form)

    fnode = Formula()

    # Define flags. 'depth' is bracket level.
    start         = False
    found_op      = False
    is_number     = False
    is_func       = False
    num_var_func  = False
    all_blank     = True
    exp_sign_next = False
    exp_next      = False
    had_dot       = False
    had_exp       = False
    nc            = 0
    depth         = 0
    buff          = ''

    while nc < len(form):
        c = form[nc]
        if c != ' ':
            all_blank = False

            # Don't check anything if we are inside brackets
            if depth == 0:

                # First check for leading + or - signs. Precedence: larger
                # equals stronger. e.g. binary op * beat binary op -

                if c == '-' and not start:
                    lop  = 'u-' # unary minus
                    posn = nc
                    found_op = True
                    precedence = 10

                elif c == '+' and not start:
                    lop  = 'u+' # unary plus
                    posn = nc
                    found_op = True
                    precedence = 10

                elif (c == '*' or c == '/' or c == '.' or c == ',') and not start:
                    raise FormulaError('Formula._parser: one of */., in illegal position in ' + form)

                else:
                    if c != '(' and c != ')':

                        if c == '*' and nc < len(form)-1 and form[nc+1] == '*':
                            # exponentiation operator '**'

                            if num_var_func and is_number and (exp_sign_next or (exp_next and not had_exp)):
                                raise FormulaError('Formula._parser: character ' + str(nc+1) + ' = ' + c + ' is invalid within a number.')

                            is_func = False
                            num_var_func = False
                            if not found_op or precedence >= 15:
                                lop        = '**'
                                posn       = nc
                                found_op   = True
                                precedence = 15
                            nc += 1

                        elif c == '*' or c == '/':
                            # multiplication and division (with a check against '**'

                            if num_var_func and is_number and \
                                    (exp_sign_next or (exp_next and not had_exp)):
                                raise FormulaError('Formula._parser: character ' + str(nc+1) + ' = ' + c + ' is invalid within a number.')

                            is_func = False
                            num_var_func = False
                            if not found_op or precedence >= 5:
                                lop        = c
                                posn       = nc
                                found_op   = True
                                precedence = 5

                        elif not exp_sign_next and (c == '+' or c == '-'):
                            # addition and subtraction. Guard against being in a number
                            # and just abou to get the exponent as in "2.0e+01" as indicated
                            # by exp_sign_next

                            is_func = False
                            num_var_func = False
                            if not found_op or precedence >= 1:
                                lop   = c
                                posn  = nc
                                found_op = True
                                precedence = 1

                        else:

                            if num_var_func:

                                # We are in the process of building a number, a variable
                                # or a function. Must past a few tests on the way.
                                # Numbers are the tough ones; variables etc can be 
                                # almost anything

                                if is_number:

                                    # A digit is always OK, so only check if not.
                                    # checks are:
                                    #
                                    # 1) + or - after an e (if not a digit)
                                    # 2) only digits after e
                                    # 3) only e or . if not after an exponent
                                    # 4) no more than one.

                                    if not c.isdigit() and \
                                            ((exp_sign_next and c != '+' and c != '-') \
                                                 or exp_next or (not exp_sign_next and c != 'e' and c != '.') \
                                                 or (had_dot and c == '.')):
                                        raise FormulaError('Formula._parser: invalid number (character = ' + c + ') in formula = ' + form)

                                    if exp_next:
                                        had_exp = True

                                    if exp_sign_next:
                                        exp_sign_next = False
                                        exp_next      = True

                                    exp_sign_next = c == 'e'
                                    had_dot       = c == '.'

                                buff += c

                            else:

                                # First element which is not one of ()+-*/ (unless it
                                # is the start of '**'), so it should be a number, variable 
                                # or function. numbers must start with a digit, variables 
                                # or functions must start with [a-z] or [A-Z], with the
                                # exception of '**'

                                is_number = c.isdigit()
                                if not is_number and not c.isalpha() and c != '*':
                                    raise FormulaError('Formula._parser: character ' + str(nc+1) + ' = ' + c + ' is invalid')

                                num_var_func = True
                                buff = c
                                if is_number:
                                    exp_sign_next, exp_next = False, False
                                    had_exp, had_dot = False, False

                    elif num_var_func and is_number:
                            raise FormulaError('Formula._parser: character ' + str(nc+1) + ' = ' + c + ' is invalid')

            if c == '(':
                depth += 1

                if num_var_func:
                    # We have a function. Need to check that it is one we know about.
                    if buff not in FUNCTIONS:
                        raise FormulaError('Formula._parser: function = ' + buff + ' not recognised.')
                    else:
                        narg = FUNCTIONS[buff]

                    if not found_op or precedence >= 20:
                        lop        = buff
                        found_op   = True
                        precedence = 20
                        is_func    = True
                        arg_first  = nc+1
                        arg_last   = 0

                    num_var_func = False

            elif c == ')':
                depth -= 1
                if depth < 0:
                    raise FormulaError('Formula._parser: unmatched ) found in expression = ' + form)

            if is_func and depth == 0 and not arg_last:
                arg_last = nc - 1

        start = True
        nc += 1

    if depth != 0:
        raise FormulaError('Formula._parser: unmatched ( found in expression = ' + form)

    if all_blank:
        raise FormulaError('Formula._parser: expression blank')

    if not found_op:
        # If no operation is found, then the expression is either a simple
        # number or variable and the parsing stops, and we make fnode an end
        # node.

        try:
            fnode._ovn = float(form)
        except ValueError:
            fnode._ovn = form

        return fnode

    # At this stage we have gone through the whole string and should
    # have found the last operation. We now need to extract its arguments
    # which may themselves be other expressions. If the last operation
    # is a function, its arguments should be comma separated as in
    # pow((x+y),3). We know the positions of the brackets from
    # arg_first and arg_last, so we just need to locate commas
    # and check that number of arguments is OK. If the last operation
    # is */+- (binary) then arguments are everything to the left and right
    # etc.

    if lop == '+' or lop == '-' or lop == '*' or lop == '/':
        # binary +,-,*,/
        args = [form[0:posn],form[posn+1:]]

    elif lop == '**':
        # exponentiation operator
        args = [form[0:posn],form[posn+2:]]

    elif lop == 'u-' or lop == 'u+':
        # unary - or +
        args = [form[posn+1:],]

    else:

        nc = arg_first
        na = 0
        args = []
        while nc <= arg_last and na < narg:
            c = form[nc]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1

            if depth == 0:
                if c == ',':
                    args.append(form[arg_first:nc])
                    arg_first = nc + 1
                    na += 1
                elif nc == arg_last:
                    args.append(form[arg_first:nc+1])
                    na += 1

            nc += 1

        if na < narg:
            raise FormulaError('Formula._parser: too few arguments for operation = ' + lop + ' in expression = ' + form)

    # Edit node recursively
    fnode._ovn  = lop

    for n in range(len(args)):
        fnode._args.append(_parser(args[n]))

    return fnode

def _stripper(form):
    """
    Recursively strips redundant pairs of brackets from an expression
    including any leading and trailing blanks.
    """

    if form == '': return form

    depth, last, n = 0, 0, 0

    form = form.strip()
    all_blank = True

    while n < len(form):
        c = form[n]
        if c != ' ':
            all_blank = False
            if c == '(':
                depth += 1
                if depth == 1:
                    first = n + 1

            elif c == ')':
                depth -= 1
                if depth < 0:
                    raise FormulaError('Formula._stripper: unmatched ) found in expression = ' + form)
                if depth == 0:
                    last = n

            elif depth == 0:
                return form
        n += 1

    if depth != 0:
        raise FormulaError('Formula._stripper: unmatched ( found in expression = ' + form)

    if all_blank:
        return form

    # If we have got here, then there are a pair of enclosing brackets.
    # Strip them off
    form = form[first:last]

    # Now recursively look for any other such pairs.
    form = _stripper(form)

    return form

def _lister(fnode, level, ostr):
    """
    Recursive formula lister.

    Arguments:

    fnode  : Formula node (top level node at the start)
    level  : bracket depth (set = 0 at the start)
    ostr   : string (set = '' at the start)

    Returns:

    level, ostr -- updated versions
    """

    level += 1

    if fnode.narg():

        if fnode.is_op('*') or fnode.is_op('/') or fnode.is_op('+') or fnode.is_op('-'):
            if level > 1:
                ostr += '('
            level, ostr = _lister(fnode.arg(0),level,ostr)
            level -= 1
            ostr += fnode._ovn
            level, ostr = _lister(fnode.arg(1),level,ostr)
            level -= 1

            if level != 1:
                ostr += ')'
        else:

            if fnode.is_op('u+'):
                ostr += '+('
            elif fnode.is_op('u-'):
                ostr += '-('
            elif fnode.is_op('**'):
                ostr += 'pow('
            else:
                ostr += fnode._ovn + '('

            for n in range(fnode.narg()):
                if n:
                    ostr += ','
                level, ostr = _lister(fnode.arg(n),level,ostr)
                level -= 1
            ostr += ')'

    else:
        ostr += str(fnode._ovn)

    return level, ostr

def _nnodes(fnode, nnode):
    """
    Recursive node counter

    Arguments:

    fnode  : Formula node (top level node at the start)
    nnode  : number of nodes, 0 at the start

    Returns:

    nnode : total number of nodes
    """

    nnode += 1

    for n in range(fnode.narg()):
        nnode = _nnodes(fnode.arg(n), nnode)

    return nnode

def _deriv(fnode, var):
    """
    Recursive function to produce a derivative of a Formula

    Arguments:

      fnode : Formula to take the derivative of. Initialise to the root formula.
      var   : the variable to take the derivative of, e.g. 'a'.

    Returns:

      dnode : the derivative of fnode with respect to var
    """

    if fnode.narg():

        # Function node

        if fnode.is_op('*'):

            # Derivative is sum of two terms, da1*a2 + a1*da2.
            # Technique is to build from the bottom up,
            # combining final Formula arguments at the end.

            # da1*a2
            tmp1 = Formula('*', _deriv(fnode.arg(0), var), fnode.arg(1))

            # a1*da2
            tmp2 = Formula('*', fnode.arg(0), _deriv(fnode.arg(1), var))

            dnode = Formula('+', tmp1, tmp2)

        elif fnode.is_op('/'):

            # Derivative is difference of two terms da1/a2 - da2*(a1/sqr(a2))

            # da1/a2
            tmp1 = Formula('/', _deriv(fnode.arg(0), var), fnode.arg(1))

            # da2*(a1/sqr(a2)). Build in stages: sqr(a2), then a1/sqr(2), then da2*(a1/sqr(a2))
            tmp2 = Formula('sqr', fnode.arg(1))
            tmp3 = Formula('/', fnode.arg(0), tmp2)
            tmp4 = Formula('*', _deriv(fnode.arg(1), var), tmp3)

            dnode = Formula('-', tmp1, tmp4)

        elif fnode.is_op('+'):
            dnode = Formula('+', _deriv(fnode.arg(0), var), _deriv(fnode.arg(1), var))

        elif fnode.is_op('-'):
            dnode = Formula('-', _deriv(fnode.arg(0), var), _deriv(fnode.arg(1), var))

        elif fnode.is_op('u+'):
            dnode = Formula('u+', _deriv(fnode.arg(0), var))

        elif fnode.is_op('u-'):
            dnode = Formula('u-', _deriv(fnode.arg(0), var))

        elif fnode.is_op('sqrt'):
            # da/(2.*sqrt(a))
            tmp1  = Formula('sqrt', fnode.arg(0))
            tmp2  = Formula('*', Formula('TWO'), tmp1)
            dnode = Formula('/',_deriv(fnode.arg(0), var), tmp2)

        elif fnode.is_op('sqr'):
            # da*(2*a)
            tmp1  = Formula('*', Formula('TWO'), fnode.arg(0))
            dnode = Formula('*',_deriv(fnode.arg(0), var), tmp1)

        elif fnode.is_op('cos'):
            # da*(-sin(a))
            tmp1  = Formula('sin', fnode.arg(0))
            tmp2  = Formula('u-', tmp1)
            dnode = Formula('*',_deriv(fnode.arg(0), var), tmp2)

        elif fnode.is_op('sin'):
            # da*cos(a)
            tmp1  = Formula('cos', fnode.arg(0))
            dnode = Formula('*',_deriv(fnode.arg(0), var), tmp1)

        elif fnode.is_op('exp'):
            # da*exp(a)
            tmp1  = Formula('exp', fnode.arg(0))
            dnode = Formula('*',_deriv(fnode.arg(0), var), tmp1)

        elif fnode.is_op('pow') or fnode.is_op('**'):
            # da1*(a2*pow(a1,(a2-1))) + da2*(ln(a1)*pow(a1,a2))
            tmp1  = Formula('-', fnode.arg(1), Formula('UNIT'))
            tmp2  = Formula('pow', fnode.arg(0), tmp1)
            tmp3  = Formula('*', fnode.arg(1), tmp2)
            tmp4  = Formula('*',_deriv(fnode.arg(0), var),tmp3)
            tmp5  = Formula('pow', fnode.arg(0), fnode.arg(0))
            tmp6  = Formula('ln', fnode.arg(0))
            tmp7  = Formula('*',tmp6,tmp5)
            tmp8  = Formula('*',_deriv(fnode.arg(1), var),tmp7)
            dnode = Formula('+',tmp4,tmp8)

        elif fnode.is_op('ln'):
            # da/a
            dnode = Formula('/',_deriv(fnode.arg(0), var), fnode.arg(0))

        else:
            raise FormulaError('Formula._deriv: unrecognised operation = ' + fnode._ovn)

    else:

        # Variable/number node. Either the variable matches in which
        # case the derivative = 1, or it does not, in which case the
        # derivative = 0.

        if fnode._ovn == var:
            dnode = Formula('UNIT')
        else:
            dnode = Formula('ZERO')

    return dnode

def _pruner(fnode):
    """
    Tries to simplify a Formula. Recursive

    Argument:

      fnode : Formula to simplify

    Returns (fnode, again) : modified Formula and boolean to indicate
    whether any changes have been made and therefore whether another
    go is worthwhile.
    """

    again = False
    if fnode.narg():

        all_const = True
        for n in range(fnode.narg()):
            fnode._args[n], ag = _pruner(fnode._args[n])
            again = again or ag
            if not fnode._args[n].is_const(): all_const = False

        if all_const:
            # all inputs to the node are constants and so we can carry out
            # the operation and close out the node

            fnode._ovn  = fnode.value({})
            fnode._args = []
            again = True

        elif (fnode.is_op('sqr') and fnode.arg(0).is_op('sqrt')) or \
                (fnode.is_op('sqrt') and fnode.arg(0).is_op('sqr')) or \
                (fnode.is_op('u-') and fnode.arg(0).is_op('u-')) or \
                (fnode.is_op('ln') and fnode.arg(0).is_op('exp')) or \
                (fnode.is_op('exp') and fnode.arg(0).is_op('ln')):
            fnode = fnode.arg(0).arg(0)
            again = True

        elif fnode.is_op('*'):

            if fnode.arg(0).is_zero() or fnode.arg(1).is_zero():
                fnode._ovn = 'ZERO'
                fnode._args = []
                again = True

            elif fnode.arg(0).is_unit():
                fnode = fnode.arg(1)
                again = True

            elif fnode.arg(1).is_unit():
                fnode = fnode.arg(0)
                again = True

            elif fnode.arg(0).is_munit():
                fnode._ovn  = 'u-'
                fnode._args = [fnode.arg(1),]
                again = True

            elif fnode.arg(1).is_munit():
                fnode._ovn  = 'u-'
                fnode._args = [fnode.arg(0),]
                again = True

        elif fnode.is_op('/'):

            if fnode.arg(0).is_zero():
                fnode._ovn = 'ZERO'
                fnode._args = []
                again = True

            elif fnode.arg(1).is_unit():
                fnode = fnode.arg(0)
                again = True

            elif fnode.arg(1).is_munit():
                fnode._ovn = 'u-'
                fnode.args = [fnode.arg(0),]
                again = True

        elif fnode.is_op('+'):

            if fnode.arg(0).is_zero():
                fnode = fnode.arg(1)
                again = True

            elif fnode.arg(1).is_zero():
                fnode = fnode.arg(0)
                again = True

            elif fnode.arg(1).is_op('u-'):
                fnode._ovn = '-'
                fnode._args[1] = fnode.arg(1).arg(0)
                again = True

        elif fnode.is_op('-'):

            if fnode.arg(1).is_zero():
                fnode = fnode.arg(0)
                again = True

            elif fnode.arg(0).is_zero():
                fnode._ovn  = 'u-'
                fnode._args = [fnode.arg(1),]
                again = True

            elif fnode.arg(1).is_op('u-'):
                fnode._ovn  = '+'
                fnode._args[1] = fnode.arg(1).arg(0)
                again = True

        elif fnode.is_op('**') or fnode.is_op('pow'):

            if fnode.arg(1).is_unit():
                fnode = fnode.arg(0)
                again = True

            elif fnode.arg(1).is_zero():
                fnode._ovn  = 'UNIT'
                fnode._args = []
                again = True

    return fnode, again

class Fit (object):
    """
    Class to represent a 1D function fit. A Fit object is (usually) initialised 
    from a file of the form (between the dashes):

    ---------------------------
    # comments

    Equation = 2*a*(x+1) - b + h*exp(-(x-c)**2/2.)
    + PI*pow(1-x,4)

    a = 1.
    b = 2. f
    h = 1.
    c = 10. f
    ---------------------------

    This will load everything after the 'Equation =' as a fitting function
    until a blank line is reached then use the following lines to define which
    variables to fix and which to vary. Note that it must start precisely as
    'Equation ='; this will be searched for. Everything before the Equation
    line and any line starting with '#' will be ignored. Checks are made that
    all variables and no more are defined and the derivatives with respect to
    the variable values are calculated.

    A fit can then be as easy as:

    --------------------------

    [some code to load data into arrays x, y and e]

    tfit   = formula.Fit('fitfile')
    result = tfit.fit(x,y,e)
    yfit   = tfit(x)
    ---------------------------

    A few results such as the final chi**2 are contained in the tuple
    'result'.  The optimised parameters and covariances are stored inside the
    Fit. The last line used the optimised model to calculate fitted y values
    which can then be plotted.

    You can also construct a Fit from a file-like object which allows one to
    build Fits from in-script strings accessed via StringIO.
    """

    def __init__(self, fno):
        """
        Creates a Fit given a file name or file-like object (to allow use of StringIO)
        containing the definition as outlined in the introduction to the class.

        Arguments::

          fno : (string or file-like object)
              file name or a file-like object defininig the fit

        Attributes::

        f    : (Formula)
             The Formula containing the equation to be fitted.

        vvv  : (dict)
             Dictionary of variable : (value , variable or not). i.e. vvv['a'][0]
             is the value of 'a' while vvv['a'][1] is a boolean indicating whether
             'a' is variable or not.

        df : (dict)
             Dictionary of Formulas representing the derivatives with
             respect to any variable parameters

        cov : (2D numpy array)
             Covariances, ordered in the same way as iterating through vvv, keeping the variable
             parameters only. This will only set if optimisation has happened, otherwise None
        """

        if isinstance(fno, str):
            fp = open(fno)
            close_at_end = True
        else:
            fp = fno
            close_at_end = False

        found_eq = False
        found_end_eq = False

        self.vvv = {}
        for line in fp:
            if not line.startswith('#'):

                if line.startswith('Equation ='):
                    if found_eq:
                        raise FormulaError('Formula.Fit.__init__: two lines start with "Equation ="')
                    found_eq = True
                    express = line[10:].strip()

                elif found_eq and not found_end_eq:
                    if line.isspace():
                        self.f = Formula(express)
                        found_end_eq = True
                    else:
                        express += line.strip()

                elif found_end_eq and not line.isspace():
                    cp = line.find('=')
                    if cp > -1:
                        # Extract the variable and its definition
                        var  = line[:cp].strip()
                        vals = line[cp+1:].strip().split()

                        if len(vals) == 1:
                            self.vvv[var] = [float(vals[0]),True]
                        elif len(vals) == 2:
                            if vals[1].upper() == 'F':
                                self.vvv[var] = [float(vals[0]),False]
                            else:
                                raise FormulaError("Formula.Fit.__init__: expected 'f' or 'F' as second entry in line = " + line)
                        else:
                            raise FormulaError('Formula.Fit.__init__: incorrect number of values in line = ' + line)

        if close_at_end:
            fp.close()

        vnames = self.f.vnames()

        # Some checks
        if 'x' in  vnames:
            vnames.remove('x')

        enames = set(self.vvv.keys())

        for vname in enames:
            if vname not in vnames:
                raise FormulaError('Formula.Fit.__init__: defined variable = "' + vname + '" not found in equation = ' + express)

        for vname in vnames:
            if vname not in enames:
                raise FormulaError('Formula.Fit.__init__: variable = "' + vname + '" not defined')

        # construct derivatives
        self.df = {}
        for key,val in self.vvv.iteritems():
            if val[1]:
                self.df[key] = self.f.deriv(key)

        self.cov = None

    def results(self,full=False):
        """
        Reports results of a fit
        """
        if self.cov is not None:

            vvars = []
            n = 0
            for key, val in self.vvv.iteritems():
                if val[1]:
                    print key,'=',val[0],'+/-',sqrt(self.cov[n,n])
                    n += 1
                    vvars.append(key)

            if full:
                for n,nvar in enumerate(vvars[:-1]):
                    for m,mvar in enumerate(vvars[n+1:]):
                        print 'r(' + nvar + ',' + mvar + ') =',self.cov[n,n+m+1]/sqrt(self.cov[n,n]*self.cov[n+m+1,n+m+1])

        else:
            print 'No fit has been carried out yet.'

    def __call__(self, x):
        """
        Returns the values equivalent to the current Fit

        Argument:

          x : position or array of positions to compute the Fit
        """
        vd = {'x' : x}
        n = 0
        for key, val in self.vvv.iteritems():
            vd[key] = val[0]

        return self.f.value(vd)

    def get(self, par):
        """
        Returns the values associated with the parameter var.
        The value and its error (if available) are returned.
        If the error is not available, None is returned.
        Raises an exception if the parameter cannot be matched

        Argument::

          par : the parameter name
        """

        if par not in self.vvv:
            raise FormulaError('Could not find any parameter "' + par + '"')

        if self.cov is not None:
            n = 0
            for key, val in self.vvv.iteritems():
                if val[1]:
                    if key == par:
                        return (val[0],sqrt(self.cov[n,n]))
                    n += 1
                elif key == par:
                    return (val[0],None)
        else:
            return (self.vvv[par][0],None)

    def set(self, par, value):
        """
        Sets the current value of parameter par to value.
        Raises an exception if the parameter cannot be matched

        Argument::

          par : the parameter name
          value : the floating point value to set it to.
        """

        if par not in self.vvv:
            raise FormulaError('Could not find any parameter "' + par + '"')
        self.vvv[par][0] = value

    def fix(self, par):
        """
        Prevent parameter par from varying in fits.
        """

        if par not in self.vvv:
            raise FormulaError('Could not find any parameter "' + par + '"')
        self.vvv[par][1] = False

    def vary(self, par):
        """
        Allow parameter par to vary in fits. Be careful: this will only
        work if the Fit was originally constructed allowing par to vary.
        If not, you will get a FormulaError. 
        """

        if par not in self.vvv:
            raise FormulaError('Could not find any parameter "' + par + '"')
        if par not in self.df:
            raise FormulaError('Could not find parameter "' + par + '" amongst the derivatives.')

        self.vvv[par][1] = True

    def fit(self, x, y, e, uderiv=True, ftol=1.5e-8, xtol=1.5e-8):
        """
        Uses scipy.optimize.leastsq to optimize Fit. The optimised values are
        stored in the 'vvv' attribute, covariances are stored in the 'cov'
        attribute and can be printed to screen using the 'results' method.

        Arguments:

         x, y, e : x, y, and y-error arrays.
         uderiv : use analytical derivatives or not.
         ftol : as used by leastsq
         xtol : as used by leastsq

        Returns:

          chisq,ier,msg,neval

          chisq : chi**2
          ier   : error flag, see scipy.optimize.leastsq doc.
          msg   : message from scipy.optimize.leastsq
          neval : number of function evaluations.
        """

        p = []
        for key, val in self.vvv.iteritems():
            if val[1]:
                p.append(val[0])

        def _func(p, x, y, e, ft):
            # construct dictionary
            vd = {'x' : x}
            n = 0
            for key, val in ft.vvv.iteritems():
                if val[1]:
                    vd[key] = p[n]
                    n += 1
                else:
                    vd[key] = val[0]
            return (y-ft.f.value(vd))/e

        def _dfunc(p, x, y, e, ft):
            # construct dictionary
            vd = {'x' : x}
            n = 0
            for key, val in ft.vvv.iteritems():
                if val[1]:
                    vd[key] = p[n]
                    n += 1
                else:
                    vd[key] = val[0]

            # space for derivatives
            jac = empty((len(x),n),float)

            n = 0
            for key, val in ft.vvv.iteritems():
                if val[1]:
                    jac[:,n] = -ft.df[key].value(vd)/e
                    n += 1

            return jac

        if uderiv:
            result = leastsq(_func, p, (x,y,e,self), _dfunc, 
                             full_output=1, ftol=ftol, xtol=xtol)
        else:
            result = leastsq(_func, p, (x,y,e,self), 
                             full_output=1, ftol=ftol, xtol=xtol)

        n = 0
        for key, val in self.vvv.iteritems():
            if val[1]:
                self.vvv[key][0] = result[0][n]
                n += 1

        chisq = (result[2]['fvec']**2).sum()

        self.cov = result[1]
        return (chisq, result[4], result[3], result[2]['nfev'])

class FormulaError(Exception):
    pass
