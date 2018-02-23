from functools import partial
import numpy as np
import matplotlib.pyplot as plt

class memoize(object):
    """
       Cache the return value of a method

       This class is meant to be used as a decorator of methods. The return value
       from a given method invocation will be cached on the instance whose method
       was invoked. All arguments passed to a method decorated with memoize must
       be hashable.

       If a memoized method is invoked directly on its class the result will not
       be cached. Instead the method will be invoked like a static method:
       class Obj(object):
           @memoize
           def add_to(self, arg):
               return self + arg
       Obj.add_to(1) # not enough arguments
       Obj.add_to(1, 2) # returns 3, result is not cached

       Script borrowed from here:
       MIT Licensed, attributed to Daniel Miller, Wed, 3 Nov 2010
       http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


class Bspline():
    """
       Numpy implementation of Cox - de Boor algorithm in 1D

       inputs:
           t: Python list or Numpy array containing knot vector
                        entries
           order: Order of interpolation, e.g. 0 -> piecewise constant between
                  knots, 1 -> piecewise linear between knots, etc.
       outputs:
           basis object that is callable to evaluate basis functions at given
           values of knot span
    """

    def __init__(self, t, order):
        """Initialize attributes"""
        self.t = np.array(t)
        self.k = order
        
        #Dummy calls to the functions for memory storage
        self.__call__(0.0)
        self.d(0.0)


    def __basis1(self, xi):
        """Order zero basis"""
        return np.where(np.all([self.t[:-1] <=  xi, xi < self.t[1:]],axis=0), 1.0, 0.0)

    def __basis(self, xi, k, compute_derivatives=False, compute_second_derivatives=False):
        """
           Recursive Cox - de Boor function to compute basis functions and
           optionally their derivatives
        """
        if k == 1:
            return self.__basis1(xi)
        elif compute_second_derivatives and k == self.k-1:
            basis_k_minus_1 = self.__basis(xi, k - 1, compute_second_derivatives=compute_second_derivatives)
            first_term_numerator = (k-2)*(k-1)
            first_term_denominator = (self.t[k-1:]-self.t[:-k+1])*(self.t[k-2:-1]-self.t[:-k+1])
            second_term_numerator = -(k-2)*(k-1)*(self.t[k:]-self.t[1:-k+1] + self.t[k-1:-1] - self.t[:-k])
            second_term_denominator = (self.t[k:]-self.t[1:-k+1])*(self.t[k-1:-1]-self.t[1:-k+1])*(self.t[k-1:-1]-self.t[:-k])
            third_term_numerator = (k-2)*(k-1)
            third_term_denominator = (self.t[k:]-self.t[1:-k+1])*(self.t[k:]-self.t[2:-k+2])

            #Disable divide by zero error because we check for it
            with np.errstate(divide='ignore', invalid='ignore'):
                first_term = np.where(first_term_denominator != 0.0,
                                      (first_term_numerator /
                                       first_term_denominator), 0.0)
                second_term = np.where(second_term_denominator != 0.0,
                                       (second_term_numerator /
                                        second_term_denominator), 0.0)
                third_term = np.where(third_term_denominator != 0.0,
                                       (third_term_numerator/
                                        third_term_denominator), 0.0)
            return (first_term[:-2]*basis_k_minus_1[:-2]
                    + second_term[:-1]*basis_k_minus_1[1:-1]
                    + third_term[:-1]*basis_k_minus_1[2:])
        else:
            basis_k_minus_1 = self.__basis(xi, k - 1, compute_second_derivatives=compute_second_derivatives)

        first_term_numerator = xi - self.t[:-k+1]
        first_term_denominator = self.t[k-1:] - self.t[:-k+1]

        second_term_numerator = self.t[k:] - xi
        second_term_denominator = (self.t[k:] - self.t[1:-k+1])

        #Change numerator in last recursion if derivatives are desired
        if compute_derivatives and k == self.k:
            first_term_numerator = (k-1)
            second_term_numerator = -(k-1)

        #Disable divide by zero error because we check for it
        with np.errstate(divide='ignore', invalid='ignore'):
            first_term = np.where(first_term_denominator != 0.0,
                                  (first_term_numerator /
                                   first_term_denominator), 0.0)
            second_term = np.where(second_term_denominator != 0.0,
                                   (second_term_numerator /
                                    second_term_denominator), 0.0)
        if compute_second_derivatives and k == self.k:
            return basis_k_minus_1
        else:
            return  (first_term[:-1] * basis_k_minus_1[:-1] + second_term * basis_k_minus_1[1:])


    @memoize
    def __call__(self, xi):
        """
           Convenience function to make the object callable.  Also 'memoized'
           for speed.
        """
        return self.__basis(xi, self.k, compute_derivatives=False, compute_second_derivatives=False)

    @memoize
    def d(self, xi):
        """
           Convenience function to compute derivate of basis functions.
           'Memoized' for speed.
        """
        return self.__basis(xi, self.k, compute_derivatives=True, compute_second_derivatives=False)

    @memoize
    def d2(self, xi):
        """
           Convenience function to compute derivate of basis functions.
           'Memoized' for speed.
        """
        return self.__basis(xi, self.k, compute_derivatives=False, compute_second_derivatives=True)