# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:52:39 2016

@author: nknezek
"""

from scipy.misc import factorial as _factorial
from scipy.special import lpmv as _lpmv
from numpy import sin as _sin
from numpy import cos as _cos
from numpy import zeros as _zeros
from numpy import sum as _sum
from numpy import exp as _exp
import numpy as _np
from .bspline2 import Bspline as _Bspline
import os as _os
import pyshtools as _sht
import warnings as _warnings

_gufm1_data_file = _os.path.dirname(_os.path.abspath(__file__)) + '/data/gufm1_data.txt'
_gufmsatE3_data_file = _os.path.dirname(_os.path.abspath(__file__)) + '/data/gufm-sat-E3.txt'
_gufmsatQ2_data_file = _os.path.dirname(_os.path.abspath(__file__)) + '/data/gufm-sat-Q2.txt'
_gufmsatQ3_data_file = _os.path.dirname(_os.path.abspath(__file__)) + '/data/gufm-sat-Q3.txt'
_chaos6_data_file = _os.path.dirname(_os.path.abspath(__file__)) + '/data/chaos6_data.txt'

class SphereHarmBase():
    def __init__(self):
        pass

    def SHrcmb2vs(self, Clm_rcmb, r_cmb=3480., r_s=6371.2, l_max=None):
        '''converts radial 4pm SH at CMB to potential SH at surface

        Converts a radial field measurement in 4pi normalized spherical harmonics at the CMB to the potential field
        spherical harmonics at the surface.

        Parameters
        ----------
        Clm_rcmb
        r_cmb
        r_s
        l_max

        Returns
        -------
        Clm_s = coefficients array with 4pi, csphase=1 normalization
        '''
        if l_max is None:
            l_max = self.l_max
        Clm_in = self._convert_SHin(Clm_rcmb, l_max=l_max)
        Clm_s = _np.zeros_like(Clm_in)
        for l in range(Clm_in.shape[1]):
            vs2rc = (r_s / r_cmb) ** (l + 2) * (l + 1)
            Clm_s[:, l, :] = Clm_rcmb[:, l, :] / vs2rc
        return Clm_s

    def _convert_SHin(self, SHin, l_max=None):
        '''helper function to convert sht classes and arrays for functions

        Parameters
        ----------
        SHin: _sht.SHCoeffs class or numpy array
        l_max: l_max

        Returns
        -------
            np.array of SH coefficients with normalization 4pi, csphase=1
        '''
        if type(SHin) is _np.ndarray:
            if SHin.shape[0] == 2 and len(SHin.shape) == 3:
                if l_max is None:
                    l_max = SHin.shape[1] - 1
                if SHin.shape[1] >= l_max + 1:
                    SHout = SHin[:, :l_max + 1, :l_max + 1]
                else:
                    raise TypeError('SHin does not contain enough coefficients for requested l_max')
            elif len(SHin.shape) < 3:
                if l_max is None:
                    l_max = int(SHin.shape[0] ** 0.5 - 1)
                if SHin.shape[0] >= (l_max + 1) ** 2:
                    SHout = _sht.shtools.SHVectorToCilm(SHin[:(l_max + 1) ** 2])
                else:
                    raise TypeError('SHin does not contain enough coefficients for requested l_max')
            else:
                raise TypeError('SHin not in a recognizable format or the wrong size for l_max')
        elif (type(SHin) is _sht.shclasses.SHRealCoeffs) or (type(SHin) is _sht.shclasses.SHComplexCoeffs):
            SHout = SHin.get_coeffs(normalization='4pi', csphase=1, lmax=SHin.lmax)
        else:
            raise TypeError('SHin not in a recognizable format or the wrong size for l_max')
        return SHout

    def get_thvec_phvec_DH(self, Nth=None, l_max=None):
        '''returns theta and ph coordinate vectors for DH grid

        :param Nth:
        :param l_max:
        :return:
        '''
        if l_max is None:
            l_max = self.l_max
        if Nth is None:
            Nth = l_max * 2 + 2
        Nph = 2 * Nth
        dth = 180 / Nth
        th = _np.linspace(dth / 2, 180 - dth / 2, Nth)
        ph = _np.linspace(dth / 2, 360 - dth / 2, Nph)
        return th, ph

class MagModel(SphereHarmBase):
    def __init__(self, data_file):
        self.data_file = data_file
        self.gt, self.tknts, self.l_max, self.bspl_order = self._read_data(data_file=self.data_file)
        self.dT = self.tknts[len(self.tknts)//2+1]-self.tknts[len(self.tknts)//2]
        self.bspline = self._make_bspline_basis(self.tknts)
        self.T_start = self.tknts[self.bspl_order-1]
        self.T_end = self.tknts[-self.bspl_order]

    def _read_data(self, data_file=None):
        '''

        Parameters
        ----------
        filename

        Returns
        -------

        '''
        if data_file is None:
            datafile = self.data_file
        with open(data_file,'rb') as f:
            f.readline()
            line1 = f.readline().split()

            l_max = int(line1[0])
            nspl = int(line1[1])
            if float(line1[2]) < 1000:
                bspl_order = int(line1[2])
                l1_tknt_loc = 3
            else:
                bspl_order = 4
                l1_tknt_loc = 2

            n = l_max*(l_max+2)

            gt = _zeros(n*nspl)
            tknts = _zeros(nspl+bspl_order)
            tknt_l1 = [float(x) for x in line1[l1_tknt_loc:]]
            tknts[:len(tknt_l1)] = tknt_l1
            ti = len(tknt_l1)
            gi = 0
            for line in f:
                l_tmp = [float(x) for x in line.split()]
                nl = len(l_tmp)
                if ti+nl <= len(tknts):
                    tknts[ti:ti+nl] = l_tmp
                    ti += nl
                else:
                    gt[gi:gi+nl] = l_tmp
                    gi += nl
        gt_out = gt.reshape(n, nspl, order='F')
        return gt_out, tknts, l_max, bspl_order

    def _read_coeffs(self, data_file=None):
        '''

        Parameters
        ----------
        filename

        Returns
        -------

        '''
        if data_file is None:
            datafile = self.data_file
        with open(data_file,'rb') as f:
            f.readline()
            line1 = f.readline().split()

            l_max = int(line1[0])

            raw = []
            for line in f:
                l_tmp = [float(x) for x in line.split()]
                for l in l_tmp:
                    raw.append(l)

        g_raw = _np.array(raw)
        return g_raw, l_max

    def _make_bspline_basis(self, tknts, order=None):
        if order is None:
            order = self.bspl_order
        bspline = _Bspline(tknts, order)
        return bspline

    def _interval(self, time):
        '''
        Calculates nleft: the index of the timeknot on the left of the interval
            tknts[nleft] < tknts[nleft+1]
            tknts[nleft] <= time <= tknts[nleft+1]

        Parameters
        ----------
        tknts:
            a numpy array containing the timestamps for all knots in the model
        time:
            the time to calculate the field

        Returns
        -------
        the index of the time knot on the left of the interval
        '''
        tknts = self.tknts
        if (time >= tknts[self.bspl_order-1] and time <= tknts[-self.bspl_order]):
            for n in range(self.bspl_order-1,len(tknts)-self.bspl_order+1):
                if time >= tknts[n]:
                    nleft = n-self.bspl_order+1
                else:
                    break
        else:
            raise IndexError("The time you've chosen is outside this model")
        return nleft

    def _Pml(self, x, l, m):
        """
        Associated Legendre Polynomial - Schmidt Quasi-Normalization
        ============================================================
        Returns the evaulated Associated Legendre Polynomial of degree n and order m at location x.

        This function evaluates the Associated Legendre Polynomials with Schmidt Quasi Normalization as defined in Schmidt (1917, p281).
        It uses the scipy built in associated legendre polynomials which have Ferrer's normalization and converts the normalization.

        Inputs
        -------
        x:
            Location of evaluation
        l:
            Degree of associated legendre polynomial
        m:
            Order of associated legendre polynomial

        Returns
        -------
        The value of the polynomial at location specified. (float)

        Associated Legendre Polynomial Normalizations:
        ------

        Schmidt Quasi-Normalized:
            P^m_l(x) = sqrt{2*(l-m)!/(l+m)!}(1-x^2)^{m/2}(d/dx)^2 P_l(x)

        Ferrer's (only for reference):
            P^m_n(x) = (-1)^m(1-x^2)^{m/2}(d/dx)^2 P_n(x)

        """
        if m == 0:
            return (_factorial(l-m)/_factorial(l+m))**0.5/(-1)**m*_lpmv(m,l,x)
        else:
            return (2*_factorial(l-m)/_factorial(l+m))**0.5/(-1)**m*_lpmv(m,l,x)

    def _dtheta_Pml(self, x, l, m):
        """
        Theta derivative of Associated Legendre Polynomial - Schmidt Quasi-Normalization
        ============================================================
        Returns the theta derivative of the Associated Legendre Polynomial of degree n and order m at location x=cos(theta).

        Inputs
        -------
        x:
            Location of evaluation ( cos(theta) )
        l:
            Degree of associated legendre polynomial
        m:
            Order of associated legendre polynomial

        Returns
        -------
        The theta derivative of the polynomial at location specified. (float)

        Associated Legendre Polynomial Normalizations:
        ------

        Schmidt Quasi-Normalized:
            P^m_l(x) = sqrt{2*(l-m)!/(l+m)!}(1-x^2)^{m/2}(d/dx)^2 P_l(x)

        Theta derivative:
            d_theta P^m_l( x=cos(th) ) = -1/sqrt{1-x^2} * [ (l+1)*x*P^m_l(x) + (l+1-m)*P^m_(l+1)(x) ]
        """
        if m == 0:
            return 1/(1-x**2)**0.5 * ( -(l+1)*x*self._Pml(x, l, m) + (l+1-m)*(_factorial(l-m)/_factorial(l+m))**0.5/(-1)**m*_lpmv(m,l+1,x))
        else:
            return 1/(1-x**2)**0.5 * ( -(l+1)*x*self._Pml(x, l, m) + (l+1-m)*(2*_factorial(l-m)/_factorial(l+m))**0.5/(-1)**m*_lpmv(m,l+1,x))

    def _calculate_g_raw_at_t(self, time):
        '''
        Calculates the Gauss Coefficients in raw ordering given the parameters calculated by inverval() and _bspline().

        Parameters
        ----------
        gt:
            raw data from gufm1 ((number of coeffs x number of tknts) numpy array)
        spl:
            B-spline basis (jorder numpy array)
        nleft:
            coordinate of the timeknot to the left of desired time
        l_max:
            spherical harmonic degree included in model (14)
        jorder:
            order of B-splines (4)
        Returns
        -------
            Gauss Coefficients for a particular time in raw ordering.
        '''
        gt = self.gt
        b = self.bspline(time)
        i = self._interval(time)
        bo = self.bspl_order
        g_raw = _sum(b[i:i+bo]*gt[:, i:i+bo], axis=1)
        return g_raw

    # def _convert_g_raw_to_shtarray(self, g_raw, l_max=None):
    #     '''
    #     Converts g_raw computed for a time to shtools formatted array
    #
    #     Inputs
    #     ------
    #     g_raw:
    #         numpy array of g_raw, standard ordering as on single-time g_raw files from website.
    #     l_max:
    #         spherical harmonic degree included in model (automatically taken from data_file)
    #     Returns
    #     -------
    #     coeffs:
    #         (2,l_max+1, l_max+1) size array of Gauss coefficients where e.g. coeffs[0,2,1] = g(l=2, m=1), coeffs[1,2,0] = h(l=2, m=0)
    #
    #     '''
    #     if not l_max:
    #         l_max = self.l_max
    #     coeffs = _np.zeros((2,l_max+1, l_max+1))
    #     coeffs[0,1,0] = g_raw[0]
    #     coeffs[0,1,1] = g_raw[1]
    #     coeffs[1,1,1] = g_raw[2]
    #     i = 3
    #     for l in range(2,l_max+1):
    #         coeffs[0,l,0] = g_raw[i]
    #         i += 1
    #         for m in range(1,l+1):
    #             coeffs[0,l,m] = g_raw[i]
    #             i += 1
    #             coeffs[1,l,m] = g_raw[i]
    #             i += 1
    #     return coeffs

    def read_SH_from_file_gufm_form(self, file):
        '''
        file:
            filename of file in gufm single-epoch form. First line is title line, second line is l_max, then coefficients

        Returns
        -------
            SH_gufm, l_max : _np.array of the raw coefficients and l_max
        '''
        with open(file, 'rb') as f:
            f.readline()
            line1 = f.readline().split()

            l_max = int(line1[0])

            raw = []
            for line in f:
                l_tmp = [float(x) for x in line.split()]
                for l in l_tmp:
                    raw.append(l)
        SH_gufm = _np.array(raw)
        return SH_gufm, l_max

    def write_SH_to_file_gufm_form(self, SH, file, heading, l_max, num_per_line=4):
        '''

        Parameters
        ----------
        SH: _sht.SHCoeffs class
        file: filename to write to
        heading: heading to write on line #1
        l_max: l_max to write on line #2
        num_per_line: number of coefficients to list per line

        Returns
        -------
            none
        '''
        Bschmidt = SH.get_coeffs(normalization='schmidt', csphase=-1)
        SHv = self._convert_shtarray_to_gufm_form(Bschmidt, l_max=SH.lmax)
        with open(file, 'w') as f:
            f.write(heading + '\n')
            f.write('{0:.0f}\t0\n'.format(l_max))
            i = 0
            while (i < len(SHv)):
                f.write('{0:.8e}'.format(SHv[i]))
                if (i+1) % num_per_line == 0:
                    f.write('\n')
                else:
                    f.write('\t')
                i += 1

    def _convert_gufm_form_to_shtarray(self, g_raw, l_max=None):
        '''
        Converts g_raw computed for a time to shtools formatted array

        Inputs
        ------
        g_raw:
            numpy array of g_raw, standard ordering as on single-time g_raw files from website.
        l_max:
            spherical harmonic degree included in model (automatically taken from data_file)
        Returns
        -------
        coeffs:
            _sht.SHCoeffs class: (2,l_max+1, l_max+1) size array of Gauss coefficients where e.g. coeffs[0,2,1] = g(l=2, m=1), coeffs[1,2,0] = h(l=2, m=0)

        '''
        if not l_max:
            l_max = self.l_max
        coeffs = _np.zeros((2, l_max + 1, l_max + 1))
        coeffs[0, 1, 0] = g_raw[0]
        coeffs[0, 1, 1] = g_raw[1]
        coeffs[1, 1, 1] = g_raw[2]
        i = 3
        for l in range(2, l_max + 1):
            coeffs[0, l, 0] = g_raw[i]
            i += 1
            for m in range(1, l + 1):
                coeffs[0, l, m] = g_raw[i]
                i += 1
                coeffs[1, l, m] = g_raw[i]
                i += 1
        return _sht.SHCoeffs.from_array(coeffs, normalization='schmidt', csphase=-1)

    def _convert_shtarray_to_gufm_form(self, shtarray, l_max=None):
        '''
        Converts g_raw computed for a time to shtools formatted array

        Inputs
        ------
        shtarray:
            (2,l_max+1, l_max+1) size array of Gauss coefficients where e.g. coeffs[0,2,1] = g(l=2, m=1), coeffs[1,2,0] = h(l=2, m=0)
        l_max:
            spherical harmonic degree included in model (automatically taken from data_file)
        Returns
        -------
        gufm_raw:
            _np.array of length (l_max+1)**2-1 ordered according to the gufm standard
        '''
        shtarray = self._convert_SHin(shtarray, l_max=l_max)
        if l_max is None:
            l_max = shtarray.lmax
        raw = []
        raw.append(shtarray[0, 1, 0])
        raw.append(shtarray[0, 1, 1])
        raw.append(shtarray[1, 1, 1])
        for l in range(2, l_max + 1):
            raw.append(shtarray[0, l, 0])
            for m in range(1, l + 1):
                raw.append(shtarray[0, l, m])
                raw.append(shtarray[1, l, m])
        return _np.array(raw)

    def _get_lmax(self, SHin):
        '''helper function to get the l_max of a set of spherical harmonics

        :param SH:
        :return:
        '''
        if type(SHin) is _np.ndarray:
            if SHin.shape[0] == 2 and len(SHin.shape) == 3:
                l_max = int(SHin.shape[1]-1)
                if not SHin.shape == (2,l_max + 1,l_max+1):
                    raise TypeError('SH is not the right shape l_max')
            elif len(SHin.shape) < 3:
                l_max = int(SHin.shape[0]**0.5-1)
                if not SHin.shape[0] == (l_max+1)**2:
                    raise TypeError('SHin does not contain enough coefficients for requested l_max')
            else:
                raise TypeError('SHin not in a recognizable format or the wrong size for l_max')
        elif (type(SHin) is _sht.shclasses.SHRealCoeffs) or (type(SHin) is _sht.shclasses.SHComplexCoeffs):
            l_max = int(SHin.lmax)
        else:
            raise TypeError('SHin not in a recognizable format or the wrong size for l_max')
        return l_max

    def _calculate_SVgh_raw_at_t(self, time):
        '''
        Calculates the time derivatives of Gauss Coefficients in raw ordering given the parameters calculated by inverval() and _bspline().

        Parameters
        ----------
        time:
            date in years to calculate SVg_raw
        Returns
        -------
        SVg_raw:
            Time derivatives of Gauss Coefficients for a particular time in raw ordering.
        '''
        gt = self.gt
        SVb = self.bspline.d(time)
        i = self._interval(time)
        bo = self.bspl_order
        SVg_raw = _sum(SVb[i:i+bo]*gt[:, i:i+bo], axis=1)
        return SVg_raw

    def _calculate_SAgh_raw_at_t(self, time):
        '''
        Calculates the second time derivatives of Gauss Coefficients in raw ordering given the parameters calculated by inverval() and _bspline().

        Parameters
        ----------
        time:
            date in years to calculate SAg_raw
        Returns
        -------
        SAg_raw:
            Second time derivatives of Gauss Coefficients for a particular time in raw ordering.
        '''
        gt = self.gt
        SAb = self.bspline.d2(time)
        i = self._interval(time)
        bo = self.bspl_order
        SAg_raw = _sum(SAb[i:i+bo]*gt[:, i:i+bo], axis=1)
        return SAg_raw

    def _convert_g_raw_to_gh(self, g_raw, l_max=None):
        '''
        Converts g_raw computed for a time to g, h dictionaries

        Inputs
        ------
        g_raw:
            numpy array of g_raw, standard ordering as on single-time g_raw files from website.
        l_max:
            spherical harmonic degree included in model (automatically taken from data_file)
        Returns
        -------
        g, h:
            dictionaries of Gauss coefficients ordered as g[l][m] and h[l][m]

        '''
        if not l_max:
            l_max = self.l_max
        g = {}
        h = {}
        g[1] = {0:g_raw[0]}
        g[1][1] = g_raw[1]
        h[1] = {0:0, 1:g_raw[2]}
        i = 3
        for l in range(2,l_max+1):
            g[l] = {}
            h[l] = {}
            g[l][0] = g_raw[i]
            i += 1
            h[l][0] = 0.
            for m in range(1,l+1):
                g[l][m] = g_raw[i]
                i += 1
                h[l][m] = g_raw[i]
                i += 1
        return g, h

    def convert_gh_to_complex(self, g, h, l_max=None):
        '''
        converts g,h real spherical harmonics to A complex spherical harmonic.

        V(r,th,ph) = a sum_m,l{ \norm(a/r)**(l+1) * ( g_m,l * cos(m*ph) + h_m,l * sin(m*ph) ) * P_m,l(cos(th)) }
        c(th,ph) = sum_m,l{ c_m,l * exp(i*m*ph) * P_m,l(cos(th)) }

        Parameters
        ----------
        g: cos(m\phi) term
        h: sin(m\phi) term

        Returns
        -------
        c: complex spherical harmonics coefficients
        '''
        if not l_max:
            l_max = max(g.keys())
        c = {}
        for l in range(1,l_max+1):
            c[l] = {}
            for m in range(0,l+1):
                c[l][m] = g[l][m] - 1j*h[l][m]
        return c

    def convert_complex_to_gh(self, c, l_max=None):
        '''
        converts c complex spherical harmonic to g,h real spherical harmonics.

        V(r,th,ph) = a sum_m,l{ \norm(a/r)**(l+1) * ( g_m,l * cos(m*ph) + h_m,l * sin(m*ph) ) * P_m,l(cos(th)) }
        c(th,ph) = sum_m,l{ c_m,l * exp(i*m*ph) * P_m,l(cos(th)) }

        Parameters
        ----------
        c: complex spherical harmonics coefficients

        Returns
        -------
        g,h: real spherical harmonics coefficients
        '''
        if not l_max:
            l_max = max(c.keys())
        g = {}
        h = {}
        for l in range(1,l_max+1):
            g[l] = {}
            h[l] = {}
            for m in range(0,l+1):
                g[l][m] = _np.real(c[l][m])
                h[l][m] = -_np.imag(c[l][m])
        return g,h

    def get_shtcoeffs_at_t(self, time, l_max=None):
        '''
        Calculates Gauss coefficients at time T

        Parameters
        ----------
        time:
            time to calculate parameters
        l_max:
            spherical harmonic degree included in model (14)
        Returns
        -------
        coeffs:
            array containing Gauss coefficients at time time
        '''
        g_raw = self._calculate_g_raw_at_t(time)
        return self._convert_gufm_form_to_shtarray(g_raw, l_max=l_max)

    def get_sht_allT(self, T, l_max=None):
        '''
        Calculates Gauss coefficients of secular acceleration at a list or _np.array of times T

        :param T: list of _np.array of times (yr)
        :param l_max:
        :return:
        '''
        sht_list = []
        for t in T:
            sht_list.append(self.get_shtcoeffs_at_t(t,l_max=l_max))
        return sht_list

    def get_SVshtcoeffs_at_t(self, time, l_max=None):
        '''
        Calculates Gauss coefficients of secular variation at time T

        Parameters
        ----------
        time:
            time to calculate parameters
        l_max:
            spherical harmonic degree included in model (14)
        Returns
        -------
        coeffs:
            array containing Gauss coefficients of secular variation at time time
        '''
        g_raw = self._calculate_SVgh_raw_at_t(time)
        return self._convert_gufm_form_to_shtarray(g_raw, l_max=l_max)

    def get_SVsht_allT(self, T, l_max=None):
        '''
        Calculates Gauss coefficients of secular variation at a list or _np.array of times T

        :param T: list of _np.array of times (yr)
        :param l_max:
        :return:
        '''
        sht_list = []
        for t in T:
            sht_list.append(self.get_SVshtcoeffs_at_t(t,l_max=l_max))
        return sht_list

    def get_SAshtcoeffs_at_t(self, time, l_max=None):
        '''
        Calculates Gauss coefficients of secular acceleration at time T

        Parameters
        ----------
        time:
            time to calculate parameters
        l_max:
            spherical harmonic degree included in model (14)
        Returns
        -------
        coeffs:
            array containing Gauss coefficients of secular acceleration at time time
        '''
        g_raw = self._calculate_SAgh_raw_at_t(time)
        return self._convert_gufm_form_to_shtarray(g_raw, l_max=l_max)

    def get_SAsht_allT(self, T, l_max=None):
        '''
        Calculates Gauss coefficients of secular acceleration at a list or _np.array of times T

        :param T: list of _np.array of times (yr)
        :param l_max:
        :return:
        '''
        sht_list = []
        for t in T:
            sht_list.append(self.get_SAshtcoeffs_at_t(t,l_max=l_max))
        return sht_list

    def B_sht(self, shtcoeffs, r=3480, Nth=None, l_max=None, a=6371.2):
        '''
        Calculates the radial magnetic field on a Driscoll-Healy grid given spherical harmonics coefficients, using shtools

        :param shtcoeffs: spherical harmonics coefficients in SHT format
        :param r: radius of caculation (km)
        :param Nth: Number of latitudinal grid point points to output (number of longitudinal points Nph = 2*Nth)
        :param l_max: maximum spherical harmonic degree to use in computation
        :param a: radius of data (6371.2 km by default)
        :return:
            data on a Nth x Nph grid
        '''

        if l_max is None:
            l_max = shtcoeffs.lmax

        # compute parameters for SHTools
        if Nth is None:
            lm = l_max
        else:
            lm = (Nth-2)//2

        # catch bad inputs
        if l_max > shtcoeffs.lmax:
            l_max = shtcoeffs.lmax
            _warnings.warn('l_max set to {} as maximum degree available in provided coefficients'.format(l_max), UserWarning)
        if l_max > lm:
            lm = l_max
            _warnings.warn('grid size increased to Nth={} as must have Nth >= 2*l_max+2'.format(lm*2+2), UserWarning)
        coeffs = self._convert_SHin(shtcoeffs, l_max=l_max)
        out = _sht.shtools.MakeGravGridDH(coeffs, a**2, a, a=r, sampling=2, lmax=lm, lmax_calc=l_max)
        return -out[0]

    def B_sht_allT(self, sht_allT, r=3480, Nth=None, l_max=None, a=6371.2):
        '''
        Calculates the radial magnetic field on a Driscoll-Healy grid given a list of spherical harmonics coefficients, using shtools

        :param sht_allT:
        :param r:
        :param Nth:
        :param l_max:
        :param a:
        :return:
        '''
        B0 = self.B_sht(sht_allT[0], r=r, Nth=Nth, l_max=l_max, a=a)
        B_t = _np.empty((len(sht_allT), B0.shape[0], B0.shape[1]))
        for i,sh in enumerate(sht_allT):
            B_t[i,:,:] = self.B_sht(sh, r=r, Nth=Nth, l_max=l_max, a=a)
        return B_t

    def gradB_sht(self, shtcoeffs, r=3480, Nth=None, l_max=None, a=6371.2):
        '''
        Find the gradient of the B field, given a list of spherical harmonics coefficients

        :param shtcoeffs: spherical harmonics of the field in [nT]
        :param r:
        :param Nth:
        :param l_max:
        :param a:
        :return: drB, dthB, dphB
            drB : radial gradient of the field in [nT/km]
            dthB : latitudinal gradient of the field [nT/km]
            dphB : longitudinal gradient of the field [nT/km]
        '''

        if type(shtcoeffs) is not _np.ndarray:
            coeffs = shtcoeffs.get_coeffs(normalization='4pi', csphase=1)
        else:
            coeffs = shtcoeffs
        if l_max is None:
            l_max = self.l_max
        if Nth is None:
            lm = l_max
        else:
            lm = (Nth-2)/2
        out = _sht.shtools.MakeGravGradGridDH(coeffs, a ** 2, a, a=r, sampling=2, lmax=lm, lmax_calc=l_max)
        drB = out[2]
        dthB = out[4]
        dphB = out[5]
        return drB, dthB, dphB

    def gradB_sht_allT(self, sht_allT, r=3480, Nth=None, l_max=None, a=6371.2):
        drB0, dthB0, dphB0 = self.gradB_sht(sht_allT[0], r=r, Nth=Nth, l_max=l_max, a=a)
        drB_t = _np.empty((len(sht_allT), drB0.shape[0], drB0.shape[1]))
        dthB_t = _np.empty((len(sht_allT), dthB0.shape[0], dthB0.shape[1]))
        dphB_t = _np.empty((len(sht_allT), dphB0.shape[0], dphB0.shape[1]))
        for i,sh in enumerate(sht_allT):
            drB_t[i,:,:], dthB_t[i,:,:], dphB_t[i,:,:] = self.gradB_sht(sh, r=r, Nth=Nth, l_max=l_max, a=a)
        return drB_t, dthB_t, dphB_t

    def get_gh_at_t(self, time, l_max=None):
        '''
        Calculates Gauss coefficients at time T

        Parameters
        ----------
        time:
            time to calculate parameters
        gt:
            raw data from gufm1 ((number of coeffs x number of tknts) numpy array)
        tknts:
            array of time-knots
        l_max:
            spherical harmonic degree included in model (14)
        jorder:
            order of B-splines (4)
        filename:
            location of raw GUFM1 text file
        Returns
        -------
        g_dict, h_dict:
            dictionaries containing Gauss coefficients at time time
        '''
        g_raw = self._calculate_g_raw_at_t(time)
        g_dict, h_dict = self._convert_g_raw_to_gh(g_raw, l_max=l_max)
        return g_dict, h_dict

    def get_SVgh_at_t(self, time, l_max=None):
        '''
        Calculates Gauss coefficients at time T

        Parameters
        ----------
        time:
            time to calculate parameters
        gt:
            raw data from gufm1 ((number of coeffs x number of tknts) numpy array)
        tknts:
            array of time-knots
        l_max:
            spherical harmonic degree included in model (14)
        jorder:
            order of B-splines (4)
        filename:
            location of raw GUFM1 text file
        Returns
        -------
        g_dict, h_dict:
            dictionaries containing Gauss coefficients at time time
        '''
        SVg_raw = self._calculate_SVgh_raw_at_t(time)
        SVg_dict, SVh_dict = self._convert_g_raw_to_gh(SVg_raw, l_max=l_max)
        return SVg_dict, SVh_dict

    def get_SAgh_at_t(self, time, l_max=None):
        '''
        Calculates Gauss coefficients at time T

        Parameters
        ----------
        time:
            time to calculate parameters
        gt:
            raw data from gufm1 ((number of coeffs x number of tknts) numpy array)
        tknts:
            array of time-knots
        l_max:
            spherical harmonic degree included in model (14)
        jorder:
            order of B-splines (4)
        filename:
            location of raw GUFM1 text file
        Returns
        -------
        g_dict, h_dict:
            dictionaries containing Gauss coefficients at time time
        '''
        SAg_raw = self._calculate_SAgh_raw_at_t(time)
        SAg_dict, SAh_dict = self._convert_g_raw_to_gh(SAg_raw, l_max=l_max)
        return SAg_dict, SAh_dict

    def _Br_for_ml(self,r,th,ph,g,h,m,l, a=6371.2):
        """
        Calculates the Br contribution for one set of m,l, using the potential field.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        g:
            Gauss coefficient (cos term)
        h:
            Gauss coefficient (sin term)
        m:
            Order of calculation
        l:
            Degree of calculation
        a:
            Radius (km) at which Gauss coefficients are calculated

        Returns
        -------
        Br contribution in Tesla at a particular point from a particular degree and order.
        """
        return (l+1.)*a**(l+2.)/abs(r)**(l+2.)*(g*_cos(m*ph) + h*_sin(m*ph))*self._Pml(_cos(th), l, m)

    def _Br_for_ml_complex(self,r,th,ph,c,m,l, a=6371.2):
        """
        Calculates the Br contribution for one set of m,l, using the potential field.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        c:
            complex gauss coefficient
        m:
            Order of calculation
        l:
            Degree of calculation
        a:
            Radius (km) at which Gauss coefficients are calculated

        Returns
        -------
        Br contribution in Tesla at a particular point from a particular degree and order.
        """
        return (l+1.)*a**(l+2.)/abs(r)**(l+2.)*c*_exp(1j*m*ph)*self._Pml(_cos(th), l, m)

    def _dphi_Br_for_ml_complex(self, r, th, ph, c, m, l, a=6371.2):
        """
        Calculates the d_phi(Br)/(R*sin(th)) contribution for one set of m,l, using the potential field.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        c:
            complex gauss coefficient
        m:
            Order of calculation
        l:
            Degree of calculation
        a:
            Radius (km) at which Gauss coefficients are calculated

        Returns
        -------
        d_phi(Br)/(R*sin(th)) in Tesla/m at a particular point from a particular degree and order.
        """
        return 1j*m/(r*_sin(th))*self._Br_for_ml_complex(r, th, ph, c, m, l, a=a)

    def _dtheta_Br_for_ml_complex(self, r, th, ph, c, m, l, a=6371.2):
        """
        Calculates the d_theta(Br)/R contribution for one set of m,l, using the potential field.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        c:
            complex gauss coefficient
        m:
            Order of calculation
        l:
            Degree of calculation
        a:
            Radius (km) at which Gauss coefficients are calculated

        Returns
        -------
        d_theta(Br)/R in Tesla at a particular point from a particular degree and order.
        """
        return (l+1.)*a**(l+2.)/abs(r)**(l+2.)*c*_exp(1j*m*ph)*self._dtheta_Pml(_cos(th), l, m)/r

    def _SVBr_for_ml(self,r,th,ph,SVg,SVh,m,l, a=6371.2):
        """
        Calculates the SVBr contribution for one set of m,l, using the potential field.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        g:
            Gauss coefficient (cos term)
        h:
            Gauss coefficient (sin term)
        m:
            Order of calculation
        l:
            Degree of calculation
        a:
            Radius (km) at which Gauss coefficients are calculated

        Returns
        -------
        SVBr contribution in Tesla at a particular point from a particular degree and order.
        """
        return self._Br_for_ml(r,th,ph,SVg,SVh,m,l, a=a)

    def _SABr_for_ml(self,r,th,ph,SAg,SAh,m,l, a=6371.2):
        """
        Calculates the SABr contribution for one set of m,l, using the potential field.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        g:
            Gauss coefficient (cos term)
        h:
            Gauss coefficient (sin term)
        m:
            Order of calculation
        l:
            Degree of calculation
        a:
            Radius (km) at which Gauss coefficients are calculated

        Returns
        -------
        SVBr contribution in Tesla at a particular point from a particular degree and order.
        """
        return self._Br_for_ml(r,th,ph,SAg,SAh,m,l, a=a)

    def Br(self,r,th,ph, g_dict, h_dict, l_max=None):
        '''
        Calculates the total radial magnetic field at a particular location, given a dictionary of gauss coefficients.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        g_dict:
            dictionary of g (cos) Gauss coefficients, ordered as g[l][m].
        h_dict:
            dictionary of h (sin) Gauss coefficients, ordered as h[l][m]. h coefficients for m=0 should be explicitly included as 0.0
        l_max:
            maximum degree to use in calculation. By default uses all supplied degrees.

        Returns
        -------
        Total Br at a particular point (Tesla)
        '''
        if l_max is None:
            l_max = max(g_dict.keys())
        Br_sum = 0
        for l in range(1,l_max+1):
            for m in range(l+1):
                Br_sum += self._Br_for_ml(r,th,ph, g_dict[l][m], h_dict[l][m], m, l)
        return Br_sum

    def Br_complex(self, r, th, ph, c_dict, l_max=None):
        '''
        Calculates the total radial magnetic field at a particular location, give a dictionary of complex gauss coefficients.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        c_dict:
            dictionary of complex Gauss coefficients, ordered as c[l][m].
        l_max:
            maximum degree to use in calculation. By default uses all supplied degrees.

        Returns
        -------
        Total Br at a particular point (Tesla, complex)
        '''
        if l_max is None:
            l_max = max(c_dict.keys())
        Br_sum = 0
        for l in range(1,l_max+1):
            for m in range(l+1):
                Br_sum += self._Br_for_ml_complex(r,th,ph, c_dict[l][m], m, l)
        return Br_sum

    def grad_Br_complex(self,r,th,ph, c_dict, l_max=None):
        '''
        Calculates the hoizontal gradient of the total radial magnetic field at a particular location, give a dictionary of complex gauss coefficients.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        c_dict:
            dictionary of complex Gauss coefficients, ordered as c[l][m].
        l_max:
            maximum degree to use in calculation. By default uses all supplied degrees.

        Returns
        -------
        grad_theta(Br), grad_phi(Br) at a particular point (Tesla, complex)
        '''
        if l_max is None:
            l_max = max(c_dict.keys())
        dth_Br_sum = 0
        dph_Br_sum = 0
        for l in range(1,l_max+1):
            for m in range(l+1):
                dth_Br_sum += self._dtheta_Br_for_ml_complex(r,th,ph, c_dict[l][m], m, l)
                dph_Br_sum += self._dphi_Br_for_ml_complex(r,th,ph, c_dict[l][m], m, l)
        return dth_Br_sum, dph_Br_sum

    def SVBr(self,r,th,ph, SVg_dict, SVh_dict, l_max=None):
        '''
        Calculates the total radial magnetic field at a particular location, give a dictionary of gauss coefficients.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        g_dict:
            dictionary of g (cos) Gauss coefficients, ordered as g[l][m].
        h_dict:
            dictionary of h (sin) Gauss coefficients, ordered as h[l][m]. h coefficients for m=0 should be explicitly included as 0.0
        l_max:
            maximum degree to use in calculation. By default uses all supplied degrees.

        Returns
        -------
        Total Br at a particular point (Tesla)
        '''
        if l_max is None:
            l_max = max(SVg_dict.keys())
        SVBr_sum = 0
        for l in range(1,l_max+1):
            for m in range(l+1):
                SVBr_sum += self._SVBr_for_ml(r,th,ph, SVg_dict[l][m], SVh_dict[l][m], m, l)
        return SVBr_sum

    def SABr(self,r,th,ph, SAg_dict, SAh_dict, l_max=None):
        '''
        Calculates the total radial magnetic field at a particular location, give a dictionary of gauss coefficients.

        Inputs
        ------
        r:
            radius location (km)
        th:
            latitude location (radians)
        ph:
            longitude location (radians)
        g_dict:
            dictionary of g (cos) Gauss coefficients, ordered as g[l][m].
        h_dict:
            dictionary of h (sin) Gauss coefficients, ordered as h[l][m]. h coefficients for m=0 should be explicitly included as 0.0
        l_max:
            maximum degree to use in calculation. By default uses all supplied degrees.

        Returns
        -------
        Total Br at a particular point (Tesla)
        '''
        if l_max is None:
            l_max = max(SAg_dict.keys())
        SABr_sum = 0
        for l in range(1,l_max+1):
            for m in range(l+1):
                SABr_sum += self._SABr_for_ml(r,th,ph, SAg_dict[l][m], SAh_dict[l][m], m, l)
        return SABr_sum

class Gufm1(MagModel):
    def __init__(self, data_file = _gufm1_data_file):
        self.data_file = data_file
        self.gt, self.tknts, self.l_max, self.bspl_order = self._read_data(data_file=self.data_file)
        self.dT = self.tknts[len(self.tknts)//2+1]-self.tknts[len(self.tknts)//2]
        self.bspline = self._make_bspline_basis(self.tknts)
        self.T_start = self.tknts[self.bspl_order-1]
        self.T_end = self.tknts[-self.bspl_order]
        self.name = 'GUFM-1'

class GufmSatE3(MagModel):
    def __init__(self, data_file = _gufmsatE3_data_file):
        self.data_file = data_file
        self.gt, self.tknts, self.l_max, self.bspl_order = self._read_data(data_file=self.data_file)
        self.dT = self.tknts[len(self.tknts)//2+1]-self.tknts[len(self.tknts)//2]
        self.bspline = self._make_bspline_basis(self.tknts)
        self.T_start = self.tknts[self.bspl_order-1]
        self.T_end = self.tknts[-self.bspl_order]
        self.name = 'GUFM-SAT-E3'

class GufmSatQ2(MagModel):
    def __init__(self, data_file = _gufmsatQ2_data_file):
        self.data_file = data_file
        self.gt, self.tknts, self.l_max, self.bspl_order = self._read_data(data_file=self.data_file)
        self.dT = self.tknts[len(self.tknts)//2+1]-self.tknts[len(self.tknts)//2]
        self.bspline = self._make_bspline_basis(self.tknts)
        self.T_start = self.tknts[self.bspl_order-1]
        self.T_end = self.tknts[-self.bspl_order]
        self.name = 'GUFM-SAT-Q2'

class GufmSatQ3(MagModel):
    def __init__(self, data_file = _gufmsatQ3_data_file):
        self.data_file = data_file
        self.gt, self.tknts, self.l_max, self.bspl_order = self._read_data(data_file=self.data_file)
        self.dT = self.tknts[len(self.tknts)//2+1]-self.tknts[len(self.tknts)//2]
        self.bspline = self._make_bspline_basis(self.tknts)
        self.T_start = self.tknts[self.bspl_order-1]
        self.T_end = self.tknts[-self.bspl_order]
        self.name = 'GUFM-SAT-Q3'

class Chaos6(MagModel):
    def __init__(self, data_file = _chaos6_data_file):
        self.data_file = data_file
        self.gt, self.tknts, self.l_max, self.bspl_order = self._read_data(data_file=self.data_file)
        self.dT = self.tknts[len(self.tknts)//2+1]-self.tknts[len(self.tknts)//2]
        self.bspline = self._make_bspline_basis(self.tknts)
        self.T_start = self.tknts[self.bspl_order-1]
        self.T_end = self.tknts[-self.bspl_order]
        self.name = 'CHAOS-6'

