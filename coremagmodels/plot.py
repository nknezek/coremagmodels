import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

def pcolormesh_DH(DH, newfig=True, symmetric_color_scale=True, clbl='scale', title='title', zmax=None):
    """
    plots data on a Driscoll-Healy grid

    :param DH:
    :param fig:
    :param sym:
    :param clbl:
    :param tit:
    :param zmax:
    :return:
    """
    cm = mpl.cm.PuOr_r
    ph = np.linspace(-180, 180, DH.shape[1], endpoint=False)
    th = np.linspace(90, -90, DH.shape[0], endpoint=False)
    pp, tt = np.meshgrid(ph, th)
    if newfig:
        plt.figure(figsize=(8, 5))
    if zmax is None:
        zmax = np.max(np.abs(DH))
    if symmetric_color_scale:
        vmin = -zmax
        vmax = zmax
    else:
        vmin = 0
        vmax = zmax
    plt.pcolormesh(pp, tt, DH, vmin=vmin, vmax=vmax, cmap=cm)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xticks(np.linspace(-180, 180, 9))
    plt.yticks(np.linspace(-90, 90, 5))
    cbar = plt.colorbar()
    cbar.set_label(clbl)
    plt.title(title)
    plt.grid()

def quiver_DH(zth, zph, newfig=True, title='title'):
    """
    plots a vector field

    :param zth:
    :param zph:
    :param fig:
    :param title:
    :return:
    """
    ph = np.linspace(-180,180,zph.shape[1],endpoint=False)
    th = np.linspace(90,-90,zph.shape[0],endpoint=False)
    if newfig:
        plt.figure(figsize=(8,5))
    Q = plt.quiver(ph,th,zph,-zth)
    plt.title(title)
    qarr_scale = np.max((zph**2 + zth**2)**0.5)
    if qarr_scale > 1:
        qarr_scale = int(qarr_scale)
        qk = plt.quiverkey(Q, 0.9, 1.03, qarr_scale, r'{0:.0f} '.format(qarr_scale)+ r'$\frac{km}{yr}$', labelpos='E',
                       fontproperties={'weight': 'bold'})
    else:
        qk = plt.quiverkey(Q, 0.9, 1.03, qarr_scale, r'{0:.0e} '.format(qarr_scale)+ r'$\frac{km}{yr}$', labelpos='E',
                       fontproperties={'weight': 'bold'})

    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.xticks(np.linspace(-180,180,9))
    plt.yticks(np.linspace(-90,90,5))
    plt.grid()

def base_quiver(zth, zph, newfig=True, title='title', proj='robin', lon_0=0., qarr_scale_mod=1, qkey=10., units='xy'):
    """
    Plots a vector field on a map of the earth

    :param zth:
    :param zph:
    :param newfig:
    :param title:
    :param proj:
    :param lon_0:
    :return:
        plot of the earth with vector field
    """
    if newfig:
        plt.figure(figsize=(8,5))
    ph = np.linspace(-180,180,zph.shape[1],endpoint=False)
    th = np.linspace(90,-90,zph.shape[0],endpoint=False)
    m = Basemap(projection=proj, lon_0=lon_0)
    m.drawmeridians(np.linspace(-180, 180, 9))
    m.drawparallels(np.linspace(-90, 90, 5))
    pp, tt = np.meshgrid(ph,th)
    # qarr_scale = int(np.max((zph**2 + zth**2)**0.5)/10)*10 * qarr_scale_mod
    Q = m.quiver(pp, tt, zph, -zth, latlon=True, scale=qarr_scale_mod, units=units)
    m.drawcoastlines()
    if proj=='robin':
        qk = plt.quiverkey(Q, 0.12, 0.03, qkey, '{0:.1f} km/yr'.format(qkey), labelpos='W')
    else:
        qk = plt.quiverkey(Q, 0.12, 0.1, qkey, '{0:.1f} km/yr'.format(qkey), labelpos='W')
    plt.title(title)

def base_DH(DH, newfig=True, title='title', symmetric_color_scale=True, clbl='scale', proj='moll', lon_0=0., zmax=None, cbar=True, coastlw=0.5):
    """plots data on a Driscoll-Healy grid onto a map of the Earth

    :param DH:
    :param newfig:
    :param title:
    :param symmetric_color_scale:
    :param clbl:
    :param proj:
    :param lon_0:
    :param zmax:
    :param cbar:
    :param coastlw:
    :return:
    """
    if newfig:
        plt.figure(figsize=(8, 5))
    cm = mpl.cm.PuOr_r
    ph = np.linspace(-180, 180, DH.shape[1], endpoint=False)
    th = np.linspace(90, -90, DH.shape[0], endpoint=False)
    pp, tt = np.meshgrid(ph, th)
    if zmax is None:
        zmax = np.max(np.abs(DH))
    if symmetric_color_scale:
        vmin = -zmax
        vmax = zmax
    else:
        vmin = 0
        vmax = zmax
    m = Basemap(projection=proj, lon_0=lon_0)
    m.drawmeridians(np.linspace(-180, 180, 9))
    m.drawparallels(np.linspace(-90, 90, 9))
    im = m.pcolormesh(pp, tt, DH, vmin=vmin, vmax=vmax, cmap=cm, latlon=True)
    m.drawcoastlines(linewidth=coastlw)
    plt.title(title)
    if cbar:
        cb = m.colorbar(im, "bottom", size="5%", pad="2%")
        cb.set_label(clbl)
    return im

def contourf_DH(DH, newfig=True, title='title', symmetric_color_scale=True, clbl='scale', proj='robin',
                     lon_0=0., zmax=None, cbar=True, coastlw=0.5, cfmt=None):
    """plots data on a Driscoll-Healy grid onto a map of the Earth

    :param DH:
    :param newfig:
    :param title:
    :param symmetric_color_scale:
    :param clbl:
    :param proj:
    :param lon_0:
    :param zmax:
    :param cbar:
    :param coastlw:
    :param cfmt:
    :return:
    """
    if newfig:
        plt.figure(figsize=(8, 5))
    cm = mpl.cm.PuOr_r
    ph = np.linspace(-180, 180, DH.shape[1], endpoint=False)
    th = np.linspace(90, -90, DH.shape[0], endpoint=False)
    pp, tt = np.meshgrid(ph, th)
    if zmax is None:
        zmax = np.max(np.abs(DH))
    if symmetric_color_scale:
        contours = np.linspace(-zmax, zmax, 16)
    else:
        contours = np.linspace(0, zmax, 16)
    m = Basemap(projection=proj, lon_0=lon_0)
    m.drawmeridians(np.linspace(-180, 180, 9))
    m.drawparallels(np.linspace(-90, 90, 9))
    im = m.contourf(pp, tt, DH, contours, cmap=cm, latlon=True)
    m.drawcoastlines(linewidth=coastlw)
    plt.title(title)
    if cbar:
        cb = m.colorbar(im, "bottom", size="5%", pad="2%", format=cfmt)
        cb.set_label(clbl)
    return im

def two_pcolor(z1, z2, newfig=True, title1='title 1', title2='title 2', symmetric_color_scale=True,
                 clbl='scale', proj='moll', lon_0=0., zmax=None, cbar=True, savename=None, cfmt='%.1f'):
    if newfig:
        fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    base_DH(z1, newfig=False, title=title1, proj=proj, zmax=zmax, cbar=False, lon_0=lon_0,
                      symmetric_color_scale=symmetric_color_scale)
    plt.subplot(122)
    f = base_DH(z2, newfig=False, title=title2, proj=proj, zmax=zmax, cbar=False, lon_0=lon_0,
                          symmetric_color_scale=symmetric_color_scale)

    plt.tight_layout()
    if cbar:
        plt.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        fig.colorbar(f, cax=cbar_ax, label=clbl, format=cfmt)
    if savename:
        plt.savefig(savename)


def two_contourf(z1, z2, newfig=True, title1='title 1', title2='title 2', symmetric_color_scale=True,
                 clbl='scale', proj='moll', lon_0=0., zmax=None, cbar=True, savename=None, cfmt='%.1f'):
    if newfig:
        fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    contourf_DH(z1, newfig=False, title=title1, proj=proj, zmax=zmax, cbar=False, lon_0=lon_0,
                          symmetric_color_scale=symmetric_color_scale)
    plt.subplot(122)
    f = contourf_DH(z2, newfig=False, title=title2, proj=proj, zmax=zmax, cbar=False, lon_0=lon_0,
                              symmetric_color_scale=symmetric_color_scale)

    plt.tight_layout()
    if cbar:
        plt.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        fig.colorbar(f, cax=cbar_ax, label=clbl, format=cfmt)
    if savename:
        plt.savefig(savename)

def lm_contourf(svrcilm, l_pltmax=14, vmax=None, title='Residual Spectral power at l and m'):
    """ plots spherical harmonic power at each l,m in a contour plot

    :param svrcilm:
    :param l_pltmax:
    :param vmax:
    :param title:
    :return:
    """
    svr_mag = (svrcilm[0, :, :] ** 2 + svrcilm[1, :, :] ** 2) ** 0.5
    if vmax is None:
        vmax = np.max(np.abs(svr_mag))
    contours = np.linspace(0, vmax, 16)
    plt.contourf(svr_mag[:l_pltmax + 1, :l_pltmax + 1].T, contours, cmap=mpl.cm.inferno_r)
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title(title)
    plt.colorbar(label='coefficient magnitude in orthonormal norm')

def lm_dots(svrcilm, l_pltmax=14, ymax=None, title='Residual Spectral power at l and m'):
    """plots spherical harmonics power at each l,m vs l with points for each m

    :param svrcilm:
    :param l_pltmax:
    :param ymax:
    :param title:
    :return:
    """
    for l in range(0, l_pltmax + 1):
        ls = (svrcilm[0, l:l_pltmax + 1, l] ** 2 + svrcilm[1, l:l_pltmax + 1, l] ** 2) ** 0.5
        plt.plot(range(l, l_pltmax + 1), ls, '.')
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.xlabel('l')
    plt.ylabel('power')
    plt.title(title)

def fft_coeffs_mag(lm_fft, svn, Nfft=None, fits=None, sdls=None, sdhs=None, deg_fits=(1, 2, 2, 2, 2), log=False,
                        title='Actual', ls='.-', ymax=None):
    """ plots the magnitude of individual fft coefficients

    :param lm_fft:
    :param Nfft:
    :param fits:
    :param sdls:
    :param sdhs:
    :param deg_fits:
    :param log:
    :param title:
    :param ls:
    :return:
    """
    l_max = lm_fft.shape[0]-1

    if Nfft is None:
        Nfft = lm_fft.shape[-1]
    plt.figure(figsize=(16, 5))
    if log:
        for i in range(Nfft):
            plt.subplot(1, Nfft, i + 1)

            data = np.abs(lm_fft[:, :, i])
            for l in range(lm_fft.shape[0]):
                plt.semilogy(range(l, l_max + 1), data[l:, l], ls)

            if (fits is None) or (sdls is None) or (sdhs is None):
                fit, sdl, sdh = svn.fit_lm_sd_in_log(data, deg=deg_fits[i])
            else:
                fit = fits[i, :]
                sdl = sdls[i, :]
                sdh = sdhs[i, :]
            plt.semilogy(range(l_max + 1), fit, 'k-')
            plt.semilogy(range(l_max + 1), sdl, 'k--')
            plt.semilogy(range(l_max + 1), sdh, 'k--')
            plt.ylim(1, 2e4)
            plt.grid()
            plt.xlabel('l')
            plt.title('{} fft c{}'.format(title, i))
    else:
        for i in range(Nfft):
            plt.subplot(1, Nfft, i + 1)

            data = np.abs(lm_fft[:, :, i])
            for l in range(lm_fft.shape[0]):
                plt.plot(range(l, l_max + 1), data[l:, l], ls)

            if (fits is None) or (sdls is None) or (sdhs is None):
                fit, sdl, sdh = svn.fit_lm_sd_in_linear(data, deg=deg_fits[i])
            else:
                fit = fits[i, :]
                sdl = sdls[i, :]
                sdh = sdhs[i, :]
            plt.plot(range(l_max + 1), fit, 'k-')
            plt.plot(range(l_max + 1), sdl, 'k--')
            plt.plot(range(l_max + 1), sdh, 'k--')
            if ymax is None:
                ymax = np.max(np.abs(lm_fft)) * 1.1
            plt.ylim(0, ymax)
            plt.grid()
            plt.xlabel('l')
            plt.title('{} fft c{}'.format(title, i))

def fft_coeffs_phase(lm_fft, svn, Nfft=None, title='Actual'):
    if Nfft is None:
        Nfft = lm_fft.shape[-1]
    l_max = lm_fft.shape[0]-1
    plt.figure(figsize=(16,5))
    _, phases = svn.get_lm_magphase(lm_fft)
    for i in range(Nfft):
        plt.subplot(1,Nfft,i+1)
        for l in range(lm_fft.shape[0]):
            plt.plot(range(l,l_max+1), phases[l:,l,i], '.')
        plt.ylim(-3.2, 3.2)
        plt.grid()
        plt.xlabel('l')
        plt.title('{} fft c{}'.format(title, i))
