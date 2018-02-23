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
    """
    plots data on a Driscoll-Healy grid onto a map of the Earth

    :param DH:
    :param fig:
    :param title:
    :param sym:
    :param clbl:
    :param proj:
    :param lon_0:
    :param zmax:
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
    """
    plots data on a Driscoll-Healy grid onto a map of the Earth

    :param DH:
    :param fig:
    :param title:
    :param sym:
    :param clbl:
    :param proj:
    :param lon_0:
    :param zmax:
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

def period_wavenumber(m, freq, yf, Tmin=2.5, Tmax=24, m_max=12, Nylabels=10, title='period-wavenumber',
                           savefig=False, logfft=False, vmin=None, vmax=None, newfig=False, colorbar=True, cblbl=None, cbfmt=None):
    if newfig:
        plt.figure(figsize=(7, 5))
    if logfft:
        yplt = np.log(np.abs(yf[:yf.shape[0] // 2, :]))
    else:
        yplt = np.abs(yf[:yf.shape[0] // 2, :])
    f = plt.pcolormesh(m, freq, yplt, cmap=mpl.cm.afmhot_r, vmin=vmin, vmax=vmax)
    plt.ylim(1 / Tmax, 1 / Tmin)
    ylabel_loc = np.linspace(1 / Tmax, 1 / Tmin, Nylabels)
    plt.yticks(ylabel_loc, ['{0:.1f}'.format(1 / x) for x in ylabel_loc])
    plt.xlim(-m_max, m_max)
    xticks = np.linspace(-m_max, m_max, m_max + 1)
    plt.xticks(xticks)
    plt.grid()
    plt.title(title)
    if cblbl is None:
        if logfft:
            cblbl = 'log |FFT|'
        else:
            cblbl = '|FFT|'
    if cbfmt is None:
        cbfmt = '%.1e'
    if colorbar:
        plt.colorbar(label=cblbl, format=cbfmt)
    plt.ylabel('period (yrs)')
    plt.xlabel('m (longitudinal wavenumber)')
    if savefig:
        plt.savefig(title + '.png')
    return f

def period_wavenumber_contourf(m, freq, yf, Tmin=2.5, Tmax=24, m_max=12, Nylabels=10, title='period-wavenumber',
                           savefig=False, logfft=False, vmin=None, vmax=None, newfig=False, colorbar=True, cblbl=None, cbfmt=None,
                                    over_color='black', under_color='white', extend='both'):

    if newfig:
        plt.figure(figsize=(7, 5))
    if logfft:
        yplt = np.log(np.abs(yf[:yf.shape[0] // 2, :]))
    else:
        yplt = np.abs(yf[:yf.shape[0] // 2, :])
    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = np.max(yplt)
    contours = np.linspace(vmin, vmax, 16)

    f = plt.contourf(m, freq, yplt, contours, cmap=mpl.cm.afmhot_r, extend=extend)

    f.cmap.set_over(over_color)
    f.cmap.set_under(under_color)
    plt.ylim(1 / Tmax, 1 / Tmin)
    ylabel_loc = np.linspace(1 / Tmax, 1 / Tmin, Nylabels)
    plt.yticks(ylabel_loc, ['{0:.1f}'.format(1 / x) for x in ylabel_loc])
    plt.xlim(-m_max, m_max)
    xticks = np.linspace(-m_max, m_max, m_max + 1)
    plt.xticks(xticks)
    plt.grid()
    plt.title(title)
    if cblbl is None:
        if logfft:
            cblbl = 'log |FFT|'
        else:
            cblbl = '|FFT|'
    if cbfmt is None:
        cbfmt = '%.1e'
    if colorbar:
        plt.colorbar(f, label=cblbl, format=cbfmt)
    plt.ylabel('period (yrs)')
    plt.xlabel('m (longitudinal wavenumber)')
    if savefig:
        plt.savefig(title + '.png')
    return f

def longitudetime(z, T, title='Longitude vs Time', newfig=False, vmin=None, vmax=None):
    T_plt = T
    ph_plt = np.linspace(0, 360, z.shape[1])
    xx, yy = np.meshgrid(ph_plt, T_plt)
    if vmax is None:
        vmax = np.max(np.abs(z))
    if vmin is None:
        vmin = -vmax
    if newfig:
        plt.figure()
    plt.pcolormesh(xx, yy, z, cmap=mpl.cm.PuOr, vmin=vmin, vmax=vmax)
    plt.xlim(0, 360)
    plt.ylim(T[0], T[-1])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('Time (yr)')
    plt.title(title)

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

def two_pwn_contourf(m, freq, yf1, yf2, title1='title 1', title2='title 2', newfig=True, Tmin=2.5, Tmax=24, m_max=12, Nylabels=10,
                        logfft=False,over_color='black', under_color='white', extend='both', vmin=0.1, vmax=10.,
                       cblbl=None, cbar=True, savename=None, cbfmt=None):
    if newfig:
        fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    period_wavenumber_contourf(m, freq, yf1, Tmin=Tmin, Tmax=Tmax, m_max=m_max, Nylabels=Nylabels, title=title1,
                           savefig=False, logfft=logfft, vmin=vmin, vmax=vmax, newfig=False, colorbar=False, cblbl=None, cbfmt=None,
                                    over_color=over_color, under_color=under_color, extend=extend)
    plt.subplot(122)
    f = period_wavenumber_contourf(m, freq, yf2, Tmin=Tmin, Tmax=Tmax, m_max=m_max, Nylabels=Nylabels, title=title2,
                           savefig=False, logfft=logfft, vmin=vmin, vmax=vmax, newfig=False, colorbar=False, cblbl=None, cbfmt=None,
                                    over_color=over_color, under_color=under_color, extend=extend)
    if cblbl is None:
        if logfft:
            cblbl = 'log |FFT|'
        else:
            cblbl = '|FFT|'
    if cbfmt is None:
        cbfmt = '%.1e'
    plt.tight_layout()
    if cbar:
        plt.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
        fig.colorbar(f, cax=cbar_ax, label=cblbl, format=cbfmt)
    if savename:
        plt.savefig(savename)
    return fig

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

def vs_latitude(th, z, title='', savename=None):
    lat = th-90
    plt.figure(figsize=(8,4))
    plt.plot(lat, z)
    plt.xlim(-90,90)
    plt.title(title)
    plt.grid()
    if savename:
        plt.savefig(savename)
    plt.xlabel('degrees latitude')

def correlation_contourf(phases, periods, corr, title='Correlation', newfig=False, savename=None,
                              colorbar=True, cblbl=None, cbfmt=None, vmin=None, vmax=None,
                              over_color='white', under_color='black', extend='neither', cmap='RdBu_r', real_period=None, real_phase=None, real_markersize=10):
    """ plot the reuslts of sweeping the correlation across phase and period

    :param periods:
    :param corr:
    :param title:
    :param newfig:
    :param savename:
    :param colorbar:
    :param cblbl:
    :param cbfmt:
    :param vmin:
    :param vmax:
    :param over_color:
    :param under_color:
    :param extend:
    :param cmap:
    :return:
    """
    z = np.array(corr.T)
    z = np.concatenate((z,-z), axis=1)
    phase_plt = np.linspace(0,360,len(phases)*2, endpoint=False)
    zpeak = np.max(z)
    peri,phsi = np.where(z==zpeak)
    phase_val = phase_plt[phsi[0]]
    per_val = periods[peri[0]]
    print("Peak Correlation phase={0:.1f} degrees, period={1:.2f} yrs".format(phase_val, per_val))
    if vmax is None:
        vmax = np.max(np.abs(z))
    if vmin is None:
        vmin = -vmax
    contours = np.linspace(vmin, vmax, 16)
    if newfig:
        plt.figure()
    f = plt.contourf(phase_plt, periods, z, contours, cmap=cmap, extend=extend)
    if real_period is not None and real_phase is not None:
        plt.plot(real_phase, real_period, '*', color='white', markeredgecolor='black', markersize=10)
    f.cmap.set_over(over_color)
    f.cmap.set_under(under_color)
    plt.title(title)
    plt.grid()
    if cblbl is None:
        cblbl = 'correlation'
    if cbfmt is None:
        cbfmt = '%.1f'
    if colorbar:
        plt.colorbar(f, label=cblbl, format=cbfmt)
    plt.xlabel('phase (degrees)')
    plt.ylabel('period (years)')
    if savename:
        plt.savefig(savename)
    return f

def amplitude_fit_2waves(amp_swept, amp_min, amp_max, Namps, newfig=False, title='Amplitude Fit for Two Waves',
                              savename=None, clbl=None, cfmt=None, over_color='Black', under_color='white',
                              extend='both',
                              vmin=None, vmax=None, real_amp1=None, real_amp2=None, xlbl=None, ylbl=None):
    if newfig:
        plt.figure(figsize=(6, 4))
    amp = np.linspace(amp_min, amp_max, Namps)
    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = np.max(amp_swept)
    contours = np.linspace(vmin, vmax, 16)
    f = plt.contourf(amp, amp, amp_swept.T, contours, cmap=mpl.cm.afmhot_r, extend=extend)
    f.cmap.set_over(over_color)
    f.cmap.set_under(under_color)
    plt.ylim(amp_min, amp_max)
    plt.xlim(amp_min, amp_max)
    ticks = np.linspace(amp_min, amp_max, (amp_max - amp_min) * 2 + 1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.gca().set_aspect('equal', adjustable='box')
    if real_amp1 is not None and real_amp2 is not None:
        plt.plot(real_amp1, real_amp2, '*', color='white', markeredgecolor='black', markersize=20)
    if xlbl is not None:
        plt.xlabel(xlbl)
    if ylbl is not None:
        plt.ylabel(ylbl)
    if clbl is None:
        clbl = 'goodness of fit'
    if cfmt is None:
        cfmt = '%.1f'
    plt.colorbar(f, label=clbl, format=cfmt)
    plt.title(title)
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
