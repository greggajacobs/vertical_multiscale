# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:55:21 2025

@author: ga_ja
"""

import numpy as np
from numpy import ma
import scipy
from pathlib import Path
from astropy.convolution import convolve
import gsw
import nc_util
import plotting_3dvar as plt3dvr
import plotting_cov as pltcov
import plotting_profile as pltprof
import properties

#%%
def vert_cov_delta(Cvf):
    """
    Convert vertical covariance into a delta vertically

    Args:
        Cvf (np array): covariance ordered as [position][nv,nv].

    Returns:
        None.

    """
    for icov in range(0, Cvf.shape[0]):
        diag = np.diag(Cvf[icov])
        Cvf[icov] = np.diag(diag)

#%%
def vert_cov_gauss(covariance_form, Cvf, nz, vscale_points):
    """
    Convert vertical covariance into Gaussian vertically

    Args:
        Cvf (np array): covariance ordered as [position][nv,nv].
        nz (int): number of vertical z points.
        vscale_points (float): decorrelation scale in points.

    Returns:
        None.

    """
    if covariance_form == 'histo':
        return

    if covariance_form == '2gauss':
        for icov in range(0, Cvf.shape[0]):
            idepth = np.arange(0, nz)
            diff = np.subtract.outer(idepth, idepth)
            localizer = np.exp(-(diff * diff) / (vscale_points * vscale_points))
            for ibeg in [0, nz]:
                for jbeg in [0, nz]:
                    if ibeg == jbeg:
                        Cvf[icov][ibeg:ibeg+nz,jbeg:jbeg+nz] = Cvf[icov][ibeg:ibeg+nz,jbeg:jbeg+nz] * localizer
                    else:
                        Cvf[icov][ibeg:ibeg+nz,jbeg:jbeg+nz] = 0.0
    elif covariance_form == '2gausscross':
        for icov in range(0, Cvf.shape[0]):
            idepth = np.arange(0, nz)
            diff = np.subtract.outer(idepth, idepth)
            localizer = np.exp(-(diff * diff) / (vscale_points * vscale_points))
            for ibeg in [0, nz]:
                for jbeg in [0, nz]:
                    Cvf[icov][ibeg:ibeg+nz,jbeg:jbeg+nz] = Cvf[icov][ibeg:ibeg+nz,jbeg:jbeg+nz] * localizer

#%%
def read_Cvf(args, month_n, depth_edges):
    """
    Read and prepare the vertical covariances
    Read data only in args beg&end lat&lon range
    Read in only within beg/end_alat/alon range

    Args:
        args (dict): input arguments.
        month_n (int): month number (Jan = 1).
        depth_edges (np array): edges of vertical bins used in analysis.

    Returns:
        Cvf (np array): covariance functions arranged as Cvf[position][TSdepth,TSdepth].
        Cv_lon (np array): longitude of netcdf dataset.
        Cv_lat (np array): latitude of netcdf dataset.
        Cv_depth (np array): depth of netcdf dataset.
        Cv_lonf (np array): Cv_lon flattened.
        Cv_latf (np array): Cv_lat flattened.
        Cv_nh (np array): number of horizontal points after flattening.
        Cv_nv (np array): number of vertical points in netcdf (twice Cv_nz because it is T and S).
        Cv_nz (np array): number of z points in either T or S.

    """
    infile = Path(args.in_covariance_netcdf_format.format(month_n))
    # read in just the dims
    vcov = nc_util.nc_ds()
    vcov.read(infile, 'covariance', read_data = False, read_atts = False)
    
    # determine the slice objects of lat, lon dims
    sox = vcov.ind_range('lon', args.beg_alon, args.end_alon, stride = 1, pad = 0)
    soy = vcov.ind_range('lat', args.beg_alat, args.end_alat, stride = 1, pad = 0)
    soz1 = np.s_[:]
    soz2 = np.s_[:]
    
    # read the data with the dimensions according to slice objects for lat and lon
    # read in all depths, subset afterward
    vcov.read(infile, 'covariance', so = (soy, sox, soz1, soz2))
    meants = nc_util.nc_ds()
    meants.read(infile, 'mean', so = (soy, sox, soz1))
    stdts = nc_util.nc_ds()
    stdts.read(infile, 'std', so = (soy, sox, soz1))
    
    # keep only the z values in the args.beg_depth to args.end_depth + 100.0 range
    
    (idim, dname, dim) = vcov.find_dim('double_depth1')
    z = dim.vals
    keepers2 = (args.beg_adepth <= z) & (z <= args.end_adepth + 100.0)
    keepers1 = keepers2[0: int(len(keepers2) / 2)]
    vcov.sub_dim_flag('double_depth1', keepers2)
    vcov.sub_dim_flag('double_depth2', keepers2)
    
    meants.sub_dim_flag('double_depth1', keepers2)
    stdts.sub_dim_flag('double_depth1', keepers2)
    
    # flatten the first two dimensions (lat, lon) leaving others
    Cvf = vcov.vals.reshape(-1, *vcov.vals.shape[2:]) # use reshape to flatten just the first two dims
    meantsf = meants.vals.reshape(-1, *meants.vals.shape[2:]) # use reshape to flatten just the first two dims
    stdtsf = stdts.vals.reshape(-1, *meants.vals.shape[2:]) # use reshape to flatten just the first two dims
        
    (i, dname, dim) = vcov.find_dim('lon')
    Cv_lon = dim.vals
    (i, dname, dim) = vcov.find_dim('lat')
    Cv_lat = dim.vals
    (i, dname, dim) = vcov.find_dim('depth')
    Cv_depth = dim.vals
    (Cv_long, Cv_latg) = np.meshgrid(Cv_lon, Cv_lat)
    Cv_lonf = Cv_long.flatten()
    Cv_latf = Cv_latg.flatten()
    
    Cv_nh = len(Cv_lonf)
    Cv_nv = len(meants.dims[dname].vals)
    Cv_nz = int(len(meants.dims[dname].vals) / 2)    # number of depths since T&S are included together
    
    # construct the depth edges
    keep_de = keepers1.copy()
    last_indx = np.where(keep_de)[0][-1]
    keep_de[last_indx + 1] = True
    keep_de = ma.append(keep_de, keep_de[-1])
    depth_edges_use = depth_edges[keep_de]

    return (Cvf, Cv_lon, Cv_lat, Cv_depth, Cv_lonf, Cv_latf, Cv_nh, Cv_nv, Cv_nz, meants, stdts, meantsf, stdtsf, depth_edges_use)

    
def clean_Cvf(Cvf):
        
    # filter the covariances if necessary
    # This assumes a transect is in memory, not a lat,lon area of Cvf
    kernel = np.ones((3,3,3))
    kernel /= np.sum(kernel)
    for i in range(0, 2):
        Cvf_orig = Cvf.copy()
        Cvf = convolve(Cvf, kernel, boundary = 'extend')
        print(i, 'max conv change: ', np.max(np.abs(Cvf - Cvf_orig)))

    # ensure covariances positive semi-definite
    # make them symmetric and eliminate small eigenvalues
    # print('cutting eigenvalues')
    eiglimit = 1.0e-8
    for icov in range(0, Cvf.shape[0]):
        c = Cvf[icov][:,:].copy()
        c = (c + c.T) / 2.0
        maxasym = np.max(np.abs(c - Cvf[icov]) / np.abs(Cvf[icov]))
        Cvf[icov] = c
        print(f'icov {icov} max asym {maxasym}')
        # eigendecomposition, sort eigenvalues
        (evals, evecs) = np.linalg.eig(c)
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]
        print(np.min(evals))
        neigvals = np.argmax(evals <= np.max(evals) * eiglimit)
        print(icov, neigvals, evals[neigvals] / np.max(evals))
        evals[evals <= np.max(evals) * eiglimit] = 0.0
        cov2 = evecs @ np.diag(evals) @ evecs.T
        Cvf[icov] = cov2
        
    return Cvf


#%%
def horizontal_correlation_gauss(args, plot_dir, An_lonf, An_latf, hlength_scale):
    Bh = properties.distsq_deg(An_latf, An_lonf, An_latf, An_lonf, hlength_scale)
    outfile = plot_dir / 'Bh.png'
    title = 'Bh matrix'
    plt3dvr.matrix(args, Bh, title, outfile,
                   x_label = 'Analysis grid index', y_label = 'Analysis grid index',
                   cb_label = 'Correlation')
    return Bh

#%%
def K(args, plot_dir, An_nh, S, Cvf, cov_amplitude):
    """
    Compute K matrix from vertical observation operator and vertical covariance

    Args:
        args (dict): runtime arguments.
        An_nh (int): number of analysis positions.
        S (np array): stht operator organized as [position][twice number of depths].
        Cvf (np array): covariance functions arranged as Cvf[position][TSdepth,TSdepth].
        cov_amplitude (float): multiply historical covariance by this to get background error covariance.

    Returns:
        K (TYPE): DESCRIPTION.
        K_avg (TYPE): DESCRIPTION.

    """
    K = np.zeros((An_nh, An_nh))
    for i in range(0, An_nh):
        for j in range(0, An_nh):
            K[i,j] = S[i][np.newaxis,:] @ ((Cvf[i,:,:] + Cvf[j,:,:]) / 2.0 * cov_amplitude) @ S[j][:,np.newaxis]

    # K matrix, shortcut by averaging
    K_avg = np.zeros((An_nh, An_nh))
    for i in range(0, An_nh):
        for j in range(0, An_nh):
            K_avg[i,j] = (K[i,i] + K[j,j]) / 2.0


    if not args.no_plotting:
        outfile = plot_dir / 'K.png'
        title = 'K matrix (S_i B_ij S_j)'
        plt3dvr.matrix(args, K, title, outfile,
                       x_label = 'Analysis grid index', y_label = 'Analysis grid index',
                       cb_label = 'amplitude')
    
        outfile = plot_dir / 'K_avg.png'
        title = 'K matrix (K_ii + K_jj) / 2'
        plt3dvr.matrix(args, K_avg, title, outfile,
                       x_label = 'Analysis grid index', y_label = 'Analysis grid index',
                       cb_label = 'amplitude')

    return (K, K_avg)

#%%
def B(args, plot_dir, n_state, Bh, An_nh, An_nv, Cvf, cov_amplitude):
    B = np.zeros((n_state, n_state))

    for i in range(0, An_nh):
        for j in range(0, An_nh):
            B[i * An_nv: (i + 1) * An_nv, j * An_nv: (j + 1) * An_nv] = Bh[i,j] * (Cvf[i,:,:] + Cvf[j,:,:]) / 2.0 * cov_amplitude
    
    outfile = plot_dir / 'B.png'

    title = 'Background error covariances'
    plt3dvr.matrix(args, B, title, outfile, vmin = B.min(), vmax = B.max(),
                   x_label = 'State index', y_label = 'State index',
                   cb_label = 'Variance', dpi = 300)
    
    return B

#%%
def do_3dvar(args, plot_dir, B, BHT, HBHT, H, R, analysis_var, innovation, An_lat, An_lon, An_latf, An_lonf, An_nz,
             depths, depth_edges_use, depth_range,
             An_nv, An_nh, ob_list, bg_state, interp_state_to_obs, cov_amplitude, fontsize, decim_vert,
             incr_tmin = -5.0, incr_tmax = 5.0, incr_smin = -0.5, incr_smax = 0.5):

    print(f'B max asym {np.max(np.abs(B - (B + B.T) / 2.0))}, {np.max(np.abs(B - (B + B.T) / 2.0)) / np.max(B)}')
    print(f'HBHT max asym {np.max(np.abs(HBHT - (HBHT + HBHT.T) / 2.0))}, {np.max(np.abs(HBHT - (HBHT + HBHT.T) / 2.0)) / np.max(B)}')

    # ensure B and HBHT are symmetric
    B = (B + B.T) / 2.0
    HBHT = (HBHT + HBHT.T) / 2.0

    # make B and HBHT positive semi-definite, they've been through a lot and are a bit out of shape
    eigval_limit = 1.0e-8
    (bevals, bevecs) = np.linalg.eig(B)
    idx = np.real(bevals).argsort()[::-1]
    bevals = np.real(bevals[idx])
    bevecs = np.real(bevecs[:,idx])
    neigvals = np.argmax(bevals <= np.max(bevals) * eigval_limit)
    print('do_3dvar, B cutting neigvals', neigvals)
    if (neigvals > 0) and (neigvals < B.shape[0] - 1):
        bevals[neigvals:] = 0.0
        B = bevecs @ np.diag(bevals) @ bevecs.T

    (evals, evecs) = np.linalg.eig(HBHT)
    idx = np.real(evals).argsort()[::-1]
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:,idx])
    neigvals = np.argmax(evals <= np.max(evals) * eigval_limit)
    print('do_3dvar, HBHT cutting neigvals', neigvals)
    if (neigvals > 0) and (neigvals < HBHT.shape[0] - 1):
        evals[neigvals:] = 0.0
        HBHT = evecs @ np.diag(evals) @ evecs.T
    
    hbhtpr_inv = np.linalg.inv(HBHT + R)
    KG = BHT @ hbhtpr_inv
    dx = KG @ innovation
    B_a = (np.eye(B.shape[0]) - KG @ H * 0.5) @ B

    # prevent negative covariances along the diagonal
    min_b_a = 1.0e-8
    d = np.diag(B_a).copy()
    d[d < min_b_a] = min_b_a
    np.fill_diagonal(B_a, d)

# CG approach
    # we are solving X = (HBHT+R)^-1 @ innovation for the vector X
    # this implies (HBHT+R) X = innovation
    # use conjugate gradient to solve for X
    # increment = BHT @ X
    
    # hbhtpr = HBHT + R
    # x0 = np.zeros(innovation.shape)
    # X = scipy.sparse.linalg.cg(hbhtpr, innovation, x0=x0, rtol=1e-010, atol=0.0, maxiter=None, M=None, callback=None)[0]
    # print('cg vs inv diff')
    # print(hbhtpr_inv @ innovation - X)

    print('max off-diagonal inv check:',
          np.max(np.abs(hbhtpr_inv @ (HBHT + R) - np.eye(HBHT.shape[0]))))

    for p in [None, 'fro', np.inf, -np.inf, 1, -1, 2, -2]:
        print('{:6s}\t{}'.format(str(p), np.linalg.cond(HBHT + R, p = p)))

#------------------------------------------------------------------------------
# plotting
#------------------------------------------------------------------------------

# eigenvalues
    outfile = plot_dir / 'B_eigenvalues.png'
    title = 'B eigenvalues'
    plt3dvr.eigvals(args, bevals, title, outfile, x_label = 'Eigenvalue Index', y_label = 'log10(Eigenvalue)', x_range = [0, 500])
    outfile = plot_dir / 'HBHT_eigenvalues.png'
    title = 'HBHT eigenvalues'
    plt3dvr.eigvals(args, evals, title, outfile)


# transect of analysis increment
    (dx_t, dx_s) = properties.ts_vec_2_mat(dx, An_nv, An_nh)
    outfile = plot_dir / 'analysis_increment_transect.png'
    
    
    if args.An_type == 'lat_transect':
        palong = An_lon
        pat = An_lat[0]
        xlabel = 'Longitude ($^\circ$E)'
        super_title1 = 'Analysis increment T&S along {}$^\circ$N ({})'.format(pat, analysis_var)
    else:
        palong = An_lat
        pat = An_lon[0]
        xlabel = 'Latitude ($^\circ$N)'
        super_title1 = 'Analysis increment T&S along {}$^\circ$E ({})'.format(pat, analysis_var)

    plt3dvr.lonslice(args, dx_t, dx_s, palong, depth_edges_use, [args.beg_adepth, args.end_adepth],
                 'Temperature', 'Salinity',
                 super_title1, outfile,
                 vmin_temp = incr_tmin, vmax_temp = incr_tmax,
                 vmin_salt = incr_smin, vmax_salt = incr_smax,
                 xlabel = xlabel, ylabel = 'Depth (m)')

# individual plots of obs and analysis

    analysis = bg_state + dx
    analysis_at_obs = interp_state_to_obs @ analysis
    (ano_t, ano_s) = properties.ts_vec_2_mat(analysis_at_obs, An_nv, len(ob_list))

    bg_at_obs = interp_state_to_obs @ bg_state
    (bg_t, bg_s) = properties.ts_vec_2_mat(bg_at_obs, An_nv, len(ob_list))

    ob_t = np.array([p.temp_pch[0: An_nz] for p in ob_list]).T
    ob_s = np.array([p.salt_pch[0: An_nz] for p in ob_list]).T

# original profiles
    for iob in range(0, len(ob_list)):
        profile = ob_list[iob]
        profile.add_sigma0()

        press = gsw.p_from_z(-depths, profile.lat)

        SA = gsw.SA_from_SP(bg_s[:,iob], press, profile.lon, profile.lat)
        CT = gsw.CT_from_t(SA, bg_t[:,iob], press)
        clim_sigma0 = gsw.sigma0(SA, CT)
        
        varnames = [r'T', r'S', r'$\sigma_o$']
        xlabels = [r'Temperature ($^\circ$C)', r'Salinity (PSU)', r'Potential Density $\sigma_o$ (kg m$^{-3})$']
        colors = ['b', 'r', 'k']
        source_labels = ['Ob', 'Cl']
        linestyles = ['solid', 'dotted']
        values = [[profile.temp, bg_t[:,iob]],
                  [profile.salt, bg_s[:,iob]],
                  [profile.sigma0, clim_sigma0]]
        depthsin = [profile.depth, depths]

        title = 'Profile at {:.2f}$^\circ$N {:.2f}$^\circ$\n{:4d}-{:02d}-{:02d}'.format(profile.lat, profile.lon,
                                                                                             profile.time.year,
                                                                                             profile.time.month,
                                                                                             profile.time.day,
                                                                                             )
        fontsize = 14
        outfile = plot_dir / 'original_profiles_{:.2f}_{:.2f}.png'.format(profile.lat, profile.lon)

        pltprof.plot_prof(varnames, xlabels, colors,
                      source_labels, linestyles, values, depthsin, outfile,
                      title = title, fontsize = fontsize)

# profile results
# get T,S,sigma0 all on the analysis vertical grid...

    for iob in range(0, len(ob_list)):
        profile = ob_list[iob]
        profile.add_sigma0()

        press = gsw.p_from_z(-depths, profile.lat)

        SA = gsw.SA_from_SP(bg_s[:,iob], press, profile.lon, profile.lat)
        CT = gsw.CT_from_t(SA, bg_t[:,iob], press)
        clim_sigma0 = gsw.sigma0(SA, CT)

        SA = gsw.SA_from_SP(ano_s[:,iob], press, profile.lon, profile.lat)
        CT = gsw.CT_from_t(SA, ano_t[:,iob], press)
        ano_sigma0 = gsw.sigma0(SA, CT)

        SA = gsw.SA_from_SP(profile.salt_pch[:An_nz], press, profile.lon, profile.lat)
        CT = gsw.CT_from_t(SA, profile.temp_pch[:An_nz], press)
        prf_sigma0 = gsw.sigma0(SA, CT)

        varnames = [r'T', r'S', r'$\sigma_o$']
        xlabels = [r'Temperature ($^\circ$C)', r'Salinity (PSU)', r'Potential Density $\sigma_o$ (kg m$^{-3})$']
        colors = ['b', 'r', 'k']
        source_labels = ['Ob', 'Bg', 'An']
        linestyles = ['solid', 'dotted', 'dashed']
        values = [[ob_t[:,iob], bg_t[:,iob], ano_t[:,iob]],
                  [ob_s[:,iob], bg_s[:,iob], ano_s[:,iob]],
                  [prf_sigma0, clim_sigma0, ano_sigma0],
                  ]
        depthsin = [depths, depths, depths]
        
        title = '3DVar analysis at {:.2f}$^\circ$N {:.2f}$^\circ$'.format(profile.lat, profile.lon)

        fontsize = 14
        outfile = plot_dir / 'result_profiles_{:.2f}_{:.2f}.png'.format(profile.lat, profile.lon)

        pltprof.plot_prof(varnames, xlabels, colors,
                      source_labels, linestyles, values, depthsin, outfile,
                      title = title, fontsize = fontsize)

# prior background standard deviation

    (prior_tstndev, prior_sstndev) = properties.ts_vec_2_mat(np.sqrt(np.diag(B)), An_nv, An_nh)
    
    outfile = plot_dir / 'prior_stndev.png'
    
    if args.An_type == 'lat_transect':
        palong = An_lon
        pat = An_lat[0]
        xlabel = 'Longitude ($^\circ$E)'
        super_title1 = 'Background standard deviation along {}$^\circ$N ({})'.format(pat, analysis_var)
    else:
        palong = An_lat
        pat = An_lon[0]
        xlabel = 'Latitude ($^\circ$N)'
        super_title1 = 'Background standard deviation along {}$^\circ$E ({})'.format(pat, analysis_var)
    
    ampf = np.sqrt(cov_amplitude)
    plt3dvr.lonslice(args, prior_tstndev, prior_sstndev, palong, depth_edges_use, [args.beg_adepth, args.end_adepth],
                 'Temperature', 'Salinity',
                 super_title1, outfile,
                 vmin_temp = args.vmin_temp_stndev * ampf, vmax_temp = args.vmax_temp_stndev * ampf,
                 vmin_salt = args.vmin_salt_stndev * ampf, vmax_salt = args.vmax_salt_stndev * ampf,
                 xlabel = xlabel, ylabel = 'Depth (m)')

# posterior background standard deviation

    (postr_tstndev, postr_sstndev) = properties.ts_vec_2_mat(np.sqrt(np.diag(B_a)), An_nv, An_nh)
    
    outfile = plot_dir / 'postr_stndev.png'
    
    if args.An_type == 'lat_transect':
        palong = An_lon
        pat = An_lat[0]
        xlabel = 'Longitude ($^\circ$E)'
        super_title1 = 'Posterior standard deviation along {}$^\circ$N ({})'.format(pat, analysis_var)
    else:
        palong = An_lat
        pat = An_lon[0]
        xlabel = 'Latitude ($^\circ$N)'
        super_title1 = 'Posterior standard deviation along {}$^\circ$E ({})'.format(pat, analysis_var)
    
    ampf = np.sqrt(cov_amplitude)
    plt3dvr.lonslice(args, postr_tstndev, postr_sstndev, palong, depth_edges_use, [args.beg_adepth, args.end_adepth],
                 'Temperature', 'Salinity',
                 super_title1, outfile,
                 vmin_temp = args.vmin_temp_stndev * ampf, vmax_temp = args.vmax_temp_stndev * ampf,
                 vmin_salt = args.vmin_salt_stndev * ampf, vmax_salt = args.vmax_salt_stndev * ampf,
                 xlabel = xlabel, ylabel = 'Depth (m)')

# prior depth-depth covariance and correlation

    depth_range = [args.beg_adepth, args.end_adepth]
    for icov, (atlat, atlon) in enumerate(zip(An_latf, An_lonf)):
        bacov = B[icov * An_nv: (icov + 1) * An_nv, icov * An_nv: (icov + 1) * An_nv]
        title = 'Prior Covariance latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        outfile = plot_dir / 'prior_Cv_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)
        pltcov.cov22(bacov, depths, depth_edges_use, depth_range, title, outfile, fontsize = 14)
    
        title = 'Prior Correlation latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        outfile = plot_dir / 'prior_Cv_corr_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)
        pltcov.cov22(bacov, depths, depth_edges_use, depth_range, title, outfile, fontsize = 14, plot_correlation = True)

# posterior depth-depth covariance and correlation

    depth_range = [args.beg_adepth, args.end_adepth]
    for icov, (atlat, atlon) in enumerate(zip(An_latf, An_lonf)):
        bacov = B_a[icov * An_nv: (icov + 1) * An_nv, icov * An_nv: (icov + 1) * An_nv]
        title = 'Posterior Covariance latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        outfile = plot_dir / 'postr_Cv_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)
        pltcov.cov22(bacov, depths, depth_edges_use, depth_range, title, outfile, fontsize = 14)
    
        title = 'Posterior Correlation latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        outfile = plot_dir / 'postr_Cv_corr_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)
        pltcov.cov22(bacov, depths, depth_edges_use, depth_range, title, outfile, fontsize = 14, plot_correlation = True)


    return (dx, B_a)

