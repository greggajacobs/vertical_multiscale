# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:52:59 2025

@author: ga_ja
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import copy
import properties
import plotting_3dvar as plt3dvr

#%%
def prep_profiles(args, plot_dir, attime, plist, hlength_scale, An_latf, An_lonf, depths, meantsf, depth_range, decim_vert, bathy_dat, bathy_interp, moving_profiles = True):
    """
    Prepare the data
    Thin based on lat,lon distance between profiles
    Keep one profile for each analysis lat,lon

    Args:
        args (dict): runtime arguments.
        attime (datetime): time of analysis.
        plist (list): list of profiles.
        hlength_scale (float): horizontal length scale in degrees latitude.
        An_latf (np array): latitude positions flattened.
        An_lonf (np array): longitude positions flattened.
        depths (np array): depths.
        meantsf (np array): mean temperature and salinity, flattened, for plotting.
        depth_range (list of 2 floats): lower and upper limits for plotting.

    Returns:
        None.

    """
    # thin data to prevent large horizontal correlations
    # thinning is based on horizontal correlation
    # no dependence on lat slice or lon slice
    gauss_corr_limit = 0.95
    
    # step 1 ------------------------------------------------------------------
    # remove profiles within too large of a correlation
    # this is done in a 2d sense
    # use only profiles extending beyone 1000 m
    # use only profiles with all good T&S
    if args.An_type == 'lat_transect':
        cutting_plist = [p for p in plist
                         if (args.beg_alon <= p.lon) and
                         (p.lon <= args.end_alon) and
                         (p.depth[-1] > 1000.0) and
                         (np.count_nonzero(p.temp_pch.mask[0:len(depths)]) == 0) and
                         (np.count_nonzero(p.salt_pch.mask[0:len(depths)]) == 0)
                         ]
    else:
        cutting_plist = [p for p in plist
                         if (args.beg_alat <= p.lat) and
                         (p.lat <= args.end_alat) and
                         (p.depth[-1] > 1000.0) and
                         (np.count_nonzero(p.temp_pch.mask[0:len(depths)]) == 0) and
                         (np.count_nonzero(p.salt_pch.mask[0:len(depths)]) == 0)
                         ]
 
    plat = np.array([p.lat for p in cutting_plist])
    plon = np.array([p.lon for p in cutting_plist])
    d2 = properties.distsq_deg(plat, plon, plat, plon, hlength_scale, An_type = '2d')
    d2 = d2 - np.diag(np.ones((len(plon))))
    while np.max(d2) > gauss_corr_limit:
        s = np.sum(d2, axis = 0)
        irem = np.argmax(s)

        flat_index_of_max = np.argmax(d2)

        # Convert the flat index to multi-dimensional indices
        row, col = np.unravel_index(flat_index_of_max, d2.shape)
        if s[row] > s[col]:
            irem = row
        else:
            irem = col
        #print('{:.3f} removing {}'.format(np.max(d2), irem))

        a = cutting_plist[0:irem]
        a.extend(cutting_plist[irem + 1: ])
        cutting_plist = a
        plon = np.delete(plon, irem)
        plat = np.delete(plat, irem)

        d2 = properties.distsq_deg(plat, plon, plat, plon, hlength_scale, An_type = '2d')
        d2 = d2 - np.diag(np.ones((len(plon))))

    # remaining after thinning
    plat1 = np.array([p.lat for p in cutting_plist])
    plon1 = np.array([p.lon for p in cutting_plist])


    # pick profile closest to each analysis point
    # choose profiles to minimize distance to each analysis point
    # maximum one profile per analysis point
    d2 = properties.distsq_deg(An_latf, An_lonf, plat1, plon1, hlength_scale, An_type = '2d') # rows are An, cols are profiles
    row_ind, col_ind = linear_sum_assignment(d2, maximize = True)
    
    puse = [cutting_plist[icol] for icol in col_ind]

    if moving_profiles:
        ob_list = puse
        ob_lat = np.array([An_latf[irow] for irow in row_ind])
        ob_lon = np.array([An_lonf[irow] for irow in row_ind])
        n_obs = len(ob_list)

    # to be used before final thinning
    plat2 = np.array([p.lat for p in puse])
    plon2 = np.array([p.lon for p in puse])

    # thin data to prevent large horizontal correlations
    plat3 = np.array([p.lat for p in puse])
    plon3 = np.array([p.lon for p in puse])
    
    puse3 = copy.copy(puse)
    d2 = properties.distsq_deg(plat3, plon3, plat3, plon3, hlength_scale, An_type = args.An_type)
    d2 = d2 - np.diag(np.ones((len(plon3))))
    while np.max(d2) > gauss_corr_limit:
        s = np.sum(d2, axis = 0)
        flat_index_of_max = np.argmax(d2)
        # Convert the flat index to multi-dimensional indices
        row, col = np.unravel_index(flat_index_of_max, d2.shape)
        if s[row] > s[col]:
            irem = row
        else:
            irem = col
        #print('{:.3f} removing {}'.format(np.max(d2), irem))

        a = puse3[0:irem]
        a.extend(puse3[irem + 1: ])
        puse3 = a
        plon3 = np.delete(plon3, irem)
        plat3 = np.delete(plat3, irem)

        d2 = properties.distsq_deg(plat3, plon3, plat3, plon3, hlength_scale, An_type = args.An_type)
        d2 = d2 - np.diag(np.ones((len(plon3))))

    if not moving_profiles:
        ob_list = puse3
        ob_lat = np.array([p.lat for p in ob_list])
        ob_lon = np.array([p.lon for p in ob_list])
        n_obs = len(ob_list)
    
    # need to set ob_lat or ob_lon to analysis lat or lon if doing a transect
    # this puts all the obs on the transect
    if args.An_type == 'lat_transect':
        ob_lat[:] = An_latf[0]
    if args.An_type == 'lon_transect':
        ob_lon[:] = An_lonf[0]

    if not args.no_plotting:
        # plot in order
        # * All points (red)
        # * Thinned points (after 2D horizontal thinning) (magenta)
        # * Used points (after selecting closest) (yellow)
        # * Used points (after longitude thinning) (green)
        all_plats = np.array([p.lat for p in plist])
        all_plons = np.array([p.lon for p in plist])
    
        outfile = plot_dir / 'data_distribution_used1_{}.png'.format(attime.strftime('%Y%m%d'))
        title = 'Data availability, {:d} of {:d} profiles'.format(len(plat), len(plist))
        plt3dvr.data_distribution(args, all_plats, all_plons, title, outfile, bathy_dat, bathy_interp, lats1 = plat1, lons1 = plon1)
    
        outfile = plot_dir / 'data_distribution_used2_{}.png'.format(attime.strftime('%Y%m%d'))
        plt3dvr.data_distribution(args, all_plats, all_plons, title, outfile, bathy_dat, bathy_interp, lats1 = plat1, lons1 = plon1, lats2 = plat2, lons2 = plon2)
    
        outfile = plot_dir / 'data_distribution_used3_{}.png'.format(attime.strftime('%Y%m%d'))
        plt3dvr.data_distribution(args, all_plats, all_plons, title, outfile, bathy_dat, bathy_interp, lats1 = plat1, lons1 = plon1, lats2 = plat2, lons2 = plon2, lats3 = plat3, lons3 = plon3)

        title = 'Observed profiles'
        outfile = plot_dir / 'observed_ts.png'
        plt3dvr.obs_prof(args, ob_list, meantsf, depths, depth_range, title, outfile)
    
        title = 'Observed profiles after pchip interpolation'
        outfile = plot_dir / 'observed_ts_pchip.png'
        plt3dvr.obs_prof(args, ob_list, meantsf, depths, depth_range, title, outfile, pchip = True)

    return(ob_list, ob_lat, ob_lon, n_obs)

#------------------------------------------------------------------------------
# TS obs operator
#------------------------------------------------------------------------------
def ts_point_operator(ob_list, Hh, An_nz, An_nv, decim_vert):
    """
    TS observation operator for obs at individual points in the vertical

    Parameters
    ----------
    ob_list : list of profile objects
        The profiles for which to make observations.
    Hh : np array
        Horizontal interpolation operator.
    An_nz : int
        Number of analysis z grid points.
    An_nv : int
        Number of analysis vertical grid points, which should be 2 * An_nz because it is both t and s.
    decim_verSt : int
        Vertical decimation of data.  This assume observations are already at the Analysis vertical points.

    Returns
    -------
    Hh_ts : np array
        Horizontal observation operator.
    S_ts : np array
        Vertical observation operator.
    """

    t1 = np.eye(An_nz)[0::decim_vert,:]
    tz = np.zeros(t1.shape)
    tsamp = np.append(t1, tz, axis = 1)
    ssamp = np.append(tz, t1, axis = 1)
    sample_prof = np.append(tsamp, ssamp, axis = 0)

    for iprof, prof in enumerate(ob_list):

        if iprof == 0:
            S_ts = sample_prof.copy()
            Hh_ts = np.repeat(Hh[iprof][np.newaxis,:], sample_prof.shape[0], axis = 0)
        else:
            S_ts = np.append(S_ts, sample_prof, axis = 0)
            Hh_ts = np.append(Hh_ts, np.repeat(Hh[iprof][np.newaxis,:], sample_prof.shape[0], axis = 0), axis = 0)

    return (Hh_ts, S_ts)

#------------------------------------------------------------------------------
# Steric Height perturbation observation operator for TS perturbation
# One operator for each analysis position
# This assumes the meantsf is on the analysis grid
#------------------------------------------------------------------------------
def stht_vertint_operator(args, plot_dir, An_latf, An_lonf, An_lat, An_lon, An_depth, An_nh, An_nz, meantsf, depth_range):
    """
    Build steric height observation operator as a perturbation from the meantsf

    Args:
        args (dict): runtime arguments.
        An_latf (np array): An_lat flattened.
        An_lonf (np array): An_lon flattened.
        An_lon (np array): longitude of netcdf dataset for plotting.
        An_depth (np array): depth of netcdf dataset.
        An_nh (np array): number of horizontal points after flattening.
        An_nz (np array): number of z points in either T or S.
        meantsf (np array): mean temperature and salinity, flattened, for plotting.
        depth_range (list of 2 floats): lower and upper limits for plotting.

    Returns:
        S (np array): observation operator organized as [position][twice number of depths for T&S].

    """

    S = []
    for i, (bg_lat, bg_lon) in enumerate(zip(An_latf, An_lonf)):
        S.append(properties.d_stht_d_ts(meantsf[i, 0: An_nz],
                             meantsf[i, An_nz:],
                             An_depth[0: An_nz],
                             bg_lat, bg_lon,
                             pref = args.reference_pressure, check = False))

    S = np.array(S)
    if not args.no_plotting:
        title = 'S, derivative of steric height wrt T&S'
        outfile = plot_dir / 'S_derivative.png'
        plt3dvr.S(args, S, An_depth[0:An_nz], depth_range, title, outfile, fontsize = 14)

    # linear estimate errors
    errs = [0.1, 0.5, 1.0, 2.0, 3.0]
    stht_err = np.zeros((An_nh, len(errs)))
    stht_anom = np.zeros((An_nh, len(errs)))
    
    # don't perturb the last few values
    # The stht() calculation inserts the reference pressure and therefore computes stht relative to pref exactly
    # The linear form integrates over all pressure values with the assumption that points at p > pref have 0 anomaly.
    # The result is the two approaches provide different results if linear form has non-zero values in p > pref
    delta_ones = np.ones((An_nz))
    delta_ones[An_nz-4:] = 0.0

    for ierr, delta_t in enumerate(errs):
        for ilon, (bg_lat, bg_lon) in enumerate(zip(An_latf, An_lonf)):
            bg_sthtv = properties.stht(meantsf[ilon, 0: An_nz],
                            meantsf[ilon, An_nz:],
                            An_depth[0: An_nz],
                            bg_lat, bg_lon,
                            pref = args.reference_pressure)
            stht_nl = properties.stht(meantsf[ilon, 0: An_nz] + delta_ones * delta_t,
                            meantsf[ilon, An_nz:],
                            An_depth[0: An_nz],
                            bg_lat, bg_lon,
                            pref = args.reference_pressure)
            stht_delta = S[ilon].T @ np.append(delta_ones * delta_t, np.zeros((An_nz)))
            stht_ln = bg_sthtv + stht_delta
    
            #print('{:5.2f} {:8.3f} {:8.3f} {:8.3f} {:5.3f}'.format(bg_lon, bg_sthtv, stht_nl, stht_ln, stht_nl - stht_ln))
    
            stht_err[ilon, ierr] = stht_ln - stht_nl
            stht_anom[ilon, ierr] = stht_nl - bg_sthtv

    stht_err_percent = stht_err / stht_anom

    if not args.no_plotting:
        title = 'Error in linear steric height operatror\nfor temperature deviation over entire water column'
        outfile = plot_dir / 'stht_operator_err.png'
        plt3dvr.S_err(args, stht_err_percent, errs, An_lon, An_lat, title, outfile,
                  fontsize = 14)

    return S

#%%
def horizontal_interpolation(args, plot_dir, An_lon, An_lat, ob_lon, ob_lat, An_nh, n_obs):

    if args.An_type != 'lon_transect':
        idxs = np.searchsorted(An_lon, ob_lon, side = 'right') - 1
        idxs[idxs == len(An_lon) - 1] -= 1
        fsx = (ob_lon - An_lon[idxs]) / (An_lon[idxs + 1] - An_lon[idxs])
    elif args.An_type != 'lat_transect':
        idys = np.searchsorted(An_lat, ob_lat, side = 'right') - 1
        idys[idys == len(An_lat) - 1] -= 1
        fsy = (ob_lat - An_lat[idys]) / (An_lat[idys + 1] - An_lat[idys])
    Hh = np.zeros((n_obs, An_nh))
    if args.An_type == 'lat_transect':
        for iprofile, (idx, fx) in enumerate(zip(idxs, fsx)):
            Hh[iprofile, idx] = 1.0 - fx
            Hh[iprofile, idx + 1] = fx
    elif args.An_type == 'lon_transect':
        for iprofile, (idy, fy) in enumerate(zip(idys, fsy)):
            Hh[iprofile, idy] = 1.0 - fy
            Hh[iprofile, idy + 1] = fy
    else:
        for iprofile, (idx, idy, fx, fy) in enumerate(zip(idxs, idys, fsx, fsy)):
            flat_index = np.ravel_multi_index((idys, idxs), (len(An_lat, len(An_lon))))
            Hh[iprofile, flat_index] = (1.0 - fx) * (1.0 - fy)
            flat_index = np.ravel_multi_index((idys, idxs + 1), (len(An_lat, len(An_lon))))
            Hh[iprofile, idx + 1] = fx * (1.0 - fy)
            flat_index = np.ravel_multi_index((idys + 1, idxs), (len(An_lat, len(An_lon))))
            Hh[iprofile, idx + 1] = (1.0 - fx) * fy
            flat_index = np.ravel_multi_index((idys + 1, idxs + 1), (len(An_lat, len(An_lon))))
            Hh[iprofile, idx + 1] = fx * fy
        

    if not args.no_plotting:
        outfile = plot_dir / 'horizontal_observation_operator.png'
        title = 'Horizontal observation operator'
        plt3dvr.matrix(args, Hh, title, outfile, vmin = 0.0, vmax = 1.0,
                       x_label = 'Analysis grid index', y_label = 'Observation index',
                       cb_label = 'amplitude')

    return Hh

#%%
def ts_obs_operator(Hh_ts, S_ts, n_obs):
    """
    Combine horizontal and vertical observation functions

    Args:
        Hh_ts (np array): horizontal interpolation organized as [obs][An horizontal posn].
        S_ts (np array): vertical function organized as [obs][An depth].
        An_nh (int): number of horizontal positions.

    Returns:
        H_ts (np array): horizontal and vertical observation operator [obs][state posn].

    """
    for i in range(0, Hh_ts.shape[0]):
        a = np.kron(Hh_ts[i,:][np.newaxis,:], S_ts[i][np.newaxis,:])
        if i == 0:
            H_ts = a
        else:
            H_ts = np.append(H_ts, a, axis = 0)
            
    return H_ts

#%%
def stht_obs_operator(Hh, S, An_nh):
    """
    Combine horizontal and vertical observation functions

    Args:
        Hh (np array): horizontal interpolation organized as [obs][An horizontal posn].
        S (np array): vertical function organized as [An horizontal posn][An depth].
        An_nh (int): number of horizontal positions.

    Returns:
        H (np array): horizontal and vertical observation operator [obs][state posn].

    """
    for i in range(0, An_nh):
        a = np.kron(Hh[:,i][:,np.newaxis], S[i][np.newaxis,:])
        if i == 0:
            H = a
        else:
            H = np.append(H, a, axis = 1)
            
    return H

#%%
def R(args, stht_obs_error_variance, t_obs_error_variance, s_obs_error_variance,
      n_obs_ts, n_profiles, An_nz, An_nv, decim_vert):

    # R = np.diag(np.ones(n_profiles)) * stht_obs_error_variance

    # n_obs_ts = n_profiles * An_nv
    # R_ts = np.zeros((n_obs_ts, n_obs_ts))
    # for i in range(0, n_profiles):
    #     for iz in range(i*An_nv, i*An_nv + An_nz):
    #         R_ts[iz,iz] = t_obs_error_variance
    #         R_ts[iz + An_nz ,iz + An_nz] = s_obs_error_variance

    # R_stht_ts = np.zeros((n_profiles + n_obs_ts, n_profiles + n_obs_ts))
    # R_stht_ts[0:n_profiles,0:n_profiles] = R
    # R_stht_ts[n_profiles:n_profiles + n_obs_ts,n_profiles:n_profiles + n_obs_ts] = R_ts

    t_or_s_obs_per_profile = int(np.floor(An_nz / decim_vert + 0.51))
    t_and_s_obs_per_profile = t_or_s_obs_per_profile * 2
    if (n_obs_ts / t_or_s_obs_per_profile != n_profiles * 2):
        errstr = f'error in obs.R, t_or_s_obs_per_profile not correct.  {n_obs_ts}, {t_or_s_obs_per_profile}, {n_profiles}'
        raise Exception(errstr)

    R = np.diag(np.ones(n_profiles)) * stht_obs_error_variance

    t1 = np.eye(t_or_s_obs_per_profile)
    tz = np.zeros(t1.shape)
    t3 = np.append(t1 * t_obs_error_variance, tz, axis = 1)
    t4 = np.append(tz, t1 * s_obs_error_variance, axis = 1)
    R_ts1 = np.append(t3, t4, axis = 0)

    R_ts = np.zeros((n_obs_ts, n_obs_ts))
    for i in range(0, n_profiles):
        ibeg = t_and_s_obs_per_profile * i
        iend = ibeg + t_and_s_obs_per_profile
        R_ts[ibeg:iend,ibeg:iend] = R_ts1

    R_stht_ts = np.zeros((R.shape[0] + R_ts.shape[0], R.shape[0] + R_ts.shape[0]))
    R_stht_ts[0:R.shape[0],0:R.shape[0]] = R
    R_stht_ts[R.shape[0]:R.shape[0] + R_ts.shape[0],R.shape[0]:R.shape[0] + R_ts.shape[0]] = R_ts

    return(R, R_ts, R_stht_ts)

#%%    
def ts_innovation(args, ob_list, H_ts, bg_state, An_depth, An_nz, decim_vert):
    #------------------------------------------------------------------------------
    # Innovation
    # Going by the definition, the innovation is (y - H x^bg)
    # The observation y is observed steric height minus climo steric height
    #   This requires first interpolating climo from the An grid to the obs locations
    # The observation of background is H @ (bg_state - climo_state)
    #------------------------------------------------------------------------------

    y = np.array(())
    for p in ob_list:
        y = np.append(y, p.temp_pch[0:An_nz:decim_vert], axis = 0)
        y = np.append(y, p.salt_pch[0:An_nz:decim_vert], axis = 0)
    
    innovation_ts = y - H_ts @ bg_state
    
    return innovation_ts

#%%    
def stht_innovation(args, plot_dir, ob_list, H, bg_state, climo_state, interp_state_to_obs, An_depth, An_nz, An_nv):
    #------------------------------------------------------------------------------
    # Innovation
    # Going by the definition, the innovation is (y - H x^bg)
    # The observation y is observed steric height minus climo steric height
    #   This requires first interpolating climo from the An grid to the obs locations
    # The observation of background is H @ (bg_state - climo_state)
    #------------------------------------------------------------------------------

    # HhS -> Map from background (TS - climo_TS) and locations
    #        to stht at observation locations
    bg_at_obs_stht_anom = H @ (bg_state - climo_state) # this turns out to be 0 because the background is climo
    
    climo_state_at_obs = interp_state_to_obs @ climo_state
    obs_stht_anom = []
    for iob, p in enumerate(ob_list):
        # This is using S_at_obs * (obs_TS - climo_TS), not correct
        # tsprof = (np.append(p.temp_pch[0: An_nz], p.salt_pch[0: An_nz]))
        # ts_anom = tsprof - climo_state_at_obs[iob * An_nv: (iob + 1) * An_nv]
        # innovation.append(np.dot(S_at_obs[iob * An_nv: (iob + 1) * An_nv], ts_anom))
        # This is using stht(ob) - stht(climo_state_at_obs), correct
        prof_stht = properties.stht(p.temp_pch[0: An_nz], p.salt_pch[0: An_nz], An_depth[0: An_nz], p.lat, p.lon,
                                    pref = args.reference_pressure)
        climo_stht = properties.stht(climo_state_at_obs[iob * An_nv: iob * An_nv + An_nz],
                                     climo_state_at_obs[iob * An_nv + An_nz: (iob + 1) * An_nv],
                                     An_depth[0: An_nz], p.lat, p.lon,
                                     pref = args.reference_pressure)
        obs_stht_anom.append(prof_stht - climo_stht)
    obs_stht_anom = np.array(obs_stht_anom)
    
    innovation = obs_stht_anom - bg_at_obs_stht_anom

    if not args.no_plotting:
        data = innovation
        if args.An_type == 'lat_transect':
            posns = np.array([p.lon for p in ob_list])
            x_label = 'Longitude ($^\circ$E)'
        else:
            posns = np.array([p.lat for p in ob_list])
            x_label = 'Latitude ($^\circ$N)'
        sorted_indices = np.argsort(posns)
        labels = ['innovation']
        title = 'Observed stht - climo stht: (obs_stht - climo_stht) - H S (bkg_TS - climo_TS) (m)'
        outfile = plot_dir / 'innovations.png'
        y_label = 'Steric height (m)'
        plt3dvr.foflon(args, posns[sorted_indices], data[sorted_indices], labels, title, outfile,
                       x_label = x_label, y_label = y_label)
    
    return innovation

