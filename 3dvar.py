# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:04:34 2025

@author: ga_ja
"""

include_dirs = ['D:\\OneDrive\\Projects\\3D_to_2D_var\\src_code',
                'D:\OneDrive\Projects\g_modules',
                'D:\OneDrive\Projects\WOD\src_code']

import torch
import os
import sys
import numpy as np
from numpy import ma
import pickle
from datetime import datetime as dtm
from pathlib import Path
import copy
import gsw
import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta
for d in include_dirs:
    if d not in sys.path:
        print(f'adding {d} to path')
        sys.path.append(d)
import startup
import nc_util
import plotting_3dvar as plt3dvr
import plotting_cov as pltcov
import plotting_profile as pprof
import WOD
import properties
import covariances
import obs
import bathy
import gplt

def loops(gom_info, natl_info):
    run_data =[
        {                                   # 2-step as it should be done
         'name': 'propr_2step',
         'stht': {'do_anly': True,
                  'covariance_form': 'histo', 'obs_error_scale': 1.0},
         'ts': {'do_anly': True,
                'covariance_form': '2gauss', 'obs_error_scale': 0.25},
         },
        {                                   # stht covariance test
         'name': 'stht_2gauss',
         'stht': {'do_anly': True,
                  'covariance_form': '2gauss', 'obs_error_scale': 1.0},
         'ts': {'do_anly': False},
         },
        {                                   # stht covariance test
         'name': 'stht_2gausscross',
         'stht': {'do_anly': True,
                  'covariance_form': '2gausscross', 'obs_error_scale': 1.0},
         'ts': {'do_anly': False},
         },
        {                                   # ts covariance test
         'name': 'ts_histo',
         'stht': {'do_anly': False,},
         'ts': {'do_anly': True,
                'covariance_form': 'histo', 'obs_error_scale': 1.0}
         },
        {                                   # ts covariance test
         'name': 'ts_2gauss',
         'stht': {'do_anly': False,},
         'ts': {'do_anly': True,
                'covariance_form': '2gauss', 'obs_error_scale': 1.0}
         },
        {                                   # ts covariance test
         'name': 'ts_2gausscross',
         'stht': {'do_anly': False,},
         'ts': {'do_anly': True,
                'covariance_form': '2gausscross', 'obs_error_scale': 1.0}
         },
    ]
    for area_name, area_info in zip(['gom', 'natl'], [gom_info, natl_info]):
        for r in run_data:
                yield(area_name, area_info, r)

def set_plot_dir(area_name, expt, analysis_var = None):
    plot_dir = args.plot_dir / area_name / expt
    if analysis_var is not None:
        plot_dir = plot_dir / analysis_var
    plot_dir.mkdir(parents = True, exist_ok = True)
    return plot_dir

sys.argv = ['3dvar',
            '-base_drive', 'D:/',
            '-data_dir', 'Projects/WOD23', '-search_filename', 'wod_\d{9}O.nc',
            '-qc_max_top_depth', '20.0', '-qc_min_bot_depth', '20.0',
            '-qc_mean_sampling_dz', '50.0',
            '-qc_require_salt', '-qc_require_temp', '-qc_sort_depth',
            '-qc_remove_masked',
            '-bathy_filename', 'Projects/GEBCO/GEBCO_2024_sub_ice_topo.nc',
            '-bathy_variable', 'elevation', '-bathy_stride', '10',
            '-reference_pressure', '1.0e7',
#            '-no_plotting'
            '-plot_dir', 'projects/3D_to_2D_var/plots',
            ]
args = startup.args()

gom_info = {
            'beg_lat': 24.0, 'end_lat': 31.0,     # used in plotting data distribution
            'beg_lon': 264.0, 'end_lon': 278.0,
            'beg_t': '20210801', 'end_t': '20210901',
            'beg_alat': 26.5, 'end_alat': 26.5,
            'beg_alon': 265.0, 'end_alon': 275.0,
            'beg_adepth': 0.0, 'end_adepth': 1000.0,
            'profile_pickle_file': 'projects/mean_TS/TS_profiles_gom.pkl',
            'in_covariance_netcdf_format': 'projects/vcov/cov_gom_080_{:02d}.nc',
            'vmin_temp': 4.0, 'vmax_temp': 32.0,      # colorbar ranges for plotting
            'vmin_salt': 34.3, 'vmax_salt': 37.3,     # colorbar ranges for plotting
            'vmin_incr_temp1': -10.0, 'vmax_incr_temp1': 10.0,      # colorbar ranges for plotting
            'vmin_incr_salt1': -1.0, 'vmax_incr_salt1': 1.0,     # colorbar ranges for plotting
            'vmin_incr_temp2': -5.0, 'vmax_incr_temp2': 5.0,      # colorbar ranges for plotting
            'vmin_incr_salt2': -1.0, 'vmax_incr_salt2': 1.0,     # colorbar ranges for plotting
            'vmax_temp_stndev': 5.0, 'vmax_salt_stndev': 1.0, # colorbar ranges for plotting
            'An_type': 'lat_transect',
            }

natl_info = {
            'beg_lat': 43.0, 'end_lat': 57.0,     # used in plotting data distribution
            'beg_lon': 310.0, 'end_lon': 330.0,
            'beg_t': '20180501', 'end_t': '20180601',
            'beg_alat': 45.0, 'end_alat': 55.0,
            'beg_alon': 322.0, 'end_alon': 322.0,
            'beg_adepth': 0.0, 'end_adepth': 1000.0,
            'profile_pickle_file': 'projects/mean_TS/TS_profiles_natl.pkl',
            'in_covariance_netcdf_format': 'projects/vcov/cov_natl_080_{:02d}.nc',
            'vmin_temp': 3.0, 'vmax_temp': 18.0,      # colorbar ranges for plotting
            'vmin_salt': 34.5, 'vmax_salt': 37.0,     # colorbar ranges for plotting
            'vmin_incr_temp1': -5.0, 'vmax_incr_temp1': 5.0,      # colorbar ranges for plotting
            'vmin_incr_salt1': -1.0, 'vmax_incr_salt1': 1.0,     # colorbar ranges for plotting
            'vmin_incr_temp2': -5.0, 'vmax_incr_temp2': 5.0,      # colorbar ranges for plotting
            'vmin_incr_salt2': -1.0, 'vmax_incr_salt2': 1.0,     # colorbar ranges for plotting
            'vmax_temp_stndev': 3.0, 'vmax_salt_stndev': 0.4, # colorbar ranges for plotting
            'An_type': 'lon_transect',
    }

for info in [gom_info, natl_info]:
    args.profile_pickle_file = args.base_drive / info['profile_pickle_file']
    with open(args.profile_pickle_file, 'rb') as f:
        (args_prev, casts, profiles, depth_edges) = pickle.load(f)
        info['args_prev'] = args_prev
        info['casts'] = casts
        info['profiles'] = profiles
        info['depth_edges'] = depth_edges


results = {}
for (area_name, area_info, run_info) in loops(gom_info, natl_info):

    # area_name = 'gom'
    # area_info = gom_info
    # run_info = {                                   # ts covariance test
    #      'name': 'ts_histo',
    #      'stht': {'do_anly': False,},
    #      'ts': {'do_anly': True,
    #             'covariance_form': 'histo', 'obs_error_scale': 1.0}
    #      }

    if area_name not in results.keys():
        results.update({area_name: {}})
    results[area_name].update({run_info['name']: {}})

    print(area_name, run_info)

    for k, v in area_info.items():
        setattr(args, k, v)
    profiles = area_info['profiles']
    depth_edges = area_info['depth_edges']

    (bathy_dat, bathy_interp) = bathy.startup(args.bathy_filename,
                                              args.bathy_variable, args.bathy_stride,
                                              beg_lat = args.beg_lat, end_lat = args.end_lat,
                                              beg_lon = args.beg_lon, end_lon = args.end_lon)

    args.in_covariance_netcdf_format = str(args.base_drive / args.in_covariance_netcdf_format)
    args.beg_t = dtm.strptime(args.beg_t, '%Y%m%d')
    args.end_t = dtm.strptime(args.end_t, '%Y%m%d')

    ssh_errvar = 0.02
    pref = 1000.0 * 1.0e4 # reference pressure for steric height
    hlength_scale = 0.5     # degrees latitutde
    cov_amplitude = 0.2
    stht_obs_error_variance = 9.0e-4
    t_obs_error_variance = 1.0e-2
    s_obs_error_variance = 1.0e-4
    vscale_points = 2.0
    decim_vert = 2
    fontsize = 14
    depth_range = [0.0, 1000.0]         # for plotting

    plot_dir = set_plot_dir(area_name, run_info['name'], analysis_var = 'setup')

    #------------------------------------------------------------------------------
    #find a good month
    #------------------------------------------------------------------------------
    # read in all datad
    # plot last few years of float positions by month
    attime = args.beg_t
    while attime < args.end_t:
        print(attime)
        plist = [p for p in profiles if ((attime <= p.time) and (p.time <= attime + relativedelta(months=1))
                                         and ('PFL' in str(p.filename)))]
        # lats = np.array([p.lat for p in plist])
        # lons = np.array([p.lon for p in plist])
        # tstr = 'data_distribution_{}.png'.format(attime.strftime('%Y%m%d'))
        # title = tstr + ' {:3d} profiles'.format(len(lats))
        # outfile = plot_dir / tstr
        # plt3dvr.data_distribution(args, lats, lons, title, outfile, bathy_dat, bathy_interp)
        #attime = attime + relativedelta(months=1)
        attime = attime + relativedelta(years=1)

    attime = args.beg_t
    month_n = attime.month
    plist = [p for p in profiles if ((args.beg_t <= p.time) and (p.time <= args.end_t)
                                         and ('PFL' in str(p.filename)))]

    #------------------------------------------------------------------------------
    # Vertical covariance Cv
    #------------------------------------------------------------------------------
    (Cvf, Cv_lon, Cv_lat, Cv_depth, Cv_lonf, Cv_latf, Cv_nh, Cv_nv, Cv_nz,
     meants, stdts, meantsf, stdtsf, depth_edges_use) = covariances.read_Cvf(args, month_n, depth_edges)
    depths = Cv_depth[0: Cv_nz]


    # for icov, c in enumerate(Cvf):
    #     print(properties.is_pos_semidef(c))

    # let's look at the diagonal of Cvf rather than the read in standard deviation
    for icov in range(0, Cvf.shape[0]):
        stdtsf[icov,:] = np.sqrt(np.diag(Cvf[icov,:,:]))

    # plot mean and standard deviations
    plt3dvr.Cvf_mean_stdev_prof(args, plot_dir, Cvf, Cv_latf, Cv_lonf, Cv_nz, Cv_depth, depths, depth_edges_use, meantsf, stdtsf)

    ## plot slice of TS covariances
    plt3dvr.transect_mean_stndev_ts(args, plot_dir, np.squeeze(meants.vals), np.squeeze(stdts.vals),
                                    Cv_nz, Cv_nv, Cv_lat, Cv_lon, depth_edges_use, depth_range)
    
    Cvf = covariances.clean_Cvf(Cvf)

    #------------------------------------------------------------------------------
    # Set up analysis grid
    #------------------------------------------------------------------------------
    # Analysis grid spacing is same as covariance grid, going to a different grid requires interpolation that is not implemented
    # Vertical coordinate includes both T&S
    An_lon = Cv_lon
    An_lat = Cv_lat
    An_depth = Cv_depth
    (An_long, An_latg) = np.meshgrid(An_lon, An_lat)
    An_lonf = An_long.flatten()
    An_latf = An_latg.flatten()

    An_nh = len(An_lonf) # number of horizontal analysis points
    An_nv = len(An_depth) # number of vertical analysis points
    An_nz = int(len(An_depth) / 2) # number of z points (half the number because of both T&S at each depth)
    n_state = An_nh * An_nv

    #------------------------------------------------------------------------------
    # Subselect profiles, one profile for each analysis point
    #------------------------------------------------------------------------------
    (ob_list, ob_lat, ob_lon, n_obs) = obs.prep_profiles(args, plot_dir, attime, plist, hlength_scale,
                                                         An_latf, An_lonf, depths, meantsf,
                                                         depth_range, decim_vert, bathy_dat, bathy_interp, moving_profiles = True)

    #------------------------------------------------------------------------------
    # Horizontal interpolation operator to obs locations
    #------------------------------------------------------------------------------
    Hh = obs.horizontal_interpolation(args, plot_dir, An_lon, An_lat, ob_lon, ob_lat, An_nh, n_obs)

    # A useful operator to interpolate from state points to obs points
    # This interpolates all vertical levels independently at once
    interp_state_to_obs = np.kron(Hh, np.eye((An_nv)))

    #------------------------------------------------------------------------------
    # Steric Height perturbation observation operator for TS perturbation
    #------------------------------------------------------------------------------
    S = obs.stht_vertint_operator(args, plot_dir, An_latf, An_lonf, An_lat, An_lon, An_depth, An_nh, An_nz, meantsf, depth_range)

    #------------------------------------------------------------------------------
    # T and S observation operators in horizontal and vertical
    #------------------------------------------------------------------------------
    (Hh_ts, S_ts) = obs.ts_point_operator(ob_list, Hh, An_nz, An_nv, decim_vert)

    #------------------------------------------------------------------------------
    # Observation operator
    # H_stht: Map from TS anomalies at state points to stht anomalies at obs points
    # H_ts: Map from TS anomalies at state points to TS anomalies at obs points
    #------------------------------------------------------------------------------
    H_stht = obs.stht_obs_operator(Hh, S, An_nh)
    H_ts = obs.ts_obs_operator(Hh_ts, S_ts, n_obs)

    #------------------------------------------------------------------------------
    # Horizontal correlation
    #------------------------------------------------------------------------------
    Bh = covariances.horizontal_correlation_gauss(args, plot_dir, ma.getdata(An_lonf),  ma.getdata(An_latf), hlength_scale)

    #------------------------------------------------------------------------------
    # Get everything into state space (one vector)
    #------------------------------------------------------------------------------
    climo_state = meantsf.flatten()
    bg_state = meantsf.flatten()
    S_state = S.flatten()

    #------------------------------------------------------------------------------
    # Conduct analysis steps
    #------------------------------------------------------------------------------
    decrease_covar = True
    for ivar, (analysis_var, anly_props) in enumerate(run_info.items()):
        if analysis_var == 'name':
            continue
        if not anly_props['do_anly']:
            continue
        
        print(ivar, analysis_var, anly_props)

        plot_dir = set_plot_dir(area_name, run_info['name'], analysis_var = analysis_var)

        # set vertical covariance to Gaussian diagonal values if needed
        covariances.vert_cov_gauss(anly_props['covariance_form'], Cvf, Cv_nz, vscale_points)

        #------------------------------------------------------------------------------
        # Innovation
        #------------------------------------------------------------------------------
        innovation_stht = obs.stht_innovation(args, plot_dir, ob_list, H_stht, bg_state, climo_state,
                                              interp_state_to_obs, An_depth, An_nz, An_nv)
        innovation_ts = obs.ts_innovation(args, ob_list, H_ts, bg_state, An_depth, An_nz, decim_vert)
        n_obs_ts = len(innovation_ts)

        #------------------------------------------------------------------------------
        # Observation error
        #------------------------------------------------------------------------------
        n_profiles = len(ob_list)
        (R_stht, R_ts, R_stht_ts) = obs.R(args, stht_obs_error_variance,
                                          t_obs_error_variance * anly_props['obs_error_scale'],
                                          s_obs_error_variance * anly_props['obs_error_scale'],
                                          n_obs_ts, n_profiles, An_nz, An_nv, decim_vert)

        #------------------------------------------------------------------------------
        # Background covariance
        #------------------------------------------------------------------------------
        if decrease_covar:
            amp = cov_amplitude
            decrease_covar = False
        else:
            amp = 1.0
        B = covariances.B(args, plot_dir, n_state, Bh, An_nh, An_nv, Cvf, amp)

        (K, K_avg) = covariances.K(args, plot_dir, An_nh, S, Cvf, amp)

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        
        if analysis_var == 'stht':
            H = H_stht
            R = R_stht
            innovation = innovation_stht
        elif analysis_var == 'ts':
            H = H_ts
            R = R_ts
            innovation = innovation_ts
        else:
            print(f'bad analysis_var: {analysis_var}')

        BHT = B @ H.T
        HBHT = H @ B @ H.T
        #HBHT = Hh @ (Bh * K) @ Hh.T

        (dx_full, B_a) = covariances.do_3dvar(args, plot_dir, B, BHT, HBHT, H, R, analysis_var, innovation,
                                       An_lat, An_lon, An_latf, An_lonf, An_nz,
                                       depths, depth_edges_use, depth_range,
                                       An_nv, An_nh, ob_list, bg_state, interp_state_to_obs, cov_amplitude, fontsize, decim_vert,
                                       incr_tmin = args.vmin_incr_temp1, incr_tmax = args.vmax_incr_temp1,
                                       incr_smin = args.vmin_incr_salt1, incr_smax = args.vmax_incr_salt1)
        plt3dvr.innovation_increment_stht(args, plot_dir, innovation, H, dx_full, ob_list, analysis_var)

        # BHT =  B @ H.T
        # HBHT = Hh @ (Bh * K) @ Hh.T
        # analysis_var = 'redu_stht'

        # (dx_redu, B_a) = covariances.do_3dvar(args, plot_dir, BHT, HBHT, H, R, analysis_var, innovation, An_lat,
        #                                An_lon, An_latf, An_lonf, An_nz,
        #                                depths, depth_edges_use, depth_range,
        #                                An_nv, An_nh, ob_list, bg_state, interp_state_to_obs, cov_amplitude, fontsize, decim_vert)
        # plt3dvr.innovation_increment_stht(args, innovation, H, dx_redu, ob_list, analysis_var)


        # save information for RMS plotting

        analysis = bg_state + dx_full
        analysis_at_obs = interp_state_to_obs @ analysis
        (ano_t, ano_s) = properties.ts_vec_2_mat(analysis_at_obs, An_nv, len(ob_list))

        bg_at_obs = interp_state_to_obs @ bg_state
        (bgo_t, bgo_s) = properties.ts_vec_2_mat(bg_at_obs, An_nv, len(ob_list))

        ob_t = np.array([p.temp_pch[0: An_nz] for p in ob_list]).T
        ob_s = np.array([p.salt_pch[0: An_nz] for p in ob_list]).T

        results[area_name][run_info['name']][analysis_var] = {}
        
        res = results[area_name][run_info['name']][analysis_var]

        res['ano_t'] = ano_t
        res['ano_s'] = ano_s
        res['bg_t'] = bgo_t
        res['bg_s'] = bgo_s
        res['ob_t'] = ob_t
        res['ob_s'] = ob_s

        #------------------------------------------------------------------------------
        # Updates for next cycle
        #------------------------------------------------------------------------------
        # update background
        bg_state = bg_state + dx_full

        # replace Cvf with B_a
        for icov in range(0, Cvf.shape[0]):
            Cvf[icov] = B_a[icov * An_nv: (icov + 1) * An_nv, icov * An_nv: (icov + 1) * An_nv]




#------------------------------------------------------------------------------
# Plots - RMS error over depth
#   gom and natl
#       stht and ts
#           
#------------------------------------------------------------------------------
from astropy.convolution import convolve

labels_all = {'propr_2step': '2-step',
          'stht_2gauss': '2Gauss',
          'stht_2gausscross': '2CGauss',
          'ts_histo': 'Histo',
          'ts_2gauss': '2Gauss',
          'ts_2gausscross': '2CGauss'}
t_ranges = {'gom': {'stht': [0.0, 4.0], 'ts': [0.0, 0.8]}, 'natl': {'stht': [0.0, 2.0], 'ts': [0.0, 0.5]}}
s_ranges = {'gom': {'stht': [0.0, 2.0], 'ts': [0.0, 0.2]}, 'natl': {'stht': [0.0, 0.6], 'ts': [0.0, 0.1]}}
kernel = np.array([0.25, 0.5, 0.25])
for area_name in ['gom', 'natl']:
    print(f'area {area_name}')
    for analysis_var in ['stht', 'ts']:
        print(f'analysis_var {analysis_var}')
        outfile = args.plot_dir / f'{area_name}_{analysis_var}.png'
        
        # collate all the runs for this analysis_variable
        all_temp, all_salt, labels = [], [], []
        for run, run_vals in results[area_name].items():
            print(f'run {run}')
            if (analysis_var in run) or (run == 'propr_2step'):
                print(f'run {run} has the analysis var {analysis_var}')
                res = results[area_name][run][analysis_var]
                all_temp.append(np.std(res['ano_t'] - res['ob_t'], axis=1))
                all_salt.append(np.std(res['ano_s'] - res['ob_s'], axis=1))
                labels.append(labels_all[run])

            nlines = 6
            cmap = plt.cm.jet
            nprofs = len(ob_list)
            colors = [cmap(i / (nlines - 1)) for i in range(nlines-1, -1, -1)]
            [(i / (nlines - 1)) for i in range(nlines-1, -1, -1)]
            fprops = gplt.fig_grid(nx = 2, ny = 1, xinter = 0.10, padtop = 0.1, figsize = (11.0, 8.5))
        
            for (i, xlabel, ranges, datasets) in zip(range(0, 2),
                                               [r'Temperature ($^\circ$C)', r'Salinity (PSU)'],
                                               [t_ranges, s_ranges],
                                               [all_temp, all_salt]):
        
                ax = gplt.ax_grid(fprops, row = 0, col = i, plt_type = 'linearx_lineary')
        
                for (iline, dat, label, c) in zip(range(0, len(all_temp)), datasets, labels, colors):
                    p = all_temp[iline]
                    ax.plot(convolve(dat, kernel, boundary = 'extend'), depths, color = c, label=label)
                    
        
                legend = plt.legend(fontsize = fontsize - 2)
                for lin in legend.get_lines():
                    lin.set_linewidth(3)
        
                xrange = ranges[area_name][analysis_var]
                # ax.plot(all_temp[0], depths)
                gplt.endax1(fprops, ax, title = None,
                            xlabel = xlabel, ylabel = 'Depth (m)',
                            xrange = xrange, yrange = [depth_range[1], depth_range[0]], fontsize = 14)
        
            title = f'{area_name} {analysis_var} RMS errors'
            plt.suptitle(title, fontsize = fontsize)
        
            plt.savefig(outfile, dpi = fprops['dpi'])
            fprops['fig'].clear()
            plt.close(fprops['fig'])



#     #------------------------------------------------------------------------------
#     #------------------------------------------------------------------------------
#     #------------------------------------------------------------------------------
#     # second step, just ts
#     #------------------------------------------------------------------------------

#     analysis_var = 'ts'
#     plot_dir = set_plot_dir(vertical_covariance_form, var_descr, analysis_var = analysis_var)

#     # #------------------------------------------------------------------------------
#     # # Get vertical covariance Cv
#     # #------------------------------------------------------------------------------
#     # replace Cvf with B_a
#     for icov in range(0, Cvf.shape[0]):
#         Cvf[icov] = B_a[icov * An_nv: (icov + 1) * An_nv, icov * An_nv: (icov + 1) * An_nv]

#     # reduce vertical covariance to Gaussian diagonal values
#     covariances.vert_cov_gauss(vertical_covariance_form, var_descr, Cvf, Cv_nz, vscale_points)

#     plt3dvr.Cvf_mean_stdev_prof(args, plot_dir, Cvf, Cv_latf, Cv_lonf, Cv_nz, Cv_depth, depths, depth_edges_use, meantsf, stdtsf)

#     #------------------------------------------------------------------------------
#     # Analysis grid does not change
#     #------------------------------------------------------------------------------

#     #------------------------------------------------------------------------------
#     # Subselect profiles, one profile for each analysis point
#     #------------------------------------------------------------------------------

#     #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     # ob_list_orig = ob_list.copy()
#     # ob_list = ob_list[12:14]
#     # ob_lat = ob_lat[12:14]
#     # ob_lon = ob_lon[12:14]
#     # n_obs = len(ob_list)

#     # ob_list = np.array([ob_list[0]])
#     # ob_lat = np.array([ob_lat[0]])
#     # ob_lon = np.array([270.3333333333333333333])
#     # n_obs = 1

#     #------------------------------------------------------------------------------
#     # Horizontal interpolation operator to obs locations
#     #------------------------------------------------------------------------------
#     Hh = obs.horizontal_interpolation(args, plot_dir, An_lon, An_lat, ob_lon, ob_lat, An_nh, n_obs)

#     #------------------------------------------------------------------------------
#     # Steric Height perturbation observation operator for TS perturbation
#     #------------------------------------------------------------------------------
#     S = obs.stht_vertint_operator(args, plot_dir, An_latf, An_lonf, An_lat, An_lon, An_depth, An_nh, An_nz, meantsf, depth_range)

#     #------------------------------------------------------------------------------
#     # T and S observation operators in horizontal and vertical
#     #------------------------------------------------------------------------------
#     (Hh_ts, S_ts) = obs.ts_point_operator(ob_list, Hh, An_nz, An_nv, decim_vert)
#     #!!!!!!!!!!!!!!!!!!!!!
#     # Hh_ts = Hh_ts[7,:][np.newaxis,:]
#     # S_ts = S_ts[7,:][np.newaxis,:]

#     #------------------------------------------------------------------------------
#     # Observation operator
#     # H_stht: Map from TS anomalies at state points to stht anomalies at obs points
#     # H_ts: Map from TS anomalies at state points to TS anomalies at obs points
#     #------------------------------------------------------------------------------
#     H_stht = obs.stht_obs_operator(Hh, S, An_nh)
#     H_ts = obs.ts_obs_operator(Hh_ts, S_ts, n_obs)

#     # A useful operator to interpolate from state points to obs points
#     # This interpolates all vertical levels independently at once
#     interp_state_to_obs = np.kron(Hh, np.eye((An_nv)))

#     #------------------------------------------------------------------------------
#     # Get everything into state space (one vector)
#     #------------------------------------------------------------------------------
#     climo_state = meantsf.flatten()
#     #bg_state = meantsf.flatten() # this has been updated, don't change it!
#     S_state = S.flatten()

#     #------------------------------------------------------------------------------
#     # Innovation
#     #------------------------------------------------------------------------------
#     innovation_stht = obs.stht_innovation(args, plot_dir, ob_list, H_stht, bg_state, climo_state,
#                                           interp_state_to_obs, An_depth, An_nz, An_nv)
#     innovation_ts = obs.ts_innovation(args, ob_list, H_ts, bg_state, An_depth, An_nz, decim_vert)
#     n_obs_ts = len(innovation_ts)
#     #!!!!!!!!!!!!!
#     # innovation_ts = np.array([1.0])
#     # n_obs_ts = 1

#     #------------------------------------------------------------------------------
#     # Observation error
#     #------------------------------------------------------------------------------
#     n_profiles = len(ob_list)
#     if var_descr == 'both':
#         (R_stht, R_ts, R_stht_ts) = obs.R(args, stht_obs_error_variance, t_obs_error_variance / 4.0, s_obs_error_variance / 4.0,
#                                           n_obs_ts, n_profiles, An_nz, An_nv, decim_vert)
#     else:
#         (R_stht, R_ts, R_stht_ts) = obs.R(args, stht_obs_error_variance, t_obs_error_variance, s_obs_error_variance,
#                                           n_obs_ts, n_profiles, An_nz, An_nv, decim_vert)
# #!!!
#     # R_ts = np.zeros((1,1))
#     # R_ts[0,0] = t_obs_error_variance

#     #------------------------------------------------------------------------------
#     # Error covariance
#     #------------------------------------------------------------------------------
#     # Horizontal correlation
#     Bh = covariances.horizontal_correlation_gauss(args, plot_dir, ma.getdata(An_lonf),  ma.getdata(An_latf), hlength_scale)

#     #------------------------------------------------------------------------------
#     # Background covariance
#     #------------------------------------------------------------------------------
#     # don't reduce the amplitude on step 2.  The analysis error from step 1 is now the background error.
#     B = covariances.B(args, plot_dir, n_state, Bh, An_nh, An_nv, Cvf, 1.0)

#     (K, K_avg) = covariances.K(args, plot_dir, An_nh, S, Cvf, cov_amplitude)

#     #------------------------------------------------------------------------------

#     H = H_ts.copy()
#     R = R_ts
#     innovation = innovation_ts

#     BHT = B @ H.T
#     HBHT = H @ B @ H.T

#     outfile = plot_dir / 'HBHT.png'

#     title = 'HBHT'
#     plt3dvr.matrix(args, HBHT, title, outfile, vmin = B.min(), vmax = B.max(),
#                    x_label = 'Index', y_label = 'Index',
#                    cb_label = 'Variance', dpi = 150)

#     a = HBHT.copy()
#     d = np.diag(1.0 / np.sqrt(np.diag(a).copy()))
#     a = d @ a @ d
#     print('max normalized hbht offdiagonal:', np.max(np.abs(a-np.diag(np.ones(a.shape[0])))))
#     outfile = plot_dir / 'HBHT_normalized.png'
#     title = 'HBHT normalized'
#     plt3dvr.matrix(args, a, title, outfile, vmin = B.min(), vmax = B.max(),
#                    x_label = 'Index', y_label = 'Index',
#                    cb_label = 'correlation', dpi = 150)

#     (dx_ts, B_a_ts) = covariances.do_3dvar(args, plot_dir, B, BHT, HBHT, H, R, analysis_var, innovation,
#                                    An_lat, An_lon, An_latf, An_lonf, An_nz,
#                                    depths, depth_edges_use, depth_range,
#                                    An_nv, An_nh, ob_list, bg_state, interp_state_to_obs,
#                                    cov_amplitude, fontsize, decim_vert,
#                                    incr_tmin = args.vmin_incr_temp2, incr_tmax = args.vmax_incr_temp2,
#                                    incr_smin = args.vmin_incr_salt2, incr_smax = args.vmax_incr_salt2)

# This was from GOM ts 2-Gauss.
# An off-diagonal correlation in HBHT was rather large (0.91).
# It seems to have been caused by a high variance in the bottom of salinity in Cvf at one point.
# Smoothing Cvf across space helped alleviate this.
# b = a-np.diag(np.ones(a.shape[0]))
# print(np.max(np.abs(b)))
# print(np.where(b == np.max(np.abs(b))))

# H_ts_orig = H_ts.copy()
# H_ts = H_ts[[53,107],:]
    
# 488,542
# 491,545
# 701,755 - 0.92

# 54 T and S obs per profile
# p1 = 701 / 54 # profile 12 (starting at 0), starting at analysis point 15, state elements 1695, 1801
# p2 = 755 / 54 # profile 13 (starting at 0), starting at analysis point 16, state elements 1801, 1907

# Within HBHT, points 53 & 107
# i10 = 12 * 54
# i20 = 13 * 54

