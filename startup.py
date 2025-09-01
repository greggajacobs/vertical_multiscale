# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:03:05 2025

@author: ga_ja
"""

import argparse
from pathlib import Path
from datetime import datetime as dtm

def args():
    parser = argparse.ArgumentParser(
                    prog='reader',
                    description='Read WOD data and save as pickle',
                    epilog='For more information, call Gregg')

    parser.add_argument('-base_drive', type = str, default = 'D:/', help='drive letter to prepend to all directories')
    
    parser.add_argument('-beg_lat', type = float, default = -90.0, help='minimum latitdue of data to retain for processing')
    parser.add_argument('-end_lat', type = float, default = 90.0, help='minimum latitdue of data to retain for processing')
    parser.add_argument('-beg_lon', type = float, default = -360.0, help='minimum longitude of data to retain for processing')
    parser.add_argument('-end_lon', type = float, default = 360.0, help='minimum longitude of data to retain for processing')
    parser.add_argument('-beg_t', type = str, default = '17000101', help='Minimum time YYYYMMDD')
    parser.add_argument('-end_t', type = str, default = '17000101', help='Maximum time YYYYMMDD')
    parser.add_argument('-beg_depth', type = float, default = 0.0, help='Upper depth (near surface)')
    parser.add_argument('-end_depth', type = float, default = 2000.0, help='Lower depth (in deep water)')

    parser.add_argument('-beg_alat', type = float, default = -90.0, help='minimum latitdue of analysis domain')
    parser.add_argument('-end_alat', type = float, default = 90.0, help='minimum latitdue of analysis domain')
    parser.add_argument('-beg_alon', type = float, default = -360.0, help='minimum longitude of analysis domain')
    parser.add_argument('-end_alon', type = float, default = 360.0, help='minimum longitude of analysis domain')
    parser.add_argument('-beg_adepth', type = float, default = 0.0, help='Upper depth (near surface) of analysis domain')
    parser.add_argument('-end_adepth', type = float, default = 2000.0, help='Lower depth (in deep water) of analysis domain')

    parser.add_argument('-data_dir', type = str, default = 'fred', help='directory location of data files')

    parser.add_argument('-search_filename', type = str, default = r'wod_\d{9}O.nc', help='regular expression for filename matching')

    parser.add_argument('-qc_max_top_depth', type = float, default = 50.0, help='Shallowest sample must have depth less than this.')
    parser.add_argument('-qc_min_bot_depth', type = float, default = 1000.0, help='Deepest sample must have depth greater than this.')
    parser.add_argument('-qc_mean_sampling_dz', type = float, default = 50.0, help='Average vertical spacing must be less than this.')
    parser.add_argument('-qc_require_salt', action = 'store_true', help='Profile must have observed salinity.')
    parser.add_argument('-qc_require_temp', action = 'store_true', help='Profile must have observed temperature.')
    parser.add_argument('-qc_sort_depth', action = 'store_true', help='Sort profile observations based on reported depth.')
    parser.add_argument('-qc_remove_masked', action = 'store_true', help='Sort profile observations based on reported depth.')

    parser.add_argument('-bathy_filename', type = str, default = 'D:/GEBCO/GEBCO_2024_sub_ice_topo.nc', help='Full path to bathymetry netcdf file')
    parser.add_argument('-bathy_variable', type = str, default = 'elevation', help='Variable name in netcdf file for bathymetry')
    parser.add_argument('-bathy_stride', type = int, default = 10, help='Decimation in lat and lon direction for bathymetry data')

    parser.add_argument('-plot_dir', type = str, default = 'D:\MLD_analysis\plots\3D_to_2D_var', help='Path to output plots')

    parser.add_argument('-dlon_agrid', type = float, default = 0.5, help='Longitude bin size for analysis grid.')
    parser.add_argument('-dlat_agrid', type = float, default = 0.5, help='Latitude bin size for analysis grid.')
    parser.add_argument('-no_plotting', action = 'store_true', help='Do not make plots.')

    parser.add_argument('-profile_pickle_file', type = str, default = 'D:\MLD_analysis\plots', help='Path to pickle file for profiles.')

    parser.add_argument('-in_covariance_netcdf_format', type = str, default = 'D:\mean_TS\cov{:02d}.nc', help='Path to input covariance netcdf files')

    parser.add_argument('-reference_pressure', type = float, default = 1.0e7, help='Reference pressure for steric height (Pascals)')
    parser.add_argument('-vmin_temp', type = float, default = 5.0, help='plotting colorbar min temperature')
    parser.add_argument('-vmax_temp', type = float, default = 32.0, help='plotting colorbar max temperature')
    parser.add_argument('-vmin_salt', type = float, default = 5.0, help='plotting colorbar min salinity')
    parser.add_argument('-vmax_salt', type = float, default = 32.0, help='plotting colorbar max salinity')
    parser.add_argument('-vmin_temp_stndev', type = float, default = 0.0, help='plotting colorbar min temperature standard deviation')
    parser.add_argument('-vmax_temp_stndev', type = float, default = 4.0, help='plotting colorbar max temperature standard deviation')
    parser.add_argument('-vmin_salt_stndev', type = float, default = 0.0, help='plotting colorbar min salinity standard deviation')
    parser.add_argument('-vmax_salt_stndev', type = float, default = 1.0, help='plotting colorbar max salinity standard deviation')
    parser.add_argument('-An_type', type = str, default = 'lat_transect', help='can be a lat_transect or lon_trasect, anything else is a lat,lon grid')

    args = parser.parse_args()

    args.base_drive = Path(args.base_drive)
    if not args.base_drive.exists():
        estr = 'base drive directory does not exist: {}'.format(args.base_drive)
        print(estr)
        raise(Exception(estr))

    args.data_dir = args.base_drive / args.data_dir
    if not args.data_dir.exists():
        estr = 'data directory does not exist: {}'.format(args.data_dir)
        print(estr)
        raise(Exception(estr))

    args.bathy_filename = args.base_drive / args.bathy_filename
    if not args.bathy_filename.exists():
        estr = 'bathymetry file does not exist: {}'.format(args.bathy_filename)
        print(estr)
        raise(Exception(estr))

    args.beg_lon = (args.beg_lon + 720.0) % 360.0
    args.end_lon = (args.end_lon + 720.0) % 360.0

    args.beg_t = dtm.strptime(args.beg_t, '%Y%m%d')
    args.end_t = dtm.strptime(args.end_t, '%Y%m%d')

    # args.bathy_filename = Path(args.bathy_filename)
    # if not args.bathy_filename.exists():
    #     estr = 'bathymetry file does not exist: {}'.format(args.bathy_filename)
    #     print(estr)
    #     raise(Exception(estr))

    args.plot_dir = args.base_drive / args.plot_dir
    args.plot_dir.mkdir(parents = True, exist_ok = True)
    
    args.in_covariance_netcdf_format = str(args.base_drive) + args.in_covariance_netcdf_format

    args.profile_pickle_file = args.base_drive / args.profile_pickle_file

    return args
