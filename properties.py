# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:06:35 2025

@author: ga_ja
"""

import gsw
from scipy.interpolate import pchip_interpolate
import numpy as np
from numpy import ma
from pathlib import Path
import plotting_3dvar as plt3dvr
import plotting_cov as pltcov
import plotting_profile as pltprof
import nc_util

def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def spcvol(t, s, z, lat, lon):
    p = gsw.p_from_z(-z, lat)
    SA = gsw.SA_from_SP(s, p, lon, lat)
    CT = gsw.CT_from_t(SA ,t, p)
    sv = gsw.specvol(SA, CT, p)
    return sv


def stht(t, s, z, lat, lon, pref = 1.0e7, add_surface = True):
    p = gsw.p_from_z(-z, lat)
    SA = gsw.SA_from_SP(s, p, lon, lat)
    CT = gsw.CT_from_t(SA ,t, p)

    p = p * 1.0e4 # convert to Pascals

    sint = pchip_interpolate(p, SA, pref)
    tint = pchip_interpolate(p, CT, pref)

    iins = np.searchsorted(p, pref)
    p = np.append(p[0:iins], pref)
    SA = np.append(SA[0:iins], sint)
    CT = np.append(CT[0:iins], tint)
    
    # add surface 0 pressure if necessary by just copying first T&S
    if add_surface and p[0] > 0:
        SA = np.insert(SA, 0, SA[0])
        CT = np.insert(CT, 0, CT[0])
        p = np.insert(p, 0, 0)

    sv = gsw.specvol(SA, CT, p / 1.0e4)

# v_i * (p_i   - p_i-1)/2 +
# v_i * (p_i+1 - p_i  )/2

# interior
# = v_i * (p_i+1 - p_i-1)/2
# first
# v_0 * (p_1 - p_0) / 2
# last
# v_n-1 * (p_n-1 - p_n-2) / 2

    vint = press_intgrl_total(p)

    return vint.T @ sv / 9.81

def press_intgrl_total(p):
    if p[0] == 0:   # if first pressure is at surface, use normal trapezoidal integration
        vint = (p[2:] - p[0:-2]) / 2.0
        vint = np.insert(vint, 0, (p[1] - p[0]) / 2.0)
        vint = np.append(vint, (p[-1] - p[-2]) / 2.0)
    else:   # if first pressure is not at surface, assume integration to surface by extending top-most value
        vint = (p[2:] - p[0:-2]) / 2.0
        vint = np.insert(vint, 0, (p[1] - p[0]) / 2.0 + p[0])
        vint = np.append(vint, (p[-1] - p[-2]) / 2.0)

    return vint

# v[i] * (p[i] - p[i-1]) / 2.0 + v[i] * (p[i+1] - p[i]) / 2.0
# v[i] * (p[i+1] - p[i-1]) / 2.0

def d_stht_d_ts(t, s, z, lat, lon, pref, check = False):
    delta = 0.001

    spcvol_00 = spcvol(t, s, z, lat, lon)

    spcvol_p0 = spcvol(t + delta, s        , z, lat, lon)
    spcvol_m0 = spcvol(t - delta, s        , z, lat, lon)
    spcvol_0p = spcvol(t        , s + delta, z, lat, lon)
    spcvol_0m = spcvol(t        , s - delta, z, lat, lon)

    d_spcvol_dt = (spcvol_p0 - spcvol_m0) / (2.0 * delta)
    d_spcvol_ds = (spcvol_0p - spcvol_0m) / (2.0 * delta)

    d_spcvol_dts = np.append(d_spcvol_dt, d_spcvol_ds)

    p = gsw.p_from_z(-z, lat) * 1.0e4 # convert to Pascals
    vint = press_intgrl_total(p)

    S = d_spcvol_dts * np.append(vint, vint) / 9.81

# check derivative by computing d_stht_dt rather than propagating d_spcvol_dt through
    if check:
        nz = len(t)
        print('d_stht_dtemp check (idepth, deriv, S[iz]')
        for iz in range(0, nz):
            dt = np.zeros((nz))
            dt[iz] = delta
            stht_p = stht(t + dt, s, z, lat, lon, pref = pref)
            stht_m = stht(t - dt, s, z, lat, lon, pref = pref)

            deriv = (stht_p - stht_m) / (2.0 * delta)

            print(iz, deriv, S[iz])

        print('d_stht_dsalt check (idepth, deriv, S[iz]')
        for iz in range(0, nz):
            dt = np.zeros((nz))
            dt[iz] = delta
            stht_p = stht(t, s + dt, z, lat, lon, pref = pref)
            stht_m = stht(t, s - dt, z, lat, lon, pref = pref)

            deriv = (stht_p - stht_m) / (2.0 * delta)

            print(iz, deriv, S[iz + nz])

    return S
    
#%%
def distsq_deg(p1lat, p1lon, p2lat, p2lon, hlength_scale, An_type = '2d'):
    latdiff = np.subtract.outer(p1lat, p2lat)
    latavg = np.deg2rad(np.add.outer(p1lat, p2lat) / 2.0)
    londiff = np.subtract.outer(p1lon, p2lon) * np.cos(latavg)
    dist2 = np.zeros(latdiff.shape)
    if An_type != 'lon_transect':      # transect at a fixed lat
        dist2 += londiff * londiff
    if An_type != 'lat_transect':      # transect at a fixed lon
        dist2 += latdiff * latdiff
    return np.exp(- dist2 / (hlength_scale * hlength_scale))


#%%
def ts_vec_2_mat(data, An_nv, An_nh):
    """
    Convert a state vector to a 2D array of TS as [depth_depth, An_position]

    Args:
        data (np array): state vector ordered as [position][T_depth_S_depth]
        An_nv (int): number of vertical points in state (T_depth_S_depth).
        An_nh (int): number of horizontal points in state.

    Returns:
        array of temperature and array of salinity ordered as [depth, position].

    """
    temp = data.reshape(An_nh, An_nv)
    nz = int(An_nv / 2)
    t = temp[:, 0:nz].T
    s = temp[:, nz:].T # return array of [depth, obs]
    return(t, s)
