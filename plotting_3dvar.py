# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:16:22 2025

@author: ga_ja
"""

import numpy as np
import gplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from cartopy import crs as ccrs, feature as cfeature
import gc
import plotting_cov as pltcov
import plotting_profile as pltprof

#%%
# data is separate lines, each row is a separate data set
def foflon(args, posns, data, labels, title, outfile, ymin = None, ymax = None,
           x_label = None, y_label = None,
           padleft = 0.07, padright = 0.03, padtop = 0.07, padbot = 0.12,
           figsize = (11.0, 5.0), dpi = 150,
           fontsize = 14):
    if args.no_plotting:
        return

    (fprops, ax) = gplt.fig_grid11(padleft = padleft, padright = padright,
                                   padtop = padtop, padbot = padbot,
                                   figsize = figsize, dpi = dpi,
                                   plt_type = 'linearx_lineary')

    linestyles = ['-k',
                  '--k',
                  '..k',
                  '-r',
                  '--r',
                  '..r',
                  '-b',
                  '--b',
                  '..b',
                  ]
    if len(posns) == 1:
        x_range = [posns[0] - 0.5, posns[0] + 0.5]
    else:
        x_range = [posns.min(), posns.max()]
    if len(data) == 1:
        y_range = [data[0] - 0.1, data[0] + 0.1]
    else:
        y_range = [data.min(), data.max()]
    if (ymin is None) or (ymax is None):
        ydif = y_range[1] - y_range[0]
        y_range = [y_range[0] - 0.10 * ydif, y_range[1] + 0.10 * ydif]
    else:
        y_range = [ymin, ymax]

    if data.ndim > 1:
        ndatasets = data.shape[0]
    else:
        ndatasets = 1
        data = data[np.newaxis,:]


    for i, label in enumerate(labels):
        d = data[i,:]
        ls = linestyles[i]

        ax.plot(posns, d, ls, label = label)
        
    legend = plt.legend(fontsize = fontsize - 2)
    for lin in legend.get_lines():
        lin.set_linewidth(3)


    gplt.endax(fprops, ax, outfile = outfile, title = title,
               xlabel = x_label, ylabel = y_label,
               xrange = x_range, yrange = y_range,
               fontsize = 14)

#%%
# just a line plot
def eigvals(args, data, title, outfile, ymin = None, ymax = None,
           x_label = None, y_label = None,
           x_range = None,
           padleft = 0.10, padright = 0.03, padtop = 0.07, padbot = 0.15,
           figsize = (8.0, 4.0), dpi = 150,
           fontsize = 14):
    if args.no_plotting:
        return

    (fprops, ax) = gplt.fig_grid11(padleft = padleft, padright = padright,
                                   padtop = padtop, padbot = padbot,
                                   figsize = figsize, dpi = dpi,
                                   plt_type = 'linearx_logy')

    xs = np.arange(0, len(data))
    ax.plot(xs, data)
    
    y_range = [0.0, np.max(data) * 1.1]
    
    if x_range is None:
        x_range = [0, len(data) - 1]

    gplt.endax(fprops, ax, outfile = outfile, title = title,
               xlabel = x_label, ylabel = y_label,
               xrange = x_range, yrange = y_range,
               fontsize = 14)

#%%
def lonslice(args, temp, salt, palong, depth_edges, depth_range,
             title_temp, title_salt,
             super_title, outfile,
             vmin_temp = None, vmax_temp = None,
             vmin_salt = None, vmax_salt = None,
             xlabel = None, ylabel = None,
             padleft = 0.08, padright = 0.25, padtop = 0.11, padbot = 0.10,
             xinter = 0.03, yinter = 0.05,
             figsize = (11.0, 6.0), dpi = 150,
             fontsize = 14):
    if args.no_plotting:
        return

    fprops = gplt.fig_grid(nx = 2, ny = 1,
                           padleft = padleft, padright = padright,
                           padtop = padtop, padbot = padbot,
                           figsize = figsize, dpi = dpi)

    dlon = (palong.max() - palong.min()) / (len(palong) - 1.0)
    edges_x = np.arange(palong.min() - dlon / 2.0, palong.max() + dlon * 0.5001, dlon)
    edges_y = depth_edges
    xrange = [palong.min(), palong.max()]
    

    if vmin_temp is None:
        vmin_temp = np.min(temp)
    if vmax_temp is None:
        vmax_temp = np.max(temp)

    if vmin_salt is None:
        vmin_salt = np.min(salt)
    if vmax_salt is None:
        vmax_salt = np.max(salt)

    for (i, title) in enumerate([title_temp, title_salt]):

        ax = gplt.ax_grid(fprops, row = 0, col = i, plt_type = 'linearx_lineary')
        
        do_ylabels = True
        if i == 0:
            data = temp
            vmin = vmin_temp
            vmax = vmax_temp
            ylab = ylabel
        else:
            do_ylabels = False
            data = salt
            vmin = vmin_salt
            vmax = vmax_salt
            ylab = None

        mappable = ax.pcolormesh(edges_x, edges_y, data,
                                         cmap = 'jet',
                                         vmin = vmin, vmax = vmax)

        gplt.endax1(fprops, ax, title = title,
                    xlabel = xlabel, ylabel = ylab,
                    xrange = xrange, yrange = [depth_range[1], depth_range[0]],
                    do_ylabels = do_ylabels, fontsize = 14)

    cbheight_scale = 0.8
    cbheight = (1.0 - padbot - padtop) * cbheight_scale
    bb = [1.0 - padright + xinter,
          padbot + ((1.0 - padbot - padtop) - cbheight) / 2.0,
          0.03,
          cbheight]

    cax = fprops['fig'].add_axes(bb)
    ne = 101
    es = np.linspace(0.0, 1.0, ne)
    vs = np.linspace(0.0, 1.0, ne - 1)
    vs = np.append(vs[:,np.newaxis], vs[:,np.newaxis], axis = 1)
    cax.pcolormesh([0.0, 0.5, 1.0], es, vs, cmap = 'jet', vmin = 0.0, vmax = 1.0)

    cax.set_ylim(0.0, 1.0)
    # ylabel_posn, ylabels = gplt.nrange(0.0, 1.0)
    # cax.set_yticks(ylabel_posn, labels = ylabels, fontsize = fontsize)
    cax.tick_params(axis='x', labeltop=False, labelbottom=False, top=False, bottom=False)
    cax.tick_params(axis='y', labelleft=False, labelright=False, left=False, right=False)
    # cax.yaxis.set_label_position("right")
    # cax.yaxis.tick_right()
    # cax.set_ylabel('Temperature ($^\circ$C)', fontsize = fontsize)

    cax1 = cax.twinx()
    cax1.spines['top'].set_visible(False)
    cax1.spines['bottom'].set_visible(False)
    cax1.spines['left'].set_visible(False)
    cax1.spines['right'].set_visible(True)

    cax1.set_ylim(vmin_temp, vmax_temp)
    cax1.tick_params(axis='x', labeltop=False, labelbottom=False, top=False, bottom=False)
    cax1.tick_params(axis='y', labelleft=False, labelright=True, right=True, left=False)
    cax1.yaxis.tick_right()
    ylabel_posn, ylabels = gplt.nrange(vmin_temp, vmax_temp)
    cax1.set_yticks(ylabel_posn, labels = ylabels, fontsize = fontsize)
    cax1.yaxis.set_label_position('right')
    cax1.set_ylabel('Temperature ($^\circ$C)', fontsize = fontsize)

    cax2 = cax.twinx()
    cax2.spines['top'].set_visible(False)
    cax2.spines['bottom'].set_visible(False)
    cax2.spines['left'].set_visible(False)
    cax2.spines['right'].set_visible(True)
    cax2.spines.right.set_position(("axes", 4.0))

    cax2.set_ylim(vmin_salt, vmax_salt)
    cax2.tick_params(axis='x', labeltop=False, labelbottom=False, top=False, bottom=False)
    cax2.tick_params(axis='y', labelleft=False, labelright=True, right=True, left=False)
    cax2.yaxis.tick_right()
    ylabel_posn, ylabels = gplt.nrange(vmin_salt, vmax_salt)
    cax2.set_yticks(ylabel_posn, labels = ylabels, fontsize = fontsize)
    cax2.yaxis.set_label_position('right')
    cax2.set_ylabel('Salinity (PSU)', fontsize = fontsize)

    plt.suptitle(super_title, fontsize = fontsize)

    plt.savefig(outfile, dpi = fprops['dpi'])
    fprops['fig'].clear()
    plt.close(fprops['fig'])

#%%
def matrix(args, m, title, outfile, vmin = None, vmax = None,
           x_label = None, y_label = None, cb_label = None,
           padleft = 0.08, padright = 0.10, padtop = 0.10, padbot = 0.07,
           figsize = (11.0, 8.5), dpi = 150,
           fontsize = 14):
    if args.no_plotting:
        return

    (fprops, ax) = gplt.fig_grid11(padleft = padleft, padright = padright,
                                   padtop = padtop, padbot = padbot,
                                   figsize = figsize, dpi = dpi,
                                   plt_type = 'linearx_lineary')

    edges_x = np.arange(-0.5, m.shape[1] - 0.49, 1.0)
    edges_y = np.arange(-0.5, m.shape[0] - 0.49, 1.0)

    if vmin is None:
        vmin = np.min(m)
    if vmax is None:
        vmax = np.max(m)

    mappable = ax.pcolormesh(edges_x, edges_y, m,
                                     cmap = 'jet',
                                     vmin = vmin, vmax = vmax)

    cb = plt.colorbar(mappable, location = 'right', shrink = 0.5, pad = 0.02)
    ylabel_posn, ylabels = gplt.nrange(vmin, vmax)
    cb.ax.set_yticks(ylabel_posn, labels = ylabels, fontsize = fontsize)
    if cb_label is not None:
        cb.ax.set_ylabel(cb_label, fontsize = fontsize)

    gplt.endax(fprops, ax, outfile = outfile, title = title,
               xlabel = x_label, ylabel = y_label,
               xrange = [edges_x[0], edges_x[-1]], yrange = [edges_y[0], edges_y[-1]],
               fontsize = 14)

#%%
def K(args, vals, vmin, vmax, title, outfile):
    if args.no_plotting:
        return

    fprops = gplt.fig_grid(figsize = (9.0, 8.5))
    ax = gplt.ax_grid(fprops, plt_type = 'linearx_lineary')
    edges = np.arange(0.0, vals.shape[1] + 1)

    mappable = ax.pcolormesh(edges, edges, vals,
                                     cmap = 'jet',
                                     vmin = vmin, vmax = vmax)

    cb = plt.colorbar(mappable, location = 'right', shrink = 0.5, pad = 0.02)
    cb.ax.set_ylabel('K (S B_v S^T)')

    gplt.endax(fprops, ax, outfile = outfile, title = title, xlabel = 'Analysis grid point', ylabel = 'Analysis grid point', xrange = [0, vals.shape[1]], yrange = vals.shape[0], fontsize = 14)

#%%
def obs_prof(args, ob_list, meantsf, depths, depth_range, title, outfile,
             padleft = 0.10, padright = 0.07, padtop = 0.07, padbot = 0.07,
             figsize = (11.0, 8.5), dpi = 150, fontsize = 14, pchip = False):
    if args.no_plotting:
        return

    cmap = plt.cm.jet
    nprofs = len(ob_list)
    colors = [cmap(i / (nprofs - 1)) for i in range(nprofs, 0, -1)]

    if args.An_type == 'lat_transect':
        posns = [p.lon for p in ob_list]
    else:
        posns = [p.lat for p in ob_list]
    sorted_indices = np.argsort(posns)

    fprops = gplt.fig_grid(nx = 2, ny = 1, xinter = 0.10, padtop = 0.1, figsize = (11.0, 8.5))

    for (i, xlabel, xrange) in zip(range(0, 2),
                                   [r'Temperature ($^\circ$C)', r'Salinity (PSU)'],
                                   [[args.vmin_temp, args.vmax_temp],
                                    [args.vmin_salt, args.vmax_salt]]):

        ax = gplt.ax_grid(fprops, row = 0, col = i, plt_type = 'linearx_lineary')

        for (iprof, c) in zip(reversed(sorted_indices), colors):
            p = ob_list[iprof]
            posn = posns[iprof]
            if i == 0:
                if pchip:
                    data = p.temp_pch[0: len(depths)]
                    ddepth = depths
                else:
                    data = p.temp
                    ddepth = p.depth
            else:
                if pchip:
                    data = p.salt_pch[0: len(depths)]
                    ddepth = depths
                else:
                    data = p.salt
                    ddepth = p.depth

            ax.plot(data, ddepth, color = c, label='{:5.2f}$^\circ$'.format(posn))

        mts = np.average(meantsf, axis = 0)
        nmts = int(len(mts) / 2)

        ax.plot(mts[i * nmts: (i + 1) * nmts], depths, 'k', linewidth = 3, label = 'clim')

        legend = plt.legend(fontsize = fontsize - 2)
        for lin in legend.get_lines():
            lin.set_linewidth(3)

        gplt.endax1(fprops, ax, title = None,
                    xlabel = xlabel, ylabel = 'Depth (m)',
                    xrange = xrange, yrange = [depth_range[1], depth_range[0]], fontsize = 14)

    plt.suptitle(title, fontsize = fontsize)

    plt.savefig(outfile, dpi = fprops['dpi'])
    fprops['fig'].clear()
    plt.close(fprops['fig'])

#%%
def S(args, S, depths, depth_range, title, outfile, fontsize = 14):
    if args.no_plotting:
        return

    ylabel = 'Depth (m)'
    cmap = plt.cm.jet
    nv = S.shape[0]
    ndepth = len(depths)
    colors = [cmap(i / (nv - 1)) for i in range(nv)]

    fprops = gplt.fig_grid(nx = 2, ny = 1, xinter = 0.10, padtop = 0.1, figsize = (11.0, 8.5))

    for (i, xlabel, xrange) in zip(range(0, 2),
                                   ['d stht / d T(z_i)', 'd stht / d S(z_i)'],
                                   [[0.0, 0.008], [-0.05, 0.0]]):

        ax = gplt.ax_grid(fprops, row = 0, col = i, plt_type = 'linearx_lineary')

        for (r, c) in zip(range(0, S.shape[0]), colors):

            ax.plot(S[r, ndepth * i: ndepth * (i + 1)], depths, color = c)

        gplt.endax1(fprops, ax, title = None,
                    xlabel = xlabel, ylabel = 'Depth (m)',
#                    xrange = xrange, yrange = depth_range, fontsize = 14)
                    xrange = xrange, yrange = [depth_range[1], depth_range[0]], fontsize = 14)

    plt.suptitle(title, fontsize = fontsize)
    plt.savefig(outfile, dpi = fprops['dpi'])
    fprops['fig'].clear()
    plt.close(fprops['fig'])

#%%
def S_err(args, stht_err_percent, errs, An_lon, An_lat, title, outfile,
          padleft = 0.07, padright = 0.03, padtop = 0.10, padbot = 0.07,
          figsize = (11.0, 5.0), dpi = 150,
          fontsize = 14):
    if args.no_plotting:
        return

    nerr = len(errs)
    nlon = stht_err_percent.shape[0]

    cmap = plt.cm.jet
    colors = [cmap(i / (nerr - 1)) for i in range(nerr)]
    
    if args.An_type == 'lat_transect':
        posns = An_lon
        x_label = 'Longitude ($^\circ$E)'
    else:
        posns = An_lat
        x_label = 'Latitude ($^\circ$N)'


    (fprops, ax) = gplt.fig_grid11(padleft = padleft, padright = padright,
                                   padtop = padtop, padbot = padbot,
                                   figsize = figsize, dpi = dpi,
                                   plt_type = 'linearx_lineary')

    for ierr, (err, color) in enumerate(zip(errs, colors)):
        ax.plot(posns, stht_err_percent[:,ierr] * 100.0,
                color = color, label='{:5.2f} $^\circ$'.format(err))

    legend = plt.legend(fontsize = fontsize)
    for lin in legend.get_lines():
        lin.set_linewidth(3)

    gplt.endax(fprops, ax, outfile = outfile, title = title,
               xlabel = x_label, ylabel = 'Percent error',
               xrange = [min(posns), max(posns)], yrange = [-20.0, 30.0],
               fontsize = fontsize)



#%%
def data_distribution(args, lats, lons, title, outfile, bathy_dat, bathy_interp,
                s = None,
                lats1 = None, lons1 = None,
                lats2 = None, lons2 = None,
                lats3 = None, lons3 = None):
    if args.no_plotting:
        return

    if s is None:
        s = mpl.rcParams['lines.markersize']**2

    clon = (args.beg_lon + args.end_lon) / 2.0
    clat = (args.beg_lat + args.end_lat) / 2.0

    projPC = ccrs.PlateCarree()
    projStr = ccrs.Stereographic(central_longitude=clon, central_latitude=clat)

    plot_proj = projStr

    fprops = gplt.fig_grid(figsize = (11.0, 8.5))
    ax = gplt.ax_grid(fprops, plt_type = 'latlon',
                      projection = plot_proj
                      )

    gl = ax.gridlines(
        draw_labels=True, linewidth=1, color='gray', alpha=0.5,
        linestyle='--',
        # ylabels_right = False, xlabels_top = False,
        # ylabel_style = {'fontsize': 14}, xlabel_style = {'fontsize': 14}
    )
    # gl.ylabels_right = False
    # gl.xlabels_top = False
    # gl.xlabel_style = {'size': 14}
    # gl.ylabel_style = {'fontsize': 14}

    # for ea in gl.ylabel_artists:
    #     right_label = ea.get_position()[0] > 0
    #     # print(ea, ea.get_position()[0], ea.get_visible())
    #     if right_label:
    #         ea.set_visible(False)

    ax.add_feature(cfeature.LAND)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='brown')

    ax.set_extent([args.beg_lon, args.end_lon, args.beg_lat, args.end_lat], crs = projPC)

    plt.title(title)

    (_, _, lon_dim) = bathy_dat.find_dim('lon')
    (_, _, lat_dim) = bathy_dat.find_dim('lat')

    ax.contour(lon_dim.vals, lat_dim.vals, bathy_dat.vals,
               [-5000.0, -4000.0, -3000.0, -2000.0, -1000.0, -500.0, -100.0],
               colors = 'k', linestyles = ['solid'], linewidths = 0.7, transform = projPC)
    
    ax.plot([args.beg_alon, args.beg_alon, args.end_alon, args.end_alon, args.beg_alon],
            [args.beg_alat, args.end_alat, args.end_alat, args.beg_alat, args.beg_alat],
            'k', linewidth = 3.0, transform = projPC, zorder = 5)

    ax.scatter(lons, lats, transform = projPC, s = s, color = 'r', zorder = 10)

    if (lats1 is not None) and (lons1 is not None):
        #print('scatter1')
        ax.scatter(lons1, lats1, transform = projPC, s = s, color = 'm', zorder = 20)

    if (lats2 is not None) and (lons2 is not None):
        #print('scatter2')
        ax.scatter(lons2, lats2, transform = projPC, s = s, color = 'y', zorder = 30)

    if (lats3 is not None) and (lons3 is not None):
        #print('scatter3')
        ax.scatter(lons3, lats3, transform = projPC, s = s+2, color = 'g',
                   linewidth=1, edgecolor='black', zorder = 40)


    plt.savefig(outfile, dpi = fprops['dpi'])
    fprops['fig'].clf()
    plt.close(fprops['fig'])

# #%%
# def cr8_pdat_ts(varnames, xlabels, colors, source_labels, linestyles, values, depthsin,
#                 profile = None, title = None, depth_range = None):
# # inputs are all lists

#     if depth_range is None:
#         depth_range = [0.0, np.array(depthsin).max()]

#     pdat = {
#         'depth_range': depth_range,
#         'depth_label': 'Depth (m)',
#         'vars': {
#         },
#     }

#     if profile is not None:
#         pdat.update({'profiles': profile})

#     if title is not None:
#         pdat.update({'title': title})
        
#     for (varname, xlab, color, linestyle, vals) in zip(varnames, xlabels, colors, linestyles, values):
#         labs = [varname + ' ' + source_label for source_label in source_labels]
#         pdat['vars'].update({
#             varname: {
#                 'depth': depthsin,
#                 'vals': vals,
#                 'x_label': xlab,
#                 'colors': color,
#                 'linestyles': linestyle,
#                 'labels': labs
#                 }
#             })

#     return pdat

# #%%
# def plot_prof(varnames, xlabels, colors,
#               source_labels, linestyles, values, depthsin, outfile,
#               depth_range = None, x_ranges = None, title = None,
#               padleft = 0.15, padright = 0.1, padtop = 0.1, padbot = 0.3,
#               figsize = (8.5, 11.0), dpi = 150, plt_type = 'linearx_lineary',
#               fontsize = 14):
    
# # each variable can have multiple data sets
# # Therefore each variable has a set of linestyles, colors, and labels for its data sets

#     spine_offset = -0.12 # in axis coordinates [0,1]
#     if depth_range is None:
#         dmin = 1.0e10
#         dmax = -1.0e10
#         for d in depthsin:
#             depth_range = [min(dmin, d.min()), max(dmax, d.max())]

#     if x_ranges is None:
#         x_ranges = [None] * len(values)

#     (fprops, ax0) = gplt.fig_grid11(padleft = padleft, padright = padright,
#                                    padtop = padtop, padbot = padbot,
#                                    figsize = figsize, dpi = dpi,
#                                    plt_type = plt_type)

#     fig = fprops['fig']
#     all_axes = []
#     lines = []
#     for ivar, (varname, xlabel, color, x_range) in enumerate(zip(varnames, xlabels, colors, x_ranges)):
#         #print(ivar, vname)
#         if ivar == 0:
#             ax = ax0
#             #print('orig axis')
#         else:
#             ax = ax0.twiny()
#             ax.spines['top'].set_visible(False)
#             ax.spines['bottom'].set_visible(True)
#             ax.spines['left'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.spines['bottom'].set_position(("axes", ivar * spine_offset))
#             #print('twinned axis')
#         all_axes.append(ax)
        
#         x_range_data = [1.0e10, -1.0e10]

#         for source_label, linestyle, value, depths in zip(source_labels, linestyles, values, depthsin):

#             #print(iline, color, linestyle, label)
#             l = ax.plot(value, depths, color = color, linestyle = linestyle, label = xlabel)
#             lines.append(l[0])

#             r = (value.max() - value.min()) * 0.05
#             x_range_data = [min(x_range_data[0], value.min()) - r, max(x_range_data[1], value.max()) + r]

#         # Decorate the x axis -------------------------------------------------
#         if x_range is None:
#             x_range = x_range_data
#         ax.set_xlim(x_range)

#         ax.tick_params(axis='x',
#                        labeltop = False, labelbottom = True,
#                        top = False, bottom = True,
#                        color = color, labelcolor = color)
#         xlabel_posn, xlabels = gplt.nrange(x_range[0], x_range[1])
#         ax.set_xticks(xlabel_posn, labels = xlabels, fontsize = fontsize,
#                       rotation = 40.0, horizontalalignment = 'right',
#                       verticalalignment = 'top')
#         offset = mtransforms.ScaledTranslation(7/72.,3/72., fprops['fig'].dpi_scale_trans)
#         for label in ax.xaxis.get_majorticklabels():
#             label.set_transform(label.get_transform() + offset)

#         ax.spines['bottom'].set_color(color)
#         ax.set_xlabel(xlabel, fontsize = fontsize, color = color, labelpad = 2)
#         ax.xaxis.set_label_position('bottom') # seems to be necessary to force it, otherwise it's at the top ...

#         # set y range for all axes
#         ax.set_ylim(depth_range[-1], depth_range[0])

#     # Decorate the y axis -------------------------------------------------
#     ylabel_posn, ylabels = gplt.nrange(depth_range[0], depth_range[1])
#     ax0.set_yticks(ylabel_posn, labels = ylabels, fontsize = fontsize)
#     ax0.tick_params(axis='y',
#                    labelleft = True, labelright = False,
#                    left = True, right = False)

#     y_label = 'Depth (m)'
#     ax0.set_ylabel(y_label, fontsize = fontsize)

#     if title is not None:
#         ax0.set_title(title, fontsize = fontsize)

#     # add legend for all variables and data sets --------------------------
#     labs = [l.get_label() for l in lines]
#     legend = ax0.legend(lines, labs, fontsize = fontsize - 2)

#     for lin in legend.get_lines():
#             lin.set_linewidth(3)

#     plt.savefig(outfile)
#     fig.clear()
#     plt.close(fig)

#%%
def Cvf_mean_stdev_prof(args, plot_dir, Cvf, Cv_latf, Cv_lonf, Cv_nz, Cv_depth, depths, depth_edges_use, meantsf, stdtsf):
    if args.no_plotting:
        return

    depth_range = [args.beg_adepth, args.end_adepth]
    for icov, (atlat, atlon) in enumerate(zip(Cv_latf, Cv_lonf)):
        title = 'Covariance latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        outfile = plot_dir / 'Cv_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)
        pltcov.cov22(Cvf[icov], depths, depth_edges_use, depth_range, title, outfile, fontsize = 14)
    
        title = 'Correlation latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        outfile = plot_dir / 'Cv_corr_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)
        pltcov.cov22(Cvf[icov], depths, depth_edges_use, depth_range, title, outfile, fontsize = 14, plot_correlation = True)
    
        # title = 'Mean latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        # outfile = plot_dir / 'TS_mean_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)
        # pdat = pltprof.cr8_pdat_ts(Cv_depth[0: Cv_nz],
        #                          meantsf[icov, 0:Cv_nz],
        #                          meantsf[icov, Cv_nz:],
        #                          title = title, depth_range = depth_range)
        # pdat['vars']['temp'].update({'x_range': [3.0, 32.0]})
        # pdat['vars']['salt'].update({'x_range': [34.3, 36.7]})
        # pltprof.plot(pdat, outfile, title, fontsize = 14)
    
        # title = 'Standev latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        # outfile = plot_dir / 'TS_std_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)
        # pdat = pltprof.cr8_pdat_ts(Cv_depth[0: Cv_nz],
        #                          stdtsf[icov, 0:Cv_nz],
        #                          stdtsf[icov, Cv_nz:],
        #                          title = title, depth_range = depth_range)
        # pdat['vars']['temp'].update({'x_range': [0.0, 5.0]})
        # pdat['vars']['salt'].update({'x_range': [0.0, 2.5]})
        # pltprof.plot(pdat, outfile, title, fontsize = 14)

        varnames = ['T', 'S']
        xax_labels = ['Temperature ($^\circ$C)', 'PSU']
        colors = ['b', 'r']
        x_ranges = [[args.vmin_temp, args.vmax_temp], [args.vmin_salt, args.vmax_salt]]
        source_labels = [' ']
        linestyles = ['solid']
        values = [[meantsf[icov, 0:Cv_nz]],
                  [meantsf[icov, Cv_nz:]]]
        depthsin = [Cv_depth[0: Cv_nz]]

        title = 'Mean at latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        fontsize = 14
        outfile = plot_dir / 'TS_mean_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)

        pltprof.plot_prof(varnames, xax_labels, colors,
                      source_labels, linestyles, values, depthsin, outfile,
                      x_ranges = x_ranges, title = title, fontsize = fontsize)

        varnames = ['T', 'S']
        xax_labels = ['Temperature ($^\circ$C)', 'PSU', 'kg m$^-3$']
        colors = ['b', 'r']
        x_ranges = [[0.0, args.vmax_temp_stndev], [0.0, args.vmax_salt_stndev]]
        source_labels = [' ']
        linestyles = ['solid']
        values = [[stdtsf[icov, 0:Cv_nz]],
                  [stdtsf[icov, Cv_nz:]]]
        depthsin = [Cv_depth[0: Cv_nz]]

        title = 'Standev at latlon {:5.2f}, {:6.2f}'.format(atlat, atlon)
        fontsize = 14
        outfile = plot_dir / 'TS_std_{:5.2f}_{:6.2f}.png'.format(atlat, atlon)

        pltprof.plot_prof(varnames, xax_labels, colors,
                      source_labels, linestyles, values, depthsin, outfile,
                      x_ranges = x_ranges, title = title, fontsize = fontsize)

#%%
def transect_mean_stndev_ts(args, plot_dir, means, stndvs, Cv_nz, Cv_nv, Cv_lat, Cv_lon, depth_edges_use, depth_range):
    # means and stndvs should be arranged [position,depth] with T&S appended along depth
    if args.no_plotting:
        return

    temp = means[:, 0: Cv_nz].T
    salt = means[:, Cv_nz: Cv_nv].T
    depth_edges = depth_edges_use
    depth_range = [args.beg_adepth, args.end_adepth]
    title_temp = 'Temperature'
    title_salt = 'Salinity'
    outfile = plot_dir / 'climo_mean_transect.png'
    
    if args.An_type == 'lat_transect':
        palong = Cv_lon
        pat = Cv_lat[0]
        xlabel = 'Longitude ($^\circ$E)'
        super_title1 = 'Climatological T&S mean along {}$^\circ$E'.format(pat)
        super_title2 = 'Climatological T&S stndev along {}$^\circ$E'.format(pat)
    else:
        palong = Cv_lat
        pat = Cv_lon[0]
        xlabel = 'Latitude ($^\circ$N)'
        super_title1 = 'Climatological T&S mean along {}$^\circ$N'.format(pat)
        super_title2 = 'Climatological T&S stndev along {}$^\circ$N'.format(pat)
    
    lonslice(args, temp, salt, palong, depth_edges, depth_range,
             title_temp, title_salt,
             super_title1, outfile,
             vmin_temp = args.vmin_temp, vmax_temp = args.vmax_temp,
             vmin_salt = args.vmin_salt, vmax_salt = args.vmax_salt,
             xlabel = xlabel, ylabel = 'Depth (m)')
    
    temp = stndvs[:, 0: Cv_nz].T
    salt = stndvs[:, Cv_nz: Cv_nv].T
    depth_edges = depth_edges_use
    depth_range = [args.beg_adepth, args.end_adepth]
    title_temp = 'Temperature'
    title_salt = 'Salinity'
    outfile = plot_dir / 'climo_stndev_transect.png'

    lonslice(args, temp, salt, palong, depth_edges, depth_range,
             title_temp, title_salt,
             super_title2, outfile,
             vmin_temp = args.vmin_temp_stndev, vmax_temp = args.vmax_temp_stndev,
             vmin_salt = args.vmin_salt_stndev, vmax_salt = args.vmax_salt_stndev,
             xlabel = xlabel, ylabel = 'Depth (m)')

#%%
def innovation_increment_stht(args, plot_dir, innovation, H, dx, ob_list, prefix):
    if args.no_plotting:
        return

    # line plot of innovation and increment

    if args.An_type == 'lat_transect':
        posns = np.array([p.lon for p in ob_list])
        x_label = 'Longitude ($^\circ$E)'
    else:
        posns = np.array([p.lat for p in ob_list])
        x_label = 'Latitude ($^\circ$N)'
    sorted_indices = np.argsort(posns)

    data = np.array([innovation, H @ dx])
    labels = ['innovation', 'increment']
    title = 'Innovation and Increment ({})'.format(prefix)
    outfile = plot_dir / 'innovation_increment_{}.png'.format(prefix)
    y_label = 'Steric height (m)'
    foflon(args, posns[sorted_indices], data[:,sorted_indices], labels, title, outfile,
                   x_label = x_label, y_label = y_label)
























