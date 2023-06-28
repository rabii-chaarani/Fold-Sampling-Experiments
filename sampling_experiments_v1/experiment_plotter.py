import matplotlib.colors as mcolors
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from LoopStructural.visualisation import MapView
import seaborn as sns


class ExpPlotter:
    def __init__(self, wspace=0.25, hspace=0.25, col=1,
                 row=2, figsize=(20, 6.5)):
        """

        """
        self.fig, self.ax = plt.subplots(col, row,
                                         figsize=figsize,  # (11.69, 8.27)
                                         dpi=300)
        self.fig.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.rcParams['font.size'] = 12
        plt.rcParams['lines.markersize'] = 30

        # self.ax[1][0].set_ylim(-90, 90)
        # self.feature = feature

    def plot_multiple_arrays(self, x, y, ix=0, iy=0, **kwargs):
        # np.random.seed(180)
        # c = np.random.rand(len(y), 3)

        for i, yy in enumerate(y):
            self.ax[ix].plot(x, yy, **kwargs)

    def plot_single_array(self, x, y, ix=0, iy=0, **kwargs):

        self.ax[ix].plot(x, y, **kwargs)

    def plot_multiple_arrays_xy(self, x, y, symbol='r-', ix=0, iy=0, **kwargs):
        # np.random.seed(180)
        # c = np.random.rand(len(y), 3)

        for xx, yy in zip(x, y):
            self.ax[ix].scatter(xx, yy, **kwargs)

    def plot_svariograms(self, x, y, wv, symbol='r-', ix=0, iy=0, **kwargs):
        for xx, yy in zip(x, y):
            # wl = wv/2
            # mask = xx == wl
            # wly = yy[mask]
            self.ax[ix].plot(xx[np.nanargmax(yy)], yy[np.nanargmax(yy)], symbol, **kwargs)
            # self.ax[ix].plot(xx, yy, symbol, **kwargs)

    def plot_wavelengths(self, wv, symbol='r-', ix=0, iy=0, **kwargs):
        names = np.array(['Wavelength guess', 'Fitted wavelength'])
        names = np.tile(names, (len(wv), 1))
        for xx, yy in zip(names, wv):
            # wl = wv/2
            # mask = xx == wl
            # wly = yy[mask]
            # self.ax[ix].plot(xx[np.nanargmax(yy)], yy[np.nanargmax(yy)], symbol, **kwargs)
            self.ax[ix].scatter(xx, yy, **kwargs)

    def plot_dataset_map(self, data, ref_feature, grid, ix=0, scale=20,
                         xmin=-20, xmax=1020, ymin=-20, ymax=1020,
                         inter=40, axial_inter=40, **kwargs):

        np.random.seed(180)

        for i in range(len(data)):
            # c = np.random.randint(low=1, high=1000, size=len(data))
            # c = np.random.randint(low=1, high=255, size=(len(data), 3))

            c = np.random.rand(len(data), 3)

            gradient_data = data[i]
            # gradient_data = np.hstack(ori_data)
            gradient_data[:, :3] = ref_feature[1].model.rescale(gradient_data[:, :3], inplace=False)
            gradient_data[:, 3:5] /= np.linalg.norm(gradient_data[:, 3:5], axis=1)[:, None]
            t = gradient_data[:, [4, 3]] * np.array([1, -1]).T
            n = gradient_data[:, 3:5]
            t *= scale
            n *= 0.5 * scale
            p1 = gradient_data[:, [0, 1]] - t
            p2 = gradient_data[:, [0, 1]] + t
            # plt.scatter(val[:,0],val[:,1],c='black')
            self.ax[0].plot([p1[:, 0], p2[:, 0]], [p1[:, 1], p2[:, 1]],
                            c=mcolors.to_rgb(c[i]), zorder=1)
            p1 = gradient_data[:, [0, 1]]
            p2 = gradient_data[:, [0, 1]] + n
            self.ax[0].plot([p1[:, 0], p2[:, 0]], [p1[:, 1], p2[:, 1]],
                            c=mcolors.to_rgb(c[i]), zorder=1)

        ref_val = ref_feature[1].evaluate_value(grid[grid[:, 2] == 0])
        ref_s1_val = ref_feature[0][0].evaluate_value(grid[grid[:, 2] == 0])
        mapview = MapView(model=ref_feature[1].model, ax=self.ax[0])
        mapview.xmax = xmax
        mapview.ymax = ymax
        mapview.xmin = xmin
        mapview.ymin = ymin
        # mapview._update_grid()
        mapview.add_contour(ref_feature[0][0], np.sort(ref_s1_val)[::axial_inter], colors=['black'],
                            linestyles='dashdot', alpha=0.25, linewidths=2)

        mapview = MapView(model=ref_feature[1].model, ax=self.ax[0])
        mapview.xmax = xmax
        mapview.ymax = ymax
        mapview.xmin = xmin
        mapview.ymin = ymin
        # mapview._update_grid()
        # mapview.add_scalar_field(ref_s0, cmap='tab20b')
        mapview.add_contour(ref_feature[1], np.sort(ref_val)[::inter], colors=['black'], alpha=0.25, linestyles='solid')

    def plot_sampling_patterns_and_density_maps(self, xyz, ref_feature, grid, ix1=0, ix2=1, ix3=2,
                                                xmin=-20, xmax=1020, ymin=-20, ymax=1020, cbar_frac=.05,
                                                cbar_shrink=.9, cbar_pad=0.05, kde_tresh=0,
                                                kde_levels=20, inter=40, axial_inter=40, **kwargs):

        # for i, line in enumerate(xyz):
        #     distances, neighb_indices = find_neighbours(line, len(line))
        #     linex = line[neighb_indices][0]
        #     self.ax[ix1].plot(linex[:, 0], linex[:, 1], linewidth=3, alpha=1)
        #     self.ax[ix1].scatter(line[:, 0], line[:, 1], zorder=len(xyz)+1, alpha=1)
        # ref_val = ref_feature.evaluate_value(grid[grid[:, 2] == 0])
        ref_val = ref_feature[1].evaluate_value(grid[grid[:, 2] == 0])
        ref_s1_val = ref_feature[0][0].evaluate_value(grid[grid[:, 2] == 0])
        # ref_s2_val = ref_feature[0][0].evaluate_value(grid[grid[:, 2] == 0])
        xys = np.vstack(xyz)
        # bounds = np.arange(len(recovery[ind]))
        kde = sns.kdeplot(x=xys[:, 0], y=xys[:, 1], shade=True, ax=self.ax,
                          thresh=kde_tresh, levels=kde_levels,
                          cbar=True, **kwargs)  # cbar_kws=dict(shrink=cbar_shrink, fraction=cbar_frac,
        # pad=cbar_pad, label='Kernel density')cbar_ax=self.ax[ix2]
        # self.ax[ix2].
        mapview = MapView(model=ref_feature[1].model, ax=self.ax)
        mapview.xmax = xmax
        mapview.ymax = ymax
        mapview.xmin = xmin
        mapview.ymin = ymin
        # mapview._update_grid()
        mapview.add_contour(ref_s0, np.sort(ref_val)[::inter], colors=['black'], linestyles='solid')

        # mapview = MapView(model=ref_feature[1].model, ax=self.ax[ix2])
        # mapview.xmax = xmax
        # mapview.ymax = ymax
        # mapview.xmin = xmin
        # mapview.ymin = ymin
        # # mapview._update_grid()
        # mapview.add_contour(ref_s0, np.sort(ref_val)[::inter], colors=['black'], linestyles='solid')

        # ref_val = ref_feature[2].evaluate_value(grid[grid[:, 2] == 0])
        # ref_s1_val = ref_feature[1][0].evaluate_value(grid[grid[:, 2] == 0])
        # ref_s2_val = ref_feature[0][0].evaluate_value(grid[grid[:, 2] == 0])
        mapview = MapView(model=ref_feature[1].model, ax=self.ax)
        mapview.xmax = xmax
        mapview.ymax = ymax
        mapview.xmin = xmin
        mapview.ymin = ymin
        # mapview._update_grid()
        mapview.add_contour(ref_feature[0][0], np.sort(ref_s1_val)[::axial_inter], colors=['black'],
                            linestyles='dashdot', linewidths=2)

        # mapview = MapView(model=ref_feature[1].model, ax=self.ax[ix2])
        # mapview.xmax = xmax
        # mapview.ymax = ymax
        # mapview.xmin = xmin
        # mapview.ymin = ymin
        # # mapview._update_grid()
        # mapview.add_contour(ref_feature[0][0], np.sort(ref_s1_val)[::axial_inter], colors=['black'],
        #                      linestyles='dashdot', linewidths=2)

    def plot(self, x, y, ix=0, symbol='r-', **kwargs):
        """

        Parameters
        ----------
        x : np.array
            vector of x
        y
        ix
        iy
        symb

        Returns
        -------

        """
        return self.ax[ix].plot(x, y, symbol, **kwargs)

    def splot_kernel_density(self, x, y, b=0.4, tresh=0.1, ix=0, **kwargs):

        x1 = np.hstack(x)
        y1 = np.hstack(y)
        # bounds = np.arange(len(recovery[ind]))
        bw = (len(x) * (2 + 2) / 4.) ** (-1. / (2 + 4))
        kde = sns.kdeplot(x=x1, y=y1, shade=True,
                          thresh=tresh, levels=100,
                          cbar=True, bw_adjust=bw * b,
                          **kwargs)

    def fold_axis_titles(self):
        self.ax[0].set_title('A. Fold Axis S-Plot')
        self.ax[1].set_title('B. Fold Axis S-Variogram')
        # self.ax[2].set_title('C. 2D histogram of observations')

        self.ax[1].set_xlabel('Variogram Steps')
        # self.ax[1].set_ylabel('Fold Axis S-Variogram')
        self.ax[1].set_ylabel('Fitted and guessed wavelength')
        self.ax[0].set_ylabel('Fold Axis Rotation Angle')
        self.ax[0].set_xlabel('Fold Frame Axis Direction Field')
        # self.ax[2].set_ylabel('Fold axis rotation angle (°)')
        # self.ax[2].set_xlabel('Fold Frame Axis Direction Field (m)')

    def fold_limb_titles(self):
        self.ax[1].set_title('B. Fold Limb S-Plot')
        self.ax[2].set_title('C. Wavelengths comparison')
        self.ax[0].set_title('A. Datasets')

        # self.ax[1].set_xlabel('Variogram Steps')
        # self.ax[1].set_ylabel('Fold Limb S-Variogram')
        self.ax[2].set_ylabel('Fitted and guessed wavelengths (m)')
        self.ax[0].set_ylabel('Y')
        self.ax[0].set_xlabel('X')
        self.ax[1].set_ylabel('Fold limb rotation angle (°)')
        self.ax[1].set_xlabel('Fold Frame Axial Surface Field (m)')

    def sampling_paths_density_titles(self):
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        # self.ax[0].set_xlabel('X (m)')
        # self.ax[0].set_ylabel('Y (m)')

        # self.ax[1].set_xlabel('X (m)')
        # self.ax[1].set_ylabel('Y (m)')

        # self.ax[0].set_title('A. Sampling locations')
        # self.ax[1].set_title('B. Sampling locations density')
        self.ax.set_title('B. Sampling locations density')

    def plot_limits(self):
        self.ax[1].set_ylim(-90, 90)
        # self.ax[0].set_xlim(-500, 500)

        # self.ax[2].set_ylim(-90, 90)
        # self.ax[1].set_xlim(-550, 550)
        # self.ax[1].set_xlim(-500, 500)
