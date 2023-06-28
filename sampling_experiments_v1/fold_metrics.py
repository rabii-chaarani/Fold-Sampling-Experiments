import numpy as np
from LoopStructural.utils.helper import create_box


class FoldMetrics:

    def __init__(self, geological_feature, bounding_box):
        self.geological_feature = geological_feature
        self.model = geological_feature.model
        self.bounding_box = bounding_box
        self.points, self.tri = create_box(self.bounding_box, np.array([40, 40, 40]))
        self.scaled_points = self.points * self.model.scale_factor
        self.geological_feature.fold.foldframe[0].set_model(self.model)
        self.geological_feature.fold.foldframe[1].set_model(self.model)
        self.z = self.geological_feature.fold.foldframe[0].evaluate_value(self.points)
        self.x = np.linspace(self.geological_feature.fold.foldframe[0].min(),
                             self.geological_feature.fold.foldframe[0].max(), 1000)
        self.fitted_limb_rotation = self.geological_feature.fold.fold_limb_rotation(self.x)
        self.amin = np.deg2rad(self.fitted_limb_rotation.min())
        self.amax = np.deg2rad(self.fitted_limb_rotation.max())
        self.value = self.geological_feature.evaluate_value(self.points)

    def fold_tightness(self):
        # Calculate tightness using method of Grose et al. (2019)
        tightness = 180 - np.rad2deg(2 * np.tan((np.arctan(self.amax) - np.arctan(self.amin)) / 2))

        return tightness

    def fold_wavelength(self):
        wv = self.geological_feature.fold.fold_limb_rotation.fitted_params[3]

        return wv

    def fold_asymmetry(self):
        # Calculate asymmetry index using method of Grose et al. (2019)
        amin = np.abs(self.fitted_limb_rotation.min())
        amax = np.abs(self.fitted_limb_rotation.max())
        median = np.median(self.fitted_limb_rotation)
        limb_rotation_angle_range = amax + amin
        # calculate the asymmetry
        # the value of the asymmetry index could be positive or negative
        # for S or Z shaped asymmetry folds, respectively
        asymmetry = median / limb_rotation_angle_range

        return asymmetry

    def fold_noncylindricity_index(self):
        wv = self.geological_feature.fold.fold_axis_rotation.fitted_params[3]

        return wv

    def hinge_angle(self):
        # Calculate hinge angle
        fold_frame_y = self.geological_feature.fold.foldframe[1]
        x = np.linspace(fold_frame_y.min(), fold_frame_y.max(), 100)
        fitted_axis_rotation = self.geological_feature.fold.fold_axis_rotation(x)
        amin = np.deg2rad(fitted_axis_rotation.min())
        amax = np.deg2rad(fitted_axis_rotation.max())
        hinge_angle = 180 - np.rad2deg(2 * np.tan((np.arctan(amax) - np.arctan(amin)) / 2))

        return hinge_angle

    def all_metrics(self):

        try:
            tightness = self.fold_tightness()
            asymmetry = self.fold_asymmetry()
            wl = self.fold_wavelength()
            wla = self.geological_feature.fold.fold_axis_rotation.fitted_params[3]
            hinge_angle = self.hinge_angle()
            metrics = [asymmetry, tightness, wl, hinge_angle, wla]
        except:

            tightness = self.fold_tightness()
            asymmetry = self.fold_asymmetry()
            wl = self.fold_wavelength()
            metrics = [tightness, asymmetry, wl]

        return np.array(metrics)
