import numpy as np
import pandas as pd
from LoopStructural.modelling.features.fold import fourier_series


def create_dict(x=None, y=None, z=None, strike=None, dip=None,
                feature_name=None, coord=None, ):
    """
    builds a dictionary that conforms with loopstructural format
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param strike: strike of foliation
    :param dip: dip of folitation
    :param feature_name: name of foliation e.g., s1
    :param coord: fold frame coordinate e.g., 0, 1 or 2
    :return:
    """
    fn = np.empty(len(x)).astype(str)
    fn.fill(feature_name)
    c = np.empty((len(x))).astype(int)
    c.fill(coord)
    dictionary = {'X': x,
                  'Y': y,
                  'Z': z,
                  'strike': strike,
                  'dip': dip,
                  'feature_name': fn,
                  'coord': c}
    return dictionary


def create_fold_frame_dataset(model, strike=0, dip=0):
    s1_ori = np.array([strike, dip])
    xyz = model.regular_grid(nsteps=[10, 10, 10]) * model.scale_factor
    s1_orientation = np.tile(s1_ori, (len(xyz), 1))
    s1_dict = create_dict(x=xyz[:, 0][0:10:2],
                          y=xyz[:, 1][0:10:2],
                          z=xyz[:, 2][0:10:2],
                          strike=s1_orientation[:, 0][0:10:2],
                          dip=s1_orientation[:, 1][0:10:2],
                          feature_name='s1',
                          coord=0)
    # Generate a dataset using s1 dictionary
    dataset = pd.DataFrame(s1_dict, columns=['X', 'Y', 'Z', 'strike', 'dip', 'feature_name', 'coord'])
    # Add y coordinate axis orientation. Y coordinate axis always perpendicular
    # to the axial surface and roughly parallel to the fold axis
    s2y = dataset.copy()
    s2s = s2y[['strike', 'dip']].to_numpy()
    s2s[:, 0] += 90
    s2s[:, 1] = dip
    s2y['strike'] = s2s[:, 0]
    s2y['dip'] = s2s[:, 1]
    s2y['coord'] = 1
    # Add y coordinate dictionary to s1 dataframe
    dataset = dataset.append(s2y)

    return dataset, xyz


def create_gradient_dict(x=None, y=None, z=None, nx=None, ny=None, nz=None, feature_name=None, coord=None,
                         data_type=None,
                         **kwargs):
    """

    :rtype: object
    """

    fn = np.empty(len(x)).astype(str)
    fn.fill(feature_name)
    c = np.empty((len(x))).astype(int)
    c.fill(coord)
    dictionary = {'X': x,
                  'Y': y,
                  'Z': z,
                  'gx': nx,
                  'gy': ny,
                  'gz': nz,
                  'feature_name': fn,
                  'coord': c}
    return dictionary


def sample_random_datasets(grid, samples=1000, sample_size=2, seed=180):
    np.random.seed(seed)
    # select all xy points with z = 0
    xyz = grid[grid[:, 2] == 0]
    # genereate an array xyz indices
    indices = np.arange(len(xyz))
    # create array to store datasets
    comb_indices = np.zeros((samples, sample_size), dtype=int)
    iseed = np.linspace(1, 2 ** 16, samples, dtype='u8')
    seeds = np.random.randint(iseed)
    # select randomly n datasets that contain N data points
    for i in range(samples):
        np.random.seed(seeds[i])
        comb_indices[i] = np.random.choice(indices, sample_size, replace=False)

    return xyz[comb_indices], comb_indices


def calculate_splot(ref_fold_frame, popt):
    return np.rad2deg(np.arctan(fourier_series(ref_fold_frame, *popt)))
