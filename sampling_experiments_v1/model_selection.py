from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA


def generate_training_dataset(data, p=2, size=100):
    np.random.seed(180)
    dataset = []
    for i in range(len(data)):
        percent = (abs(data[i]) * p) / 100
        dist = np.random.normal(data[i], percent, size=size)
        dataset.append(dist)
    dataset = np.asarray(dataset).T

    return dataset


def find_optimal_number_of_clusters(data, smin=1, smax=100):
    Sum_of_squared_distances = []
    K = range(smin, smax)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def _preprocess(x_train, x_data, method='scale'):
    data = np.concatenate([x_train, x_data])

    if method == 'scale':

        # scaler = MaxAbsScaler()   #MinMaxScaler()StandardScaler()
        scaler = MinMaxScaler()
        scaler.fit_transform(data)
        data_transformed = scaler.transform(data)
        # data = np.concatenate([training_data, data_transformed])
    # if method == 'normalise':

    # data_transformed = normalize(data_transformed, norm='l2')
    else:
        data_transformed = data
    return data_transformed


def reduce_dimensions_kpca(data, method='PCA'):
    if method == 'PCA':
        pca = KernelPCA()
        reduced = pca.fit_transform(data)

    if method == 'KPCA':
        kpca = KernelPCA(n_components=2,
                         kernel='rbf',
                         fit_inverse_transform=True)

        # t_reduced = kpca.fit_transform(x_train)
        reduced = kpca.fit_transform(data)

        # X_back = kpca.inverse_transform(X_kpca)
    return reduced


def find_model_clusters(data, k=None):
    km = KMeans(n_clusters=k)
    km = km.fit(data)

    return km.labels_


def detect_outliers(x_train, x_data,
                    p=10, size=10000,
                    preprocessing=False, reduce=False,
                    reduction_method='KPCA'):

    ax_train = generate_training_dataset(x_train, p=p, size=size,
                                         )
    data = np.concatenate([ax_train, x_data])

    # data preprocessing with scaling or normalising
    if preprocessing:

        if preprocessing == 'scale':
            data = _preprocess(ax_train, x_data, method='scale')
            # data = np.concatenate([training_data, p_data])

        if preprocessing == 'normalise':
            data = _preprocess(data, method='normalise')

    # data reduction
    if reduce:
        data = reduce_dimensions_kpca(data,
                                      method=reduction_method)

    # lof = svm.OneClassSVM(nu=0.095, kernel='rbf', gamma=2)#0.095
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True,
                             algorithm='auto', contamination='auto')
    length = len(ax_train)
    train = lof.fit(data[:length])
    # test = lof.fit(reduced_tra)
    out_in = lof.predict(data[length:])

    return out_in, data, lof


def as_bool(array):
    array = array.copy()
    array[array == -1] = 0
    x = array.astype(bool)
    return x
