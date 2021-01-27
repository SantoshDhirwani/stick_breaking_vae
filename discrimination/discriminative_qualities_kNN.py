from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def main(models_dict):
    # models_dict: (dict) of zip(model_names, features_dict)
    # features_dict: (dict) of zip(xy_sets: [train_data, train_labels, test_data, test_labels])
    # train_labels, test_labels: (array-like) ground truth labels all samples in partition
    # test_data, train_data: (array-like) latent space features for all samples in partition

    model_names = ['SB-VAE', 'Gauss VAE', 'Raw Pixels']
    xy_sets = ['train_data', 'train_labels', 'test_data', 'test_labels']
    k = [3, 5, 10]

    def fit_kNN_classifier(n_neighbors, features_dict):
        train_y = np.squeeze(features_dict[xy_sets[1]])
        n_samples = train_y.shape[0]
        train_x = np.reshape(features_dict[xy_sets[0]], (n_samples, -1))

        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifier.fit(train_x, train_y)

        return classifier

    def score_kNN_classifier(classifier, features_dict):
        test_y = np.squeeze(features_dict[xy_sets[3]])
        n_samples = test_y.shape[0]
        test_x = np.reshape(features_dict[xy_sets[2]], (n_samples, -1))

        score = classifier.score(test_x, test_y)

        return score

    def get_kNN_test_error(model, n_neighbors):
        features_dict = models_dict[model]
        classifier = fit_kNN_classifier(n_neighbors=n_neighbors, features_dict=features_dict)
        score = score_kNN_classifier(classifier, features_dict=features_dict)
        error = 100 - score

        return error

    def creat_error_table(error):
        rows = error.models.values
        columns = ['k={}'.format(k) for k in error.k.values]
        data = error.values.reshape(len(rows), len(columns))

        fig, axs = plt.subplots()
        the_table = axs.table(cellText=data,
                              rowLabels=rows,
                              colLabels=columns,
                              loc='center')

        axs.axis('off')
        plt.savefig('test_error.png', bbox_inches='tight')

    test_error = xr.DataArray(np.full((len(model_names), len(k)), np.nan),
                              coords=dict(models=model_names, k=k),
                              dims=['model_classes', 'k'])

    for model in model_names:
        if model in models_dict.keys():
            for n_neighbors in k:
                test_error.loc[dict(models=model, k=n_neighbors)] = get_kNN_test_error(model, n_neighbors)

    creat_error_table(test_error)
    test_error.to_netcdf('test_error.nc')


if __name__ == '__main__':
    # running test
    import cPickle as cp

    data = cp.load(open('datasets/svhn_pca.pkl', 'rb'))
    for key, val in data.items():
        data[key] = val[:100]  # pruning for memory inexpensive test
    models_dict = {'Gauss VAE': data}

    main(models_dict=models_dict)
