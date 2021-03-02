from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import torch
from model_classes.VAEs_pytorch import GaussianVAE, StickBreakingVAE
from utils.util_vars import CUDA, parametrizations, train_data, test_data, train_dataset, test_dataset, \
    train_valid_test_splits, input_ndims

model_names = ['StickBreakingVAE_Kumaraswamy', 'GaussianVAE', 'RawPixels']
xy_sets = ['train_data', 'train_labels', 'test_data', 'test_labels']
k = [3, 5, 10]
checkpoint_paths = ['trained_models\GaussianVAE/best_checkpoint_GaussianVAE_Feb_24_2021_07_53',
                    'trained_models\StickBreakingVAE_Kumaraswamy/best_checkpoint_StickBreakingVAE_Kumaraswamy_Feb_21_2021_07_46']


def fit_kNN_classifier(n_neighbors, features_dict):
    train_y = features_dict[xy_sets[1]].squeeze()
    n_samples = train_y.shape[0]
    train_x = features_dict[xy_sets[0]].reshape(n_samples, -1)

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(train_x, train_y)

    return classifier


def score_kNN_classifier(classifier, features_dict):
    test_y = features_dict[xy_sets[3]].squeeze()
    n_samples = test_y.shape[0]
    test_x = features_dict[xy_sets[2]].reshape(n_samples, -1)

    score = classifier.score(test_x, test_y)

    return score


def get_kNN_test_error(features_dict, n_neighbors):
    classifier = fit_kNN_classifier(n_neighbors=n_neighbors, features_dict=features_dict)
    score = score_kNN_classifier(classifier, features_dict=features_dict)
    error = 1 - score

    return error


def create_error_table(error):
    rows = error.models.values
    columns = ['k={}'.format(k) for k in error.k.values]
    data = error.values.reshape(len(rows), len(columns)).round(2)

    fig, axs = plt.subplots()
    the_table = axs.table(cellText=data,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center')
    axs.axis('off')
    plt.savefig('test_error.png', bbox_inches='tight')


def load_model(checkpoint_path):
    if 'GaussianVAE' in checkpoint_path:
        model = GaussianVAE().cuda() if CUDA else GaussianVAE()
    else:
        parametrization = [x for x in parametrizations.values() if x in checkpoint_path]
        model = StickBreakingVAE(*parametrization).cuda() if CUDA else StickBreakingVAE(*parametrization)

    model_state_dict = torch.load(checkpoint_path)['model_state_dict']
    model.load_state_dict(model_state_dict)

    return model


def get_models_dict(train_data, test_data):
    # create nested dict
    models_dict = dict(zip(model_names, [{} for x in model_names]))

    # get data and labels
    train_data = train_data.reshape(-1, 1, input_ndims)[:train_valid_test_splits[0]]
    test_data = test_data.reshape(-1, 1, input_ndims)
    train_labels = train_dataset.targets[:train_valid_test_splits[0]]
    test_labels = test_dataset.targets

    if CUDA:
        train_data = train_data.cuda()
        test_data = test_data.cuda()

    # get raw data features
    features_dict = dict(zip(xy_sets, [train_data, train_labels,
                                       test_data, test_labels]))
    models_dict['RawPixels'] = features_dict

    # get latent space data features
    for checkpoint_path in checkpoint_paths:
        model = load_model(checkpoint_path)
        model_name = [x for x in model_names if x in checkpoint_path][0]

        latent_train_data = model.reparametrize(*model.encode(train_data), parametrization=model.parametrization)
        latent_test_data = model.reparametrize(*model.encode(test_data), parametrization=model.parametrization)

        features_dict = dict(zip(xy_sets, [latent_train_data.detach().numpy(), train_labels.detach().numpy(),
                                           latent_test_data.detach().numpy(), test_labels.detach().numpy()]))
        models_dict[model_name] = features_dict

    return models_dict


def main():

    test_error = xr.load_dataarray('test_error.nc')
    test_error *= 100

    if test_error is None:
        test_error = xr.DataArray(np.full((len(model_names), len(k)), np.nan),
                                  coords=dict(models=model_names, k=k),
                                  dims=['models', 'k'])

        models_dict = get_models_dict(train_data, test_data)
        for model in model_names:
            if model in models_dict.keys():
                for n_neighbors in k:
                    print(f'\nFitting and scoring {n_neighbors}-neighbor kNN trained on {model}...')
                    test_error.loc[dict(models=model, k=n_neighbors)] = get_kNN_test_error(models_dict[model], n_neighbors)

        test_error.to_netcdf('test_error.nc', mode='a')

    create_error_table(test_error)


if __name__ == '__main__':
    main()
