import datetime
import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.util_vars import CUDA, learning_rate, print_interval, train_loader, test_loader, n_train_epochs, input_shape,\
    latent_ndims, parametrizations, lookahead, n_monte_carlo_samples
from model_classes.VAEs_pytorch import GaussianVAE, StickBreakingVAE


# init model and optimizer
time_now = datetime.datetime.now().__format__('%b_%d_%Y_%H_%M')
parametrization = parametrizations['Kumar']
model = GaussianVAE().cuda() if CUDA else GaussianVAE()
# model = StickBreakingVAE(parametrization).cuda() if CUDA else StickBreakingVAE(parametrization)
optimizer = optim.Adam(model.parameters(), betas=(0.95, 0.999), lr=learning_rate)
parametrization_str = parametrization if model._get_name() == "StickBreakingVAE" else ''
model_name = '_'.join(filter(None, [model._get_name(), parametrization_str]))
model_path = 'trained_models'
tb_writer = SummaryWriter(f'logs/{model_name}')

best_test_epoch = None
best_test_loss = None
stop_training = None


def train(epoch):
    print(f'\nTrain Epoch: {epoch}')
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.cuda() if CUDA else data
        optimizer.zero_grad()
        recon_batch, param1, param2 = model(data)
        mc_sample_idx = torch.randint(high=len(data), size=(n_monte_carlo_samples,))
        loss = model.ELBO_loss(recon_batch[mc_sample_idx], data[mc_sample_idx], param1, param2, model.kl_divergence)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % print_interval == 0:
            print(f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item() / len(data):.6f}')

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    tb_writer.add_scalar("Loss/train", loss.item() / len(data), epoch)


def test(epoch):
    global best_test_epoch, best_test_loss, stop_training
    model.eval()
    test_loss = 0

    for batch_idx, data in enumerate(test_loader):
        data = data.cuda() if CUDA else data
        recon_batch, param1, param2 = model(data)
        mc_sample = torch.randint(high=len(data), size=(1,))  # draw random monte carlo sample
        test_loss += model.ELBO_loss(recon_batch[mc_sample], data[mc_sample], param1, param2, model.kl_divergence).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    tb_writer.add_scalar("Loss/test", test_loss, epoch)

    if epoch == 1:
        best_test_epoch = epoch
        best_test_loss = test_loss
    else:
        best_test_epoch = epoch if test_loss < best_test_loss else best_test_epoch
        best_test_loss = test_loss if best_test_epoch else best_test_loss
        stop_training = True if epoch - best_test_epoch > lookahead else False


for epoch in range(1, n_train_epochs + 1):
    train(epoch)
    test(epoch)
    n_random_samples = 16

    if epoch == best_test_epoch:

        sample = torch.randn(n_random_samples, latent_ndims)  # is randn normalized to latent space?
        sample = sample.cuda() if CUDA else sample
        sample = model.decode(sample).cpu()

        # save latent space samples
        tb_writer.add_images(f'{n_random_samples}_samples_from_latent_space_{time_now}',
                             img_tensor=sample.view(n_random_samples, 1, *input_shape),
                             global_step=epoch,
                             dataformats='NCHW')

        random_idxs = torch.randint(0, int(test_loader.dataset.shape[0]), size=(n_random_samples,))
        samples = test_loader.dataset[random_idxs]

        # save originals
        tb_writer.add_images(f'{n_random_samples}_original_test_samples_{time_now}',
                             img_tensor=samples.view(n_random_samples, 1, *input_shape),
                             global_step=epoch,
                             dataformats='NCHW')

        samples = samples.cuda() if CUDA else samples
        samples = torch.stack([model(x)[0] for x in samples])

        # save reconstructed
        tb_writer.add_images(f'{n_random_samples}_reconstructed_test_samples_{time_now}',
                             img_tensor=samples.view(n_random_samples, 1, *input_shape),
                             global_step=epoch,
                             dataformats='NCHW')

        # save trained weights
        best_test_model = model.state_dict().copy()

    elif stop_training:
        break

tb_writer.close()
torch.save(best_test_model, os.path.join(model_path, f'{model_name}_epoch{epoch}_{time_now}'))


