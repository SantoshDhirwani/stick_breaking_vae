import datetime
import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from util_vars import CUDA, learning_rate, print_interval, train_loader, test_loader, n_train_epochs, input_shape,\
    latent_ndims
from VAEs_pytorch import GaussianVAE, StickBreakingVAE

# init model and optimizer
# model = GaussianVAE().cuda() if CUDA else GaussianVAE()
model = StickBreakingVAE().cuda() if CUDA else StickBreakingVAE()
optimizer = optim.Adam(model.parameters(), betas=(0.95, 0.999), lr=learning_rate)
model_name = model._get_name()
tb_writer = SummaryWriter(f'logs/{model_name}')


def train():
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.cuda() if CUDA else data
        optimizer.zero_grad()
        recon_batch, param1, param2 = model(data)
        loss = model.ELBO_loss(recon_batch, data, param1, param2)
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


def test():
    model.eval()
    test_loss = 0

    for batch_idx, data in enumerate(test_loader):
        data = data.cuda() if CUDA else data
        recon_batch, param1, param2 = model(data)
        test_loss += model.ELBO_loss(recon_batch, data, param1, param2).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    tb_writer.add_scalar("Loss/test", test_loss, epoch)


for epoch in range(1, n_train_epochs + 1):
    print(f'\nTrain Epoch: {epoch}')
    train()
    test()

    if epoch % 10 == 0:
        # generate random samples
        n_random_samples = 16
        sample = torch.randn(n_random_samples, latent_ndims)  # is randn normalized to latent space?
        sample = sample.cuda() if CUDA else sample
        sample = model.decode(sample).cpu()

        tb_writer.add_images(f'{n_random_samples}samples_from_latent_space',
                             img_tensor=sample.view(n_random_samples, 1, *input_shape),
                             global_step=epoch,
                             dataformats='NCHW')

    if epoch == n_train_epochs - 1:  # for final epoch
        n_random_samples = 16
        random_idxs = torch.randint(0, int(test_loader.dataset.shape[0]), size=(n_random_samples,))
        samples = test_loader.dataset[random_idxs]

        # save originals
        tb_writer.add_images(f'{n_random_samples}original_test_samples',
                             img_tensor=samples.view(n_random_samples, 1, *input_shape),
                             global_step=epoch,
                             dataformats='NCHW')

        samples = samples.cuda() if CUDA else samples
        samples = torch.stack(model(samples))

        # save reconstructed
        tb_writer.add_images(f'{n_random_samples}reconstructed_test_samples',
                             img_tensor=samples.view(n_random_samples, 1, *input_shape),
                             global_step=epoch,
                             dataformats='NCHW')

tb_writer.close()

# save trained weights
model_path = 'trained_models'
time_now = datetime.datetime.now().__format__('%b_%d_%Y_%H_%M')
torch.save(model.state_dict(), os.path.join(model_path, f'{model_name}_{time_now}'))
