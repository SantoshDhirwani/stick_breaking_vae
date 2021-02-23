import datetime
import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.util_vars import CUDA, learning_rate, print_interval, train_loader, test_loader, n_train_epochs, input_shape,\
    latent_ndims, parametrizations, lookahead, n_monte_carlo_samples, n_random_samples, model_path, checkpoint_path
from model_classes.VAEs_pytorch import GaussianVAE, StickBreakingVAE

# init model and optimizer
time_now = datetime.datetime.now().__format__('%b_%d_%Y_%H_%M')
parametrization = parametrizations['Kumar']
model = GaussianVAE().cuda() if CUDA else GaussianVAE()
# model = StickBreakingVAE(parametrization).cuda() if CUDA else StickBreakingVAE(parametrization)
optimizer = optim.Adam(model.parameters(), betas=(0.95, 0.999), lr=learning_rate, eps=1e-4)
parametrization_str = parametrization if model._get_name() == "StickBreakingVAE" else ''
model_name = '_'.join(filter(None, [model._get_name(), parametrization_str]))
start_epoch = 1

if checkpoint_path:  # load checkpoint state
    assert model_name in checkpoint_path, 'Mismatch between specified and checkpoint parametrization'
    checkpoint = torch.load(checkpoint_path)
    start_epoch, model_state_dict, optimizer_state_dict = list(checkpoint.values())
    optimizer.load_state_dict(optimizer_state_dict)
    model.load_state_dict(model_state_dict)

# init save directories
tb_writer = SummaryWriter(f'logs/{model_name}')
if not os.path.exists(os.path.join(model_path, model_name)):
    os.mkdir(os.path.join(model_path, model_name))

best_test_epoch = None
best_test_loss = None
best_test_model = None
best_test_optimizer = None
stop_training = None


def train(epoch):
    print(f'\nTrain Epoch: {epoch}')
    model.train()
    reconstruction_loss = 0
    regularization_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.cuda() if CUDA else data
        mc_sample_idx = torch.randint(high=len(data), size=(n_monte_carlo_samples,))
        mc_sample = data[mc_sample_idx]
        optimizer.zero_grad()
        recon_batch, param1, param2 = model(mc_sample)
        batch_reconstruction_loss, batch_regularization_loss = \
            model.ELBO_loss(recon_batch, mc_sample, param1, param2, model.kl_divergence)
        reconstruction_loss += batch_reconstruction_loss
        regularization_loss += batch_regularization_loss
        loss = batch_reconstruction_loss + batch_regularization_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # to prevent exploding gradient

        # check for NaN values in all .grad attributes
        for name, param in model.named_parameters():
            isfinite = torch.isfinite(param).all()
            if not isfinite:
                print(name, isfinite)
                torch.save({'epoch': best_test_epoch,
                            'model_state_dict': best_test_model,
                            'optimizer_state_dict': best_test_optimizer},
                           os.path.join(model_path, model_name, f'finite_loss_checkpoint_{model_name}_{time_now}'))
                break

        optimizer.step()

        if batch_idx % print_interval == 0:
            print(f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item() / len(data):.6f}')

    train_loss = reconstruction_loss + regularization_loss

    reconstruction_loss /= len(train_loader.dataset)
    regularization_loss /= len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

    tb_writer.add_scalar(f"{time_now}/Loss/train", train_loss.item(), epoch)
    tb_writer.add_scalar(f"{time_now}/Regularization_Loss/train", regularization_loss.item(), epoch)
    tb_writer.add_scalar(f"{time_now}/Reconstruction_Loss/train", reconstruction_loss.item(), epoch)


def test(epoch):
    global best_test_epoch, best_test_loss, stop_training
    model.eval()
    reconstruction_loss = 0
    regularization_loss = 0

    for batch_idx, data in enumerate(test_loader):
        data = data.cuda() if CUDA else data
        mc_sample_idx = torch.randint(high=len(data), size=(n_monte_carlo_samples,))  # draw random monte carlo sample
        mc_sample = data[mc_sample_idx]
        recon_batch, param1, param2 = model(mc_sample)
        batch_reconstruction_loss, batch_regularization_loss = \
            model.ELBO_loss(recon_batch, mc_sample, param1, param2, model.kl_divergence)

        reconstruction_loss += batch_reconstruction_loss
        regularization_loss += batch_regularization_loss

    test_loss = reconstruction_loss + regularization_loss

    reconstruction_loss /= len(test_loader.dataset)
    regularization_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    tb_writer.add_scalar(f"{time_now}/Loss/test", test_loss.item(), epoch)
    tb_writer.add_scalar(f"{time_now}/Regularization_Loss/test", regularization_loss.item(), epoch)
    tb_writer.add_scalar(f"{time_now}/Reconstruction_Loss/test", reconstruction_loss.item(), epoch)

    if epoch == start_epoch:
        best_test_epoch = epoch
        best_test_loss = test_loss
    else:
        best_test_epoch = epoch if test_loss < best_test_loss else best_test_epoch
        best_test_loss = test_loss if best_test_epoch else best_test_loss
        stop_training = True if epoch - best_test_epoch > lookahead else False


for epoch in range(start_epoch, n_train_epochs + 1):
    try:
        train(epoch)
        test(epoch)
    except AssertionError:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(model_path, model_name, f'failed_checkpoint_{model_name}_{time_now}'))
        print('Stick segments do not sum to one/reconstructed_x.log() is not finite. Stopping training.')
        break

    if epoch == best_test_epoch:

        sample = torch.randn(n_random_samples, latent_ndims)  # TODO: normalize random sample to latent space bounds
        sample = sample.cuda() if CUDA else sample
        sample = model.decode(sample).cpu()

        # save latent space samples
        tb_writer.add_images(f'{n_random_samples}_random_latent_space_samples_{time_now}',
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
        best_test_optimizer = optimizer.state_dict().copy()

    elif stop_training:
        break

tb_writer.close()
torch.save({'epoch': best_test_epoch,
            'model_state_dict': best_test_model,
            'optimizer_state_dict': best_test_optimizer},
           os.path.join(model_path, model_name, f'best_checkpoint_{model_name}_{time_now}'))


