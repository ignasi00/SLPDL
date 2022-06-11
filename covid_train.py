# If this was a little serious, this code would have looked better

from datetime import datetime
import numpy as np
import random
from types import SimpleNamespace
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from frameworks.pytorch.loggers.wandb_logger import WandbLogger
from SLPDL_utils.build_hubert import build_hubert
from SLPDL_utils.build_vgg import  build_vgg
from SLPDL_utils.covid_dataset import build_covid_dataset, build_covid_collate


###############################################################################################

model_type = ['VGG', 'HUBERT'][0]

data_params = SimpleNamespace(
    train_scp_path = './data_lists/wavs16k/train/wav.scp',        # train data folder
    train_labels_path = './data_lists/wavs16k/train/text',
    valid_scp_path = './data_lists/wavs16k/valid/wav.scp',        # valid data folder
    valid_labels_path = './data_lists/wavs16k/valid/text',
    test_scp_path = './data_lists/wavs16k/test/wav.scp',          # test data folder
    wavs_path = './data/wavs16k/',
    checkpoint = './model_parameters/covid/',                            # checkpoints directory
)

train_params = SimpleNamespace(
    batch_size = 22,                             # training and valid batch size
    epochs = 50,                                 # maximum number of epochs to train
    lr = 0.0002,                                 # learning rate
    momentum = 0.9,                              # SGD momentum, for SGD only
    optimizer = 'adam',                          # optimization method: sgd | adam
    log_interval = 5,                            # how many batches to wait before logging training status
    patience = 5,                                # how many epochs of no loss improvement should we wait before stop training
    cuda = True,                                 # use gpu
    num_workers = 2,                             # how many subprocesses to use for data loading
    seed = 1234                                  # random seed
)

param_loader_vgg_params = SimpleNamespace(
    window_size = .04,                           # window size for the stft
    window_stride = .02,                         # window stride for the stft
    window_type = 'hamming',                     # window type for the stft
    normalize = True,                            # use spect normalization
    max_len = 1000
)

param_loader_hubert_params = SimpleNamespace(
    MODEL = "facebook/hubert-base-ls960",
    max_seconds = 10
)

model_vgg_params = SimpleNamespace(
    arc = 'VGG13',                               # VGG11, VGG13, VGG16, VGG19
    hidden=64,
    dropout=0.4,
    batch_norm=True
)

model_hubert_params = SimpleNamespace(
    adapter_hidden_size = 64
)

###############################################################################################


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(dataloader, model, criterion, optimizer, epoch, cuda, log_interval, weight=None, max_norm=None, verbose=True):
    model.train()

    global_epoch_loss = 0
    samples = 0

    for batch_idx, (_, data, target) in enumerate(dataloader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target.float())
        loss.backward()

        if max_norm is not None : nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        global_epoch_loss += loss.data.item() * len(target)
        samples += len(target)

        if verbose:
            if batch_idx % log_interval == 0:
                progress = 100 * samples / len(dataloader.dataset)
                loss = global_epoch_loss / samples
                print(f'Train Epoch: {epoch} [{samples}/{len(dataloader.dataset)} ({progress:.0f}%)]\tLoss: {loss:.6f}')

    return global_epoch_loss / samples

def valid(loader, model, criterion, cuda, verbose=True, data_set='Validation'):
    model.eval()

    test_loss = 0

    tpred = []
    ttarget = []

    with torch.no_grad():
        for keys, data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            pred = output.sigmoid()

            tpred.append(pred.cpu().numpy())

            if target[0] != -1:
                loss = criterion(output, target.float()).data.item()
                test_loss += loss * len(target) # sum up batch loss 
                ttarget.append(target.cpu().numpy())
    
    if len(ttarget) > 0:
        test_loss /= len(dataloader.dataset)
        auc, auc_ci = roc_auc_score_ci(np.concatenate(ttarget), np.concatenate(tpred))
        if verbose:
            print(f'\n{data_set} set: Average loss: {test_loss:.4f}, AUC: {100 * auc:.1f}% ({auc_ci[0] * 100:.1f}% - {auc_ci[1] * 100:.1f}%)\n')

        return test_loss, auc


def main(model_type):
    seed_everything(train_params.seed)

    train_params.cuda = train_params.cuda and torch.cuda.is_available()
    if train_params.cuda : print(f'Using CUDA with {torch.cuda.device_count()} GPUs')

    if model_type == 'VGG':
        param_loader_params = param_loader_vgg_params
        model_params = model_vgg_params
        model = build_vgg(model_vgg_params.arc, hidden=model_vgg_params.hidden, dropout=model_vgg_params.dropout, batch_norm=model_vgg_params.batch_norm)
        mel_feats = True
    elif model_type == 'HUBERT':
        param_loader_params = param_loader_hubert_params
        model_params = model_hubert_params
        model = build_hubert(adapter_hidden_size=model_hubert_params.adapter_hidden_size)
        model.freeze_feature_encoder()
        for param in model.hubert.encoder.parameters():
            param.requires_grad = False
        mel_feats = False
    else:
        raise NotImplementedError(f'The model_type "{model_type}" was not implemented yet.')

    if train_params.cuda:
        model.cuda()
    
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    train_dataset = build_covid_dataset(data_params.train_scp_path, param_loader_params, mel_feats=mel_feats, basepath=data_params.wavs_path, labels_filepath=data_params.train_labels_path)
    valid_dataset = build_covid_dataset(data_params.valid_scp_path, param_loader_params, mel_feats=mel_feats, basepath=data_params.wavs_path, labels_filepath=data_params.valid_labels_path)

    collate_fn = build_covid_collate()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_params.batch_size, shuffle=True, num_workers=train_params.num_workers, pin_memory=train_params.cuda, sampler=None, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=train_params.batch_size, shuffle=False, num_workers=train_params.num_workers, pin_memory=train_params.cuda, sampler=None, collate_fn=collate_fn)

    if train_params.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=train_params.lr)
    elif train_params.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=train_params.lr, momentum=train_params.momentum)
    else:
        raise NotImplementedError(f'The optimizer "{train_params.optimizer}" was not implemented yet')
    
    best_valid_auc = 0
    iteration = 0
    epoch = 1
    best_epoch = epoch

    experiment_name = datetime.now().strftime("%Y%m%d%H%M%S")
    wandb_logger = WandbLogger("SLPDL_COVID", experiment_name, "slpdl2022")
     
    wandb_logger.summarize(vars(train_params))
    wandb_logger.summarize(vars(param_loader_params))
    wandb_logger.summarize(vars(model_params))
    wandb_logger.summarize({'model_type' : model_type})

    t0 = time.time()
    while (epoch < train_params.epochs + 1) and (iteration < train_params.patience):

        train_loss = train(train_loader, model, criterion, optimizer, epoch, train_params.cuda, train_params.log_interval)
        valid_loss, valid_auc = valid(valid_loader, model, criterion, train_params.cuda, data_set='Validation')
        
        wandb_logger.log({'epoch' : epoch, 'train_loss' : train_loss, 'valid_loss' : valid_loss, 'valid_auc' : valid_auc})

        if not os.path.isdir(args.checkpoint):
            os.mkdir(args.checkpoint)

        torch.save(model.state_dict(), f'./{data_params.checkpoint}/model{epoch:03d}.pt')

        if valid_auc <= best_valid_auc:
            iteration += 1
            print(f'AUC was not improved, iteration {iteration}')

        else:
            print('Saving state')

            iteration = 0
            best_valid_auc = valid_auc
            best_epoch = epoch

            state = {
                'train_loss' : train_loss,
                'valid_auc': valid_auc,
                'valid_loss': valid_loss,
                'epoch': epoch,
            }

            if not os.path.isdir(args.checkpoint):
                os.mkdir(args.checkpoint)

            torch.save(state, f'./{data_params.checkpoint}/ckpt.pt')

        epoch += 1
        print(f'Elapsed seconds: ({time.time() - t0:.0f}s)')

    print(f'Best AUC: {best_valid_auc*100:.1f}% on epoch {best_epoch}')
    wandb_logger.summarize(state)
    return model, wandb_logger


if __name__ == '__main__':
    model, wandb_logger = main(model_type)

    if True:
        from covid_test import test

        params = SimpleNamespace(
            test_scp_path = './data_lists/wavs16k/test/wav.scp',        # train data folder
            wavs_path = './data/wavs16k/',
            test_batch_size = 10,
            cuda = True,
            num_workers = 2
        )

        test_dataset = build_covid_dataset(params.test_scp_path, param_loader_params, mel_feats=mel_feats, basepath=params.wavs_path)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda, sampler=None)

        # get best epoch and model
        state = torch.load(f'./{data_params.checkpoint}/ckpt.pt')
        epoch = state['epoch']
        print(f"Testing model (epoch {epoch})")

        model.load_state_dict(torch.load(f'./{data_params.checkpoint}/model{epoch:03d}.pt'))
        if args.cuda:
            model.cuda()

        results = './outputs/covid/submission.csv'
        print(f"Saving results in {results}")
        test(test_loader, model, params.cuda, save=results)

        wandb_logger.upload_submission(results, wait=True)

