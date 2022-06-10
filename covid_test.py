# If this was a little serious, this code would have looked better

from types import SimpleNamespace
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from SLPDL_utils.build_hubert import build_hubert
from SLPDL_utils.build_vgg import  build_vgg
from SLPDL_utils.covid_dataset import build_covid_dataset, build_covid_collate


###############################################################################################

model_type = ['VGG', 'HUBERT'][0]

params = SimpleNamespace(
    test_scp_path = './data_lists/wavs16k/test/wav.scp',        # train data folder
    wavs_path = './data/wavs16k/',
    checkpoint = './model_parameters/covid/',                            # checkpoints directory
    test_batch_size = 10,
    cuda = True,
    seed = 1234,
    num_workers = 2
)

param_loader_vgg_params = SimpleNamespace(
    window_size = .04,                           # window size for the stft
    window_stride = .02,                         # window stride for the stft
    window_type = 'hamming',                     # window type for the stft
    normalize = True                             # use spect normalization
)

param_loader_hubert_params = SimpleNamespace(
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


def test(loader, model, cuda, save=None):
    model.eval()

    if save is not None:
        csv = open(save, 'wt')
        print('index,prob', file=csv)

    with torch.no_grad():
        for keys, data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            output = output.squeeze(-1)
            pred = output.sigmoid()

            if save is not None:
                for i, key in enumerate(keys):
                    print(f'{key},{pred[i]}', file=csv)
    
def main():
    seed_everything(params.seed)

    params.cuda = params.cuda and torch.cuda.is_available()
    if params.cuda : print(f'Using CUDA with {torch.cuda.device_count()} GPUs')

    if model_type == 'VGG':
        param_loader_params = param_loader_vgg_params
        model = build_vgg(model_vgg_params.arc, hidden=model_vgg_params.hidden, dropout=model_vgg_params.dropout, batch_norm=model_vgg_params.batch_norm)
        mel_feats = True
    elif model_type == 'HUBERT':
        param_loader_params = param_loader_hubert_params
        model = build_hubert(adapter_hidden_size=model_hubert_params.adapter_hidden_size)
        model.freeze_feature_encoder()
        for param in model.hubert.encoder.parameters():
            param.requires_grad = False
        mel_feats = False
    else:
        raise NotImplementedError(f'The model_type "{model_type}" was not implemented yet.')

    if params.cuda:
        model.cuda()

    test_dataset = build_covid_dataset(params.test_scp_path, param_loader_params, mel_feats=mel_feats, basepath=params.wavs_path)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda, sampler=None)

    # get best epoch and model
    state = torch.load(f'./{params.checkpoint}/ckpt.pt')
    epoch = state['epoch']
    print(f"Testing model (epoch {epoch})")

    model.load_state_dict(torch.load(f'./{params.checkpoint}/model{epoch:03d}.pt'))
    if args.cuda:
        model.cuda()

    results = './outputs/covid/submission.csv'
    print(f"Saving results in {results}")
    test(test_loader, model, params.cuda, save=results)

