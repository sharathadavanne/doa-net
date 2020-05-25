import torch
import torch.nn as nn
from train_hnet import HNet, HungarianDataset
from IPython import embed
import matplotlib.pyplot as plot

use_pos_enc = True
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

model = HNet(use_pos_enc=use_pos_enc).to(device)
model.eval()
model.load_state_dict(torch.load("data/best_model_pos_enc.pt" if use_pos_enc else "data/best_model.pt" ))

test_data = HungarianDataset(train=False, use_pos_enc=use_pos_enc)

for feat, labels in test_data:
    feat = torch.tensor(feat).unsqueeze(0).to(device).float()
    pred = model(feat).squeeze().sigmoid().clone().detach().numpy()
    plot.plot(labels.reshape(-1), label='ref')
    plot.plot(pred, label='predicted')
    plot.legend()
    plot.ylim([0, 1])
    plot.show()




