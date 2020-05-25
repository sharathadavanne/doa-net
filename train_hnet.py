from generate_training_data import load_obj
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
            self.conv_Q.in_channels,
            self.conv_V.out_channels,
            self.conv_K.out_channels
            )


class HNet2D(nn.Module):
    def __init__(self, use_pos_enc=True):
        super().__init__()
        self.in_conv = nn.ReLU(nn.BatchNorm2d(nn.Conv2d(6 if use_pos_enc else 1, 32, kernel_size=(3, 3), padding=1)))
        self.attn = AttentionLayer(32, 32, 32)
        self.out_conv1 = nn.ReLU(nn.Conv1d(32, 32, kernel_size=5, padding=2))
        self.out_conv2 = nn.Conv1d(32, 1, kernel_size=5, padding=2)

    def forward(self, query):
        out = self.in_conv(query)
        out = out.view(query.shape[0], query.shape[1], -1)
        out = self.attn.forward(out)
        out = self.out_conv1(out)
        out = self.out_conv2(out)
        return out.squeeze(1)


class HNet(nn.Module):
    def __init__(self, use_pos_enc=True):
        super().__init__()
        hid_ch = 16
        self.out_conv1 = nn.Conv1d(6 if use_pos_enc else 1, hid_ch, kernel_size=5, padding=2)
        self.out_conv2 = nn.Conv1d(hid_ch, hid_ch, kernel_size=5, padding=2)
        self.out_conv3 = nn.Conv1d(hid_ch, hid_ch, kernel_size=5, padding=2)
        self.attn = AttentionLayer(hid_ch, hid_ch, hid_ch)
        self.out_conv4 = nn.Conv1d(hid_ch, hid_ch, kernel_size=5, padding=2)
        self.out_conv5 = nn.Conv1d(hid_ch, 1, kernel_size=5, padding=2)

    def forward(self, query):
        out = self.out_conv1(query)
        out = F.relu(out)
        out = self.out_conv2(out)
        out = F.relu(out)
        out = self.out_conv3(out)
        out = F.relu(out)
        out = self.out_conv4(out)
        out = F.relu(out)
        out = self.attn.forward(out)
        out = self.out_conv5(out)
        return out.squeeze(1)


class HungarianDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True, max_len=5, use_pos_enc=True):
        if train:
            self.data_dict = load_obj('data/hung_data_train')
        else:
            self.data_dict = load_obj('data/hung_data_test')
        self.max_len = max_len

        self.use_pos_enc = use_pos_enc
        if self.use_pos_enc:
            c = math.ceil(math.log(max_len**2) / math.log(2.0))
            self.pe = (torch.arange(max_len**2)[None] // 2**torch.arange(c)[:, None])%2
            self.pe = self.pe.float()

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        feat = -1*np.ones((self.max_len, self.max_len))
        label = np.zeros((self.max_len, self.max_len))
        nb_rows, nb_cols = self.data_dict[idx][2].shape
        feat[:nb_rows, :nb_cols] = self.data_dict[idx][2]
        label[:nb_rows, :nb_cols] = self.data_dict[idx][3]
        feat, label = feat.reshape(1, -1), label.reshape(-1)
        if self.use_pos_enc:
            feat = torch.cat((torch.tensor(feat).float(), self.pe), 0)
        return feat, label


def main():
    batch_size = 256
    nb_epochs = 100
    use_pos_enc = True

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        HungarianDataset(train=True, use_pos_enc=use_pos_enc),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        HungarianDataset(train=False, use_pos_enc=use_pos_enc),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = HNet(use_pos_enc=use_pos_enc).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss = 1000
    best_epoch = -1
    for epoch in range(1, nb_epochs + 1):

        # TRAINING
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            output = model(data)
            train_loss += criterion(output, target).item()  # sum up batch loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        #TESTING
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device).float(), target.to(device).float()
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
        test_loss /= len(test_loader.dataset)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "data/best_model_pos_enc.pt" if use_pos_enc else "data/best_model.pt" )
        print('Epoch: {}\ttrain_loss: {:.4f}\ttest_loss: {:.4f}\tbest_epoch: {}'.format(epoch, train_loss, test_loss, best_epoch))
    print('Best epoch: {}\nBest loss: {}'.format(best_epoch, best_loss))

if __name__ == "__main__":
    main()

