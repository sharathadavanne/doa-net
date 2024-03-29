#
# A wrapper script that trains the DOAnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import doanet_model
import doanet_parameters
import time
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
sys.path.insert(0,'/users/sadavann/hungarian-net')
# sys.path.insert(0,'/home/sharath/PycharmProjects/hungarian-net')
from train_hnet import HNetGRU
from cls_metric import doa_metric

eps = 1e-7


def plot_functions(fig_name, _tr_loss, _val_loss, _tr_hung_loss, _val_hung_loss):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(211)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(nb_epoch), _val_loss, label='test loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(nb_epoch), _tr_hung_loss, label='train hung loss')
    plot.plot(range(nb_epoch), _val_hung_loss, label='test hung loss')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()


def test_epoch(data_generator, model, hnet_model, activity_loss, criterion, metric_cls, params, device):
    nb_train_batches, train_loss, train_dMOTP_loss, train_dMOTA_loss, train_act_loss = 0, 0., 0., 0., 0.
    model.eval()
    with torch.no_grad():
        for data, target in data_generator.generate():
            # load one batch of data
            target_activity = target[:, :, -params['unique_classes']:].reshape(-1, params['unique_classes'])
            nb_framewise_doas_gt = target_activity.sum(-1)
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-params['unique_classes']]).to(device).float()

            # process the batch of data based on chosen mode
            activity_binary = None
            if params['use_hnet']:
                if params['use_dmot_only']:
                    output = model(data)
                else:
                    output, activity_out = model(data)
                    activity_out = activity_out.view(-1, activity_out.shape[-1])
                    activity_binary = (torch.sigmoid(activity_out).cpu().detach().numpy() > 0.5)
            else:
                output = model(data)

            # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
            max_nb_doas = output.shape[2]//3
            output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
            target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

            # Compute unit-vectors of predicted DoA
            # (batch, sequence, 3, max_nb_doas) to (batch*sequence, 3, max_nb_doas)
            output, target = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1])
            output_norm = torch.sqrt(torch.sum(output**2, -1) + 1e-10)
            output = output/output_norm.unsqueeze(-1)

            if params['use_hnet']:
                # get pair-wise distance matrix between predicted and reference.
                dist_mat = torch.cdist(output.contiguous(), target.contiguous())
                da_mat, _, _ = hnet_model(dist_mat)
                da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)
                da_mat = da_mat.view(dist_mat.shape)
                da_mat = (da_mat>0.5)*da_mat
                da_activity = da_mat.max(-1)[0]

                # Compute dMOTP loss for true positives
                dMOTP_loss = (torch.mul(dist_mat, da_mat).sum(-1).sum(-1) * da_mat.sum(-1).sum(-1)*params['dMOTP_wt']).sum()/da_mat.sum()
                # dMOTP_loss = torch.mul(dist_mat, da_mat).sum()/ da_mat.sum()

                # Compute dMOTA loss
                M = da_activity.sum(-1)
                N = torch.Tensor(nb_framewise_doas_gt).to(device)
                FP = torch.clamp(M-N, min=0)
                FN = torch.clamp(N-M, min=0)
                IDS = (da_mat[1:]*(1-da_mat[:-1])).sum(-1).sum(-1)
                IDS = torch.cat((torch.Tensor([0]).to(device), IDS))
                dMOTA_loss = ((FP + FN + params['IDS_wt']* IDS).sum() / (M+ torch.finfo(torch.float32).eps).sum())

                train_dMOTP_loss += dMOTP_loss.item()
                train_dMOTA_loss += dMOTA_loss.item()
                loss = dMOTP_loss+params['dMOTA_wt']*dMOTA_loss
                if not params['use_dmot_only']:
                    act_loss = activity_loss(activity_out, (da_activity>0.5).float())
                    loss = params['branch_weights'][0] * loss + params['branch_weights'][1] * act_loss
                    train_act_loss += act_loss.item()
            else:
                loss = criterion(output, target)

            # compute the angular distance matrix to estimate the localization error
            dist_mat_hung = torch.matmul(output.detach(), target.transpose(-1, -2))
            dist_mat_hung = torch.clamp(dist_mat_hung, -1+eps, 1-eps) # the +- eps is critical because the acos computation will become saturated if we have values of -1 and 1
            dist_mat_hung = torch.acos(dist_mat_hung)  # (batch, sequence, max_nb_doas, max_nb_doas)
            dist_mat_hung = dist_mat_hung.cpu().detach().numpy()
            
            metric_cls.partial_compute_metric(dist_mat_hung, target_activity, pred_activity=activity_binary)

            train_loss += loss.item()
            nb_train_batches += 1
            if params['quick_test'] and nb_train_batches == 4:
                break

        train_loss /= nb_train_batches
        if params['use_hnet']:
            train_act_loss /= nb_train_batches
            train_dMOTP_loss /= nb_train_batches
            train_dMOTA_loss /= nb_train_batches

    return metric_cls, train_loss, train_dMOTP_loss, train_dMOTA_loss, train_act_loss


def train_epoch(data_generator, optimizer, model, hnet_model, activity_loss, criterion, params, device):
    nb_train_batches, train_loss, train_dMOTP_loss, train_dMOTA_loss, train_act_loss = 0, 0., 0., 0., 0.
    model.train()
    for data, target in data_generator.generate():

        # load one batch of data
        target_activity = target[:, :, -params['unique_classes']:].reshape(-1, params['unique_classes'])
        nb_framewise_doas_gt = target_activity.sum(-1)
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-params['unique_classes']]).to(device).float()

        # process the batch of data based on chosen mode
        optimizer.zero_grad()
        if params['use_hnet']:
            if params['use_dmot_only']:
                output = model(data)
            else:
                output, activity_out = model(data)
                activity_out = activity_out.view(-1, activity_out.shape[-1])
#                activity_binary = (torch.sigmoid(activity_out).cpu().detach().numpy() > 0.5)
        else:
            output = model(data)

        # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
        max_nb_doas = output.shape[2]//3
        output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
        target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

        # Compute unit-vectors of predicted DoA
        # (batch, sequence, 3, max_nb_doas) to (batch*sequence, 3, max_nb_doas)
        output, target = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1])
        output_norm = torch.sqrt(torch.sum(output**2, -1) + 1e-10)
        output = output/output_norm.unsqueeze(-1)

        if params['use_hnet']:
            # get pair-wise distance matrix between predicted and reference.
            dist_mat = torch.cdist(output.contiguous(), target.contiguous())
            da_mat, _, _ = hnet_model(dist_mat)
            da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)
            da_mat = da_mat.view(dist_mat.shape)
            da_mat = (da_mat>0.5)*da_mat
            da_activity = da_mat.max(-1)[0]

            # Compute dMOTP loss for true positives
            dMOTP_loss = (torch.mul(dist_mat, da_mat).sum(-1).sum(-1) * da_mat.sum(-1).sum(-1)*params['dMOTP_wt']).sum()/da_mat.sum()
            # dMOTP_loss = torch.mul(dist_mat, da_mat).sum()/ da_mat.sum()
            
            # Compute dMOTA loss
            M = da_activity.sum(-1)
            N = torch.Tensor(nb_framewise_doas_gt).to(device)
            FP = torch.clamp(M-N, min=0)
            FN = torch.clamp(N-M, min=0)
            IDS = (da_mat[1:]*(1-da_mat[:-1])).sum(-1).sum(-1)
            IDS = torch.cat((torch.Tensor([0]).to(device), IDS))
            dMOTA_loss = ((FP + FN + params['IDS_wt']* IDS).sum() / (M+ torch.finfo(torch.float32).eps).sum())
            
            train_dMOTP_loss += dMOTP_loss.item()
            train_dMOTA_loss += dMOTA_loss.item()
            loss = dMOTP_loss+params['dMOTA_wt']*dMOTA_loss
            if not params['use_dmot_only']:
                act_loss = activity_loss(activity_out, (da_activity>0.5).float())
                loss = params['branch_weights'][0] * loss + params['branch_weights'][1] * act_loss
                train_act_loss += act_loss.item()
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    train_loss /= nb_train_batches
    if params['use_hnet']:
        train_act_loss /= nb_train_batches
        train_dMOTP_loss /= nb_train_batches
        train_dMOTA_loss /= nb_train_batches

    return train_loss, train_dMOTP_loss, train_dMOTA_loss, train_act_loss


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = doanet_parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # load Hungarian network for data association, and freeze all layers.
    hnet_model = HNetGRU(max_len=2).to(device)
    hnet_model.load_state_dict(torch.load("models/hnet_model.pt", map_location=torch.device('cpu')))
    for model_params in hnet_model.parameters():
        model_params.requires_grad = False
    print('---------------- Hungarian-net -------------------')
    print(hnet_model)

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        test_splits = [1]
        val_splits = [2]
        train_splits = [[3, 4, 5, 6]]

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_{}_split{}'.format(
            task_id, job_id, params['dataset'], params['mode'], split
        )
        unique_name = os.path.join(params['model_dir'], unique_name)
        model_name = '{}_model.h5'.format(unique_name)
        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False
        )

        # Collect i/o data size and load model configuration
        data_in, data_out = data_gen_train.get_data_sizes()
        model = doanet_model.CRNN(data_in, data_out, params).to(device)
#        model.load_state_dict(torch.load("models/23_5624972_mic_dev_split1_model.h5", map_location='cpu'))

        print('---------------- DOA-net -------------------')
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
            params['fnn_size']))
        print(model)

        # start training
        best_val_epoch = -1
        best_doa, best_mota, best_ids, best_recall, best_precision, best_fscore = 180, 0, 1000, 0, 0, 0
        patience_cnt = 0

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        tr_loss_list = np.zeros(nb_epoch)
        val_loss_list = np.zeros(nb_epoch)
        hung_tr_loss_list = np.zeros(nb_epoch)
        hung_val_loss_list = np.zeros(nb_epoch)

        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = torch.nn.MSELoss()
        activity_loss = nn.BCEWithLogitsLoss()

        for epoch_cnt in range(nb_epoch):
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            train_loss, train_dMOTP_loss, train_dMOTA_loss, train_act_loss = train_epoch(data_gen_train, optimizer, model, hnet_model, activity_loss, criterion, params, device)
            train_time = time.time() - start_time
            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------
            start_time = time.time()
            val_metric = doa_metric()
            val_metric, val_loss, val_dMOTP_loss, val_dMOTA_loss, val_act_loss = test_epoch(data_gen_val, model, hnet_model, activity_loss, criterion, val_metric, params, device)

            val_hung_loss, val_mota, val_ids, val_recall_doa, val_precision_doa, val_fscore_doa = val_metric.get_results()
            val_time = time.time() - start_time

            # Save model if loss is good
            if val_hung_loss <= best_doa:
                best_val_epoch, best_doa, best_mota, best_ids, best_recall, best_precision, best_fscore = epoch_cnt, val_hung_loss, val_mota, val_ids, val_recall_doa, val_precision_doa, val_fscore_doa
                torch.save(model.state_dict(), model_name)

            # Print stats and plot scores
            print(
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                'train_loss: {:0.2f} {}, val_loss: {:0.2f} {}, '
                'LE/MOTA/IDS/LR/LP/LF: {:0.3f}/{}, '
                'best_val_epoch: {} {}'.format(
                    epoch_cnt, train_time, val_time,
                    train_loss, '({:0.2f},{:0.2f},{:0.2f})'.format(train_dMOTP_loss, train_dMOTA_loss, train_act_loss) if params['use_hnet'] else '',
                    val_loss, '({:0.2f},{:0.2f},{:0.2f})'.format(val_dMOTP_loss, val_dMOTA_loss, val_act_loss) if params['use_hnet'] else '',
                    val_hung_loss, '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_mota, val_ids, val_recall_doa, val_precision_doa, val_fscore_doa),
                    best_val_epoch, '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_doa, best_mota, best_ids, best_recall, best_precision, best_fscore))
            )

            tr_loss_list[epoch_cnt], val_loss_list[epoch_cnt], hung_val_loss_list[epoch_cnt] = train_loss, val_loss, val_hung_loss
            plot_functions(unique_name, tr_loss_list, val_loss_list, hung_tr_loss_list, hung_val_loss_list)

            patience_cnt += 1
            if patience_cnt > params['patience']:
                break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        print('Load best model weights')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        print('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False
        )

        test_metric = doa_metric()
        test_metric, test_loss, test_dMOTP_loss, test_dMOTA_loss, test_act_loss = test_epoch(data_gen_test, model, hnet_model, activity_loss, criterion, test_metric, params, device)

        test_hung_loss, test_mota, test_ids, test_recall_doa, test_precision_doa, test_fscore_doa = test_metric.get_results()

        print(
            'test_loss: {:0.2f} {}, LE/MOTA/IDS/LR/LP/LF: {:0.3f}/{}'.format(
                test_loss, '({:0.2f},{:0.2f},{:0.2f})'.format(test_dMOTP_loss, test_dMOTA_loss, test_act_loss) if params['use_hnet'] else '',
                test_hung_loss, '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(test_mota, test_ids, test_recall_doa, test_precision_doa, test_fscore_doa))
        )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

