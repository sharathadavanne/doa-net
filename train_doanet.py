#
# A wrapper script that trains the DOAnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import evaluation_metrics
import doanet_model
import doanet_parameters
import time
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
sys.path.insert(0,'/users/sadavann/hungarian-net')
#sys.path.insert(0,'/home/sharath/PycharmProjects/hungarian-net')
from train_hnet import HNetGRU
from scipy.optimize import linear_sum_assignment


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

    # load Hungarian network for data association.
    hnet_model = HNetGRU(max_len=2).to(device)
    hnet_model.eval()
    hnet_model.load_state_dict(torch.load("models/hnet_model.pt" ))
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

        # Collect the reference labels for validation data
        data_in, data_out = data_gen_train.get_data_sizes()
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))

        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
            params['fnn_size']))

        model = doanet_model.CRNN(data_in, data_out, params).to(device)
#        model.load_state_dict(torch.load("models/1_1_foa_dev_split1_model.h5", map_location='cpu'))
        print('---------------- DOA-net -------------------')
        print(model)
        best_val_loss = 99999
        best_val_epoch = -1
        best_hung_loss = 99999
        best_hung_epoch = -1
        patience_cnt = 0
        eps = 1e-7
        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        tr_loss_list = np.zeros(nb_epoch)
        val_loss_list = np.zeros(nb_epoch)
        hung_tr_loss_list = np.zeros(nb_epoch)
        hung_val_loss_list = np.zeros(nb_epoch)

        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = torch.nn.MSELoss()
        activity_loss = nn.BCEWithLogitsLoss()
        # start training
        for epoch_cnt in range(nb_epoch):
            start = time.time()

            # TRAINING
            model.train()
            train_loss, train_dMOTP_loss, train_dMOTA_loss, train_act_loss, train_hung_loss, nb_train_batches, train_tp_doa, train_total_gt_doa, train_total_pred_doa, train_recall_doa, train_precision_doa = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            for data, target in data_gen_train.generate():
                optimizer.zero_grad()
                nb_framewise_doas_gt = target[:, :, -1].reshape(-1)
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()
                output, activity_out = model(data)

                # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
                max_nb_doas = output.shape[2]//3
                output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
                target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

                # get pair-wise distance matrix between predicted and reference.
                output, target, activity_out = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1]), activity_out.view(-1, activity_out.shape[-1])
                output_norm = torch.sqrt(torch.sum(output**2, -1) + 1e-10)
                output = output/output_norm.unsqueeze(-1)
                dist_mat = torch.cdist(output.contiguous(), target.contiguous())

                # hungarian loss
                dist_mat_hung = torch.matmul(output.detach(), target.transpose(-1, -2))
                dist_mat_hung = torch.clamp(dist_mat_hung, -1+eps, 1-eps) # the +- eps is critical because the acos computation will become saturated if we have values of -1 and 1
                dist_mat_hung = torch.acos(dist_mat_hung)  # (batch, sequence, max_nb_doas, max_nb_doas)

                if params['use_hnet']:
                    with torch.no_grad():
                        hidden = torch.zeros(1, dist_mat.shape[0], 128).to(device)
                        da_mat, _, _ = hnet_model(dist_mat.transpose(1, 2), hidden)
                        da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)
                        da_mat = da_mat.view(dist_mat.shape)
                        da_mat = (da_mat>0.5).float().detach()

                    # Compute dMOTP loss for true posiitives
                    dMOTP_loss = (torch.mul(dist_mat, da_mat).sum(-1).sum(-1) * da_mat.sum(-1).sum(-1)*params['dMOTP_wt']).sum()/da_mat.sum()
#                    dMOTP_loss = torch.mul(dist_mat, da_mat).sum()/ da_mat.sum()

                    # Compute dMOTA loss
                    M = da_mat.max(-1)[0].sum(-1)
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
                        act_loss = activity_loss(activity_out, da_mat.max(-1)[0])
                        loss = params['branch_weights'][0] * loss + params['branch_weights'][1] * act_loss
                        train_act_loss += act_loss.item()
                else:
                    loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                da_mat_numpy = da_mat.cpu().detach().numpy().astype(int)
                dist_mat_numpy = dist_mat_hung.cpu().detach().numpy()
                train_hung_loss += np.multiply(dist_mat_numpy, da_mat_numpy).sum()
                
                train_loss += loss.item()
                train_tp_doa += np.min((nb_framewise_doas_gt, da_mat_numpy.sum(-1).sum(-1)), 0).sum()
                train_total_pred_doa += da_mat_numpy.sum()
                train_total_gt_doa += nb_framewise_doas_gt.sum()
                nb_train_batches += 1
                if params['quick_test'] and nb_train_batches == 4:
                    break

            train_hung_loss /= train_total_pred_doa
            train_loss /= nb_train_batches
            train_act_loss /= nb_train_batches
            train_dMOTP_loss /= nb_train_batches
            train_dMOTA_loss /= nb_train_batches
            train_recall_doa = train_tp_doa / (float(train_total_gt_doa) + eps)
            train_precision_doa = train_tp_doa / (float(train_total_pred_doa) + eps)


            ## Validation
            model.eval()
            val_loss, val_act_loss, val_dMOTP_loss, val_dMOTA_loss, val_hung_loss, nb_val_batches, val_tp_doa, val_total_gt_doa, val_total_pred_doa, val_recall_doa, val_precision_doa = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            with torch.no_grad():
                for data, target in data_gen_val.generate():
                    nb_framewise_doas_gt = target[:, :, -1].reshape(-1)
                    data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()
                    output, activity_out = model(data)

                    # (batch, sequence, max_nb_doas*3) to (batch, sequence, max_nb_doas, 3)
                    max_nb_doas = output.shape[2]//3
                    output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
                    target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

                    # get pair-wise distance matrix between predicted and reference.
                    output, target, activity_out = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1]), activity_out.view(-1, activity_out.shape[-1])
                    output_norm = torch.sqrt(torch.sum(output**2, -1) + 1e-10)
                    output = output/output_norm.unsqueeze(-1)
                    dist_mat = torch.cdist(output.contiguous(), target.contiguous())

                    # hungarian loss
                    dist_mat_hung = torch.matmul(output.detach(), target.transpose(-1, -2))
                    dist_mat_hung = torch.clamp(dist_mat_hung, -1+eps, 1-eps) # the +- eps is critical because the acos computation will become saturated if we have values of -1 and 1
                    dist_mat_hung = torch.acos(dist_mat_hung)  # (batch, sequence, max_nb_doas, max_nb_doas)

                    if params['use_hnet']:
                        with torch.no_grad():
                            hidden = torch.zeros(1, dist_mat.shape[0], 128).to(device)
                            da_mat, _, _ = hnet_model(dist_mat.transpose(1, 2), hidden)
                            da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)
                            da_mat = da_mat.view(dist_mat.shape)
                            da_mat = (da_mat>0.5).float().detach()

                            # Compute dMOTP loss for true positives
                            dMOTP_loss = torch.mul(dist_mat, da_mat).sum()/ da_mat.sum()

                            M = da_mat.max(-1)[0].sum(-1)
                            N = torch.Tensor(nb_framewise_doas_gt).to(device)
                            FP = torch.clamp(M-N, min=0)
                            FN = torch.clamp(N-M, min=0)
                            IDS = (da_mat[1:]*(1-da_mat[:-1])).sum(-1).sum(-1)
                            IDS = torch.cat((torch.Tensor([0]).to(device), IDS))
                            dMOTA_loss = ((FP + FN + params['IDS_wt']* IDS).sum() / (M+ torch.finfo(torch.float32).eps).sum())

                            val_dMOTP_loss += dMOTP_loss.item()
                            val_dMOTA_loss += dMOTA_loss.item()

                            loss = dMOTP_loss+params['dMOTA_wt']*dMOTA_loss
                            if not params['use_dmot_only']:
                                act_loss = activity_loss(activity_out, da_mat.max(-1)[0])
                                loss = params['branch_weights'][0] * loss + params['branch_weights'][1] * act_loss
                                val_act_loss += act_loss.item()
                    else:
                        loss = criterion(output, target)

                    da_mat_numpy = da_mat.cpu().detach().numpy().astype(int)
                    dist_mat_numpy = dist_mat_hung.cpu().detach().numpy()

                    val_hung_loss += np.multiply(dist_mat_numpy, da_mat_numpy).sum()
                    val_loss += loss.item()
                    val_tp_doa += np.min((nb_framewise_doas_gt, da_mat_numpy.sum(-1).sum(-1)), 0).sum()
                    val_total_gt_doa += nb_framewise_doas_gt.sum()
                    val_total_pred_doa += da_mat_numpy.sum()
                    nb_val_batches += 1
                    if params['quick_test'] and nb_val_batches == 2:
                        break

            val_hung_loss /= val_total_pred_doa
            val_loss /= nb_val_batches
            val_recall_doa = val_tp_doa / (float(val_total_gt_doa) + eps)
            val_precision_doa = val_tp_doa / (float(val_total_pred_doa) + eps)
            val_dMOTP_loss /= nb_val_batches
            val_dMOTA_loss /= nb_val_batches
            val_act_loss /= nb_val_batches

            if val_hung_loss < best_val_loss:
                best_val_loss = val_hung_loss
                best_val_epoch = epoch_cnt
                torch.save(model.state_dict(), model_name)

            print(
                'epoch: {}, time: {:0.2f}, '
                'train_loss: {:0.2f} {}, val_loss: {:0.2f} {}, '
                'train_hung_loss: {:0.2f}/{:0.2f}/{:0.3f}, val_hung_loss_deg: {:0.2f}/{:0.2f}/{:0.3f}, '
                'best_val_epoch: {}'.format(
                    epoch_cnt, time.time()-start,
                    train_loss,
                    '' if params['use_hnet'] and params['use_dmot_only'] else '({:0.2f},{:0.2f},{:0.2f})'.format(train_dMOTP_loss, train_dMOTA_loss, train_act_loss),
                    val_loss,
                    '' if params['use_hnet'] and params['use_dmot_only'] else '({:0.2f},{:0.2f},{:0.2f})'.format(val_dMOTP_loss, val_dMOTA_loss, val_act_loss),
                    train_precision_doa*100.0, train_recall_doa*100.0, 180*train_hung_loss/np.pi, val_precision_doa*100.0, val_recall_doa*100.0, 180*val_hung_loss/np.pi,
                    best_val_epoch)
            )

            tr_loss_list[epoch_cnt], val_loss_list[epoch_cnt], hung_tr_loss_list[epoch_cnt], hung_val_loss_list[epoch_cnt] = train_loss, val_loss, train_hung_loss, val_hung_loss
            plot_functions(unique_name, tr_loss_list, val_loss_list, hung_tr_loss_list, hung_val_loss_list)

            patience_cnt += 1
            if patience_cnt > params['patience']:
                break


        # Evaluate on unseen test data
        print('Load best model weights')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

       
        print('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False
        )

        model.eval()
        test_loss, test_act_loss, test_dMOTP_loss, test_dMOTA_loss, test_hung_loss, nb_test_batches, test_tp_doa, test_total_gt_doa, test_total_pred_doa, test_recall_doa, test_precision_doa = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        with torch.no_grad():
            for data, target in data_gen_test.generate():
                nb_framewise_doas_gt = target[:, :, -1].reshape(-1)
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()
                output, activity_out = model(data)

                # (batch, sequence, max_nb_doas*3) to (batch, sequence, max_nb_doas, 3)
                max_nb_doas = output.shape[2]//3
                output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
                target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

                # get pair-wise distance matrix between predicted and reference.
                output, target, activity_out = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1]), activity_out.view(-1, activity_out.shape[-1])
                output_norm = torch.sqrt(torch.sum(output**2, -1) + 1e-10)
                output = output/output_norm.unsqueeze(-1)
                dist_mat = torch.cdist(output.contiguous(), target.contiguous())

                # hungarian loss
                dist_mat_hung = torch.matmul(output.detach(), target.transpose(-1, -2))
                dist_mat_hung = torch.clamp(dist_mat_hung, -1+eps, 1-eps) # the +- eps is critical because the acos computation will become saturated if we have values of -1 and 1
                dist_mat_hung = torch.acos(dist_mat_hung)  # (batch, sequence, max_nb_doas, max_nb_doas)

                if params['use_hnet']:
                    with torch.no_grad():
                        hidden = torch.zeros(1, dist_mat.shape[0], 128).to(device)
                        da_mat, _, _ = hnet_model(dist_mat.transpose(1, 2), hidden)
                        da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)
                        da_mat = da_mat.view(dist_mat.shape)
                        da_mat = (da_mat>0.5).float().detach()

                        # Compute dMOTP loss for true positives
                        dMOTP_loss = torch.mul(dist_mat, da_mat).sum()/ da_mat.sum()

                        M = da_mat.max(-1)[0].sum(-1)
                        N = torch.Tensor(nb_framewise_doas_gt).to(device)
                        FP = torch.clamp(M-N, min=0)
                        FN = torch.clamp(N-M, min=0)
                        IDS = (da_mat[1:]*(1-da_mat[:-1])).sum(-1).sum(-1)
                        IDS = torch.cat((torch.Tensor([0]).to(device), IDS))
                        dMOTA_loss = ((FP + FN + params['IDS_wt']* IDS).sum() / (M+ torch.finfo(torch.float32).eps).sum())

                        test_dMOTP_loss += dMOTP_loss.item()
                        test_dMOTA_loss += dMOTA_loss.item()

                        loss = dMOTP_loss+params['dMOTA_wt']*dMOTA_loss
                        if not params['use_dmot_only']:
                            act_loss = activity_loss(activity_out, da_mat.max(-1)[0])
                            loss = params['branch_weights'][0] * loss + params['branch_weights'][1] * act_loss
                            test_act_loss += act_loss.item()
                else:
                    loss = criterion(output, target)

                da_mat_numpy = da_mat.cpu().detach().numpy().astype(int)
                dist_mat_numpy = dist_mat_hung.cpu().detach().numpy()

                test_hung_loss += np.multiply(dist_mat_numpy, da_mat_numpy).sum()
                test_loss += loss.item()
                test_tp_doa += np.min((nb_framewise_doas_gt, da_mat_numpy.sum(-1).sum(-1)), 0).sum()
                test_total_gt_doa += nb_framewise_doas_gt.sum()
                test_total_pred_doa += da_mat_numpy.sum()
                nb_test_batches += 1

        test_hung_loss /= test_total_pred_doa
        test_loss /= nb_test_batches
        test_recall_doa = test_tp_doa / (float(test_total_gt_doa) + eps)
        test_precision_doa = test_tp_doa / (float(test_total_pred_doa) + eps)
        test_dMOTP_loss /= nb_test_batches
        test_dMOTA_loss /= nb_test_batches
        test_act_loss /= nb_test_batches
        print(
            'test_loss: {:0.2f} {}, test_hung_loss_deg: {:0.2f}/{:0.2f}/{:0.3f}'.format(
                test_loss,
                '' if params['use_hnet'] and params['use_dmot_only'] else '({:0.2f},{:0.2f},{:0.2f})'.format(test_dMOTP_loss, test_dMOTA_loss, test_act_loss),
                test_precision_doa*100.0, test_recall_doa*100.0, 180*test_hung_loss/np.pi)
        )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

