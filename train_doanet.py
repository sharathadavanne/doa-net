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
# sys.path.insert(0,'/users/sadavann/hungarian-net')
sys.path.insert(0,'/home/sharath/PycharmProjects/hungarian-net')
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
        model.load_state_dict(torch.load("/home/sharath/PycharmProjects/doa-net/models/50_4099991_foa_dev_split1_model.h5", map_location='cpu'))
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
        criterion1 = torch.nn.MSELoss()
        criterion2 = torch.nn.MSELoss()

        # start training
        for epoch_cnt in range(nb_epoch):
            start = time.time()

            # TRAINING
            model.train()
            train_loss, train_hung_loss, nb_train_batches, train_tp_doa, train_total_doa = 0., 0., 0., 0, 0
            for data, target in data_gen_train.generate():
                optimizer.zero_grad()
                nb_framewise_doas_gt = target[:, :, -1].reshape(-1)
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()
                output = model(data)


                # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
                max_nb_doas = output.shape[2]//3
                output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
                target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

                # get unit vectors
#                target_norm, output_norm = torch.sqrt(torch.sum(target**2, -1) + eps), torch.sqrt(torch.sum(output**2, -1) + eps)
#                target, output = target/target_norm.unsqueeze(-1), output/output_norm.unsqueeze(-1)

                # get pair-wise distance matrix between predicted and reference.
                dist_mat = torch.matmul(output, target.transpose(-1, -2))
                # dist_mat = torch.clamp(dist_mat, -1+eps, 1-eps) # the +- eps is critical because the acos computation will become saturated if we have values of -1 and 1
                # dist_mat = torch.acos(dist_mat)  # (batch, sequence, max_nb_doas, max_nb_doas)
                dist_mat = dist_mat.view(-1, max_nb_doas, max_nb_doas)   # (batch*sequence, max_nb_doas, max_nb_doas)

                if params['use_hnet']:
                    with torch.no_grad():
                        hidden = torch.zeros(1, dist_mat.shape[0], 128).to(device)
                        da_mat, _, _ = hnet_model(dist_mat.transpose(1, 2), hidden)
                        da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)
                        da_mat = da_mat.view(dist_mat.shape)
                        max_val, max_inds = da_mat.max(-1)
                    
                    if params['binary_da']:
                        da_mat = (da_mat>0.5).float()
                    
                    # Compute dMOTP loss for true positives
                    dist_loss = torch.mean(torch.mul(dist_mat, da_mat))
                    if params['use_dmotp_only']:
                        loss = dist_loss
                    else:
                        target = target.view(-1, target.shape[-2], target.shape[-1])
                        output = output.view(-1, output.shape[-2], output.shape[-1])

                        target1 = target[np.arange(max_inds.shape[0]), max_inds[:, 0]]
                        target2 = target[np.arange(max_inds.shape[0]), max_inds[:, 1]]

                        mse1 = criterion1(output[:, 0, :], target1)
                        mse2 = criterion2(output[:, 1, :], target2)
                        loss = params['branch_weights'][0] * dist_loss + params['branch_weights'][1] * mse1 + params['branch_weights'][1] * mse2
                else:
                    loss = criterion1(output, target)
                loss.backward()
                optimizer.step()

                da_mat_numpy = (da_mat > 0.5).cpu().detach().numpy().astype(int)
                dist_mat_numpy = dist_mat.cpu().detach().numpy()
                dist_mat_numpy = np.clip(dist_mat_numpy, -1, 1)
                dist_mat_numpy = np.arccos(dist_mat_numpy)

                loc_tp_doa = 0
                loc_hung_loss = 0.0
                for ind, loc_dist_mat in enumerate(dist_mat_numpy):
                    loc_tp = min(nb_framewise_doas_gt[ind], da_mat_numpy[ind].sum())
                    if loc_tp:
                        loc_tp_doa += loc_tp
                        row_ind, col_ind = linear_sum_assignment(loc_dist_mat)
                        loc_hung_loss += np.multiply(loc_dist_mat[row_ind, col_ind], da_mat_numpy[ind]).sum()
                        loc_hung_loss /= loc_tp

                train_hung_loss += loc_hung_loss
                train_loss += loss.item()
                train_tp_doa += loc_tp_doa
                train_total_doa += nb_framewise_doas_gt.sum()
                nb_train_batches += 1
                if params['quick_test'] and nb_train_batches == 4:
                    break
            train_hung_loss /= nb_train_batches
            train_loss /= nb_train_batches
            train_tp_doa /= (float(train_total_doa) + eps)

            ## TESTING
            model.eval()
            test_loss, test_hung_loss, nb_test_batches, test_tp_doa, test_total_doa = 0., 0., 0., 0, 0
            dMOTP, mse_b1, mse_b2 = 0., 0., 0.
            with torch.no_grad():
                for data, target in data_gen_val.generate():
                    nb_framewise_doas_gt = target[:, :, -1].reshape(-1)
                    data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()
                    output = model(data)

                    # (batch, sequence, max_nb_doas*3) to (batch, sequence, max_nb_doas, 3)
                    max_nb_doas = output.shape[2]//3
                    output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
                    target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

                    #target_norm, output_norm = torch.sqrt(torch.sum(target**2, -1) + eps), torch.sqrt(torch.sum(output**2, -1) + eps)
                    #target, output = target/target_norm.unsqueeze(-1), output/output_norm.unsqueeze(-1)

                    # get pair-wise distance matrix between predicted and reference.
                    dist_mat = torch.matmul(output, target.transpose(-1, -2))
                    # dist_mat = torch.clamp(dist_mat, -1+eps, 1-eps)
                    # dist_mat = torch.acos(dist_mat)  # (batch, sequence, max_nb_doas, max_nb_doas)
                    dist_mat = dist_mat.view(-1, max_nb_doas, max_nb_doas)   # (batch*sequence, max_nb_doas, max_nb_doas)

                    if params['use_hnet']:
                        with torch.no_grad():
                            hidden = torch.zeros(1, dist_mat.shape[0], 128).to(device)
                            da_mat, _, _ = hnet_model(dist_mat.transpose(1, 2), hidden)
                            da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)
                            da_mat = da_mat.view(dist_mat.shape)
                            max_val, max_inds = da_mat.max(-1)

                        if params['binary_da']:
                            da_mat = (da_mat>0.5).float()
                    
                        # Compute dMOTP loss for true positives
                        dist_loss = torch.mean(torch.mul(dist_mat, da_mat))
                        dMOTP += dist_loss
                        if params['use_dmotp_only']:
                            loss = dist_loss
                        else:
                            target = target.view(-1, target.shape[-2], target.shape[-1])
                            output = output.view(-1, output.shape[-2], output.shape[-1])

                            target1 = target[np.arange(max_inds.shape[0]), max_inds[:, 0]]
                            target2 = target[np.arange(max_inds.shape[0]), max_inds[:, 1]]

                            mse1 = criterion1(output[:, 0, :], target1)
                            mse2 = criterion2(output[:, 1, :], target2)
                            loss = params['branch_weights'][0] * dist_loss + params['branch_weights'][1] * mse1 + params['branch_weights'][1] * mse2

                            mse_b1 += mse1
                            mse_b2 += mse2
                    else:
                        loss = criterion1(output, target)

                    da_mat_numpy = (da_mat > 0.5).cpu().detach().numpy().astype(int)
                    dist_mat_numpy = dist_mat.cpu().detach().numpy()
                    dist_mat_numpy = np.clip(dist_mat_numpy, -1, 1)
                    dist_mat_numpy = np.arccos(dist_mat_numpy)

                    loc_tp_doa = 0
                    loc_hung_loss = 0.0
                    for ind, loc_dist_mat in enumerate(dist_mat_numpy):
                        loc_tp = min(nb_framewise_doas_gt[ind], da_mat_numpy[ind].sum())
                        if loc_tp:
                            loc_tp_doa += loc_tp
                            row_ind, col_ind = linear_sum_assignment(loc_dist_mat)
                            loc_hung_loss += np.multiply(loc_dist_mat[row_ind, col_ind], da_mat_numpy[ind]).sum()
                            loc_hung_loss /= loc_tp

                    test_hung_loss += loc_hung_loss
                    test_loss += loss.item()  # sum up batch loss
                    test_tp_doa += loc_tp_doa
                    test_total_doa += nb_framewise_doas_gt.sum()
                    nb_test_batches += 1
                    if params['quick_test'] and nb_test_batches == 2:
                        break

            test_hung_loss /= nb_test_batches
            test_loss /= nb_test_batches
            test_tp_doa /= (float(test_total_doa) + eps)
            dMOTP /= nb_test_batches
            mse_b1 /= nb_test_batches
            mse_b2 /= nb_test_batches

            if test_hung_loss < best_val_loss:
                best_val_loss = test_hung_loss
                best_val_epoch = epoch_cnt
                torch.save(model.state_dict(), model_name)

            print(
                'epoch: {}, time: {:0.2f}, '
                'train_loss: {:0.2f}, val_loss: {:0.2f} {}, '
                'train_hung_loss: {:0.2f}/{:0.3f}, test_hung_loss_deg: {:0.2f}/{:0.3f}, '
                'best_val_epoch: {}'.format(
                    epoch_cnt, time.time()-start,
                    train_loss, test_loss,
                    '' if params['use_dmotp_only'] else '({:0.2f},{:0.2f},{:0.2f})'.format(dMOTP, mse_b1, mse_b2),
                    train_tp_doa*100.0, 180*train_hung_loss/np.pi, test_tp_doa*100.0, 180*test_hung_loss/np.pi,
                    best_val_epoch)
            )

            tr_loss_list[epoch_cnt], val_loss_list[epoch_cnt], hung_tr_loss_list[epoch_cnt], hung_val_loss_list[epoch_cnt] = train_loss, test_loss, train_hung_loss, test_hung_loss
            plot_functions(unique_name, tr_loss_list, val_loss_list, hung_tr_loss_list, hung_val_loss_list)

            patience_cnt += 1
            if patience_cnt > params['patience']:
                break


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
