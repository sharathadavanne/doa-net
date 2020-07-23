#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
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
sys.path.insert(0,'/home/sharath/PycharmProjects/hungarian-net')
from train_hnet import HNetGRU
from scipy.optimize import linear_sum_assignment


def collect_test_labels(_data_gen_test, _data_out, _nb_classes, quick_test):
    # Collecting ground truth for test data
    nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0]
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[1], _data_out[2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        if _data_gen_test.get_data_gen_mode():
            doa_label = tmp_label
        else:
            doa_label = tmp_label[:, :, :-1]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = doa_label
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_doa


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
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = doanet_parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    hnet_model = HNetGRU(max_len=2).to(device)
    hnet_model.eval()
    hnet_model.load_state_dict(torch.load("models/hnet_model.pt" ))
    print('---------------- Hungarian-net -------------------')
    print(hnet_model)
    feat_cls = cls_feature_class.FeatureClass(params)
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

        nb_classes = data_gen_train.get_nb_classes()
        gt = collect_test_labels(data_gen_val, data_out, nb_classes, params['quick_test'])
        doa_gt = evaluation_metrics.reshape_3Dto2D(gt)

        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
            params['fnn_size']))

        model = doanet_model.CRNN(data_in, data_out, params)
        print('---------------- DOA-net -------------------')
        print(model)
        best_val_loss = 99999
        best_val_epoch = -1
        best_hung_loss = 99999
        best_hung_epoch = -1
        patience_cnt = 0
        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        tr_loss_list = np.zeros(nb_epoch)
        val_loss_list = np.zeros(nb_epoch)
        hung_tr_loss_list = np.zeros(nb_epoch)
        hung_val_loss_list = np.zeros(nb_epoch)

        optimizer = optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()

        # start training
        for epoch_cnt in range(nb_epoch):
            start = time.time()

            # TRAINING
            model.train()
            train_loss, train_hung_loss, nb_train_batches = 0., 0., 0.
            for data, target in data_gen_train.generate():
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()
                output = model(data)
                optimizer.zero_grad()

                # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
                max_nb_doas = output.shape[2]//3
                output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas)
                target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas)

                # get pair-wise distance matrix between predicted and reference.
                # target_norm, output_norm = torch.sqrt(torch.sum(target**2, -2) + 1e-10), torch.sqrt(torch.sum(output**2, -2) + 1e-10)
                # target, output = target/target_norm.unsqueeze(-2), output/output_norm.unsqueeze(-2)
                dist_mat = torch.matmul(torch.transpose(output, -1, -2), target)
                dist_mat[dist_mat == 0.0] = -1 # [TODO] fix this to set the distance value to -1 for default target DOA (0, 0, 0)
                dist_mat = torch.clamp(dist_mat, -1, 1)
                dist_mat = torch.acos(dist_mat)  # (batch, sequence, max_nb_doas, max_nb_doas)
                dist_mat = dist_mat.view(-1, max_nb_doas, max_nb_doas)   # (batch*sequence, max_nb_doas, max_nb_doas)

                if params['use_hnet']:
                    # get data association between predicted and reference using hungarian-net
                    hidden = torch.zeros(1, dist_mat.shape[0], 128)
                    da_mat, _, _ = hnet_model(dist_mat, hidden)
                    da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)

                    loss = torch.mean(dist_mat * da_mat.view(dist_mat.shape))
                else:
                    loss = criterion(output, target)

                    loc_hung_loss = 0.0
                    dist_mat = dist_mat.detach().numpy()
                    for loc_dist_mat in dist_mat:
                        row_ind, col_ind = linear_sum_assignment(loc_dist_mat)
                        loc_hung_loss += loc_dist_mat[row_ind, col_ind].sum()
                    loc_hung_loss /= dist_mat.shape[0]

                loss.backward()
                optimizer.step()
                train_hung_loss += loc_hung_loss
                train_loss += loss.item()
                nb_train_batches += 1
                if params['quick_test'] and nb_train_batches ==4:
                    break
            train_hung_loss /= nb_train_batches
            train_loss /= nb_train_batches

            ## TESTING
            model.eval()
            test_loss, test_hung_loss, nb_test_batches = 0., 0., 0.
            with torch.no_grad():
                for data, target in data_gen_val.generate():
                    data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()
                    output = model(data)

                    # (batch, sequence, max_nb_doas*3) to (batch, sequence, max_nb_doas, 3)
                    max_nb_doas = output.shape[2]//3
                    output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas)
                    target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas)


                    # get pair-wise distance matrix between predicted and reference.
                    # target_norm, output_norm = torch.sqrt(torch.sum(target**2, -2) + 1e-10), torch.sqrt(torch.sum(output**2, -2) + 1e-10)
                    # target, output = target/target_norm.unsqueeze(-2), output/output_norm.unsqueeze(-2)
                    dist_mat = torch.matmul(torch.transpose(output, -1, -2), target)
                    dist_mat[dist_mat == 0.0] = -1
                    dist_mat = torch.clamp(dist_mat, -1, 1)
                    dist_mat = torch.acos(dist_mat)  # (batch, sequence, max_nb_doas, max_nb_doas)
                    dist_mat = dist_mat.view(-1, max_nb_doas, max_nb_doas)   # (batch*sequence, max_nb_doas, max_nb_doas)

                    if params['use_hnet']:
                        # get data association between predicted and reference using hungarian-net
                        hidden = torch.zeros(1, dist_mat.shape[0], 128)
                        da_mat, _, _ = hnet_model(dist_mat, hidden)
                        da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)

                        loss = torch.mean(dist_mat * da_mat.view(dist_mat.shape))
                    else:
                        loss = criterion(output, target)

                        loc_hung_loss = 0.0
                        dist_mat = dist_mat.detach().numpy()
                        for loc_dist_mat in dist_mat:
                            row_ind, col_ind = linear_sum_assignment(loc_dist_mat)
                            loc_hung_loss += loc_dist_mat[row_ind, col_ind].sum()
                        loc_hung_loss /= dist_mat.shape[0]

                    test_hung_loss += loc_hung_loss
                    test_loss += loss.item()  # sum up batch loss
                    nb_test_batches += 1
                    if params['quick_test'] and nb_test_batches == 2:
                        break

            test_hung_loss /= nb_test_batches
            test_loss /= nb_test_batches

            if test_loss < best_val_loss:
                best_val_loss = test_loss
                best_val_epoch = epoch_cnt
                torch.save(model.state_dict(), "models/best_val_model.pt")
            if test_hung_loss < best_hung_loss:
                best_hung_loss = test_hung_loss
                best_hung_epoch = epoch_cnt
                torch.save(model.state_dict(), "models/best_hung_model.pt")

            print('epoch: {}, train_loss: {:0.2f}, val_loss: {:0.2f}, train_hung_loss: {:0.2f}, test_hung_loss: {:0.2f}, '
                  'best_val_epoch: {}, best_hung_epoch: {}'.format(
                epoch_cnt, train_loss, test_loss, train_hung_loss, test_hung_loss, best_val_epoch, best_hung_epoch))

            tr_loss_list[epoch_cnt], val_loss_list[epoch_cnt], hung_tr_loss_list[epoch_cnt], hung_val_loss_list[epoch_cnt] = train_loss, test_loss, train_hung_loss, test_hung_loss
            plot_functions(unique_name, tr_loss_list, val_loss_list, hung_tr_loss_list, hung_val_loss_list)

            patience_cnt += 1
            if patience_cnt > params['patience']:
                break

        # # ------------------  Calculate metric scores for unseen test split ---------------------------------
        # print('\nLoading the best model and predicting results on the testing split')
        # print('\tLoading testing dataset:')
        # data_gen_test = cls_data_generator.DataGenerator(
        #     params=params, split=split, shuffle=False, per_file=params['dcase_output'], is_eval=True if params['mode'] is 'eval' else False
        # )
        #
        # model = model.load_seld_model('{}_model.h5'.format(unique_name), params['doa_objective'])
        # pred_test = model.predict_generator(
        #     generator=data_gen_test.generate(),
        #     steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
        #     verbose=2
        # )
        #
        # test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test if params['doa_objective'] is 'mse' else pred_test[:, :, nb_classes:])
        #
        # if params['dcase_output']:
        #     # Dump results in DCASE output format for calculating final scores
        #     dcase_dump_folder = os.path.join(params['dcase_dir'], '{}_{}_{}'.format(task_id, params['dataset'], params['mode']))
        #     cls_feature_class.create_folder(dcase_dump_folder)
        #     print('Dumping recording-wise results in: {}'.format(dcase_dump_folder))
        #
        #     test_filelist = data_gen_test.get_filelist()
        #     # Number of frames for a 60 second audio with 100ms hop length = 600 frames
        #     max_frames_with_content = data_gen_test.get_nb_frames()
        #
        #     # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with
        #     # zero padding in the remaining frames
        #     frames_per_file = data_gen_test.get_frame_per_file()
        #
        #     for file_cnt in range(test_doa_pred.shape[0]//frames_per_file):
        #         output_file = os.path.join(dcase_dump_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
        #         dc = file_cnt * frames_per_file
        #         output_dict = feat_cls.regression_label_format_to_output_format(
        #             test_doa_pred[dc:dc + max_frames_with_content, :]
        #         )
        #         data_gen_test.write_output_format_file(output_file, output_dict)
        #
        # if params['mode'] is 'dev':
        #     test_data_in, test_data_out = data_gen_test.get_data_sizes()
        #     test_gt = collect_test_labels(data_gen_test, test_data_out, nb_classes, params['quick_test'])
        #     test_doa_gt = evaluation_metrics.reshape_3Dto2D(test_gt)
        #
        #     # Calculate DCASE2019 scores
        #     test_doa_loss = evaluation_metrics.compute_doa_scores_regr_xyz(test_doa_pred, test_doa_gt)
        #     print('Results on test split:')
        #     print('\tLocalization-only scores: DOA Error: {:0.1f}, Frame recall: {:0.1f}'.format(test_doa_loss[0], test_doa_loss[1]*100))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
