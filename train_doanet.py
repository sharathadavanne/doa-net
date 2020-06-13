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


def plot_functions(fig_name, _tr_loss, _val_loss, _doa_loss):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(311)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(nb_epoch), _val_loss, label='train loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(312)
    plot.plot(range(nb_epoch), _doa_loss[:, 0]/180., label='doa er / 180')
    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='doa fr')
    plot.legend()
    plot.grid(True)

    plot.subplot(313)
    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_pks')
    plot.plot(range(nb_epoch), _doa_loss[:, 3], label='good_pks')
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

        best_doa_metric = 99999
        best_epoch = -1
        patience_cnt = 0
        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        tr_loss = np.zeros(nb_epoch)
        val_loss = np.zeros(nb_epoch)
        doa_metric = np.zeros((nb_epoch, 6))

        optimizer = optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss(reduction='sum')

        # start training
        for epoch_cnt in range(nb_epoch):
            start = time.time()

            # TRAINING
            model.train()
            train_loss, nb_train_batches = 0., 0.
            for data, target in data_gen_train.generate():
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()
                optimizer.zero_grad()

                output = model(data)

                # (batch, sequence, nb_doas*3) to (batch, sequence, nb_doas, 3)
                output = output.view(output.shape[0], output.shape[1], output.shape[2]//3, 3)
                target = target.view(target.shape[0], target.shape[1], target.shape[2]//3, 3)

                # get pair-wise distance matrix between predicted and reference.
                target_norm, output_norm = torch.sqrt(torch.sum(target**2, -1) + 1e-10), torch.sqrt(torch.sum(output**2, -1) + 1e-10)
                target, output = target/target_norm.unsqueeze(-1), output/output_norm.unsqueeze(-1)
                dist_mat = torch.matmul(output, torch.transpose(target, -1, -2)).view(-1, 2, 2)
                dist_mat = torch.clamp(dist_mat, -1, 1)
                dist_mat = torch.acos(dist_mat)

                # get data association between predicted and reference using hungarian-net
                hidden = torch.zeros(1, dist_mat.shape[0], 128)
                da_mat, _, _ = hnet_model(dist_mat, hidden)
                da_mat = da_mat.sigmoid()

                loss = torch.mean(dist_mat*da_mat.view(dist_mat.shape))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                nb_train_batches += 1
                if params['quick_test'] and nb_train_batches ==4:
                    break
            train_loss /= nb_train_batches

            # #TESTING
            model.eval()
            test_loss, nb_test_batches = 0., 0.
            with torch.no_grad():
                for data, target in data_gen_val.generate():
                    data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-1]).to(device).float()

                    output = model(data)

                    # (batch, sequence, nb_doas*3) to (batch, sequence, nb_doas, 3)
                    output = output.view(output.shape[0], output.shape[1], output.shape[2]//3, 3)
                    target = target.view(target.shape[0], target.shape[1], target.shape[2]//3, 3)

                    # get pair-wise distance matrix between predicted and reference.
                    target_norm, output_norm = torch.sqrt(torch.sum(target**2, -1) + 1e-10), torch.sqrt(torch.sum(output**2, -1) + 1e-10)
                    target, output = target/target_norm.unsqueeze(-1), output/output_norm.unsqueeze(-1)
                    dist_mat = torch.matmul(output, torch.transpose(target, -1, -2)).view(-1, 2, 2)
                    dist_mat = torch.clamp(dist_mat, -1, 1)
                    dist_mat = torch.acos(dist_mat)

                    # get data association between predicted and reference using hungarian-net
                    hidden = torch.zeros(1, dist_mat.shape[0], 128)
                    da_mat, _, _ = hnet_model(dist_mat, hidden)
                    da_mat = da_mat.sigmoid()

                    loss = torch.mean(dist_mat*da_mat.view(dist_mat.shape))

                    test_loss += loss.item()  # sum up batch loss
                    nb_test_batches += 1
                    if params['quick_test'] and nb_test_batches==2:
                        break

            test_loss /= nb_test_batches

            print('epoch: {}, train_loss: {}, val_loss: {}'.format(epoch_cnt, train_loss, test_loss))
        #     doa_pred = evaluation_metrics.reshape_3Dto2D(pred)
        #
        # #     # Calculate the DOA score
        #     doa_metric[epoch_cnt, :] = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred, doa_gt)
        #
        #     # Visualize the metrics with respect to epochs
        #     plot_functions(unique_name, tr_loss, val_loss, doa_metric)
        #
        #     patience_cnt += 1
        #     if doa_metric[epoch_cnt] < best_doa_metric:
        #         best_doa_metric = doa_metric[epoch_cnt]
        #         best_epoch = epoch_cnt
        #         model.save(model_name)
        #         patience_cnt = 0
        #
        #     print(
        #         'epoch_cnt: {}, time: {:0.2f}s, tr_loss: {:0.2f}, val_loss: {:0.2f}, '
        #         'DE: {:0.1f}, FR:{:0.1f}, '
        #         'best_doa_score: {:0.2f}, best_epoch : {}\n'.format(
        #             epoch_cnt, time.time() - start, tr_loss[epoch_cnt],  val_loss[epoch_cnt],
        #             doa_metric[epoch_cnt, 0], doa_metric[epoch_cnt, 1]*100,
        #             best_doa_metric, best_epoch
        #         )
        #     )
        #     if patience_cnt > params['patience']:
        #         break
        #
        # print('\nResults on validation split:')
        # print('\tUnique_name: {} '.format(unique_name))
        # print('\tSaved model for the best_epoch: {}'.format(best_epoch))
        # print('\tDOA_score (early stopping score) : {}'.format(best_doa_metric))
        # print('\tDOA_error: {:0.1f}, Frame recall: {:0.1f}'.format(doa_metric[best_epoch, 0], doa_metric[best_epoch, 1]*100))
        #
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
