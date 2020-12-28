#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#
import numpy as np
import os
import sys
import cls_data_generator
import doanet_model
import doanet_parameters
import torch
from IPython import embed
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plot


def main(argv):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    params = doanet_parameters.get_params()

    print('\nLoading the best model and predicting results on the testing split')
    print('\tLoading testing dataset:')
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=1, shuffle=False, per_file=params['dcase_output'], is_eval=True if params['mode'] is 'eval' else False
    )
    data_in, data_out = data_gen_test.get_data_sizes()
    dump_figures = True
    checkpoint_name = "models/6_4352803_foa_dev_split1_model.h5"
    model = doanet_model.CRNN(data_in, data_out, params)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_name, map_location=torch.device('cpu')))
    model = model.to(device)
    if dump_figures:
        dump_folder = os.path.join('dump_dir', os.path.basename(checkpoint_name).split('.')[0])
        os.makedirs(dump_folder, exist_ok=True)

    with torch.no_grad():
        file_cnt = 0
        for data, target in data_gen_test.generate():
            data = torch.tensor(data).to(device).float()
            output, activity = model(data)
            output = output.cpu().detach().numpy()
            use_activity_detector = True
            if use_activity_detector:
                activity = (torch.sigmoid(activity).cpu().detach().numpy() >0.5)
            mel_spec = data[0][0].cpu()
            foa_iv = data[0][-1].cpu()
            target[target > 1] =0

            plot.figure(figsize=(20,10))
            plot.subplot(321), plot.imshow(torch.transpose(mel_spec, -1, -2))
            plot.subplot(322), plot.imshow(torch.transpose(foa_iv, -1, -2))

            plot.subplot(323), plot.plot(target[0][:, 0], 'r', label='x1r')
            plot.subplot(323), plot.plot(target[0][:, 2], 'g', label='y1r')
            plot.subplot(323), plot.plot(target[0][:, 4], 'b', label='z1r')
            plot.grid()
            plot.ylim([-1.1, 1.1]), plot.legend()

            plot.subplot(324), plot.plot(target[0][:, 1], 'r', label='x2r')
            plot.subplot(324), plot.plot(target[0][:, 3], 'g', label='y2r')
            plot.subplot(324), plot.plot(target[0][:, 5], 'b', label='z2r')
            plot.grid()
            plot.ylim([-1.1, 1.1]), plot.legend()
            if use_activity_detector:
                output[0][:, 0:5:2] = activity[0][:, 0][:, np.newaxis]*output[0][:, 0:5:2]
                output[0][:, 1:6:2] = activity[0][:, 1][:, np.newaxis]*output[0][:, 1:6:2]

            plot.subplot(325), plot.plot(output[0][:, 0], ':r', label='x1p')
            plot.subplot(325), plot.plot(output[0][:, 2], ':g', label='y1p')
            plot.subplot(325), plot.plot(output[0][:, 4], ':b', label='z1p')
            plot.grid()
            plot.ylim([-1.1, 1.1]), plot.legend()

            plot.subplot(326), plot.plot(output[0][:, 1], ':r', label='x2p')
            plot.subplot(326), plot.plot(output[0][:, 3], ':g', label='y2p')
            plot.subplot(326), plot.plot(output[0][:, 5], ':b', label='z2p')
            plot.grid()
            plot.ylim([-1.1, 1.1]), plot.legend()

            if dump_figures:
                fig_name = '{}'.format(os.path.join(dump_folder, '{}.png'.format(file_cnt)))
                print('saving figure : {}'.format(fig_name))
                plot.savefig(fig_name, dpi=100)
                plot.close()
                if file_cnt>4:
                    break
                file_cnt += 1

            else:
                plot.show()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


