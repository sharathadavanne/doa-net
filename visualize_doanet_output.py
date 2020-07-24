#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import sys
import cls_data_generator
import doanet_model
import doanet_parameters
import torch
from IPython import embed
import matplotlib
matplotlib.use('TkAgg')
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

    model = doanet_model.CRNN(data_in, data_out, params)
    model.eval()
    model.load_state_dict(torch.load("models/best_hung_model.pt", map_location=torch.device('cpu')))
    with torch.no_grad():
        for data, target in data_gen_test.generate():
            data, target = torch.tensor(data).to(device).float(), target[:, :, :-1]
            output = model(data)
            output = output.detach().numpy()
            mel_spec = data[0][0]
            foa_iv = data[0][-1]

            plot.figure()
            plot.subplot(321), plot.imshow(torch.transpose(mel_spec, -1, -2))
            plot.subplot(322), plot.imshow(torch.transpose(foa_iv, -1, -2))

            plot.subplot(323), plot.plot(target[0][:, 0], 'r', label='x1r')
            plot.subplot(323), plot.plot(target[0][:, 2], 'g', label='y1r')
            plot.subplot(323), plot.plot(target[0][:, 4], 'b', label='z1r')
            plot.ylim([-1.1, 1.1]), plot.legend()

            plot.subplot(324), plot.plot(target[0][:, 1], 'c', label='x2r')
            plot.subplot(324), plot.plot(target[0][:, 3], 'm', label='y2r')
            plot.subplot(324), plot.plot(target[0][:, 5], 'y', label='z2r')
            plot.ylim([-1.1, 1.1]), plot.legend()

            plot.subplot(325), plot.plot(output[0][:, 0], ':r', label='x1p')
            plot.subplot(325), plot.plot(output[0][:, 2], ':g', label='y1p')
            plot.subplot(325), plot.plot(output[0][:, 4], ':b', label='z1p')
            plot.ylim([-1.1, 1.1]), plot.legend()

            plot.subplot(326), plot.plot(output[0][:, 1], ':c', label='x2p')
            plot.subplot(326), plot.plot(output[0][:, 3], ':m', label='y2p')
            plot.subplot(326), plot.plot(output[0][:, 5], ':y', label='z2p')
            plot.ylim([-1.1, 1.1]), plot.legend()
            plot.show()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
