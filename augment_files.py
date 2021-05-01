import os
from itertools import combinations
from IPython import embed
import doanet_parameters


params = doanet_parameters.get_params()

final_data_size_multiplier = 4
fold_list = [3, 4, 5, 6]
wav_path = os.path.join(params['dataset_dir'], '{}_{}'.format(params['dataset'], params['mode']))
meta_path = os.path.join(params['dataset_dir'], 'metadata_{}'.format(params['mode']))

fold_file_list = {ind:{} for ind in fold_list}
for file_name in os.listdir(wav_path):
    fold_cnt = int(file_name.split('_')[0][-1])
    room_cnt = int(file_name.split('_')[1][-1])
    mix_name = file_name.split('_')[2]
    if fold_cnt in fold_list and 'ov1' in file_name and 'aug' not in mix_name:
        if room_cnt not in fold_file_list[fold_cnt]:
            fold_file_list[fold_cnt][room_cnt] = []
        fold_file_list[fold_cnt][room_cnt].append(file_name)

for fold in fold_file_list:
    print(fold)
    for room in fold_file_list[fold]:
        print(room, len(fold_file_list[fold][room]))
        max_pairs = len(fold_file_list[fold][room]) * final_data_size_multiplier

        for comb_cnt, comb in enumerate(combinations(fold_file_list[fold][room], 2)):
            # Mix the two audio files
            out_file_name = comb[0].replace(comb[0].split('_')[2], 'aug{}{}'.format(comb[0].split('_')[2][-3:], comb[1].split('_')[2][-3:]))
            os.system('sox --combine mix {} {} {}'.format(
                os.path.join(wav_path, comb[0]),
                os.path.join(wav_path, comb[1]),
                os.path.join(wav_path, out_file_name))
            )

            # Mix the metadata files
            with open(os.path.join(meta_path, out_file_name.replace('.wav', '.csv')), 'w') as outfile:
                for fname in [os.path.join(meta_path, comb[0].replace('.wav', '.csv')), os.path.join(meta_path, comb[1].replace('.wav', '.csv'))]:
                    with open(fname) as infile:
                        outfile.write(infile.read())

            if comb_cnt >= (max_pairs-1):
                break



