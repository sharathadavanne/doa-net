
# DOAnet: Differentiable Tracking-Based Training of Deep Learning Sound Source Localizers

The direction-of-arrival (DOA) Network (DOAnet) is the deep-learning-based implementation of a generic localizer that can estimate the number of active sources (multi-source localization), and further localize and track these sources with respect to time. In order to train a localizer that can a) detect unknown-number of sources and b) optimize directly on popular localization and tracking metrics such as [CLEAR MOT metrics](https://link.springer.com/content/pdf/10.1155/2008/246309.pdf) we employ our novel [deep Hungarian network (Hnet)](https://github.com/sharathadavanne/hungarian-net). There by skipping the use of objective functions and expensive permutation invariant training (PIT). If you are using this repo in any format, then please consider citing the following paper. 

> Sharath Adavanne*, Archontis Politis* and Tuomas Virtanen, "[Differentiable Tracking-Based Training of Deep Learning Sound Source Localizers](https://arxiv.org/pdf/2111.00030.pdf)" in the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2021)
 
If you want to read more about [generic approaches to sound localization and tracking then check here](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-and-tracking).

## METHOD

The overall block diagram is shown in the figure below, consisting of the localization network (DOAnet), and a [deep Hungarian network (Hnet)](https://github.com/sharathadavanne/hungarian-net) taking as input the distance matrix **D** computed from the DOAnet outputs, and predicting a (soft) differentiable association matrix **A**. Thereafter a series of differentiable matrix manipulations follow that provide further soft approximations of localization error, true/false positives/negatives. From those approximations, the differentiable dMOTp and dMOTa are constructed and their combination serves as the overall training objective.

<p align="center">
   <img src="https://github.com/sharathadavanne/doa-net/blob/master/images/DOAnet.png" width="400" title="Differentiable tracking-based training of DOAnet">
</p>

## DATASETS
We study the method on [TAU-NIGENS Spatial Sound Events 2020 dataset](https://doi.org/10.5281/zenodo.3870859),  provided in the [DCASE2020 Task 3 (SELD) challenge](http://dcase.community/challenge2020/task-sound-event-localization-and-detection)


## Getting Started

This repository consists of multiple Python scripts described below. If you have used any of my earlier repos on [SELD](https://github.com/sharathadavanne/seld-net), then this follows a similar structure.
* The `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `doanet_parameters.py` script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The `cls_feature_class.py` script has routines for labels creation, features extraction and normalization.
* The `cls_data_generator.py` script provides feature + label data in generator mode for training.
* The `doanet_model.py` script implements the DOAnet architecture.
* The `cls_metric.py` script implements the metrics for localization.
* The `train_doanet.py` is a wrapper script that trains the DOAnet.
* The `augment_files.py` script helps produce more recordings with overlapping sound events. It simply adds two random recordings which are known to have just one source active at a time. This script is specific to the dataset used in this study.  
* The `visualize_doanet_output.py` script helps visualize the DOAnet output.


### Prerequisites

The provided codebase has been tested on Python 3.8.1 and Torch 1.10


### Training the DOAnet

In order to quickly train DOAnet follow the steps below.

* For the chosen dataset (Ambisonic or Microphone), download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are Ambisonic dataset, then the 'base_folder/' should have two folders - 'foa_dev/' and 'metadata_dev/' after unzipping.

* Now update the respective dataset name and its path in `doanet_parameters.py` script. For the above example, you will change `dataset='foa'` and `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `doanet_parameters.py` script where all the features and labels will be dumped.

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. Run the script as shown below. This will dump the normalized features and labels in the `feat_label_dir` folder.

```
python3 batch_feature_extraction.py
```

You can now train the DOAnet using default parameters using
```
python3 train_doanet.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `doanet_parameters.py` script and call them as following
```
python3 train_doanet.py <task-id> <job-id>
```
Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.

* By default, the code runs in `quick_test = True` mode. This trains the network for 2 epochs on 2 mini-batches. Once you get to run the code sucessfully, set `quick_test = False` in `doanet_parameters.py` script and train on the entire data.

* In order to visualize the output of DOAnet in the `dcase_dir` directory (can be defined in `doanet_parameters.py`), first choose the model to use by setting the variable `checkpoint_name` in `visualize_doanet_output.py` script and run 
```
python3 visualize_doanet_output.py
```

## License
The repository is licensed under the [TAU License](LICENSE.md).