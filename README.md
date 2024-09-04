# NeuralSRP -- Locata evaluation
This is a fork from @DavidDiazGuerra's [Cross3D](https://github.com/DavidDiazGuerra/Cross3D) repository.
The goal of this repository is to test the Neural-SRP method on the LOCATA dataset, while comparing it with
the Cross3D model itself.

## Dependencies

* Python: it has been tested with Python 3.8
* Numpy, matplotlib, scipy, soundfile, pandas and tqdm
* Pytorch: it has been tested with Python 1.4.0
* [webrtcvad](https://github.com/wiseman/py-webrtcvad)
* [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) [[2]](#references)

## Datasets

* **LibriSpeech** The training dataset is generated during the training of the models as the trajectories are needed, 
but to simulate them you will need to have the [LibriSpeech corpus](http://www.openslr.org/12) in your machine. By 
default, the main scripts look for it in [datasets/LibriSpeech](https://github.com/DavidDiazGuerra/Cross3D/tree/master/datasets/LibriSpeech) 
but you can modify its path with the `path_train` and `path_test` variables.
* **LOCATA** In order to test the models with actual recordings, you will also need the dataset of the 
[LOCATA challenge](https://www.locata.lms.tf.fau.de/). By default, the main scripts look for it in 
[datasets/LOCATA](https://github.com/DavidDiazGuerra/Cross3D/tree/master/datasets/LOCATA) 
but you can modify its phat with the `path_locata` variable.

## Main scripts

* `train.py` trains the models. You can modify the parameters of the training in the `params.py` file.
* `visualize_locata.py` generates the figures of the paper [[1]](#references) with the results of the models on the
LOCATA dataset.

## Pretrained models

You can find the pretrained models in the [release section](https://github.com/DavidDiazGuerra/Cross3D/releases) of the 
repository.

## Other source files

`acousticTrackingDataset.py`, `acousticTrackingTrainers.py`, `acousticTrackingModels.py` and `acousticTrackingDataset.py`
contain several classes and functions employed by the main scripts. They have been published to facilitate the 
replicability of the research presented in [[1]](#references), not as a software library. Therefore, any feature included 
in them that is not used by the main scripts may be untested and could contain bugs.

## References

>[1] D. Diaz-Guerra, A. Miguel and J. R. Beltran, "Robust Sound Source Tracking Using SRP-PHAT and 3D Convolutional Neural Networks," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 300-311, 2021 [[DOI](https://doi.org/10.1109/TASLP.2020.3040031)] [[arXiv preprint](https://arxiv.org/abs/2006.09006)].
>
>[2] D. Diaz-Guerra, A. Miguel, J.R. Beltran, "gpuRIR: A python library for Room Impulse Response simulation with GPU 
acceleration," in Multimedia Tools and Applications, Oct. 2020 [[DOI](https://doi.org/10.1007/s11042-020-09905-3)] [[SharedIt](https://rdcu.be/b8gzW)] [[arXiv preprint](https://arxiv.org/abs/1810.11359)].
>
>[3] Sharath Adavanne, Archontis Politis, Joonas Nikunen, and Tuomas Virtanen, "Sound event localization and detection 
of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected Topics in Signal 
Processing (JSTSP 2018).
/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/BSS/Anechoic_earnoise