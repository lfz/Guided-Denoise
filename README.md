# Guided-Denoise
The winning submission for NIPS 2017: Defense Against Adversarial Attack of team TSAIL

# Paper 
[Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser](https://arxiv.org/abs/1712.02976)

# File Description

* prepare_data.ipynb: generate dataset

* Originset, Originset_test: the folder for original image

* toolkit: the program running the attack in batch

* Attackset: the attacks

* Advset: the adversarial images

* checkpoints: the models checkpoint used, download [here](https://pan.baidu.com/s/1kVzP9nL)

* Exps: the defense model

* GD_train, PD_train: train the defense model using guided denoise or pixel denoise

# How to use
the attacks are stored in folder Attackset 
the script is in the toolkit folder. 
in the run_attacks.sh file:
modify models to the attacks you want to generate, separate by comma, or use "all" to include all attacks in Attackset.
use the command to run:

   `bash run_attacks.sh $gpuids`
   
where gpuids is the id of the gpus you want to use, they are number separated by comma. It will generate the training set.
Then change the line `DATASET_DIR="${parentdir}/Originset"` to `DATASET_DIR="${parentdir}/Originset_test"`, and run the command    `bash run_attacks.sh $gpuids` again.

Then specify a model you want to use, the models are stored in Exp folder, there is a sample folder, it refers to a model named "sample", let's use it. Then go to GD_train if you want to use guided denoiser, 
run 

`python main --exp sample ` 

The program will load Exp/sample/model.py as a model to train. and also you can specify other parameters defined in the GD_train/main.py
