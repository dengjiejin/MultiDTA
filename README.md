# MultiDTA


## dependencies
numpy == 1.19.4 <br>
kreas == 2.4.3 <br>
Pconsc4 == 0.4 <br>
pytorch == 1.7.0 <br>
PyG (torch-geometric) == 1.4.1 <br>
hhsuite (https://github.com/soedinglab/hh-suite)<br>
rdkit == 2020.09.5 <br>
ccmpred (https://github.com/soedinglab/CCMpred) <br>

## data preparation
1. Prepare the data need for train. Get all msa files of the proteins in datasets (for more detail description of datasets, please refer to [datasets](https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md)), and using Pconsc4 to predict all the contact map. A script in the repo can be run to do all the steps: <br>
**python scripts.py** <br><br>
2. And if you want to skip the long time preparation, please directly download the contact map and msa files which we already generated from [files](https://drive.google.com/open?id=1rqAopf_IaH3jzFkwXObQ4i-6bUUwizCv). For more detailed generating information, please refer to the "scripts.py". Then copy the corresponding two folders to each dataset dir. For example:  <br>
(1) download the data.zip and unzip it. <br>
(2) copy two folders called "aln" and "pconsc4" from davis to the /data/davis of your repo, so do the KIBA. <br>


## train (cross validation)
5 folds cross validation. <br>
**python training_5folds.py 0 0 0** <br>
where the parameters are dataset selection, gpu selection, fold (0,1,2,3,4).

## test
This is to do the prediction with the models we trained. And this step is to reproduce the experiments. <br>
**python test.py 0 0** <br>
and the parameters are dataset selection, gpu selection.



