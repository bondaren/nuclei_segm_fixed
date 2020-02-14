# Nuclei segmentation of 3D stacks of mouse embryo

The pipeline for instance segmentation of the nuclei consists of 2-steps: 
1. semantic segmentation with a 3D U-Net, where the target task is predicting the nuclei masks together with the nuclei 
'boundaries' and z,y,x-affinities (multi-task learning)
2. instance segmentation with one of the following strategies:
    - thresholding on the 1st channel (nuclei mask) + connected components (baseline)
    - [Mutex Watershed](https://arxiv.org/abs/1904.12654)
    - [Multicut](https://www.nature.com/articles/nmeth.4151)
    
## Prerequisites
- Linux

## Segmentation with PlantSeg (easy)
The easiest way to segment on new data would be to use [PlantSeg](https://github.com/hci-unihd/plant-seg) application.
This method is limited to nuclei boundary predictions only (e.g. no nuceli mask + thresholding), but allows for a wide
range of segmentation strategies: `GASP (average linkage)`, `Mutext Watershed`, `MultiCut`, `Distance transform watershed` (allows for later proofreading)
from with an user-friendly graphical user interface.

Setup steps:

1.Install plantseg with conda:
```bash
conda create -n embryo-seg -c lcerrone -c abailoni -c cpape -c awolny -c conda-forge nifty=vplantseg1.0.8 plantseg=1.0.5
```
2. Activate conda environment:
```bash
conda activate embryo-seg
```
3. Checkout the repo in oder to download the networks:
```bash
git clone https://github.com/kreshuklab/vlad-nuclei.git
```
below `VLAD_NUCLEI_DIR` will refer to the directory where the repo was cloned.

4. Load models from `VLAD_NUCLEI_DIR/experiments/final` into the PlantSeg using the UI.
![Model load](https://user-images.githubusercontent.com/706781/74533911-fd02ce00-4f32-11ea-9a27-25176f008264.png)

7. Run segmentation using the PlantSeg
```bash
plantseg --gui
```
choose file to segment, set hyperparameters and run the segmentation :rocket:

For more info and troubleshooting see [PlantSeg documentation](https://github.com/hci-unihd/plant-seg).


## Segmentation with command line scripts (advanced)
For more flexibility, one may use python scripts provided in the repo. **Segmentation with thresholdding** for example
can be done only using this method.

Clone `vlad-nuclei` repository:
```bash
git clone https://github.com/kreshuklab/vlad-nuclei.git
```
below `VLAD_NUCLEI_DIR` will refer to the directory where the repo was cloned. 

Create a new conda environment:
```bash
conda create -n embryo-seg-cli --c cpape -c awolny -c conda-forge elf pytorch-3dunet
```

Activate conda environment:
```bash
conda activate embryo-seg-cli
```

Run network predictions on the Linux noded with a GPU:
```bash
predict3dunet --config PATH_TO_CONFIG
```
where `PATH_TO_CONFIG` is the path to the network prediction configuration, e.g. in order to use the network trained
to predict nuclei mask + nuclei boundaries + affinities, use [config_predict.yml](experiments/final/unet_bce_dice_ab_nuclei_boundary_aff/config_predict.yml).
Change `model_path` key to be `VLAD_NUCLEI_DIR/experiments/final/unet_bce_dice_ab_nuclei_boundary_aff/best_checkpoint.pytorch` (replace `VLAD_NUCLEI_DIR` by the actual path),
and the `file_paths` key to point to the files on which to run the prediction (if you specify directory, all the h5 files in this directory will be predicted).

For more detailed information of how to predict with `predict3dunet` see [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).


For segmentation go to `VLAD_NUCLEI_DIR` with `cd VLAD_NUCLEI_DIR`.

Wait for the prediction process to finish and then segment with one of the 3 strategies:
1. Mutex Watershed:
```bash
python segmentation/mws_segmentation.py --pmaps <PATH_TO_THE_NETWORK_PREDICTION_FILE> --mask --threshold 0.8 
```
this will save the results in the same directory as `PATH_TO_THE_NETWORK_PREDICTION_FILE`, but with the suffix `_mws.h5`.
2. Multicut:
```bash
python segmentation/mc_segmentation.py --pmaps <PATH_TO_THE_NETWORK_PREDICTION_FILE> --channel 1 
```
this will save the results in the same directory as `PATH_TO_THE_NETWORK_PREDICTION_FILE`, but with the suffix `_mc.h5`.
3. Thresholding + connected components
```bash
python segmentation/threshold_segmentation.py --pmaps <PATH_TO_THE_NETWORK_PREDICTION_FILE> --threshold 0.9
```
this will save the results in the same directory as `PATH_TO_THE_NETWORK_PREDICTION_FILE`, but with the suffix `_threshold.h5`.


## Run validation script
Activate conda environment (see [Segmentation with command line scripts (advanced)](#segmentation-with-command-line-scripts-(advanced))):
```bash
conda activate embryo-seg-cli
```

Run:
```bash
python utils/precision_recall.py --gt <LIST_OF_GROUND_TRUTH_FILES> --seg <LIST_OF_SEGMENTATION_FILES> 
```
E.g.:
```bash
python utils/precision_recall.py --gt /home/vlad/EmbryoFiles/GT_Ab1_test.h5 --seg /home/vlad/EmbryoFiles/GT_Ab1_test_threshold.h5
```
`LIST_OF_GROUND_TRUTH_FILES` and `LIST_OF_SEGMENTATION_FILES` must much. This will print out the precision/recall/accuracy values
for each intersection-over-union threshold considered, as well as save the precision-recall plots in the current directory. 

