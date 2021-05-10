## Habitat on web

### Setup

1. Follow the offical `habitat-lab` and `habitat-sim` installation instructions.

### Data

1. Download the dataset from [this](https://habitat-on-web.s3.amazonaws.com/data/assets/data.zip) URL
2. Unzip the the contents in `data/` folder

### Dataset Setup
1. Change the `DATASET_PATH` field in the file `habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml` to point to the dataset split path

2. Run
    ```
    srun -p short --constraint 2080_ti --gres gpu:1 -c 6 --job-name gen bash /path/to/habitat-lab/habitat_baselines/rearrangement/utils/generate_dataset.sh <scene_id> <path_to_episodes_gzip>
    ```

3. Run the command on step 2 for all 9 scenes
    ```
    JeFG25nYj2p
    q9vSo1VnCiC
    i5noydFURQK
    S9hNv5qa7GM
    29hnd4uzFmX
    jtcxE69GiFV
    JmbYfDe2QKZ
    TbHJrupSAjP
    zsNo4HB9uLZ
    ```

### Training setup

1. Set the `DATASET_PATH` in the file `habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml`

2. Change the folder paths for the following configs in the file `habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml`:
    ```
    TENSORBOARD_DIR
    VIDEO_DIR
    CHECKPOINT_FOLDER
    ```
3. Set the value of `MODEL.inflection_weight_coef` based on the split you are training. Find the values for each splits [here](https://www.notion.so/ab2173d31ce3425a97a4fad874920b5d?v=65c29317d4494122918b56e63e421dad)

4. Run
    ```
    cd /path/to/habitat-lab
    srun -p long --constraint rtx_6000 --gres gpu:8 -c 8 --job-name il bash /path/to/habitat-lab/scripts/run_training.sh
    ```

5. To run distributed training use the following command:
    ```
    cd /path/to/habitat-lab
    sbatch habitat_baselines/rearrangement/il/multi_node_slurm.sh
    ```
    You can set the Multi GPU configs in `habitat_baselines/rearrangement/il/multi_node_slurm.sh`


### Evaluation setup

1. Change the folder paths for the following configs in the file `habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml`:
    ```
    TENSORBOARD_DIR
    VIDEO_DIR
    ```

2. Set the checkpoint path in the file `habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml`:
    ```
    EVAL_CKPT_PATH_DIR
    ```

3. Run
    ```
    cd /path/to/habitat-lab
    srun -p short --constraint rtx_6000 --gres gpu:1 -c 6 --job-name eval bash /srv/share3/rramrakhya6/habitat-lab/scripts/run_eval.sh
    ```
    
    If you used distributed training run the following
    ```
    cd /path/to/habitat-lab
    srun -p short --constraint rtx_6000 --gres gpu:1 -c 6 --job-name eval bash /srv/share3/rramrakhya6/habitat-lab/scripts/run_eval.sh distrib
    ```
