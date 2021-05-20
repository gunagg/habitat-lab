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
    srun -p short --constraint 2080_ti --gres gpu:1 -c 6 --job-name gen bash /path/to/habitat-lab/habitat_baselines/rearrangement/utils/generate_dataset.sh <scene_id> <task> <path_to_episodes_gzip>
    ```
    Value of `task` can be `rearrangement` or `objectnav`

3. Run the command on step 2 for all 9 scenes for pick and place task. Pass `rearrangement` as `<task>` param to the script
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

4. Run the command on step 2 for all 4 splits for objectnav task. Pass `objectnav` as `<task>` param to the script
    ```
    split_1
    split_2
    split_4
    split_5
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

4. Change the `DATA_PATH` to dataset path in the file `configs/tasks/object_rearrangement.yaml` to point to `data/datasets/object_rearrangement/v4/{split}/{split}.json.gz`

5. Make sure `CHECKPOINT_INTERVAL` is set to `1` in `habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml`

6. Set `NUM_PROCESSES` to the `1` in the file `habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml`.

7. Run
    ```
    cd /path/to/habitat-lab
    srun -p long --constraint rtx_6000 --gres gpu:8 -c 8 --job-name il bash /path/to/habitat-lab/scripts/run_training.sh
    ```

9. To run distributed training use the following command:
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

3. Change the `DATA_PATH` to dataset path in the file `configs/tasks/object_rearrangement.yaml`:

4. Set `NUM_PROCESSES` to the number of scenes in the dataset in the file `habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml`.

5. Run
    ```
    cd /path/to/habitat-lab
    srun -p short --constraint rtx_6000 --gres gpu:1 -c 6 --job-name eval bash /srv/share3/rramrakhya6/habitat-lab/scripts/run_eval.sh
    ```
    
    If you used distributed training run the following
    ```
    cd /path/to/habitat-lab
    srun -p short --constraint rtx_6000 --gres gpu:1 -c 6 --job-name eval bash /srv/share3/rramrakhya6/habitat-lab/scripts/run_eval.sh distrib
    ```
