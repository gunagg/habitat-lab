## Habitat on web

### Setup

1. Follow the offical `habitat-lab` and `habitat-sim` installation instructions.

### Data

1. Download the dataset from [this](https://habitat-on-web.s3.amazonaws.com/data/assets/data.zip) URL
2. Unzip the the contents in `data/` folder


### ObjectNav DDP Training setup

1. Change the folder paths for the following configs in the file `habitat_baselines/config/objectnav/ddpil_objectnav.yaml`:
    ```
    TENSORBOARD_DIR
    VIDEO_DIR
    CHECKPOINT_FOLDER
    ```

2. Set the value of `INFLECTION_COEF` based on the split you are training in the file `configs/tasks/objectnav_mp3d_il.yaml`. Find the values for each splits [here](https://www.notion.so/ab2173d31ce3425a97a4fad874920b5d?v=65c29317d4494122918b56e63e421dad)

3. Change the `DATA_PATH` to dataset path in the file `configs/tasks/objectnav_mp3d_il.yaml` to point to `data/datasets/objectnav_mp3d_v4/{split}/{split}.json.gz`

4. Make sure `CHECKPOINT_INTERVAL` is set to `100` in `habitat_baselines/config/objectnav/ddpil_objectnav.yaml`

5. Set `NUM_PROCESSES` to `8` in the file `habitat_baselines/config/objectnav/ddpil_objectnav.yaml`. Set `--nodes` to `2`, `--gres gpu` to `8` and `--n-tasks-per-node` to `8` in `scripts/train_il.sh`
    **Note**: `NUM_PROCESS` * n_gpus should be less than or equal to total number of scenes in the dataset (i.e. 56 scenes for train split).

6. To use semantic observations in training set `USE_SEMANTICS` to `True`. To enable finetuning on predicted semantic observations set `USE_PRED_SEMANTICS` to `True`. Change `SWITCH_TO_PRED_SEMANTICS_EPOCH` to set finetuning start epoch.

7. To run distributed training use the following command:
    ```
    cd /path/to/habitat-lab
    sbatch scripts/train_il.sh /path/to/ddpil_objectnav.yaml
    ```
    You can set the Multi GPU configs in `scripts/train_il.sh`



### Evaluation setup

1. Change the folder paths for the following configs in the file `habitat_baselines/config/objectnav/ddpil_objectnav.yaml`:
    ```
    TENSORBOARD_DIR
    VIDEO_DIR
    ```

2. Set the checkpoint path in the file `habitat_baselines/config/objectnav/ddpil_objectnav.yaml`:
    ```
    EVAL_CKPT_PATH_DIR
    ```

3. Change the `DATA_PATH` to dataset path in the file `configs/tasks/objectnav_mp3d_il.yaml`:

4. Set `NUM_PROCESSES` to `8` in the file `habitat_baselines/config/objectnav/ddpil_objectnav.yaml`.

5. Run
    ```
    cd /path/to/habitat-lab
    srun -p short --constraint rtx_6000 --gres gpu:1 -c 6 --job-name eval bash /srv/share3/rramrakhya6/habitat-lab/scripts/run_eval.sh /path/to/ddpil_objectnav.yaml
    ```
