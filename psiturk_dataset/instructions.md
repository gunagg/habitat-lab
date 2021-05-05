## Habitat on web

### Data

1. Download the dataset from [this](https://drive.google.com/file/d/1lhyv4Xh4rmGeQauJBOOWpV3mJl2CbRdN/view?usp=sharing) URL
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

3. Run
    ```
    cd /path/to/habitat-lab
    sbatch habitat_baselines/rearrangement/il/multi_node_slurm.sh
    ```