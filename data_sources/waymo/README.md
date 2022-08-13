# Install env

```
# Create env
conda create  --name waymo-od python=3.8
# This includes the waymo protobuf definition/(metrics eval code).
conda activate waymo-od
pip install waymo-open-dataset-tf-2-6-0
```

An alternative is to install waymo libraries in the existing env.
```
conda activate $(cat conda_ver)
pip install waymo-open-dataset-tf-2-6-0
```

# Data
Overview: [Waymo data](https://waymo.com/open/data/motion/)

Training scenarios [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_0_0/uncompressed/scenario/training?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=1&prefix=&forceOnObjectsSortingFiltering=false)
And there's separate test/validation set.

## Download 
1. Download one item to explore manually
2. Download one item using `gsutil`
3. Download dir

### 1. Download one manually

### 2. Download one item using `gsutil`
```
# Install gsutil
snap install  google-cloud-sdk --classic
# Authenticate 
gcloud auth login
gcloud auth application-default login

# Download
gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/scenario/training/training.tfrecord-00000-of-01000 .
```
### 3. Download dir
* Follow the same step to install and auth the `gsutil` in step 2
```
# After install
# Download
gsutil -m cp  -r gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/scenario/training waymo_scenario__training_data
```

# Run visualization
```
conda activate waymo-od
python visualization_func.py --tfrecord /home/xiongyi/Downloads/uncompressed_scenario_training_training.tfrecord-00000-of-01000 
```
# Run streamlit visualization (data stats)
```
# First create data statistics file
# It output file: waymo_stats_tfrecord_1_scenario_517.pickle
python util/data_stats_class.py --tfrecord /home/xiongyi/data/waymo/training/scenario_training_training.tfrecord-00000-of-01000

# Then run streamlit service
streamlit run data_sources/waymo/util/data_exploration/app.py --  --stats [the_output_stats_file]
```


# Waymo dataset training
## Protobuf conversion
```
python convert_waymo_data.py --input-dir [WAYMO_DATA_DIR]/uncompressed/scenario/training/ --output-dir ~/intent/waymo_training_pb --dataset-label training
python convert_waymo_data.py --input-dir [WAYMO_DATA_DIR]/uncompressed/scenario/validation/ --output-dir ~/intent/waymo_validation_pb --dataset-label validation
```
Note: These commands will convert to RAD-compatible probobufs from all training/validation data. Feel free to kill at any time.

## Simple model training
```
conda activate $(cat conda_ver)
python intent/multiagents/train_pedestrian_trajectory_prediction.py --max-agents 8 --learning-rate 1e-3 --vis-interval 10 --scene-image-mode=none --agent-image-mode=none --past-timestep 6 --future-timestep 25 --past-timestep-size 0.2 --future-timestep-size 0.2 --num-visualization-images 10 --dropout 0.1 --child- 0.05 --training-set-ratio 0.8 --MoN-number-samples 6 --additional-dropout-ratio 0.2 --ignore-ego --input-training-dir ~/intent/waymo_training_pb/ --input-validation-dir ~/intent/waymo_validation_pb/ --datasets-name-lists none --epoch-size 16 --val-epoch-size 16 --vis-epoch-size 8 --num-workers 16 --val-num-workers 16 --vis-num-workers 8 --map-points-max 5000 --max-files 200 --batch-size 8 --val-batch-size 8 --vis-batch-size 8
```
Note: play with the last 4 arguments depending on your training environment.

## Train a marginal baseline model with only dynamics
Train a marginal prediction model using only dynamics (without map) with MLP decoder, to reproduce results from [Waymo paper](https://arxiv.org/pdf/2104.10133.pdf).
(Numbers in parentheses are Waymo-reported numbers.)

Train on validation set (most up to date as of Sept. 24th):
```
python multiagents/train_pedestrian_trajectory_prediction.py --input-dir ~/intent/waymo/waymo_validation_pb_aug/ --max-agents 4 --learning-rate 1e-3 --vis-interval 10 --scene-image-mode=none --agent-image-mode=none --past-timesteps 11 --future-timesteps 80 --past-timestep-size 0.1 --future-timestep-size 0.1 --num-visualization-images 20 --dropout 0.1 --child-network-dropout 0.0 --additional-dropout-ratio 0.2 --training-set-ratio 0.8 --batch-size 32 --epoch-size 2048 --raw-l2-for-mon --vis-num-workers 0 --l2-term-coeff 0.0 --augment-trajectories False --MoN-number-samples 6 --mon-term-coeff 1.0 --encoder-normalized-trajectory-only --disable-map-input --ignore-ego --use-waymo-dataset True --report-agent-type-metrics True --report-waymo-metrics True --use-marginal-error True --use-mlp-decoder True --use-discriminator False --interp-type none --disable-gnn-edges --disable-label-weights --datasets-name-lists none --logger-type weights_and_biases
```
Note: Waymo metrics computation only supports full time steps (past timesteps of 11 and future timesteps of 80).

```
python multiagents/train_pedestrian_trajectory_prediction.py --input-training-dir ~/intent/waymo/waymo_training_pb/ --input-validation-dir ~/intent/waymo/waymo_validation_pb/ --max-agents 8 --learning-rate 1e-3 --vis-interval 10 --scene-image-mode=none --agent-image-mode=none --past-timesteps 10 --future-timesteps 80 --past-timestep-size 0.1 --future-timestep-size 0.1 --num-visualization-images 20 --dropout 0.1 --child-network-dropout 0.0 --additional-dropout-ratio 0.2 --training-set-ratio 0.8 --batch-size 32 --epoch-size 2048 --raw-l2-for-mon --vis-num-workers 0 --l2-term-coeff 0.0 --augment-trajectories False --MoN-number-samples 6 --mon-term-coeff 1.0 --encoder-normalized-trajectory-only --disable-map-input --ignore-ego --use-waymo-dataset True --report-agent-type-metrics True --use-marginal-error True --use-mlp-decoder True --use-discriminator False --interp-type none --disable-label-weights --datasets-name-lists none --logger-type weights_and_biases
```


|    | minADE | minFDE |
|----|--------|-------|
| 3s | 0.584   | 1.351 |
| 5s | 1.369   | 3.446 |
| 8s | 2.881 (2.63)  | 7.490 |
