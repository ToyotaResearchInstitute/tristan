# Argoverse Dataset

## Download Argoverse v1.1 dataset
Download **Argoverse Motion Forecasting v1.1 dataset** _and_ **Argoverse HD Maps** from https://www.argoverse.org/av1.html

## Install Argoverse API
Follow instructions at https://github.com/argoai/argoverse-api#installation

## Convert to protobufs
Command usage:
```bash
$ python data_sources/argoverse/create_adovehicle_snippets_argoverse.py --input-dir ${ARGOVERSE_DATA_DIR} --output-dir ${OUTPUT_PROTOBUFs_DIR} --protobuf -s -r --smooth
```
Options include:
 - --protobuf: Save data as protobufs.
 - -s: Select candidate map centerlines close to the agent trajectory, using Argoverse API, as opposed to geofencing.
 - -r: Resample equidistant points along map centerlines.
 - --smooth: Smooth agent trajectory.
 - -v: Visualize data. 
 - -l: Maximum number of examples to save.
 - --mode: Whether the data comes from train/val/test set.

### Data conversion commands
Convert training data:
```bash
$ python data_sources/argoverse/create_adovehicle_snippets_argoverse.py --input-dir ~/argoverse_data/train/data/ --output-dir ~/argoverse_data/train_pb --protobuf -s -r --smooth
```

Convert validation data:
```bash
$ python data_sources/argoverse/create_adovehicle_snippets_argoverse.py --input-dir ~/argoverse_data/val/data/ --output-dir ~/argoverse_data/val_pb --protobuf -s -r --smooth
```

Convert test data:
```bash
$ python data_sources/argoverse/create_adovehicle_snippets_argoverse.py --input-dir ~/argoverse_data/test/data/ --output-dir ~/argoverse_data/test_pb --protobuf -s -r --smooth --mode test
```

#### Converted Version
For TRI users, see [here](../../intent/multiagents/README.md#Training) to download converted protobuf.

### Example training command
The following assumes that you have created a training environment as described [here](../../intent/multiagents/README.md#installation).

Run basic training without images
```bash
$ python intent/multiagents/train_pedestrian_trajectory_prediction.py --input-dir /home/ec2-user/converted_protobufs/argoverse/train_pb/ --scene-image-mode none --agent-image-mode none --datasets-name-lists none --use-semantic false --num-workers 32 --batch-size 32 --val-batch-size 32 --val-num-workers 32 --vis-batch-size 32 --vis-num-workers 32 --max-agents 4 --past-timesteps 20 --future-timesteps 30 --past-timestep-size 0.1 --future-timestep-size 0.1 --val-interval 10 --vis-interval 10 --disable-map-decoder false
```

To train other models, refer to README in its own folder.
- For instance, to train a hybrid model, refer to [here](../../intent/multiagents/hybrid/README.md)

---

## Augment protobufs
This is usually used for non-standard training, such as language-based prediction.

### Language
Command usage:
```bash
$ python data_sources/augment_protobuf_with_language.py --source-protobufs-dir [SOURCE_PROTOBUFS_DIR] --augmented-protobufs-dir [AUGMENTED_PROTOBUFS_DIR]
```
