# Hybrid prediction training
This is the code to train and evaluate the models in "[HYPER: Learned Hybrid Trajectory Prediction via Factored Inference and Adaptive Sampling](https://arxiv.org/abs/2110.02344)".

Tested for _Argoverse_ dataset only.

## Package Installation
See the top level README to set up the development environment.

## Convert raw Argoverse data to protobufs
See instructions in [here](../../../data_sources/argoverse/README.md)

## Train a model
Example training script for Argoverse:
```bash
python intent/multiagents/hybrid/train_hybrid_vehicle_prediction.py --input-training-dir ~/intent/argoverse_v11/train_pb/ --input-validation-dir ~/intent/argoverse_v11/val_pb/ --dropout 0.1 --child-network-dropout 0.0 --additional-dropout-ratio 0.2 --training-set-ratio 0.8 --batch-size 32 --epoch-size 512 --learn-discrete-proposal --map-points-max 3600 --map-encoder-type attention --report-sample-metrics true --trajectory-regularization-cost 1.0
```
Options for baselines and ablations:
 - --learn-discrete-proposal: Remove this option will use the transition function instead of proposal function for discrete prediction.
 - --proposal-adaptive-sampling: Set to `false` to use the non-adapative sampling method.
 - --hybrid-fixed-mode: Add this option to disable mode transitions.

## Evaluate model offline
Given a trained model with saved model directory name SESSION_ID, run
```bash
python intent/multiagents/hybrid/run_hybrid_vehicle_trajectory_prediction.py --input-training-dir ~/intent/argoverse_v11/train_pb/ --input-validation-dir ~/intent/argoverse_v11/val_pb/ --dropout 0.1 --child-network-dropout 0.0 --additional-dropout-ratio 0.2 --training-set-ratio 0.8 --batch-size 32 --epoch-size 512 --learn-discrete-proposal--map-points-max 3600 --map-encoder-type attention --resume-session-name SESSION_ID_best_fde --MoN-number-samples 50 --hybrid-runner-subsample-size 6 --hybrid-runner-save True
```
Options include:
 - --hybrid-runner-subsample-size: Number of samples to select for final prediction. Default is 6.
 - --hybrid-runner-nms-dist-threshold: Distance threshold used for NMS. Default is 2.0.
 - --hybrid-runner-dist-type: Distance type used for FPS and NMS. Option includes 'final' (default) and 'avg'.
 - --hybrid-runner-visualize: Whether to visualize examples.
 - --hybrid-runner-save: Whether to save individual prediction results into json files.


## Unit Test
Test basic training with example Argoverse data:
```bash
python intent/multiagents/hybrid/test_hybrid_training.py
```

## Citation
```
@inproceedings{huang2022:hyper,
  title={HYPER: Learned Hybrid Trajectory Prediction via Factored Inference and Adaptive Sampling},
  author={Huang, Xin and Rosman, Guy and Gilitschenski, Igor and Jasour, Ashkan and McGill, Stephen G. and Leonard, John J. and Williams, Brian C.},
  booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022}
}
```
