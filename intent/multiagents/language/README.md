# Language-based predictor

This is the code to train and evaluate the models in "[Trajectory Prediction with Linguistic Representations](https://arxiv.org/abs/2110.09741)".

Tested for _Argoverse_ dataset only.

## Package installation

See the top level README to set up the development environment.

## Augment protobufs with language tokens

1. Get the converted Argoverse/Waymo protobufs.
   See instructions in [here](../../../data_sources/argoverse/README.md)
2. Augment the protobufs using filters or the annotated data.
   - Augment using filters
     ```bash
     python data_sources/augment_protobuf_with_language.py --source-protobufs-dir [DATA_DIR]/{train/val}_pb/ --augmented-protobufs-dir [DATA_DIR]/augmented_{train/val}_pb/ --smooth --viz
     ```
     The `--smooth` flag is only needed for Argoverse data.
     The `--viz` flag saves the map with the annotation to image files.
   - Augment using the annotated captions
     ```bash
     python data_sources/augment_protobuf_with_language.py --source-protobufs-dir [DATA_DIR]/{train/val}_pb/ --source-caption-json-dir [DATA_DIR]/captions/ --augmented-protobufs-dir [DATA_DIR]/augmented_{train/val}_pb/ --viz
     ```

## Train a model

Example training script for Argoverse to get results in Fig. 5 (*Ours*):
```bash
python intent/multiagents/language/train_language_trajectory_prediction.py --input-training-dir ~/argoverse/synthetic/train_pb/ --input-validation-dir ~/argoverse/synthetic/val_pb/ --learning-rate 1e-3 --scene-image-mode=none --agent-image-mode=none --past-timesteps 20 --future-timesteps 30 --past-timestep-size 0.1 --future-timestep-size 0.1 --batch-size 32 --val-batch-size 32 --vis-batch-size 32 --map-attention-type point --use-language=true --use-semantics=false --max-language-tokens 10 --token-encoder-input-size 4 --token-encoder-hidden-size 4 --token-generator-hidden-size 4 --language-use-mlp=true --language-dropout-ratio 0.1 --child-network-dropout 0.0 --additional-dropout-ratio 0.1 --datasets-name-lists=none --latent-factors-type=none --cache-latent-factors=true --MoN-number-samples 6
```
Options for training the baseline and ablation models in Fig. 5 (Line 1-4):
 - --use-latent-factors: Set to `false` to use the baseline LSTM decoder.
 - --latent-factors-type: Set to `attention` to run the baseline multihead attention decoder.
 - --language-ablate-attention: Set to `true` to ablate the language attention.
 - --language-ablate-agent-attention: Set to `true` to ablate the agent tokens in language attention.

## Evaluate the model
Given a trained model with saved model directory name SESSION_ID, run
```bash
python intent/multiagents/language/run_language_trajectory_prediction.py --input-training-dir ~/argoverse/synthetic/train_pb/ --input-validation-dir ~/argoverse/synthetic/val_pb/ --scene-image-mode=none --agent-image-mode=none --past-timesteps 20 --future-timesteps 30 --past-timestep-size 0.1 --future-timestep-size 0.1 --batch-size 32 --epoch-size 2048 --map-attention-type point --use-language=true --use-semantics=false --max-language-tokens 10 --token-encoder-input-size 4 --token-encoder-hidden-size 4 --token-generator-hidden-size 4 --language-use-mlp=true --language-dropout-ratio 0.1 --child-network-dropout 0.0 --additional-dropout-ratio 0.1 --datasets-name-lists=none --latent-factors-type=none --resume-session-name SESSION_ID --MoN-number-samples 50
```
Options include:
 - --visualize-prediction: Visualize language token attention.
 - --compute-information-gain: Compute information gain by setting half of the samples to padded tokens.
 - --output-folder-name: Output folder that stores the computed metrics.
 
## Citation
```
@inproceedings{kuo2022:language-traj,
  title={Trajectory Prediction with Linguistic Representations},
  author={Kuo, Yen-Ling and Huang, Xin and Barbu, Andrei and McGill, Stephen G. and Katz, Boris and Leonard, John J. and Rosman, Guy},
  booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022}
}
```
