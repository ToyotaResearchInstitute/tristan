import argparse

import numpy as np
import tqdm

from triceps.protobuf.proto_arguments import nullable_str


def trajectory_pred_argument_setter(parser):
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument(
        "--learning-epsilon",
        type=float,
        default=1e-4,
        help="Term added to the denominator in the Adam optimizer to improve numerical stability. "
        "Need to see if it makes a difference beyond the Pytorch default of 1e-8.",
    )
    parser.add_argument(
        "--learning-rate-milestones",
        nargs="+",
        default=None,
        help="Learning rate milestones to use for scheduler, in epochs - e.g 100 200",
    )
    parser.add_argument(
        "--err-horizons-timepoints",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 5.0],
        help="Prediction horizon milestones for FDE computations. (s)",
    )
    parser.add_argument(
        "--err-horizons-timepoints-x",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 5.0],
        help="Prediction horizon milestones along the x-axis for FDE computations. (s) The default values come from the test vehicle definition of 'Done'.",
    )
    parser.add_argument(
        "--err-horizons-timepoints-y",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 5.0],
        help="Prediction horizon milestones along the y-axis for FDE computations. (s) The default values come from the test vehicle definition of 'Done'.",
    )
    parser.add_argument(
        "--miss-thresholds",
        type=float,
        nargs="+",
        default=list(np.arange(4.0)),
        help="Absolute distances from ground truth above which predictions are classified as misses. (m)",
    )
    parser.add_argument(
        "--miss-thresholds-x",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0],
        help="Distances from ground truth along the x-axis above which predictions are classified as misses. (m) The default values come from the test vehicle definition of 'Done'.",
    )
    parser.add_argument(
        "--miss-thresholds-y",
        type=float,
        nargs="+",
        default=[0.2, 0.2, 0.3, 0.5],
        help="Distances from ground truth along the y-axis above which predictions are classified as misses. (m) The default values come from the test vehicle definition of 'Done'.",
    )
    parser.add_argument(
        "--learning-rate-gamma",
        type=float,
        default=np.sqrt(0.1),
        help="Learning rate reduction factor for milestones scheduler",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay for the optimizer.")
    parser.add_argument(
        "--pretraining-resume-optimizer",
        type=str2bool,
        default="false",
        help="resume optimizer states between pretraining/finetuning rounds.",
    )
    parser.add_argument(
        "--type-conditioned-temporal-model", type=str2bool, default="false", help="Use a separate LSTM per agent type."
    )
    parser.add_argument(
        "--full-dataset-epochs",
        type=str2bool,
        default="false",
        help="Instead of a dataset random subset, use the whole dataset every epoch.",
    )
    parser.add_argument(
        "--disable-optimization", type=str2bool, default="false", help="Do not perform any optimization."
    )
    parser.add_argument(
        "--logger-type",
        type=str,
        choices=["tensorboard", "weights_and_biases", "none"],
        default="tensorboard",
        help="Which framework to use to log during training.",
    )
    parser.add_argument(
        "--max-training-time",
        type=float,
        default=-1.0,
        help="The max training time, in hours. If it's < 0, will train indefinitely.",
    )
    parser.add_argument("--log-residuals", type=str2bool, default="true", help="Log the residual norms of parameters.")
    parser.add_argument("--wandb-entity", type=str, default="tri", help="For weights and biases logging: entity name.")
    parser.add_argument(
        "--wandb-project", type=str, default="ma_pred", help="For weights and biases logging: project name."
    )
    parser.add_argument(
        "--use-multiagent-accelerated-decoder",
        type=str2bool,
        default="true",
        help="Whether to use multiagent decoder that run on batch*agents*samples. Note, --use-hybrid-outputs is not currently supported.",
    )
    parser.add_argument(
        "--use-batch-graph-encoder", type=str2bool, default="true", help="Use batched version of graph encoder."
    )
    parser.add_argument(
        "--periodic-save-hours",
        type=int,
        default=0,
        help="If > 0, save a persistent snapshot of the model every this many hours.",
    )
    parser.add_argument(
        "--disable-model-saving",
        type=str2bool,
        default="false",
        help="Do not save models (e.g. for statistics runner).",
    )
    parser.add_argument(
        "--discriminator-learning-ratio",
        type=float,
        default=0.25,
        help="Learning ratio to get the discriminator learning rate.",
    )
    parser.add_argument("--l2-term-coeff", type=float, default=0.0, help="Data term coefficient (L2).")
    parser.add_argument(
        "--weighted-l2-term-coeff", type=float, default=10.0, help="Data term coefficient (weighted L2)."
    )
    parser.add_argument("--mon-term-coeff", type=float, default=0.5, help="Data term coefficient (MoN).")
    parser.add_argument("--discriminator-term-coeff", type=float, default=1.0, help="Discriminator term coefficient.")
    parser.add_argument(
        "--nullify-irrelevant-discriminator-agents",
        type=str2bool,
        default="false",
        help="When discriminating, set future agents to zero.",
    )

    parser.add_argument("--init-cont-term-coeff", type=float, default=10.0, help="Initial continuous term coefficient.")
    parser.add_argument(
        "--disable-discriminator-update",
        action="store_true",
        help="Disable the update for the discriminator (for debugging purposes).",
    )
    parser.add_argument("--leaky-generator", type=str2bool, default="true", help="Make the generator ReLUs leaky.")
    parser.add_argument(
        "--zero-generator-noise",
        type=str2bool,
        default="false",
        help="Make the generator use only zero noise vectors. Should simulate MSE for a GAN's generator.",
    )
    parser.add_argument(
        "--leaky-discriminator", type=str2bool, default="true", help="Make the discriminator ReLUs leaky."
    )
    parser.add_argument(
        "--special-init", type=str2bool, default="true", help="Use specific init for weights" " (e.g xavier)."
    )
    parser.add_argument("--predictor-hidden-state-dim", type=int, default=64, help="Hidden state dim of the predictor.")
    parser.add_argument("--edge-hidden-state-dim", type=int, default=4, help="Edge hidden state dim of the predictor.")
    parser.add_argument(
        "--encoder-tbptt",
        type=int,
        default=-1,
        help="Truncated backprop through time (TBPPT) " "length for the encoders in the GAN.",
    )
    parser.add_argument(
        "--decoder-tbptt",
        type=int,
        default=0,
        help="Truncated backprop through time (TBPPT) " "length for the decoder in the GAN.",
    )
    parser.add_argument(
        "--raw-l2-for-mon",
        action="store_true",
        help="If true, uses squared error without truncation for MoN. Otherwise, use truncated squared error",
    )
    parser.add_argument(
        "--predictor-normalization-scale", type=float, default=0.1, help="Normalization scale for coordinates."
    )
    parser.add_argument(
        "--predictor-batch-norm", type=str2bool, default="false", help="Use batch norm within GraphEncoder."
    )
    parser.add_argument("--predictor-layer-norm", action="store_true", help="Use layer norm in GraphEncoder LSTMs.")
    parser.add_argument(
        "--predictor-normalize", type=str2bool, default="true", help="Normalize trajectories in predictor."
    )
    parser.add_argument(
        "--predictor-local-rotation",
        type=str2bool,
        default="true",
        help="Rotate trajectories locally with respect to each agent in predictor.",
    )
    parser.add_argument(
        "--local-rotation-noise",
        type=float,
        default=None,
        help="Add noise to the orientation after rotating trajectories locally.",
    )
    parser.add_argument(
        "--linear-discriminator",
        type=str2bool,
        default="true",
        help="Use a linear clamped discriminator instead of tanh.",
    )
    parser.add_argument("--MoN-number-samples", type=int, default=5, help="Maximum number of samples to use for MoN.")
    parser.add_argument(
        "--l2-error-only", action="store_true", help="Replace probability measures with an L2 error fitting."
    )
    parser.add_argument("--child-network-dropout", type=float, default=0.1, help="Dropout ratio for child networks.")
    parser.add_argument(
        "--coordinate-encoder-dropout", type=float, default=0.0, help="Dropout ratio for position inputs."
    )
    parser.add_argument(
        "--encoder-decoder-type",
        type=str,
        choices=["gnn", "lstm_mlp", "polynomial", "transformer"],
        default="gnn",
        help="Encoder decoder type.",
    )
    parser.add_argument("--edge-dropout", type=float, default=0.5, help="Dropout ratio for edge inputs.")
    parser.add_argument(
        "--graph-input-dropout", type=float, default=0.0, help="Dropout ratio for graph inputs tensors."
    )
    parser.add_argument("--state-dropout-ratio", type=float, default=0.0, help="Dropout of LSTM state in the decoder.")
    parser.add_argument(
        "--additional-dropout-ratio", type=float, default=0.0, help="Dropout of additional " "factors in the decoder."
    )
    parser.add_argument(
        "--trajectory-regularization-cost", type=float, default=0.0, help="Added regularization coefficient."
    )
    parser.add_argument(
        "--tailored-robust-function",
        type=str2bool,
        default="false",
        help="Tailor the robust cost function to agent type, horizon.",
    )
    parser.add_argument(
        "--robust-function-temporal-slope",
        type=float,
        default=1.0,
        help="How quickly to increase the scale in the robust function.",
    )
    parser.add_argument(
        "--robust-function-temporal-intercept",
        type=float,
        default=1.0,
        help="The scale coefficient for t=0 in the robust function.",
    )
    parser.add_argument(
        "--pedestrian-noise-scale",
        type=float,
        default=1.0,
        help="How quickly noise rises for pedestrians (as opposed to vehicles).",
    )
    parser.add_argument(
        "--pedestrian-type-id",
        type=int,
        default=3,
        help="What is the index of the pedestrian type, for noise purposes.",
    )
    parser.add_argument(
        "--dropout-ratio",
        type=float,
        default=0.25,
        help="Dropout ratio for FC layers in general, probability to be zero.",
    )
    parser.add_argument(
        "--predictor-truncation-scale", type=float, default=5.0, help="The size at which to truncate for robust L2."
    )
    parser.add_argument("--disable-map-input", action="store_true", help="Disable using maps for the predictor.")
    parser.add_argument(
        "--load-map-input-dataset",
        type=str2bool,
        default="False",
        help="Load the map inputs even if map encoding is disabled.",
    )
    parser.add_argument(
        "--disable-map-decoder", type=str2bool, default="True", help="Disable maps for the decoder " "(not stable yet)."
    )
    parser.add_argument(
        "--silent-fail-on-missing-images", action="store_true", help="Place 0 image on missing image files."
    )
    parser.add_argument(
        "--scene-inputs-detach", type=str2bool, default="false", help="Detach scene inputs in training."
    )
    parser.add_argument(
        "--scene-inputs-nullify",
        type=str,
        nargs="*",
        default=[],
        help="""
                        List of additional input keys for which the result of the 
                        additional input encoder will be replaced by zeros in 
                        fuse_scene_tensor. This allows evaluating the actual 
                        contribution of a particular scene input.
                        """,
    )
    parser.add_argument(
        "--agent-inputs-detach", type=str2bool, default="false", help="Detach agent inputs in training."
    )
    parser.add_argument(
        "--agent-inputs-nullify",
        type=str,
        nargs="*",
        default=[],
        help="""
                        List of additional agent input keys for which the result of 
                        the additional input encoder will be replaced by zeros in 
                        fuse_agent_tensor. This allows evaluating the actual 
                        contribution of a particular agent input.
                        """,
    )

    parser.add_argument(
        "--disable-position-input", action="store_true", help="Use an all-zeros tensor as the past trajectory input."
    )
    parser.add_argument("--clip-gradients", action="store_true", help="Clip gradients when optimizing")
    parser.add_argument(
        "--clip-gradients-threshold", type=float, default=1.0, help="Clip gradients threshold when optimizing"
    )
    parser.add_argument(
        "--augment-trajectories", type=str2bool, default="true", help="Rotate trajectories for data augmentation."
    )
    parser.add_argument("--cumulative-decoding", type=str2bool, default="false", help="Decode in a cumulative fashion.")
    parser.add_argument(
        "--augmentation-timestamp-scale",
        type=float,
        default=0.0,
        help="Timeshift to augment/move trajectories, in seconds.",
    )

    parser.add_argument(
        "--augmentation-angle-scale",
        type=float,
        default=np.pi,
        help="(half-)Angle by which to rotate trajectories for data augmentation, in radians.",
    )
    parser.add_argument(
        "--augmentation-translation-scale",
        type=float,
        default=3.0,
        help="Translation by which to move, scale in meters.",
    )
    parser.add_argument(
        "--augmentation-noise-scale",
        type=float,
        default=0.0,
        help="Additive noise augmentation for trajectories, in meters.",
    )
    parser.add_argument(
        "--augmentation-process-noise-spring",
        type=float,
        default=0.2,
        help="Additive process noise augmentation for trajectories - spring force coefficient.",
    )
    parser.add_argument(
        "--augmentation-process-noise-scale",
        type=float,
        default=0.0,
        help="Additive process noise augmentation for trajectories, in meters.",
    )
    parser.add_argument(
        "--log-name-string", type=str, default="", help="Postfix to add for tensorboard and Weights & Biases logging."
    )
    parser.add_argument(
        "--disable-gan-training",
        action="store_true",
        help="Are we training a GAN (if set to false) or a single-cost predictor (if set to true.",
    )
    parser.add_argument(
        "--enable-other-agents-loss",
        action="store_true",
        help="If true, include other agents in computing cost, otherwise only use ego-vehicle and the relevant pedestrian.",
    )
    parser.add_argument(
        "--map-layer-features",
        nargs="+",
        type=int,
        default=[32],
        help="Feature dimensions list for the map cnn encoder.",
    )
    parser.add_argument(
        "--coordinate-decoder-widths",
        nargs="+",
        type=int,
        default=None,
        help="A list of the coordinate decoder output layer widths, specifying the network structure.",
    )
    parser.add_argument(
        "--coordinate-encoder-widths",
        nargs="+",
        type=int,
        default=None,
        help="A list of the coordinate encoder output layer widths, specifying the network structure.",
    )
    parser.add_argument(
        "--decoder-embed-position",
        type=str2bool,
        default="true",
        help="Should the decoder embed the recent positions or feed them as-is.",
    )
    parser.add_argument(
        "--decoder-embed-input",
        type=str2bool,
        default="true",
        help="Should the decoder embed the recent positions or feed them as-is.",
    )
    parser.add_argument(
        "--num-visualization-worst-cases",
        type=int,
        default=10,
        help="The number of visualizations to do for random samples from the dataset.",
    )
    parser.add_argument(
        "--num-visualization-images",
        type=int,
        default=12,
        help="The number of visualizations to do for random samples from the dataset.",
    )
    parser.add_argument(
        "--num-map-visualization-batches",
        type=int,
        default=1,
        help="The number of batches during which to do map visualizations.",
    )
    parser.add_argument(
        "--save-cases-jsons",
        type=str2bool,
        default="false",
        help="Whether or not to save JSONs with info from the worst-case collection.",
    )
    parser.add_argument(
        "--save-cases-camera-videos",
        type=str2bool,
        default="false",
        help="Whether or not to save camera videos from the worst-case collection.",
    )
    parser.add_argument(
        "--optimizer-step-batch", type=int, default=1, help="How many optimizer steps to accumulate gradients over."
    )
    parser.add_argument(
        "--resume-optimizer",
        type=str2bool,
        default="false",
        help="Load the saved optimizer state dict when resuming training",
    )
    parser.add_argument(
        "--encoder-normalized-trajectory-only",
        action="store_true",
        help="Use only normalized trajectory in encoder (to avoid geographical overfitting).",
    )
    parser.add_argument(
        "--use-latent-factors", type=str2bool, default="false", help="Use latent factors in prediction."
    )
    parser.add_argument("--use-semantics", type=str2bool, default="false", help="Use semantics in prediction.")
    parser.add_argument(
        "--semantic-labels-balance-cap",
        type=float,
        default=10.0,
        help="The cap on label class imbalance for reweighting.",
    )
    parser.add_argument("--use-discriminator", type=str2bool, default="true", help="Predict with a discriminator.")
    parser.add_argument(
        "--ego-agent-only", type=str2bool, default="false", help="Select agents based on only ego agent."
    )
    parser.add_argument(
        "--pretrained-image-encoder",
        type=str2bool,
        default="false",
        help="Use pretrained backbone model for image encoder. "
        "Note that setting to true downloads derivative material of ImageNet "
        "-- which complicates legal status",
    )
    parser.add_argument("--disable-label-weights", action="store_true", help="Disable class weights for semantic loss")
    parser.add_argument(
        "--deterministic-prediction",
        type=str2bool,
        default="false",
        help="Generate deterministic prediction without noise samples.",
    )
    parser.add_argument(
        "--require-agent-images",
        type=str2bool,
        default="false",
        help="Whether to require all agents to have images if the cameras are available.",
    )
    parser.add_argument("--map-fps-size", type=int, default="20", help="FPS size for a map.")
    parser.add_argument(
        "--agent-image-frozen-layers", type=int, default=0, help="Number of layers to freeze in agent image encoder."
    )
    parser.add_argument(
        "--scene-image-frozen-layers", type=int, default=0, help="Number of layers to freeze in scene image encoder."
    )
    parser.add_argument(
        "--raster-map-frozen-layers", type=int, default=None, help="Number of layers to freeze in raster map encoder."
    )
    parser.add_argument(
        "--agent-image-layer-features",
        nargs="+",
        type=int,
        default=[8, 16, 16, 64, 256],
        help="The feature widths of layers for custom agent image encoder.",
    )
    parser.add_argument(
        "--scene-image-layer-features",
        nargs="+",
        type=int,
        default=[8, 16, 16, 64, 256],
        help="The feature widths of layers for custom scene image encoder.",
    )
    parser.add_argument(
        "--raster-map-layer-features",
        nargs="+",
        type=int,
        default=[8, 16, 16, 64, 256],
        help="The feature widths of layers for custom raster map image encoder.",
    )
    parser.add_argument(
        "--coeff-input-processors-costs",
        type=float,
        default=1.0,
        help="Coefficient for input processors auxiliary costs.",
    )
    parser.add_argument(
        "--interpolation-excursion-threshold",
        type=float,
        default=4,
        help="The threshold on interpolation excursion from the original data.",
    )
    parser.add_argument(
        "--map-sampling-length", type=float, default=None, help="Typical length to resample map elements."
    )
    parser.add_argument(
        "--map-sampling-minimum-length", type=float, default=None, help="Minimum length to resample map elements."
    )
    parser.add_argument(
        "--map-points-max",
        type=int,
        default="2500",
        help="Max number of points to include in data. Suggested numbers: 2500 for Argoverse, 20000 for Waymo.",
    )
    parser.add_argument(
        "--map-segment-size",
        type=int,
        default="40",
        help="Length of map segments.",
    )
    parser.add_argument(
        "--map-segment-max",
        type=int,
        default="1500",
        help="Maximum number of map segments.",
    )
    parser.add_argument(
        "--map-points-subsample-ratio",
        type=int,
        default="1",
        help="Subsample ratio of map points.",
    )
    parser.add_argument(
        "--map-elements-max",
        type=int,
        default="100",
        help="Max number of elements in map data. Suggested numbers: 100 for tlogs, 10 for Argoverse, 500 for Waymo.",
    )
    parser.add_argument(
        "--map-id-type",
        type=str,
        choices=["integer", "binary", "onehot"],
        default="integer",
        help="Representation of map id.",
    )
    parser.add_argument(
        "--map-mask-distance-threshold",
        type=float,
        default=0.0,
        help="Distance threshold (in meter) to decide what map points to use. Default 0.0 means no threshold. At 100 meters, the memory reduces by 22.6% at the cost of 2.6% FDE decrease.",
    )
    parser.add_argument("--map-polyline-feature-degree", type=int, default="0", help="Degree of map polyline feature.")
    parser.add_argument(
        "--map-representation-cache-miss-ratio",
        type=float,
        default="1.0",
        help="Cache miss ratio (lower means more caching and less updates) for a map."
        "Should not effect things if --cache-map-encoder false.",
    )
    parser.add_argument(
        "--map-representation-cache-scale",
        type=float,
        default="1.0",
        help="Cache scale (lower means caching changes at a finer resolution) for a map.",
    )
    parser.add_argument(
        "--map-input-type", type=str, choices=["point", "raster"], default="point", help="Input type for map."
    )
    parser.add_argument("--map-encoder-type", type=str, default="gnn", help="Model to use for map encoder.")
    parser.add_argument(
        "--map-gnn-layer", type=int, default=0, help="How many round of message passing to run in a map GNN."
    )
    parser.add_argument(
        "--map-num-relevant-points",
        type=int,
        default=1,
        help="Number of relevant map point embeddings to pool in total.",
    )
    parser.add_argument(
        "--map-input-normalization-scales",
        nargs="+",
        type=float,
        default=[5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        help="The scale by which to normalize map encoder inputs.",
    )
    parser.add_argument(
        "--map-num-nearest-distances",
        type=int,
        default=2,
        help="The number of nearest element distances to save for the network to handle duplicate elements.",
    )
    parser.add_argument(
        "--map-reconstruction-position-coeff",
        type=float,
        default=0.2,
        help="The coefficient for scaling map position reconstruction costs.",
    )
    parser.add_argument(
        "--map-log-reconstruction-plot",
        type=str2bool,
        default="true",
        help="log the map reconstruction to Tensorboard/WandB.",
    )
    parser.add_argument(
        "--validate-multiple-samples",
        type=str2bool,
        default="false",
        help="Compute errors on multiple samples in validation.",
    )
    parser.add_argument(
        "--validate-sample-size", type=int, default=6, help="Number of samples to validate in validation."
    )
    parser.add_argument("--view-min-span-size", type=float, default=5.0, help="Minimum span for visualization")
    parser.add_argument(
        "--map-view-margin-around-trajectories",
        type=float,
        default=5.0,
        help="Maximum margin around trajectories for visualization.",
    )
    parser.add_argument("--disable-gnn-edges", action="store_true", help="Disable edges in the GNNs.")
    parser.add_argument("--disable-self-edges", type=str2bool, default="true", help="Disable self-edges in the GNNs.")
    parser.add_argument("--pretraining-mode", type=str2bool, default="false", help="Train via finetuning schedule.")
    parser.add_argument(
        "--pretraining-timescale", type=float, default=10.0, help="Time scale for finetuning schedule, in iterations."
    )
    parser.add_argument(
        "--pretraining-relative-lengths",
        nargs="+",
        type=float,
        default=None,
        help="Pretraining schedule, to be multipled by pretraining-timescale.",
    )
    parser.add_argument(
        "--pretraining-definition",
        type=str,
        default="intent.multiagents.parameter_utils.define_pretraining_parameter_sets",
        help="Define a function to set finetuning schedule, if non-default",
    )
    parser.add_argument(
        "--params-start-level", type=int, default=None, help="The parameter level at which to start training."
    )

    parser.add_argument(
        "--layer-pred-widths",
        nargs="+",
        type=int,
        default=None,
        help="A list of the encoder output layer widths, specifying the network structure.",
    )
    parser.add_argument(
        "--edge-encoder-widths",
        nargs="+",
        default=[4, 4],
        type=int,
        help="A list of edge encoder widths, specifying the network structure.",
    )
    parser.add_argument(
        "--self-edge-encoder-widths",
        nargs="+",
        default=None,
        type=int,
        help="A list of self-edge encoder widths, specifying the network structure.",
    )
    parser.add_argument(
        "--serialize-trajectories",
        nargs="+",
        default=None,
        type=str,
        help="a list of protobuf file basenames: eg 8eca814c6ac6c6dd2324a33a8de9b61e0e312695_000005.pb. "
        "These instances will be serialized together into one pickle file which can then be plotted compared across models.",
    )
    parser.add_argument(
        "--node-encoder-widths",
        nargs="+",
        default=None,
        type=int,
        help="A list of node encoder widths, specifying the network structure.",
    )
    parser.add_argument("--disable-gnn-edges", action="store_true", help="Disable edges in the GNNs.")
    parser.add_argument("--disable-self-edges", type=str2bool, default="true", help="Disable self-edges in the GNNs.")
    parser.add_argument(
        "--runner-output-folder",
        type=str,
        default="data_runner_results",
        help="Folder to store statistics and results from the dataset. By default this folder is relative to the --artifacts_folder param.",
    )
    parser.add_argument(
        "--trajectory-validity-criterion",
        choices=["conservative-relevant", "conservative-ego-only"],
        default="conservative-ego-only",
        help="validity criterion for prediction instances.",
    )
    parser.add_argument(
        "--trajectory-time-ego-padding",
        type=float,
        default=1.5,
        help="The padding that is allowed to be missing egovehicle trajectory samples before instance is declared invalid.",
    )
    parser.add_argument(
        "--trajectory-time-ado-padding",
        type=float,
        default=2.0,
        help="The padding that is allowed to be missing relevant adoagent trajectory samples before instance is declared invalid.",
    )
    parser.add_argument(
        "--synthetic-dataset-piecewise-coefficient",
        type=float,
        default=1.0,
        help="Coefficient for piecewise linear behavior in the synthetic dataset.",
    )
    parser.add_argument(
        "--synthetic-dataset-quadratic-coefficient",
        type=float,
        default=1.0,
        help="Coefficient for quadratic behavior in the synthetic dataset.",
    )
    parser.add_argument(
        "--synthetic-dataset-process-noise-coefficient",
        type=float,
        default=0.1,
        help="Coefficient for process noice in the synthetic dataset.",
    )
    parser.add_argument(
        "--datasets-name-lists",
        type=nullable_str,
        default=None,
        help="location of json with train/validation protobuf file lists or 'none' for automatic generation. If no value is set, another default value will be set in pedestrian_trajectory_prediction_util.py.",
    )
    parser.add_argument(
        "--dataset-collision-inteval",
        type=float,
        default=30.0,
        help="The interval with which to separate examples into train/test, in seconds.",
    )
    parser.add_argument(
        "--past-only-dataloading",
        type=str2bool,
        default=False,
        help="Only load past information in dataloaders, for inference mode.",
    )
    parser.add_argument(
        "--worst-case-per-agent-vis",
        type=str2bool,
        default="False",
        help="Generate worst case images for each agent individually.",
    )
    parser.add_argument(
        "--use-waymo-dataset",
        type=str2bool,
        default=None,
        help="Whether to use variables (i.e. agent type definition) specific to Waymo dataset.",
    )
    parser.add_argument(
        "--train-waymo-interactive-agents",
        type=str2bool,
        default="False",
        help="Whether to train on only interactive agents using Waymo interactive dataset. This flag should be True only if Waymo interactive dataset is used.",
    )
    parser.add_argument(
        "--report-waymo-metrics",
        type=str2bool,
        default="False",
        help="Report Waymo metrics in Tensorboard/W&B.",
    )
    parser.add_argument(
        "--report-agent-type-metrics",
        type=str2bool,
        default="False",
        help="Report metrics per agent type in Tensorboard/W&B.",
    )
    parser.add_argument(
        "--report-sample-metrics",
        type=str2bool,
        default="False",
        help="Report metrics per sample in Tensorboard/W&B.",
    )
    parser.add_argument(
        "--relevant-agent-types",
        nargs="+",
        default=None,
        type=str,
        help="A list of relevant agent types to train and evaluate, including car, bicycle, pedestrian, etc.",
    )
    parser.add_argument(
        "--use-marginal-error",
        type=str2bool,
        default="False",
        help="Train model using marginal error.",
    )
    parser.add_argument(
        "--use-mlp-decoder",
        type=str2bool,
        default="False",
        help="Use an MLP decoder.",
    )
    parser.add_argument(
        "--use-wandb-config",
        type=str2bool,
        default="False",
        help="Use wandb config to override default parameters. Used when running wandb sweep.",
    )
    parser.add_argument(
        "--num-encoder-transformer-heads",
        type=int,
        default=4,
        help="Number of attention heas for the encoder transformer.",
    )
    parser.add_argument(
        "--num-encoder-transformer-blocks",
        type=int,
        default=2,
        help="Number of transformer blocks for the encoder.",
    )
    parser.add_argument(
        "--num-decoder-transformer-heads",
        type=int,
        default=4,
        help="Number of attention heas for the decoder transformer.",
    )
    parser.add_argument(
        "--num-decoder-transformer-blocks",
        type=int,
        default=2,
        help="Number of transformer blocks for the decoder.",
    )
    parser.add_argument("--encoder-agent-dropout", type=float, default=0.0, help="Dropout ratio for agents.")
    parser.add_argument(
        "--use-global-map",
        type=str2bool,
        default="False",
        help="Use map in the global frame as the scene input.",
    )
    parser.add_argument(
        "--use-map-transformer",
        type=str2bool,
        default="True",
        help="Use map attention in the transformer-based model.",
    )
    parser.add_argument(
        "--num-encoder-transformer-agent-attn-skips",
        type=int,
        default=0,
        help="Number of initial blocks to skip attention across agents.",
    )
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
