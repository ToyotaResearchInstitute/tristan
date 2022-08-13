from intent.trajectory_prediction_misc import str2bool


def hybrid_prediction_argument_setter(parser):
    # Define parameters for hybrid prediction.
    parser.add_argument(
        "--compute-maneuvers-online",
        type=str2bool,
        default="true",
        help="[Hybrid Inference] Whether to compute maneuvers online from positions.",
    )
    parser.add_argument(
        "--discrete-label-type",
        type=str,
        default="maneuvers",
        help="[Hybrid Inference] Types of discrete labels. Currently only support maneuvers.",
    )
    parser.add_argument(
        "--discrete-label-size",
        type=int,
        default=5,
        help="[Hybrid Inference] Domain size of discrete labels.",
    )
    parser.add_argument(
        "--learn-discrete-proposal",
        action="store_true",
        help="[Hybrid Inference] Learn discrete proposal distribution for diverse sampling.",
    )
    parser.add_argument(
        "--proposal-adaptive-sampling",
        type=str2bool,
        default="true",
        help="[Hybrid Inference] Learn proposal distribution adaptively given previously full samples.",
    )
    parser.add_argument(
        "--hybrid-discrete-dropout",
        type=float,
        default=0.0,
        help="[Hybrid Inference] Dropout ratio for discrete samples.",
    )
    parser.add_argument(
        "--hybrid-discrete-dropout-type",
        type=str,
        choices=["zero", "random"],
        default="zero",
        help="[Hybrid Inference] Dropout type for discrete samples (zero-mask vs. random noise)",
    )
    parser.add_argument(
        "--proposal-samples-lstm",
        action="store_true",
        help="[Hybrid Inference] Encode previously generated samples through LSTM.",
    )
    parser.add_argument(
        "--hybrid-teacher-forcing", action="store_true", help="[Hybrid Inference] Use ground truth position in decoder."
    )
    parser.add_argument(
        "--hybrid-dropout-validation",
        type=str2bool,
        default="true",
        help="[Hybrid Inference] Use additional dropout in validation.",
    )
    parser.add_argument(
        "--hybrid-fixed-mode",
        action="store_true",
        help="[Hybrid Inference] Use fixed mode for hybrid prediction, to reimplement ManeuverLSTM.",
    )
    parser.add_argument(
        "--hybrid-fixed-mode-type",
        type=str,
        choices=["first", "mode"],
        default="mode",
        help="[Hybrid Inference] Whether the ground truth mode is from the first mode or the most common mode.",
    )
    parser.add_argument(
        "--hybrid-disable-mode",
        action="store_true",
        help="[Hybrid Inference] Disable mode during hybrid prediction, to simulate single mode prediction.",
    )
    parser.add_argument(
        "--hybrid-smooth-mode", action="store_true", help="[Hybrid Inference] Use smoothed mode for hybrid prediction."
    )
    parser.add_argument(
        "--discrete-term-coeff",
        type=float,
        default=1.0,
        help="[Hybrid Inference] Loss coefficient for hybrid prediction losses.",
    )
    parser.add_argument(
        "--discrete-reg-term-coeff",
        type=float,
        default=0.0,
        help="[Hybrid Inference] Loss coefficient to regularize proposal distribution values.",
    )
    parser.add_argument(
        "--discrete-transition-coeff",
        type=float,
        default=0.0,
        help="[Hybrid Inference] Loss coefficient to regularize discrete transitions.",
    )
    parser.add_argument(
        "--trajectory-regularization-cost",
        type=float,
        default=1.0,
        help="Added regularization (second derivative) coefficient for continuous trajectory.",
    )
    parser.add_argument(
        "--discrete-supervised",
        type=str2bool,
        default="true",
        help="[Hybrid Inference] Supervised training using future discrete mode.",
    )
    parser.add_argument(
        "--perturb-discrete-ratio",
        type=float,
        default=0.0,
        help="[Hybrid Inference] Ratio to replace discrete labels with random labels.",
    )
    # Unsupervised.
    parser.add_argument(
        "--hybrid-observation-sigma", type=float, default=1.0, help="[Hybrid Inference] Sigma of observation noise."
    )
    parser.add_argument(
        "--use-empirical-distribution",
        action="store_true",
        help="[Hybrid Inference] Use empirical distribution as prior in hybrid inference.",
    )

    return parser


def hybrid_argoverse_prediction_argument_setter(parser):
    # Overwrite parameters that are different from default values for hybrid prediction on Argoverse.
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max-agents", type=int, default=1, help="Maximum number of agents to read.")
    parser.add_argument("--past-timesteps", type=int, default=20, help="Number of past timesteps.")
    parser.add_argument("--future-timesteps", type=int, default=30, help="Number of future timesteps.")
    parser.add_argument(
        "--past-timestep-size",
        type=float,
        default=0.1,
        help="Timestep size (i.e., sampling period) for the past. Need to update past-timesteps if changing this timestep size.",
    )
    parser.add_argument(
        "--future-timestep-size",
        type=float,
        default=0.1,
        help="Timestep size (i.e., sampling period) for the future. Need to update future-timesteps if changing this timestep size.",
    )

    parser.add_argument(
        "--scene-image-mode",
        choices=["all", "custom", "none"],
        default="none",
        help="""
            Selection mode of scene images. If this is set to 'custom',
            --scene-image-timepoints can be used for fine-grained control of
            which images to choose. If no timepoints are given, the training
            injects images at all timepoints.
            """,
    )
    parser.add_argument(
        "--agent-image-mode",
        choices=["all", "custom", "none"],
        default="none",
        help="""
            Selection mode of agent images. If this is set to 'custom',
            --agent-image-agents and --agent-image-timepoints can be used for
            fine-grained control of which images to choose. If either of these
            arguments is not given, the training behaves as if they have been
            set to use all agents or timepoints respectively.
            """,
    )
    parser.add_argument(
        "--raw-l2-for-mon",
        type=str2bool,
        default="true",
        help="If true, uses squared error without truncation for MoN. Otherwise, use truncated squared error",
    )
    parser.add_argument(
        "--ego-agent-only", type=str2bool, default="true", help="Select agents based on only ego agent."
    )
    parser.add_argument("--MoN-number-samples", type=int, default=6, help="Maximum number of samples to use for MoN.")
    parser.add_argument("--mon-term-coeff", type=float, default=1.0, help="Data term coefficient (MoN).")

    # Batch encoder and decoder not supported by HYPER.
    parser.add_argument(
        "--use-batch-graph-encoder", type=str2bool, default="false", help="Use batched version of graph encoder."
    )
    parser.add_argument(
        "--use-multiagent-accelerated-decoder",
        type=str2bool,
        default="false",
        help="Whether to use multiagent decoder that run on batch*agents*samples. Note, --use-hybrid-outputs is not currently supported.",
    )

    parser.add_argument(
        "--latent-factors-type",
        type=str,
        default="none",
        choices=["none", "const", "explicit"],
        help="Type of latent factors.",
    )
    parser.add_argument(
        "--child-network-dropout",
        type=float,
        default=0.0,
        help="Dropout ratio for child networks, including trajectory input.",
    )
    parser.add_argument(
        "--augment-trajectories", type=str2bool, default="false", help="Rotate trajectories for data augmentation."
    )
    parser.add_argument("--use-discriminator", type=str2bool, default="false", help="Predict with a discriminator.")
    parser.add_argument(
        "--map-input-normalization-scales",
        nargs="+",
        type=float,
        default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        help="The scale by which to normalize map encoder inputs.",
    )
    parser.add_argument(
        "--map-sampling-length", type=float, default=None, help="Typical length to resample map elements."
    )
    parser.add_argument(
        "--map-sampling-minimum-length", type=float, default=None, help="Minimum length to resample map elements."
    )
    parser.add_argument(
        "--map-elements-max",
        type=int,
        default="10",
        help="Max number of elements in map data. Suggested numbers: 100 for tlogs, 10 for Argoverse, 500 for Waymo.",
    )
    parser.add_argument("--use-semantics", type=str2bool, default="false", help="Use semantics in prediction.")
    parser.add_argument(
        "--disable-label-weights", type=str2bool, default="true", help="Disable class weights for semantic loss"
    )
    parser.add_argument(
        "--interp-type",
        nargs="+",
        default=["none"],
        help="kind of interpolation to use -- e.g. 'interp1d', 'interpSpline', 'interpGP'",
    )
    parser.add_argument(
        "--regression-test-early-stop",
        type=str2bool,
        default="true",
        help="When set, run regression test early stop scheme.",
    )
    parser.add_argument(
        "--regression-test-early-stop-test-error-type",
        type=str,
        default="MoN_fde_error",
        help="Specify the type of validation error to test on",
    )
    parser.add_argument(
        "--regression-test-early-stop-test-error",
        type=float,
        default=0.0,
        help="Specify the validation error, when the error is lower, stop training. Default is 0.6",
    )

    return parser
