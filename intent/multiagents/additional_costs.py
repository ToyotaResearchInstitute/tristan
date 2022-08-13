from abc import ABC, abstractmethod


class AdditionalCostCallback(ABC):
    def __init__(self) -> None:
        """A generic interface to compute additional costs for training."""

    @abstractmethod
    def update_costs(
        self,
        additional_stats: dict,
        params: dict,
        predicted: dict,
        expected: dict,
        future_encoding: dict,
        extra_context: dict,
    ) -> None:
        """Compute the cross-entropy loss of the generated token sequence.

        Parameters
        ----------
        additional_stats : dict
            A dictionary of the additional costs.
        params : dict
            A parameters dictionary.
        predicted : dict
            A dictionary of predicted values
        expected : dict
            A dictionary of expected values.
        future_encoding : dict
            A dictionary with additional encoding information for the future prediction.
        extra_context : dict
            A dictionary that provides extra context for computing costs.
        """


class TrajectoryRegularizationCost(AdditionalCostCallback):
    def __init__(self):
        super().__init__()

    def update_costs(self, additional_stats, params, predicted, expected, future_encoding, extra_context):
        """Compute cost addition according to |d^2 x/ dt^2|^2

        Parameters
        ----------
        additional_stats : dict
            A dictionary output variable, add 'acceleration_cost' value with the added cost.
        params : dict
            A parameters dictionary.
        predicted : dict
            A dictionary with predicted 'trajectories', a tensor value of size B x A x T x 2
        expected : dict
            A dictionary with expected 'trajectories', tensor value of size B x A x T x 2
        future_encoding : dict
            A dictionary with additional encoding information for the future prediction.
        extra_context : dict
            A dictionary containing "timestamps", a B x T_total tensor with timestamps for the past,future timepoints.
        """
        future_timestamps = extra_context["timestamps"][:, -predicted["trajectories"].shape[2] :]
        dt = (future_timestamps[:, 2:] - future_timestamps[:, :-2]).unsqueeze(1).unsqueeze(-1).unsqueeze(-1) / 2.0
        d2x = (
            predicted["trajectories"][:, :, 1:-1, :] * 2
            - predicted["trajectories"][:, :, 2:, :]
            - predicted["trajectories"][:, :, :-2, :]
        )  # A [1 -2 1] 2nd order derivative
        acceleration_norm = (d2x**2 / dt**4).mean()  # the norm of (d2x/dt^2)^2
        additional_stats["acceleration_cost"] = acceleration_norm * params["trajectory_regularization_cost"]
