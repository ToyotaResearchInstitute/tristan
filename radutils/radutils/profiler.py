import os
import time
from typing import Optional, Union

import torch

DEFAULT_LOG_PATH = os.path.expanduser("~/intent/logs/")


class DummyProfiler:
    """No-op profiler, shouldn't do anything."""

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            raise exc_val
        return self

    def step(self, flag_name):
        return self


class WallClockProfiler:
    """Print time for each step in wall clock time."""

    def __init__(self):
        self.previous_step = "Starting_profile"
        self.previous_time = time.time()
        self.previous_epoch_end = time.time()
        self.previous_batch_loop_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            raise exc_val
        return self

    def step(self, flag_name):
        t = time.time()
        print(f"### Time from {self.previous_step} to {flag_name}: {t-self.previous_time}")
        self.previous_time = t
        self.previous_step = flag_name
        if flag_name == "epoch_end":
            print(f"### @@@ Epoch took {t - self.previous_epoch_end}")
            self.previous_epoch_end = t
        if flag_name == "after_batch_loop":
            print(f"@@@ Batch took {t - self.previous_batch_loop_time}")
            self.previous_batch_loop_time = t
        return self


class TorchProfiler(DummyProfiler):
    """Profiling using torch.profiler.
    Only start profiling when the flag_name matches the flags defined by profile_duration.
    """

    def __init__(self, trace_folder, profile_duration=None, warmup=2):
        super().__init__()
        from torch.profiler import ProfilerActivity

        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=1, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_folder),
            with_stack=True,
            profile_memory=False,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        )
        self.profile_flags = get_profile_flag(profile_duration, warmup - 1)

    def __enter__(self):
        if self.profiler is not None:
            self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_value, traceback)
        return self

    def step(self, flag_name):
        """Do step in profiler, only when the flag_name matches the predefined flag.
        flag_name: str
            The name/location of the flag.
        """
        if self.profiler is not None and self.profile_flags:
            if self.profile_flags[0] == flag_name:
                print("@@@@@@@@@@@@ profile step", flag_name)
                len_flags = len(self.profile_flags)
                if len_flags == 2:
                    print(f"@@@@@ entering profiling at step {self.profile_flags[0]}")
                elif len_flags == 1:
                    print(f"@@@@@ exiting profiling at step {self.profile_flags[0]}")
                # Only run step() on specific flag
                self.profile_flags.pop(0)
                self.profiler.step()
        return self


class NvidiaDLProf(DummyProfiler):
    """Profiling using Nvidia's dlprof.
    Only start profiling when the flag_name matches the flags defined by profile_duration.
    """

    def __init__(self, profile_duration=None):
        super().__init__()
        self.use_nvidia_dlprof = True
        self.profiler = torch.autograd.profiler.emit_nvtx()
        self.profiler_finished = None
        self.profile_flags = get_profile_flag(profile_duration, 0)
        self.profile_flags.pop(0)

        # Initialize DLProf
        import nvidia_dlprof_pytorch_nvtx

        nvidia_dlprof_pytorch_nvtx.init()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_value, traceback)
        return self

    def step(self, flag_name):
        """Do step in profiler, start and stop profiling when flag_name matches the predefined flag.
        flag_name: str
            The name/location of the flag.
        """
        if self.profiler is not None and self.profile_flags:
            print("profile flags", self.profile_flags)
            if self.profile_flags[0] == flag_name:
                len_flag = len(self.profile_flags)
                if len_flag > 2:
                    raise RuntimeError("Nvidia DLProf can only take len(profile_flags) <= 2")
                elif len_flag == 2:
                    # Entering profiling
                    print(f"@@@@@ entering profiling at step {self.profile_flags[0]}")
                    self.profiler.__enter__()
                    self.profile_flags.pop(0)
                elif len_flag == 1:
                    print(f"@@@@@ exiting profiling at step {self.profile_flags[0]}")
                    self.profiler.__exit__(None, None, None)
                    self.profiler_finished = self.profiler
                    # Avoid calling self.profiler.__exit__() twice
                    self.profiler = None
                    self.profile_flags.pop(0)
                else:
                    pass
        return self


def create_profiler(
    profile_type: Optional[str],
    profile_duration: Optional[str],
    num_epochs: int,
    trace_folder=DEFAULT_LOG_PATH,
    warmup: int = 2,
) -> Union[TorchProfiler, NvidiaDLProf, WallClockProfiler, DummyProfiler]:
    """Create profiler

    profile_type: str
        the type of profiler, pytorch ot nvidia
    profile_duration: str
        when and where to start and stop profiling
    num_epochs: int
        the num of total epochs
    trace_folder:
        the output folder
    warmup: int
        warmup cycle for pytorch profiler.
    :return:
        Union[TorchProfiler, NvidiaDLProf, DummyProfiler]
    """
    if profile_type == "nvidia":
        return NvidiaDLProf(profile_duration)
    elif profile_type == "pytorch":
        assert num_epochs >= 2, "Profiling with PyTorch Profiler must use at least 2 epochs"
        assert warmup >= 2, "warmup must be greater than 2 to get good result"
        return TorchProfiler(trace_folder, profile_duration, warmup)
    elif profile_type == "wallclock":
        return WallClockProfiler()
    elif profile_type is None or profile_type == "none":
        return DummyProfiler()
    else:
        raise RuntimeError(f"Unrecognized profiler type: {profile_type}")


def get_profile_flag(profile_duration: str, warmups: int = 1):
    """Create a list of profile flags used to track when to start profiling.
    Profiling always starts when there's 2 flags in the array and stops when there's 1 left.

    profile_duration: str
        which process to profile.
    warmups: int
        warmup cycles, used by pytorch.
    :return:
        A list of str, denoting when to run step() in profile
    """
    # Between the last two flags is where the profiling happens. True for all types of profilers.
    profile_flags = ["trainer_start"] + ["epoch_end"] * warmups  # Multiply by the number of profile.schedule.warmup
    if profile_duration is None or "none" == profile_duration:
        profile_flags = [None, None]
    elif "all" == profile_duration:
        profile_flags += ["trainer_start", "trainer_end"]
    elif "full_epoch" == profile_duration:
        profile_flags += ["epoch_end", "epoch_end"]
    elif "dataloading" == profile_duration:
        profile_flags += ["before_dataloading", "after_dataloading"]
    elif "generator" == profile_duration:
        profile_flags += ["after_dataloading", "after_generate_trajectory"]
    elif "costs" == profile_duration:
        profile_flags += ["before_compute_cost", "after_compute_cost"]
    elif "optimization" == profile_duration:
        profile_flags += ["after_compute_cost", "after_optimization"]
    elif "log_stats" == profile_duration:
        profile_flags += ["before_log_stats", "after_dataset_loop"]
    elif "training" == profile_duration:
        profile_flags += ["after_dataloading", "after_batch_loop"]
    elif "dataloading_training" == profile_duration:
        profile_flags += ["before_dataloading", "after_batch_loop"]
    else:
        raise ValueError("Unsupported profile duration")
    return profile_flags


def print_profile_result(profile: Union[TorchProfiler, NvidiaDLProf, DummyProfiler]):
    """Print profiler result to console.
    Note: writing result to disk is done elsewhere.

    profile:
        the profile class
    """
    if isinstance(profile, TorchProfiler):
        print(profile.profiler.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=50))
