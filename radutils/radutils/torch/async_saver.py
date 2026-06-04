import logging
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from typeguard import typechecked

import radutils.session_data as sess_data

try:
    from intent.multiagents.cache_data_paths_tri import SAVED_ARTIFACT_S3_PREFIX, SAVED_MODEL_S3_PREFIX
except ModuleNotFoundError:
    # In public repo, set your own s3 path here, e.g. s3://...
    SAVED_MODEL_S3_PREFIX = ""
    SAVED_ARTIFACT_S3_PREFIX = ""

DEFAULT_ROLLING_SAVES = 5


class ResumeStrategy(Enum):
    BEST = "best"  # Best according to early stopping criterion
    LAST = "last"

    def __str__(self):
        return self.value


DEFAULT_RESUME_STRATEGY = ResumeStrategy.LAST
DEFAULT_SAVE_STRATEGY = ResumeStrategy.LAST


@typechecked
def sync_to_s3(local_dir: str, s3_prefix: str, include_files: List[str] = None) -> None:
    command = f"aws s3 sync --acl=bucket-owner-full-control {local_dir} {s3_prefix}"
    if include_files:
        include_cmd = " ".join([f"--include {file}" for file in include_files])
        command = f"{command} --exclude '*' {include_cmd}"
    print(f"Syncing to S3: {command}")
    # Do NOT add --delete here as in some cases we are syncing one dynamic subdirectory
    # Do not catch exceptions here; fail loudly if we are not saving the result.
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True, shell=True, check=True)


@typechecked
def compress_directory(path_to_directory: str) -> str:
    """
    Compresses all files in the directory, without the containing directory,
    and places the archive in the parent directory.
    Returns the path to the archive
    """
    path_to_directory = os.path.normpath(path_to_directory)
    parent_dir = pathlib.Path(path_to_directory).parent
    dir_name = os.path.basename(path_to_directory)
    archive_path = os.path.join(parent_dir, f"{dir_name}.tar.gz")

    command = f"tar -zcf {archive_path} -C {path_to_directory} ."
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True, shell=True, check=True)
    return archive_path


@typechecked
def sync_model_to_s3(folder: str) -> None:
    """This is called at every checkpoint"""
    print(f"Syncing models for directory {folder}")
    folder = os.path.normpath(folder)
    session_name = os.path.basename(folder)
    s3_url = f"{SAVED_MODEL_S3_PREFIX}/{session_name}"
    sync_to_s3(folder, s3_url)


@typechecked
def sync_session_model_to_s3(session_id: str, session_data: Dict) -> None:
    model_dir = sess_data.get_saved_path(session_id, session_data, "model_save_folder")
    # We may not have saved a model if the trainer crashed, so do not add confusing errors
    if not os.path.isdir(model_dir):
        print(f"Skipping S3 upload of models: model directory '{model_dir}' not found")
        return

    # Currently, only the latest model should not be saved, but support saving everything only at the end
    for model_subdir in os.listdir(model_dir):
        path_to_model_dir = os.path.join(model_dir, model_subdir)
        if not os.path.isdir(path_to_model_dir) or not model_subdir.startswith(session_id):
            continue
        sync_model_to_s3(path_to_model_dir)


@typechecked
def compress_and_sync_session_artifacts_to_s3(session_id: str, session_data: Dict) -> None:
    artifact_dir = sess_data.get_saved_path(session_id, session_data, "artifacts_folder")

    # We may not have created artifacts, so do not crash in that case
    if not artifact_dir:
        print("Skipping S3 upload of artifacts: artifact directory not set")
        return

    if not os.path.isdir(artifact_dir):
        print(f"Skipping S3 upload of artifacts: artifact directory '{artifact_dir}' not found")
        return

    artifact_dir = os.path.join(artifact_dir, session_id)
    archive_path = compress_directory(artifact_dir)

    parent_dir = str(pathlib.Path(archive_path).parent)
    archive_name = os.path.basename(archive_path)
    sync_to_s3(parent_dir, SAVED_ARTIFACT_S3_PREFIX, [archive_name])


@typechecked
def sync_model_and_artifacts_to_s3(session_id: str) -> None:
    """This is called only after the trainer exits."""
    session_data = sess_data.read_session_data(session_id)
    sync_session_model_to_s3(session_id, session_data)
    compress_and_sync_session_artifacts_to_s3(session_id, session_data)


@typechecked
def sync_model_from_s3_command(session_name: str, path_to_model_directory: str) -> Tuple[str, str]:
    s3_url = os.path.join(SAVED_MODEL_S3_PREFIX, session_name)
    folder = os.path.join(path_to_model_directory, session_name)
    command = f"aws s3 sync {s3_url} {folder}"
    return command, folder


@typechecked
def sync_models_from_s3(session_names: List[str], path_to_model_directory: str) -> None:
    for session_name in session_names:
        command, folder = sync_model_from_s3_command(session_name, path_to_model_directory)
        if os.path.exists(folder):
            # If local session folder exist, don't sync from s3
            print(f"Local model exists in folder {folder}, won't download from s3")
            return
        print(f"Syncing model from S3: {command}")
        # Do not catch exceptions here; fail loudly if we are not loading the model.
        subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True, shell=True, check=True)


def update_rolling_backups(fname: str, num_rolling_saves: int):
    """Save a copy of a file, up to num_rolling_saves copies."""
    for idx in reversed(range(num_rolling_saves - 1)):
        extant_rolling_save = Path(f"{fname}-{idx}")
        if extant_rolling_save.exists():
            extant_rolling_save.rename(f"{fname}-{idx+1}")
    orig_file = Path(fname)
    if orig_file.exists():
        orig_file.rename(f"{fname}-1")


@typechecked
def get_previous_basename(basename: str, previous_version: int = 0) -> str:
    if previous_version:
        basename = f"{basename}-{previous_version}"
    return basename


@typechecked
def get_model_save_basename(
    model_name: str, strategy: Optional[ResumeStrategy] = None, previous_version: int = 0
) -> str:
    if strategy is None:
        strategy = DEFAULT_SAVE_STRATEGY
    basename_suffix = ""
    if strategy == ResumeStrategy.BEST:
        basename_suffix = f".{ResumeStrategy.BEST}"
    basename = f"model_{model_name}{basename_suffix}.pth"
    basename = get_previous_basename(basename, previous_version)
    return basename


@typechecked
def get_checkpoint_basename(strategy: Optional[ResumeStrategy] = None, previous_version: int = 0) -> str:
    if strategy is None:
        strategy = DEFAULT_SAVE_STRATEGY
    basename_suffix = ""
    if strategy == ResumeStrategy.BEST:
        basename_suffix = f".{ResumeStrategy.BEST}"
    basename = f"checkpoint{basename_suffix}.tar"
    basename = get_previous_basename(basename, previous_version)
    return basename


@typechecked
def get_epoch_stamped_basename(basename: str, epoch: int) -> str:
    """Return a human-epoch-stamped save filename while preserving the original suffix."""
    human_epoch = epoch + 1
    path = Path(basename)
    return f"{path.stem}.epoch{human_epoch:04d}{path.suffix}"


def _link_or_copy(src: str, dst: str) -> None:
    """Create a cheap stable alias when possible, falling back to a copy."""
    dst_path = Path(dst)
    if dst_path.exists():
        dst_path.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def save_models(
    models_state_dict: Dict,
    checkpoint: Optional[Dict],
    folder: str,
    save_to_s3: bool,
    strategy: Optional[ResumeStrategy] = DEFAULT_SAVE_STRATEGY,
    num_rolling_saves=DEFAULT_ROLLING_SAVES,
) -> None:
    """Save model state_dict and checkpoint.
    models_state_dict: dict
        models' state_dict to save
    checkpoint: dict
        The checkpoint includes the state_dict for optimizer, epoch info
    folder: str
        The dir to save the model
    strategy: ResumeStrategy
        For saving different variants of a model; eg "best"
    num_rolling_saves: int
        Retain this many old copies of the saved model files; each file will be suffixed with a counter
    """
    print("Saving model to {}".format(folder))
    if os.environ.get("RAD_INTERRUPT_ACTIVE"):
        # Saving is not atomic, so do not do it here
        print("Skipping save due to interrupt")
        return

    os.makedirs(folder, exist_ok=True)
    saved_model_files = []
    for model_name, model_state_dict in models_state_dict.items():
        model_basename = get_model_save_basename(model_name, strategy)
        fname_model = os.path.join(folder, model_basename)
        update_rolling_backups(fname_model, num_rolling_saves)
        torch.save(model_state_dict, fname_model)
        saved_model_files.append((model_basename, fname_model))
    if checkpoint:
        checkpoint_basename = get_checkpoint_basename(strategy)
        fname_checkpoint = os.path.join(folder, checkpoint_basename)
        update_rolling_backups(fname_checkpoint, num_rolling_saves)
        torch.save(checkpoint, fname_checkpoint)
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            checkpoint_epoch_basename = get_epoch_stamped_basename(checkpoint_basename, epoch)
            _link_or_copy(fname_checkpoint, os.path.join(folder, checkpoint_epoch_basename))
            for model_basename, fname_model in saved_model_files:
                model_epoch_basename = get_epoch_stamped_basename(model_basename, epoch)
                _link_or_copy(fname_model, os.path.join(folder, model_epoch_basename))
            print(f"Saved epoch-stamped model aliases for epoch {epoch + 1}: {checkpoint_epoch_basename}")
    if save_to_s3:
        sync_model_to_s3(folder)


def copy_to_cpu(t: torch.Tensor):
    """Copy tensor to CPU, if it's already on cpu, make a copy to avoid it get changed when saving."""
    if t.device.type == "cpu":
        return torch.clone(t)
    return t.cpu()


class AsyncModelSaver(threading.Thread):
    """This class save the model and checkpoint while training asynchronous.
    Main thread (training loop) will call set_model_to_save() to set new model to save, make a copy of the model in cpu,
    then move on.
    The saver thread will check if there's any model saved, then do the writing, then loop over.
    """

    def __init__(self, rolling_saves=DEFAULT_ROLLING_SAVES, timeout: int = 10):
        super().__init__(daemon=True)
        self._pending_models_state_dict = {}
        self._pending_checkpoint = {}
        self._options = {}
        self._lock = threading.Lock()
        self.should_stop = False
        self.wait_period = 1
        self.timeout = timeout
        self.num_rolling_saves = rolling_saves
        self.start()

    def set_model_to_save(
        self,
        models: Dict[str, torch.nn.Module],
        checkpoint: Optional[Dict],
        folder: str,
        save_to_s3: bool,
        strategy: Optional[ResumeStrategy] = None,
    ):
        """Set the model to be saved.
        Note: The newly set model will override the current set model, pending to save (but not yet saved).

        models_state_dict: dict
            This must be the return from model.state_dict()
        """
        models_state_dict = {}
        for model_name, model in models.items():
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            # Copy model state_dict without move model from cuda to cpu.
            models_state_dict[model_name] = {k: copy_to_cpu(param) for k, param in model.state_dict().items()}

        with self._lock:
            # Override current pending model.
            self._pending_models_state_dict[folder] = models_state_dict
            self._pending_checkpoint[folder] = checkpoint
            self._options[folder] = {"save_to_s3": save_to_s3, "strategy": strategy}

    def save_model_sync(
        self,
        models: Dict[str, torch.nn.Module],
        checkpoint: Optional[Dict],
        folder: str,
        save_to_s3: bool,
        strategy: Optional[ResumeStrategy] = None,
    ):
        models_state_dict = {}
        for model_name, model in models.items():
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            # Copy model state_dict without move model from cuda to cpu.
            models_state_dict[model_name] = {k: param for k, param in model.state_dict().items()}

        save_models(
            models_state_dict,
            checkpoint,
            folder,
            save_to_s3,
            strategy=strategy,
            num_rolling_saves=self.num_rolling_saves,
        )

    def run(self):
        """Saver thread loop.
        Infinite loop, until self.should_stop is true.
        """
        while not self.should_stop:
            try:
                model_to_save = None
                checkpoints = None
                with self._lock:
                    if self._pending_models_state_dict:
                        # Grab and pin the model to save
                        model_to_save = self._pending_models_state_dict
                        checkpoints = self._pending_checkpoint
                        options = self._options
                        self._pending_models_state_dict = {}
                        self._pending_checkpoint = {}
                        self._options = {}
                if model_to_save:
                    # Note, there's only one thread does the writing (this thread), so no lock is needed.
                    for folder in model_to_save:
                        save_models(
                            model_to_save[folder],
                            checkpoints.get(folder, None),
                            folder,
                            **options[folder],
                            num_rolling_saves=self.num_rolling_saves,
                        )
                time.sleep(self.wait_period)
            except Exception as e:
                logging.error("Error in AsyncModelSaver, error: %s", str(e))

    def __del__(self):
        self.should_stop = True
        self.join(timeout=self.timeout)
