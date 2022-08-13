""" Miscellaneous utilities that don't fit any other category (yet) """
import hashlib
import os
import time
import uuid
from subprocess import check_output
from typing import Optional, Tuple

from google.protobuf.timestamp_pb2 import Timestamp
from typeguard import typechecked

import radutils.torch.async_saver as saver


def get_current_commit_hash() -> str:
    """Get commit hash so that we log which codebase is training.

    Returns
    -------
    str
        The abbreviated git commmit hash of HEAD.
    """
    return check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()


def get_current_branch() -> str:
    """Get git branch so that we log which codebase is training.

    Returns
    -------
    str
        The name of the current git branch
    """
    return check_output(["git", "branch", "--points-at=HEAD", "--format=%(refname:short)"]).decode().strip()


@typechecked
def generate_session_id(param_level: Optional[int], session_name: Optional[str]) -> str:
    """Generates a session id.
    Returns
    -------
    str
        A session id of the format: TIME-COMMIT-HASH[-SESSION_NAME--PARAM_LEVEL].
        An example id is 03-01T11:23:24-d894a7-d507c-Baseline--2

        Reasoning
            For time, for example '03-31T11:23:24' is generally a ISO format, year is removed for conciseness.
            '--' before PARAM_LEVEL is used to avoid having to deal with any '-' used in the SESSION_NAME, which could contain '-'.
            For example '03-01T11:23:24-d894a7-d507c-Baseline-0--2'

        If param_level and session_name isn't specified, it won't be appended.
        If session_name is specified but not param_level, session_name will be appended.
        If param_level is specified but not session_name, session_name will be set to 'session', then both will be appended.
    """
    cur_hash = hashlib.md5(str(uuid.uuid1()).encode()).hexdigest()[:5]
    commit = get_current_commit_hash()
    timestamp = time.strftime("%m-%dT%H_%M_%S")

    session_id = f"{timestamp}-{commit}-{cur_hash}"

    if param_level is not None:
        if session_name is None:
            # Must set param_level when using session_name
            session_name = "session"

    if session_name:
        session_id += f"-{session_name}"
        if param_level:
            session_id += f"--{param_level}"
    return session_id


@typechecked
def create_or_resume_session(
    param_level: int = None,
    session_name: str = None,
    resume_session_name: str = None,
    current_session_name: str = None,
    path_to_model_directory: str = None,
    use_best_fte: bool = True,
) -> str:
    if resume_session_name:
        saver.sync_models_from_s3([resume_session_name], path_to_model_directory, use_best_fte)
    session_id = current_session_name or generate_session_id(param_level, session_name)
    os.environ["RAD_SESSION_ID"] = session_id
    return session_id


@typechecked
def parse_session_name(session_id: str) -> Tuple[str, str, str, Optional[str], Optional[int]]:
    """Extracts the different session name components

    Parameters
    ----------
    session_id : str
        The name of the session which is of the format TIME-COMMIT-HASH[-SESSION_NAME--PARAM_LEVEL].
        An example session name is 03-31T11:23:24-d894a7-d507c-Baseline--2. The suffix
        is optional.

    Returns
    -------
    str
        The time_str session time
    str
        The git commit hash of the session
    str
        The uuid of the session.
    str
        The session name
    int
        The param_level
    """
    param_level_split = session_id.split("--")
    param_level = None
    if len(param_level_split) > 1:
        try:
            param_level = int(param_level_split[-1])
            name = "--".join(param_level_split[:-1])
        except ValueError:
            # Error converting param_level to int, assume no param level
            name = session_id
    else:
        # No param level
        name = session_id
    name_components = name.split("-", maxsplit=4)

    time_str = f"{name_components[0]}-{name_components[1]}"
    commit = name_components[2]
    uuid = name_components[3]
    if len(name_components) > 4:
        session_name = name_components[4]
    else:
        session_name = None

    return time_str, commit, uuid, session_name, param_level


@typechecked
def get_param_level_from_session_id(session_id: Optional[str]) -> int:
    """Always returns a valid param level, defaulting to 0 if the session ID in invalid or field is missing"""
    if not session_id:
        return 0
    session_components = parse_session_name(session_id)
    param_level = session_components[-1] or 0
    return param_level


def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return os.path.relpath(text, start=prefix)
    return text


def parse_protobuf_timestamp(timestamp_as_string: str):
    # Protobuf has a greater sub-second resolution than python, so we need to use their parser
    timestamp = Timestamp()
    timestamp.FromJsonString(timestamp_as_string)

    return timestamp
