import json
import os
import tempfile
from typing import Any, Dict, List

from typeguard import typechecked


@typechecked
def get_path_to_session_data(session_id: str) -> str:
    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, f"intent_prediction_session_data_{session_id}.json")
    return file_path


@typechecked
def read_session_data(session_id: str) -> Dict:
    file_path = get_path_to_session_data(session_id)
    if os.path.exists(file_path):
        try:
            with open(file_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            print(f"Read session data from {file_path}")
            return data
        except (FileNotFoundError, json.JSONDecodeError, AssertionError):
            print(f"Could not read session data from {file_path}")
            return {}
    else:
        print(f"Session data not found at {file_path}")
        return {}


@typechecked
def read_session_value(session_id: str, session_data: Dict, key: str) -> Any:
    if key not in session_data:
        # Give better debug message than key error
        raise ValueError(f"Key '{key}' not found for session {session_id}, data {session_data}")
    val = session_data[key]
    return val


@typechecked
def save_session_values(session_id: str, params: Dict, keys_list: List[str] = None) -> None:
    if keys_list:
        # Make dense by defaulting to None
        params = {k: params.get(k) for k in keys_list}
    file_path = get_path_to_session_data(session_id)
    session_data = read_session_data(session_id)
    session_data.update(params)
    with open(file_path, "w") as f:
        json.dump(session_data, f)


@typechecked
def save_session_value(session_id: str, key: str, val: str) -> None:
    save_session_values(session_id, {key: val})


@typechecked
def get_saved_path(session_id: str, session_data: Dict, attribute: str) -> str:
    saved_path = read_session_value(session_id, session_data, attribute)
    saved_path = os.path.normpath(os.path.expanduser(saved_path))
    return saved_path
