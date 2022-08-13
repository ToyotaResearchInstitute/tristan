import argparse

import pytest

from triceps.protobuf.proto_arguments import ensure_command_param, parse_arguments


def test_ensure_command_param() -> None:
    # '=' form
    arg_list = ["fake-bin", "-arg=val"]
    ensure_command_param(arg_list, "-arg", "val2")
    assert arg_list == ["fake-bin", "-arg=val2"]

    arg_list = ["fake-bin", "-arg=val"]
    ensure_command_param(arg_list, "-arg2", "val2", param_alias="-arg")
    assert arg_list == ["fake-bin", "-arg2=val2"]

    # 'after' form
    arg_list = ["fake-bin", "-arg", "val"]
    ensure_command_param(arg_list, "-arg", "val2")
    assert arg_list == ["fake-bin", "-arg", "val2"]

    arg_list = ["fake-bin", "-arg", "val"]
    ensure_command_param(arg_list, "-arg2", "val2", param_alias="-arg")
    assert arg_list == ["fake-bin", "-arg2", "val2"]

    # Append
    arg_list = ["fake-bin"]
    ensure_command_param(arg_list, "-arg", "val")
    assert arg_list == ["fake-bin", "-arg", "val"]

    # Missing value
    with pytest.raises(ValueError):
        arg_list = ["fake-bin", "-arg=val"]
        ensure_command_param(arg_list, "-arg2", param_alias="-arg")

    # Assign value to flag
    with pytest.raises(ValueError):
        arg_list = ["fake-bin", "-arg", "-arg2"]
        ensure_command_param(arg_list, "-arg2", "val", param_alias="-arg")

    # Past the end val
    with pytest.raises(ValueError):
        arg_list = ["fake-bin", "-arg"]
        ensure_command_param(arg_list, "-arg", "val")

    # Past the end val
    with pytest.raises(ValueError):
        arg_list = ["fake-bin", "-arg"]
        ensure_command_param(arg_list, "-arg2", "val", param_alias="-arg")


class TestParamSets:
    def setup_method(self, _):
        self.mock_param_set_name = "mock_param_set"
        self.mock_compatible_param_set_name = "mock_compatible_param_set"
        self.mock_incompatible_param_set_name = "mock_incompatible_param_set"
        self.mock_param_set = {
            "dataset_names": ["mock_dataset"],
            "cache_dir": "fake/cache/dir",
            "agent_image_mode": "none",
        }
        self.mock_compatible_param_set = {
            "dataset_names": ["mock_dataset", "mock_compatible_dataset"],  # Arrays should merge, discarding duplicates
            "cache_dir": "fake/cache/dir",  # Matching params
            "scene_image_mode": "none",  # Not set in first param set
        }
        self.mock_incompatible_param_set = {
            "dataset_names": ["mock_incompatible_dataset"],
            "cache_dir": "incompatible/cache/dir",  # Conflicts with other param sets
        }

    def mock_arg_setter(self, parser: argparse.ArgumentParser):
        parser.__dict__["param_sets"][self.mock_param_set_name] = self.mock_param_set
        parser.__dict__["param_sets"][self.mock_compatible_param_set_name] = self.mock_compatible_param_set
        parser.__dict__["param_sets"][self.mock_incompatible_param_set_name] = self.mock_incompatible_param_set

        return parser

    def test_single_param_set(self):
        expected_params = {
            **vars(parse_arguments([])),
            **self.mock_param_set,
            "param_set": [self.mock_param_set_name],
            "provided_args": ["--param-set", self.mock_param_set_name, "--current-session-name=custom_session_name"],
            "current_session_name": "custom_session_name",
        }
        actual_params = vars(
            parse_arguments(
                [
                    "--param-set",
                    self.mock_param_set_name,
                    "--current-session-name=custom_session_name",  # Not set by any param sets
                ],
                additional_arguments_setter=[self.mock_arg_setter],
            )
        )

        assert actual_params == expected_params

    def test_compatible_param_sets(self):
        """Tests that two compatible parameter sets are merged with the user commands, correctly"""
        expected_params = {
            **vars(parse_arguments([])),
            **self.mock_param_set,
            **self.mock_compatible_param_set,
            # Dataset names should be the union of both datasets, in order of first appearance
            "dataset_names": self.mock_param_set["dataset_names"] + self.mock_compatible_param_set["dataset_names"][1:],
            "param_set": [self.mock_param_set_name, self.mock_compatible_param_set_name],
            "provided_args": [
                "--param-set",
                self.mock_param_set_name,
                self.mock_compatible_param_set_name,
                "--current-session-name=custom_session_name",
            ],
            "current_session_name": "custom_session_name",
        }
        actual_params = vars(
            parse_arguments(
                [
                    "--param-set",
                    self.mock_param_set_name,
                    self.mock_compatible_param_set_name,
                    "--current-session-name=custom_session_name",  # Not set by any param sets
                ],
                additional_arguments_setter=[self.mock_arg_setter],
            )
        )

        assert actual_params == expected_params

    def test_incompatible_param_sets(self):
        """Tests that the parser will exit if incompatible param sets are called together"""
        with pytest.raises(SystemExit):
            parse_arguments(
                ["--param-set", self.mock_param_set_name, self.mock_incompatible_param_set_name],
                additional_arguments_setter=[self.mock_arg_setter],
            )

    def test_incompatible_user_params(self):
        """Tests that the parser will exit if user provides a parameter that conflicts with the param sets"""
        with pytest.raises(SystemExit):
            parse_arguments(
                [
                    "--param-set",
                    self.mock_param_set_name,
                    self.mock_compatible_param_set_name,
                    "--agent-image-mode=all",  # Set by first param to "none"
                ],
                additional_arguments_setter=[self.mock_arg_setter],
            )
