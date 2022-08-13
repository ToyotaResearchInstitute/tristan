import os


def test_public_repo_proto_files():
    for file in [
        "triceps/triceps/protobuf/prediction_training.proto",
    ]:
        assert os.path.exists(file)
