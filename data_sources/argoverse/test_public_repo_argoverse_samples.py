import os


def test_public_repo_argoverse_samples():
    for file in [
        "data_sources/argoverse/dummy_data/training/argoverse_dummy_0.pb",
        "data_sources/argoverse/dummy_data/training/argoverse_dummy_1.pb",
        "data_sources/argoverse/dummy_data/validation/argoverse_dummy_2.pb",
        "data_sources/argoverse/dummy_data/validation/argoverse_dummy_3.pb",
    ]:
        assert os.path.exists(file)
