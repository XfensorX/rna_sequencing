import os


def get_project_root_path() -> str:
    return os.popen(cmd="git rev-parse --show-toplevel").read().strip()


def get_data_dir_path() -> str:
    return os.path.join(get_project_root_path(), "data")


def get_test_data_path() -> str:
    return os.path.join(get_data_dir_path(), "splits", "test")
