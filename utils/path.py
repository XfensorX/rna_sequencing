import os


def get_project_root_path() -> str:
    return os.popen(cmd="git rev-parse --show-toplevel").read().strip()


def get_data_dir_path() -> str:
    return os.path.join(get_project_root_path(), "data")
