import pytest
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--dataroot", action="store", help="Root folder for dataset files.")
    parser.addoption("--dataset-num-workers", action="store", help="Number of parallel workers to process datasets."
                                                                   "Set to 0 to disable multiprocess.")


@pytest.fixture
def dataroot(request):
    return request.config.getoption("--dataroot", default="data/")


@pytest.fixture
def dataset_num_workers(request):
    return request.config.getoption("--dataset-num-workers", default=0)

@pytest.fixture
def V8_path(dataroot):
    return str(Path(dataroot, "V8/"))


@pytest.fixture
def lego_path(dataroot):
    return str(Path(dataroot, "lego/"))