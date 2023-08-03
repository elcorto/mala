"""Test whether the examples are still working."""
import importlib
import runpy
import os

import pytest


# There is https://github.com/RKrahl/pytest-dependency and
# https://github.com/pytest-dev/pytest-order but we like to keep in simple
# here. When running in parallel (pytest-xdist), we execute ex01 at most twice,
# which is OK.
def check_and_create_ex01_artifact():
    if not os.path.exists("be_model.zip"):
        print("Running example basic/ex01 in preparation another example")
        runpy.run_path("../examples/basic/ex01_train_network.py")


@pytest.mark.examples
class TestExamples:
    def test_basic_ex01(self):
        runpy.run_path("../examples/basic/ex01_train_network.py")

    def test_basic_ex02(self):
        check_and_create_ex01_artifact()
        runpy.run_path("../examples/basic/ex02_test_network.py")

    def test_basic_ex03(self):
        runpy.run_path("../examples/basic/ex03_preprocess_data.py")

    def test_basic_ex04(self):
        runpy.run_path("../examples/basic/ex04_hyperparameter_optimization.py")

    def test_basic_ex05(self):
        check_and_create_ex01_artifact()
        runpy.run_path("../examples/basic/ex05_run_predictions.py")

    def test_basic_ex06(self):
        check_and_create_ex01_artifact()
        runpy.run_path("../examples/basic/ex06_ase_calculator.py")

    def test_advanced_ex01(self):
        runpy.run_path("../examples/advanced/ex01_checkpoint_training.py")

    def test_advanced_ex02(self):
        runpy.run_path("../examples/advanced/ex02_shuffle_data.py")

    def test_advanced_ex03(self):
        runpy.run_path("../examples/advanced/ex03_tensor_board.py")

    def test_advanced_ex04(self):
        runpy.run_path("../examples/advanced/ex04_acsd.py")

    def test_advanced_ex05(self):
        runpy.run_path("../examples/advanced/ex05_checkpoint_hyperparameter_optimization.py")

    def test_advanced_ex06(self):
        runpy.run_path("../examples/advanced/ex06_distributed_hyperparameter_optimization.py")

    @pytest.mark.skipif(importlib.util.find_spec("oapackage") is None,
                        reason="No OAT found on this machine, skipping this "
                               "test.")
    def test_advanced_ex07(self):
        runpy.run_path("../examples/advanced/ex07_advanced_hyperparameter_optimization.py")

    def test_advanced_ex08(self):
        runpy.run_path("../examples/advanced/ex08_visualize_observables.py")

