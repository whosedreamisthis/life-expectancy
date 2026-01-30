import os

from le_package import config
from le_package.train import run_training


def test_run_training_saves_model():
    """
    Integration test to ensure the training pipeline runs
    and saves a model file to the disk.
    """
    # 1. Arrange
    # Ensure the models directory exists (usually handled by the project structure)
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    save_file_name = f"{config.MODEL_DIR}/{config.P}.pkl"

    # Remove old model if it exists to ensure a clean test
    if os.path.exists(save_file_name):
        os.remove(save_file_name)

    # 2. Act
    run_training()

    # 3. Assert
    # Check that the file was actually created
    assert os.path.exists(save_file_name)

    # Optional: Check that the file size is greater than 0 (not a corrupted save)
    assert os.path.getsize(save_file_name) > 0
