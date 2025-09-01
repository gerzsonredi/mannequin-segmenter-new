"""
Pytest configuration file for mannequin-segmenter tests.
This file contains shared fixtures and configuration for all tests.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_env_variables():
    """
    Fixture that provides a dictionary of mock environment variables commonly used for GCP and related configuration.

    Returns:
        dict: A dictionary containing mock values for:
            - GCP_PROJECT_ID
            - GCP_SA_KEY
            - GCP_BUCKET_NAME

    This fixture is useful for tests that require GCP credentials or configuration without using real secrets.
    """
    return {
        "GCP_PROJECT_ID": "test-project-id",
        "GCP_SA_KEY": "dGVzdC1zZXJ2aWNlLWFjY291bnQta2V5",  # base64 encoded test key
        "GCP_BUCKET_NAME": "test-bucket-name"
    }


@pytest.fixture
def mock_gcs_client():
    """
    Fixture that provides a mock GCS client object.

    Returns:
        MagicMock: A mock object simulating a Google Cloud Storage client, with the following methods mocked:
            - bucket
            - blob operations

    This fixture allows tests to simulate GCS interactions without making real GCP calls.
    """
    client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.upload_from_file = MagicMock()
    mock_blob.download_to_file = MagicMock()
    return client


@pytest.fixture
def mock_inferencer():
    """
    Fixture that provides a mock model inferencer instance.

    Returns:
        MagicMock: A mock inferencer object with the 'process_image_url' method mocked.

    Use this fixture to test code that depends on the model inferencer without running actual inference.
    """
    inferencer = MagicMock()
    inferencer.process_image_url = MagicMock()
    return inferencer


@pytest.fixture
def mock_logger():
    """
    Fixture that provides a mock logger object for AppLogger.

    Returns:
        MagicMock: A mock logger with the 'log' method mocked.

    This fixture is useful for capturing and asserting log calls in tests.
    """
    logger = MagicMock()
    logger.log = MagicMock()
    return logger


@pytest.fixture
def temp_directory():
    """
    Fixture that creates a temporary directory for use during a test and cleans it up afterwards.

    Yields:
        Path: A pathlib.Path object pointing to the temporary directory.

    The directory is automatically removed after the test completes, ensuring no leftover files.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_gcs_storage():
    """
    Fixture that provides a mock Google Cloud Storage module.

    Returns:
        MagicMock: A mock object simulating the google.cloud.storage module.

    Use this fixture to test code that interacts with GCS without making real GCP calls.
    """
    storage_mock = MagicMock()
    return storage_mock


@pytest.fixture(scope="session")
def project_root_path():
    """
    Fixture that provides the absolute path to the project root directory.

    Returns:
        Path: A pathlib.Path object pointing to the root of the project (parent of the tests directory).

    This can be used in tests that need to resolve paths relative to the project root.
    """
    return Path(__file__).parent.parent


# Configure pytest to handle warnings
@pytest.fixture(autouse=True)
def configure_warnings():
    """
    Auto-used fixture to configure warning handling for all tests.

    This fixture suppresses DeprecationWarning and PendingDeprecationWarning messages
    during test runs to reduce noise from dependencies or deprecated code paths.

    It is automatically applied to all tests in the session.
    """
    import warnings
    # Filter out specific warnings that may occur during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 