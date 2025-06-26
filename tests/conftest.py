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
    Fixture that provides a dictionary of mock environment variables commonly used for AWS and related configuration.

    Returns:
        dict: A dictionary containing mock values for:
            - AWS_ACCESS_KEY_ID
            - AWS_SECRET_ACCESS_KEY
            - AWS_S3_BUCKET_NAME
            - AWS_S3_REGION

    This fixture is useful for tests that require AWS credentials or configuration without using real secrets.
    """
    return {
        "AWS_ACCESS_KEY_ID": "test_access_key_id",
        "AWS_SECRET_ACCESS_KEY": "test_secret_access_key",
        "AWS_S3_BUCKET_NAME": "test-bucket-name",
        "AWS_S3_REGION": "us-east-1"
    }


@pytest.fixture
def mock_s3_client():
    """
    Fixture that provides a mock S3 client object.

    Returns:
        MagicMock: A mock object simulating a boto3 S3 client, with the following methods mocked:
            - upload_fileobj
            - download_file
            - get_paginator

    This fixture allows tests to simulate S3 interactions without making real AWS calls.
    """
    client = MagicMock()
    client.upload_fileobj = MagicMock()
    client.download_file = MagicMock()
    client.get_paginator = MagicMock()
    return client


@pytest.fixture
def mock_evfsam_inferencer():
    """
    Fixture that provides a mock EVFSAMSingleImageInferencer instance.

    Returns:
        MagicMock: A mock inferencer object with the 'process_image_url' method mocked.

    Use this fixture to test code that depends on the EVFSAMSingleImageInferencer without running actual inference.
    """
    inferencer = MagicMock()
    inferencer.process_image_url = MagicMock()
    return inferencer


@pytest.fixture
def mock_logger():
    """
    Fixture that provides a mock logger object for EVFSAMLogger.

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
def mock_boto3_session():
    """
    Fixture that provides a mock boto3 session object.

    Returns:
        MagicMock: A mock object simulating a boto3 session.

    Use this fixture to test code that interacts with boto3 sessions without making real AWS calls.
    """
    session = MagicMock()
    return session


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