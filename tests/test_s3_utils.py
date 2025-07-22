import unittest
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestS3Utils(unittest.TestCase):
    """Test cases for s3_utils module"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Clean up any existing mocks
        pass
        
    def tearDown(self):
        """Clean up after each test method."""
        pass

    @patch('tools.s3_utils.EVFSAMLogger')
    @patch('tools.s3_utils.Path')
    def test_download_evf_sam_existing_folder(self, mock_path_class, mock_logger_class):
        """
        Test that download_evf_sam_from_s3 returns immediately and does not attempt to download
        if the EVF-SAM folder already exists locally.

        This test verifies:
        - The function checks for the existence of the "EVF-SAM" directory.
        - If the directory exists, it logs a message and returns the path as a string.
        - No S3 or file operations are performed.
        """
        from tools.s3_utils import download_evf_sam_from_s3
        
        mock_logger_instance = MagicMock()
        mock_logger_class.return_value = mock_logger_instance
        
        mock_evf_sam_path = MagicMock()
        mock_evf_sam_path.exists.return_value = True
        mock_evf_sam_path.is_dir.return_value = True
        mock_path_class.return_value = mock_evf_sam_path
        
        mock_evf_sam_path.__str__.return_value = "EVF-SAM"
        
        result = download_evf_sam_from_s3()
        
        self.assertEqual(result, "EVF-SAM")
        mock_path_class.assert_called_once_with("EVF-SAM")
        mock_logger_instance.log.assert_called_once_with(
            "EVF-SAM folder already exists locally. Skipping download."
        )

    @patch('tools.s3_utils.EVFSAMLogger')
    @patch('tools.s3_utils.Path')
    @patch('tools.s3_utils.get_env_variable')
    def test_download_evf_sam_missing_credentials(self, mock_get_env, mock_path_class, mock_logger_class):
        """
        Test that download_evf_sam_from_s3 raises a ValueError and logs an error
        when required AWS credentials are missing from the environment.

        This test verifies:
        - The function checks for AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_REGION.
        - If any are missing (None), it logs an error and raises ValueError.
        - No S3 or file operations are performed.
        """
        from tools.s3_utils import download_evf_sam_from_s3
        
        mock_logger_instance = MagicMock()
        mock_logger_class.return_value = mock_logger_instance
        
        mock_evf_sam_path = MagicMock()
        mock_evf_sam_path.exists.return_value = False
        mock_evf_sam_path.is_dir.return_value = False
        mock_path_class.return_value = mock_evf_sam_path
        
        mock_get_env.return_value = None
        
        with self.assertRaises(ValueError) as context:
            download_evf_sam_from_s3()
        
        self.assertIn("Missing AWS credentials", str(context.exception))
        mock_logger_instance.log.assert_called_with(
            "Missing AWS credentials in .env file"
        )

    @patch('tools.s3_utils.EVFSAMLogger')
    @patch('tools.s3_utils.boto3')
    @patch('tools.s3_utils.Path')
    @patch('tools.s3_utils.get_env_variable')
    def test_download_evf_sam_successful_download(self, mock_get_env, mock_path_class, mock_boto3, mock_logger_class):
        """
        Test that download_evf_sam_from_s3 successfully downloads files from S3
        when the EVF-SAM folder does not exist locally and all AWS credentials are present.

        This test verifies:
        - The function creates an S3 client with the correct credentials.
        - It paginates through the S3 bucket and downloads all non-directory files.
        - It logs each download and a summary message.
        - The function returns the local path string after successful download.
        """
        from tools.s3_utils import download_evf_sam_from_s3
        
        mock_logger_instance = MagicMock()
        mock_logger_class.return_value = mock_logger_instance
        
        mock_evf_sam_path = MagicMock()
        mock_evf_sam_path.exists.return_value = False
        mock_evf_sam_path.is_dir.return_value = False
        mock_evf_sam_path.__str__.return_value = "EVF-SAM"
        
        mock_local_path = MagicMock()
        mock_local_path.parent.mkdir = MagicMock()
        mock_local_path.__str__.return_value = "EVF-SAM/test_file.py"
        
        def path_side_effect(arg):
            if arg == "EVF-SAM":
                return mock_evf_sam_path
            else:
                return mock_local_path
        mock_path_class.side_effect = path_side_effect
        
        def env_side_effect(key):
            env_vars = {
                "AWS_ACCESS_KEY_ID": "test_access_key",
                "AWS_SECRET_ACCESS_KEY": "test_secret_key", 
                "AWS_S3_REGION": "us-east-1"
            }
            return env_vars.get(key)
        mock_get_env.side_effect = env_side_effect
        
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {'Key': 'models/EVF-SAM/test_file.py'},
                    {'Key': 'models/EVF-SAM/subfolder/'},
                    {'Key': 'models/EVF-SAM/another_file.txt'}
                ]
            }
        ]
        
        result = download_evf_sam_from_s3()
        
        self.assertEqual(result, "EVF-SAM")
        mock_boto3.client.assert_called_once_with(
            's3',
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            region_name="us-east-1"
        )
        self.assertEqual(mock_s3_client.download_file.call_count, 2)
        expected_log_calls = [
            call("Downloading EVF-SAM from S3 bucket: artifactsredi/models/EVF-SAM/"),
            call("Downloading: models/EVF-SAM/test_file.py -> EVF-SAM/test_file.py"),
            call("Downloading: models/EVF-SAM/another_file.txt -> EVF-SAM/test_file.py"),
            call("Successfully downloaded 2 files from S3")
        ]
        mock_logger_instance.log.assert_has_calls(expected_log_calls, any_order=False)

    @patch('tools.s3_utils.EVFSAMLogger')
    @patch('tools.s3_utils.boto3')
    @patch('tools.s3_utils.Path')
    @patch('tools.s3_utils.get_env_variable')
    def test_download_evf_sam_no_files_found(self, mock_get_env, mock_path_class, mock_boto3, mock_logger_class):
        """
        Test that download_evf_sam_from_s3 raises a RuntimeError and logs an error
        when no files are found in the S3 bucket for the given prefix.

        This test verifies:
        - The function paginates through the S3 bucket but finds no 'Contents' key.
        - It logs a message indicating no files were found.
        - It raises a RuntimeError with an appropriate message.
        """
        from tools.s3_utils import download_evf_sam_from_s3
        
        mock_logger_instance = MagicMock()
        mock_logger_class.return_value = mock_logger_instance
        
        mock_evf_sam_path = MagicMock()
        mock_evf_sam_path.exists.return_value = False
        mock_evf_sam_path.is_dir.return_value = False
        mock_path_class.return_value = mock_evf_sam_path
        
        def env_side_effect(key):
            env_vars = {
                "AWS_ACCESS_KEY_ID": "test_access_key",
                "AWS_SECRET_ACCESS_KEY": "test_secret_key",
                "AWS_S3_REGION": "us-east-1"
            }
            return env_vars.get(key)
        mock_get_env.side_effect = env_side_effect
        
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]
        
        with self.assertRaises(RuntimeError) as context:
            download_evf_sam_from_s3()
        
        self.assertIn("No files found in S3 bucket", str(context.exception))
        mock_logger_instance.log.assert_any_call(
            "No files found in S3 bucket artifactsredi with prefix models/EVF-SAM/"
        )

    @patch('tools.s3_utils.EVFSAMLogger')
    @patch('tools.s3_utils.boto3')
    @patch('tools.s3_utils.Path')
    @patch('tools.s3_utils.get_env_variable')
    def test_download_evf_sam_s3_error_with_local_fallback(self, mock_get_env, mock_path_class, mock_boto3, mock_logger_class):
        """
        Test that download_evf_sam_from_s3 handles an S3 error gracefully by falling back
        to an existing local EVF-SAM folder if available.

        This test verifies:
        - The function attempts to paginate S3 but encounters an exception.
        - It logs the error.
        - It checks again for the local folder, and if it now exists, logs a fallback message and returns the path.
        - No files are downloaded from S3.
        """
        from tools.s3_utils import download_evf_sam_from_s3
        
        mock_logger_instance = MagicMock()
        mock_logger_class.return_value = mock_logger_instance
        
        mock_evf_sam_path = MagicMock()
        mock_evf_sam_path.exists.side_effect = [False, True]
        mock_evf_sam_path.__str__.return_value = "EVF-SAM"
        mock_path_class.return_value = mock_evf_sam_path
        
        def env_side_effect(key):
            env_vars = {
                "AWS_ACCESS_KEY_ID": "test_access_key",
                "AWS_SECRET_ACCESS_KEY": "test_secret_key",
                "AWS_S3_REGION": "us-east-1"
            }
            return env_vars.get(key)
        mock_get_env.side_effect = env_side_effect
        
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = Exception("S3 connection failed")
        
        result = download_evf_sam_from_s3()
        
        self.assertEqual(result, "EVF-SAM")
        mock_logger_instance.log.assert_any_call(
            "Error downloading EVF-SAM from S3: S3 connection failed"
        )
        mock_logger_instance.log.assert_any_call(
            "Using existing local EVF-SAM folder"
        )

    @patch('tools.s3_utils.EVFSAMLogger')
    @patch('tools.s3_utils.boto3')
    @patch('tools.s3_utils.Path')
    @patch('tools.s3_utils.get_env_variable')
    def test_download_evf_sam_s3_error_no_local_fallback(self, mock_get_env, mock_path_class, mock_boto3, mock_logger_class):
        """
        Test that download_evf_sam_from_s3 raises a RuntimeError if an S3 error occurs
        and there is no local EVF-SAM folder to fall back on.

        This test verifies:
        - The function attempts to paginate S3 but encounters an exception.
        - It logs the error.
        - It checks for the local folder, finds it does not exist, and raises a RuntimeError.
        """
        from tools.s3_utils import download_evf_sam_from_s3
        
        mock_logger_instance = MagicMock()
        mock_logger_class.return_value = mock_logger_instance
        
        mock_evf_sam_path = MagicMock()
        mock_evf_sam_path.exists.return_value = False
        mock_path_class.return_value = mock_evf_sam_path
        
        def env_side_effect(key):
            env_vars = {
                "AWS_ACCESS_KEY_ID": "test_access_key",
                "AWS_SECRET_ACCESS_KEY": "test_secret_key",
                "AWS_S3_REGION": "us-east-1"
            }
            return env_vars.get(key)
        mock_get_env.side_effect = env_side_effect
        
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = Exception("S3 connection failed")
        
        with self.assertRaises(RuntimeError) as context:
            download_evf_sam_from_s3()
        
        self.assertIn("Failed to download EVF-SAM from S3 and no local copy exists", str(context.exception))
        mock_logger_instance.log.assert_any_call(
            "Error downloading EVF-SAM from S3: S3 connection failed"
        )


if __name__ == '__main__':
    unittest.main() 