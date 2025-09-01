import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
import sys
import numpy as np
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api_app import create_app

class TestApiApp(unittest.TestCase):
    """
    Test suite for the Flask application created by the factory pattern in api_app.py.

    This class uses unittest and extensive mocking to isolate the Flask app from its
    external dependencies (GCP, model inference, logging, etc.), ensuring that only
    the API logic is tested. Each test method targets a specific endpoint or error
    scenario, and the setUp/tearDown methods manage the patching of dependencies.
    """

    def setUp(self):
        """
        Set up the test environment before each test method.

        - Patches external dependencies in api_app (get_env_variable, storage, AppLogger, 
          and load_dotenv) to prevent real GCP/model/logging/dotenv interactions.
        - Configures the mocks for GCS client, model inferencer, and logger.
        - Creates the Flask app in testing mode and injects the mock inferencer.
        - Initializes the Flask test client for making requests to the app.
        """
        # Patch external dependencies for the entire test class
        self.patcher_env = patch('api_app.get_env_variable', self._mock_get_env)
        self.patcher_storage = patch('api_app.storage')
        self.patcher_dotenv = patch('api_app.load_dotenv')

        self.mock_get_env = self.patcher_env.start()
        self.mock_boto3 = self.patcher_boto3.start()
        self.mock_inferencer_cls = self.patcher_inferencer.start()
        self.mock_logger_cls = self.patcher_logger.start()
        self.patcher_dotenv.start()

        # Configure the mocks
        self.mock_s3_client = MagicMock()
        self.mock_boto3.client.return_value = self.mock_s3_client
        self.mock_inferencer = MagicMock()
        self.mock_inferencer_cls.return_value = self.mock_inferencer
        # Configure the logger mock to have a valid level attribute
        self.mock_logger_cls.return_value.logger.level = logging.INFO
        
        # Create the app in testing mode
        self.app = create_app(testing=True)
        # Replace the placeholder inferencer with our mock
        self.app.config['INFERENCER'] = self.mock_inferencer
        
        self.client = self.app.test_client()

    def tearDown(self):
        """
        Clean up after each test method.

        Stops all patches started in setUp to ensure no side effects between tests.
        """
        self.patcher_env.stop()
        self.patcher_boto3.stop()
        self.patcher_inferencer.stop()
        self.patcher_logger.stop()
        self.patcher_dotenv.stop()

    def _mock_get_env(self, key):
        """
        Mock implementation for get_env_variable.

        Args:
            key (str): The environment variable key to retrieve.

        Returns:
            str or None: Returns a mock value for AWS-related environment variables,
                         or None for any other key.
        """
        return {
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret_key",
            "AWS_S3_BUCKET_NAME": "test-bucket",
            "AWS_S3_REGION": "us-east-1",
        }.get(key)

    def test_health_endpoint(self):
        """
        Test the /health endpoint for a successful response.

        This test mocks the datetime used in the endpoint to ensure a deterministic
        timestamp in the response. It verifies that the endpoint returns HTTP 200
        and a JSON body with status 'healthy'.
        """
        with patch('api_app.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = '2024-01-15T10:00:00'
            response = self.client.get('/health')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data)['status'], 'healthy')

    def test_infer_endpoint_missing_image_url(self):
        """
        Test the /infer endpoint when the required 'image_url' field is missing.

        Sends a POST request with an empty JSON body and asserts that the response
        status code is 400 (Bad Request) and the error message indicates that
        'image_url' was not provided.
        """
        response = self.client.post('/infer', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('image_url not provided', response.get_json()['error'])

    def test_infer_endpoint_invalid_prompt_mode(self):
        """
        Test the /infer endpoint when an invalid 'prompt_mode' value is provided.

        Sends a POST request with a valid 'image_url' but invalid 'prompt_mode'
        and asserts that the response status code is 400 (Bad Request) and the 
        error message indicates the prompt_mode is invalid.
        """
        response = self.client.post('/infer', json={
            'image_url': 'https://example.com/image.jpg',
            'prompt_mode': 'invalid_mode'
        })
        self.assertEqual(response.status_code, 400)
        error_message = response.get_json()['error']
        self.assertIn('Invalid prompt_mode', error_message)
        self.assertIn('invalid_mode', error_message)

    def test_infer_endpoint_valid_prompt_modes(self):
        """
        Test the /infer endpoint with different valid prompt_mode values.

        This test verifies that all valid prompt_mode values ('under', 'above', 'both')
        are accepted by the endpoint and passed correctly to the inferencer.
        """
        valid_modes = ['under', 'above', 'both']
        
        for mode in valid_modes:
            with self.subTest(prompt_mode=mode):
                # Mock successful inference
                self.mock_inferencer.process_image_url.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
                
                response = self.client.post('/infer', json={
                    'image_url': 'https://example.com/image.jpg',
                    'prompt_mode': mode
                })
                
                # Should not return 400 error for valid modes
                self.assertNotEqual(response.status_code, 400)
                
                # Verify the inferencer was called with the correct prompt_mode
                if response.status_code == 200:
                    self.mock_inferencer.process_image_url.assert_called_with(
                        'https://example.com/image.jpg', 
                        plot=False, 
                        prompt_mode=mode
                    )

    def test_infer_endpoint_model_not_loaded(self):
        """
        Test the /infer endpoint when the model inferencer is not loaded.

        Sets the app's INFERENCER config to None to simulate a model loading failure.
        Sends a POST request with a valid 'image_url' and asserts that the response
        status code is 500 (Internal Server Error) and the error message indicates
        the model failed to load.
        """
        # Set the app's inferencer to None to simulate loading failure
        self.app.config['INFERENCER'] = None
        response = self.client.post('/infer', json={'image_url': 'some_url'})
        self.assertEqual(response.status_code, 500)
        self.assertIn('model failed to load', response.get_json()['error'])

    @patch('api_app.Image')
    @patch('api_app.io')
    @patch('api_app.uuid')
    @patch('api_app.datetime')
    def test_infer_successful_upload(self, mock_datetime, mock_uuid, mock_io, mock_image):
        """
        Test a successful inference and S3 upload via the /infer endpoint.

        This test simulates the full inference and upload pipeline:
        - Mocks the model inferencer to return a dummy numpy array as the result.
        - Mocks uuid4 and datetime to produce deterministic S3 keys/paths.
        - Mocks PIL.Image, io.BytesIO, and S3 upload to avoid real file/network operations.
        - Sends a POST request with a valid 'image_url'.
        - Asserts that the response is HTTP 200 and contains a visualization_url with the
          expected S3 path.
        - Verifies that the S3 upload, image conversion, and buffer creation were called.
        
        Args:
            mock_datetime: Mock for api_app.datetime.
            mock_uuid: Mock for api_app.uuid.
            mock_io: Mock for api_app.io.
            mock_image: Mock for api_app.Image.
        """
        # Arrange
        self.mock_inferencer.process_image_url.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_uuid.uuid4.return_value = MagicMock(__str__=lambda s: 'test-uuid')
        mock_datetime.utcnow.return_value.strftime.return_value = '2024/01/15'
        
        # Act
        response = self.client.post('/infer', json={'image_url': 'some_url'})
        
        # Assert
        self.assertEqual(response.status_code, 200, response.get_json())
        data = response.get_json()
        self.assertIn('visualization_url', data)
        self.assertIn('s3.us-east-1.amazonaws.com/2024/01/15/test-uuid.jpg', data['visualization_url'])
        
        # Verify the inferencer was called with default prompt_mode
        self.mock_inferencer.process_image_url.assert_called_with(
            'some_url', 
            plot=False, 
            prompt_mode='both'  # Should use the default
        )
        
        self.mock_s3_client.upload_fileobj.assert_called_once()
        mock_image.fromarray.assert_called_once()
        mock_image.fromarray.return_value.save.assert_called_once()
        mock_io.BytesIO.assert_called_once()

    def test_infer_default_prompt_mode(self):
        """
        Test that the /infer endpoint uses the default prompt_mode when none is specified.

        This test verifies that when no prompt_mode is provided in the request,
        the endpoint uses the default value from the environment configuration.
        """
        # Arrange
        self.mock_inferencer.process_image_url.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Act - send request without prompt_mode
        response = self.client.post('/infer', json={'image_url': 'test_url'})
        
        # Assert
        self.assertEqual(response.status_code, 200)
        # Verify the inferencer was called with the default prompt_mode from environment
        self.mock_inferencer.process_image_url.assert_called_with(
            'test_url', 
            plot=False, 
            prompt_mode='both'  # Should use the default from _mock_get_env
        )

if __name__ == '__main__':
    unittest.main() 