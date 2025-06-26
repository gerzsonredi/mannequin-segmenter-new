import unittest
from unittest.mock import patch
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.env_utils import get_env_variable


class TestEnvUtils(unittest.TestCase):
    """Test cases for env_utils module"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Store original environment to restore later
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_get_env_variable_existing_clean_value(self):
        """
        Test that get_env_variable returns the correct value when the environment variable exists
        and contains a clean (unquoted, unpadded) value.
        """
        os.environ['TEST_VAR'] = 'clean_value'
        result = get_env_variable('TEST_VAR')
        self.assertEqual(result, 'clean_value')

    def test_get_env_variable_with_quotes(self):
        """
        Test that get_env_variable strips surrounding single or double quotes from the value.
        """
        test_cases = [
            ('"quoted_value"', 'quoted_value'),
            ("'single_quoted'", 'single_quoted'),
            ('\'mixed"quotes\'', 'mixed"quotes'),
            ('"value with spaces"', 'value with spaces')
        ]
        
        for input_value, expected_output in test_cases:
            with self.subTest(input_value=input_value):
                os.environ['TEST_VAR'] = input_value
                result = get_env_variable('TEST_VAR')
                self.assertEqual(result, expected_output)

    def test_get_env_variable_with_whitespace(self):
        """
        Test that get_env_variable strips leading and trailing whitespace, including tabs and newlines,
        and also strips quotes if present after whitespace is removed.
        """
        test_cases = [
            ('  spaced_value  ', 'spaced_value'),
            ('\tvalue_with_tabs\t', 'value_with_tabs'),
            ('\nvalue_with_newlines\n', 'value_with_newlines'),
            ('  "quoted with spaces"  ', 'quoted with spaces')
        ]
        
        for input_value, expected_output in test_cases:
            with self.subTest(input_value=input_value):
                os.environ['TEST_VAR'] = input_value
                result = get_env_variable('TEST_VAR')
                self.assertEqual(result, expected_output)

    def test_get_env_variable_nonexistent(self):
        """
        Test that get_env_variable returns None when the environment variable does not exist.
        """
        # Make sure the variable doesn't exist
        if 'NONEXISTENT_VAR' in os.environ:
            del os.environ['NONEXISTENT_VAR']
        
        result = get_env_variable('NONEXISTENT_VAR')
        self.assertIsNone(result)

    def test_get_env_variable_empty_value(self):
        """
        Test that get_env_variable returns None when the environment variable exists but is set to an empty string.
        """
        os.environ['EMPTY_VAR'] = ''
        result = get_env_variable('EMPTY_VAR')
        self.assertIsNone(result)  # Empty string is falsy, so function returns None

    def test_get_env_variable_whitespace_only(self):
        """
        Test that get_env_variable returns an empty string when the environment variable contains only whitespace.
        """
        os.environ['WHITESPACE_VAR'] = '   '
        result = get_env_variable('WHITESPACE_VAR')
        self.assertEqual(result, '')  # Whitespace-only becomes empty after strip

    def test_get_env_variable_only_quotes(self):
        """
        Test that get_env_variable returns an empty string when the environment variable contains only quotes,
        or a single quote after stripping the other type of quote.
        """
        test_cases = [
            ('""', ''),    # Empty after stripping quotes
            ("''", ''),    # Empty after stripping quotes
            ('"\'', ''),   # Single quote after stripping double quotes
            ("'\"", ''),   # Double quote after stripping single quotes
        ]
        
        for input_value, expected_output in test_cases:
            with self.subTest(input_value=input_value):
                os.environ['TEST_VAR'] = input_value
                result = get_env_variable('TEST_VAR')
                self.assertEqual(result, expected_output)

    def test_get_env_variable_complex_values(self):
        """
        Test that get_env_variable correctly handles complex values, including URLs, keys, regions,
        values with commas, and values with both spaces and quotes inside.
        """
        test_cases = [
            ('  "https://example.com/path"  ', 'https://example.com/path'),
            ("'aws_access_key_123456'", 'aws_access_key_123456'),
            ('  us-east-1  ', 'us-east-1'),
            ('"value,with,commas"', 'value,with,commas'),
            ("'value with spaces and quotes \"inside\"'", 'value with spaces and quotes "inside')  # Note: only outer quotes are stripped
        ]
        
        for input_value, expected_output in test_cases:
            with self.subTest(input_value=input_value):
                os.environ['TEST_VAR'] = input_value
                result = get_env_variable('TEST_VAR')
                self.assertEqual(result, expected_output)

    @patch('os.getenv')
    def test_get_env_variable_with_mock(self, mock_getenv):
        """
        Test that get_env_variable works correctly when os.getenv is mocked to return a quoted value
        with whitespace, and that it calls os.getenv with the correct key.
        """
        mock_getenv.return_value = '  "mocked_value"  '
        result = get_env_variable('MOCKED_VAR')
        mock_getenv.assert_called_once_with('MOCKED_VAR')
        self.assertEqual(result, 'mocked_value')

    @patch('os.getenv')
    def test_get_env_variable_mock_none(self, mock_getenv):
        """
        Test that get_env_variable returns None when os.getenv is mocked to return None,
        and that it calls os.getenv with the correct key.
        """
        mock_getenv.return_value = None
        result = get_env_variable('NONEXISTENT_VAR')
        mock_getenv.assert_called_once_with('NONEXISTENT_VAR')
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main() 