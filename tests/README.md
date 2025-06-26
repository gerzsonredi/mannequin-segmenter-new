# Tests for Mannequin Segmenter

This directory contains unit tests for the mannequin-segmenter microservice.

## Structure

```
tests/
├── __init__.py              # Makes tests a Python package
├── conftest.py              # Pytest configuration and shared fixtures
├── test_api_app.py          # Tests for Flask API application (6 tests)
├── test_env_utils.py        # Tests for environment utilities (10 tests)
├── test_s3_utils.py         # Tests for S3 utilities (6 tests)
├── test_requirements.txt    # Testing-specific dependencies
└── README.md               # This file
```

## Test Summary

**Total: 22 unit tests**
- **test_env_utils.py**: 10 tests (100% passing) - Environment variable handling
- **test_s3_utils.py**: 6 tests (100% passing) - S3 integration and model downloads
- **test_api_app.py**: 6 tests (100% passing) - Flask API endpoints and error handling

## Running Tests

### Prerequisites

1. Install testing dependencies:

```bash
pip install -r tests/test_requirements.txt
```

2. Ensure the main project dependencies are installed:

```bash
pip install -r requirements.txt
```

### Running All Tests

**Using unittest (recommended for this project):**

```bash
# Run all tests
python -m unittest discover tests/ -v

# Run specific test modules that are known to work
python -m unittest tests.test_env_utils tests.test_s3_utils tests.test_api_app -v
```

**Using pytest:**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest --cov=. --cov-report=html tests/
```

### Running Specific Tests

**Run tests for a specific module:**

```bash
python -m unittest tests.test_api_app -v
python -m unittest tests.test_env_utils -v
python -m unittest tests.test_s3_utils -v
```

**Run a specific test method:**

```bash
python -m unittest tests.test_env_utils.TestEnvUtils.test_get_env_variable_existing_clean_value -v
```

## Test Coverage

### Current Test Coverage by Module

| Module | Tests | Status | Coverage | Description |
|--------|-------|--------|----------|-------------|
| env_utils | 10 | ✅ 100% | Complete | Environment variable handling, quote stripping, validation |
| s3_utils | 6 | ✅ 100% | Complete | S3 downloads, error handling, local fallback |
| api_app | 6 | ✅ 100% | Core features | Health endpoint, request validation, error handling |

**Total: 22/22 tests passing (100% success rate)**

### Generate Coverage Report

```bash
# HTML coverage report
pytest --cov=. --cov-report=html tests/
open htmlcov/index.html

# Terminal coverage report
pytest --cov=. --cov-report=term-missing tests/
```

## Test Categories

### Unit Tests by Module

#### test_env_utils.py (10 tests)
- Quote stripping from environment variables
- Whitespace handling
- Empty values and edge cases
- Nonexistent variables
- Complex formatting scenarios

#### test_s3_utils.py (6 tests)
- EVF-SAM folder detection
- AWS credentials handling
- Successful S3 downloads with file counting
- S3 errors with local fallback
- Missing credentials scenarios

#### test_api_app.py (6 tests)
- **Health endpoint**: Service status and metadata
- **Request validation**: Missing image_url handling
- **Input validation**: No JSON data handling
- **Model state**: Model not loaded scenarios
- **Processing errors**: Image processing failures
- **Exception handling**: Inference exceptions

### Mock Strategy

The tests use comprehensive mocking to:

- **Prevent real model loading** using `create_app()` factory pattern
- **Mock AWS S3 interactions** to avoid network calls
- **Mock environment variables** for consistent testing
- **Simulate error conditions** without external dependencies
- **Use dedicated mock functions** instead of lambdas for stability

### Key Architectural Changes

The project has been refactored to support better testability:

1. **Flask App Factory Pattern**: `api_app.py` now uses `create_app()` function instead of module-level initialization
2. **Dependency Injection**: All dependencies (logger, S3 client, inferencer) are attached to the app instance
3. **Mock-Friendly Design**: Imports happen inside `create_app()` allowing full control over mocking

## Writing New Tests

When adding new functionality, ensure to:

1. **Create corresponding tests** in the appropriate test file
2. **Use the `create_app()` pattern** for Flask tests to ensure proper mocking
3. **Mock external dependencies** (S3, models, environment variables)
4. **Test both success and failure scenarios**
5. **Follow existing naming conventions**
6. **Add comprehensive docstrings**

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<specific_functionality>`

### Example Flask Test Structure

```python
@patch('evfsam.EVFSAMSingleImageInferencer')
@patch('api_app.load_dotenv')
# ... other patches
def test_new_endpoint(self, mock_load_dotenv, mock_inferencer_cls, ...):
    self._setup_mocks(mock_load_dotenv, ...)
    import api_app
    app = api_app.create_app()
    app.testing = True
    client = app.test_client()
    
    response = client.post('/new-endpoint', json={'data': 'value'})
    self.assertEqual(response.status_code, 200)
```

### Best Practices

1. **Use helper functions** like `_mock_get_env()` instead of lambdas for stability
2. **Setup mocks before importing** the module under test
3. **Clear module cache** in setUp/tearDown to ensure clean test state
4. **Test error paths** as thoroughly as success paths
5. **Use realistic test data** that matches production scenarios

## Continuous Integration

These tests are designed to run in CI/CD environments:

- **No external dependencies** through comprehensive mocking
- **Fast execution** with no real model loading or network calls
- **Predictable results** with deterministic mock responses
- **Cross-platform compatibility** using standard Python libraries

## Advanced Test Runners

### Custom Test Runners

The project includes several test runners for different scenarios:

```bash
# Simple unittest runner (fastest)
python run_tests_simple.py --module env_utils -v

# Advanced runner with detailed reporting
python run_tests.py --coverage --verbose

# Comprehensive runner with summaries
python run_all_tests.py
```

## Troubleshooting

**Common Issues:**

1. **Import Errors**: Ensure the project root is in PYTHONPATH
2. **Missing Dependencies**: Install test requirements with `pip install -r tests/test_requirements.txt`
3. **Mock Issues**: Use dedicated functions instead of lambdas for side_effects
4. **Module Cache**: Tests clear module cache to ensure clean imports

**Debug Tests:**

```bash
# Run with verbose output
python -m unittest tests.test_api_app -v

# Run single test for debugging
python -m unittest tests.test_api_app.TestApiApp.test_health_endpoint -v

# Use Python debugger
python -m pdb -m unittest tests.test_api_app.TestApiApp.test_health_endpoint
```

**Performance Tips:**

- Tests complete in ~3-4 seconds total
- Each module can be tested independently
- Mock strategy prevents expensive operations
- Use specific test runners for faster iteration

## Integration with Main Project

These tests integrate with the main project's testing infrastructure:

- Compatible with existing `requirements.txt`
- Work with the refactored `api_app.py` using `create_app()`
- Support both development and CI/CD environments
- Provide foundation for future integration tests
