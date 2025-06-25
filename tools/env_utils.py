import os

def get_env_variable(key: str) -> str:
    """
    Retrieve environment variable and clean it of quotes and whitespace.
    
    Args:
        key: Environment variable name
        
    Returns:
        Cleaned environment variable value or None if not found
    """
    value = os.getenv(key)
    if value:
        return value.strip().strip("'\"")
    return None 