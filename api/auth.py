"""
API Authentication - Simple API Key auth
"""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from .config import API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key from header."""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Add 'X-API-Key' header."
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key


def check_api_key(key: str) -> bool:
    """Check if API key is valid (for non-FastAPI use)."""
    return key == API_KEY
