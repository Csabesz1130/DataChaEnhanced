from fastapi import APIRouter
from src.version import get_current_version, APP_NAME

router = APIRouter(prefix="/version", tags=["version"]) 

@router.get("", summary="Get app version")
def get_version():
    return {"app": APP_NAME, "version": get_current_version()}