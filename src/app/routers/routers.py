from typing import Dict
import uuid

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}

@router.get("/test")
def test() -> Dict[str, str]:
    return {"test": "ok"}
