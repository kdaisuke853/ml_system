import imp
import os 

from fastapi import FastAPI
from src.app.routers import routers

app = FastAPI(
    title="Test server",
    description="api server",
)

app.include_router(routers.router, prefix="", tags=[""])

