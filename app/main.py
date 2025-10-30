from typing import Union
from fastapi import FastAPI
from app.api import router

app = FastAPI(title="DeepFake Detection API")
app.include_router(router)