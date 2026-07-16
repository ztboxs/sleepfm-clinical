import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import API_HOST, API_PORT
from api.models_manager import manager
from api.routers import documentation, health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SleepFM API – loading models …")
    manager.load_all()
    logger.info("Models loaded. API is ready.")
    yield
    logger.info("Shutting down SleepFM API.")


app = FastAPI(
    title="SleepFM-Clinical API",
    description="HTTP API for SleepFM multimodal sleep foundation model inference",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(documentation.router)


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/help")
async def help_page():
    return FileResponse(os.path.join(STATIC_DIR, "docs.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        timeout_keep_alive=300,
        h11_max_incomplete_event_size=0,
    )
