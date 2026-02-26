import os
import asyncio
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger

from api import schemas, inference
from api.utils import temp_directory, save_upload_file

router = APIRouter(prefix="/api/v1", tags=["inference"])

_gpu_lock = asyncio.Lock()


@router.post("/preprocess", response_model=schemas.PreprocessResponse)
async def preprocess(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only .edf files are accepted")

    with temp_directory() as tmp_dir:
        edf_path = os.path.join(tmp_dir, file.filename)
        hdf5_path = os.path.join(tmp_dir, file.filename.rsplit(".", 1)[0] + ".hdf5")
        await save_upload_file(file, edf_path)

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, inference.preprocess_edf, edf_path, hdf5_path)

    return schemas.PreprocessResponse(
        status="success",
        channels=schemas.ChannelInfo(**info["channels"]),
        duration_seconds=info["duration_seconds"],
        sample_rate=info["sample_rate"],
    )


@router.post("/embed", response_model=schemas.EmbedResponse)
async def embed(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only .edf files are accepted")

    try:
        with temp_directory() as tmp_dir:
            edf_path = os.path.join(tmp_dir, file.filename)
            hdf5_path = os.path.join(tmp_dir, file.filename.rsplit(".", 1)[0] + ".hdf5")
            emb_dir = os.path.join(tmp_dir, "emb")
            await save_upload_file(file, edf_path)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, inference.preprocess_edf, edf_path, hdf5_path)

            async with _gpu_lock:
                emb_dict = await loop.run_in_executor(
                    None, inference.generate_embeddings, hdf5_path, emb_dir
                )

        embeddings_info = {}
        for mod, arr in emb_dict.items():
            embeddings_info[mod] = schemas.EmbeddingInfo(shape=list(arr.shape))

        first_shape = next(iter(emb_dict.values())).shape if emb_dict else (0, 0)
        return schemas.EmbedResponse(
            status="success",
            embeddings=embeddings_info,
            num_5min_chunks=first_shape[0] // 60 if len(first_shape) > 0 else 0,
            embed_dim=int(first_shape[1]) if len(first_shape) > 1 else 0,
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"Embed error: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/predict", response_model=schemas.PredictResponse)
async def predict(
    file: UploadFile = File(...),
    age: Optional[float] = Form(None),
    gender: Optional[int] = Form(None),
    tasks: str = Form("sleep_staging,disease_prediction"),
):
    if not file.filename or not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only .edf files are accepted")

    task_list = [t.strip() for t in tasks.split(",")]
    do_staging = "sleep_staging" in task_list
    do_disease = "disease_prediction" in task_list

    if do_disease and (age is None or gender is None):
        raise HTTPException(
            status_code=400,
            detail="age and gender are required for disease_prediction task",
        )

    try:
        with temp_directory() as tmp_dir:
            edf_path = os.path.join(tmp_dir, file.filename)
            hdf5_path = os.path.join(tmp_dir, file.filename.rsplit(".", 1)[0] + ".hdf5")
            emb_dir = os.path.join(tmp_dir, "emb")
            await save_upload_file(file, edf_path)

            loop = asyncio.get_event_loop()

            logger.info("Step 0: Preprocessing EDF …")
            await loop.run_in_executor(None, inference.preprocess_edf, edf_path, hdf5_path)

            async with _gpu_lock:
                logger.info("Step 1: Generating embeddings …")
                await loop.run_in_executor(
                    None, inference.generate_embeddings, hdf5_path, emb_dir
                )

                emb_file = os.path.join(
                    emb_dir, os.path.splitext(os.path.basename(hdf5_path))[0] + ".hdf5"
                )

                staging_result = None
                disease_result = None

                if do_staging:
                    logger.info("Step 2a: Sleep staging …")
                    staging_result = await loop.run_in_executor(
                        None, inference.run_sleep_staging, emb_file
                    )

                if do_disease:
                    logger.info("Step 2b: Disease prediction …")
                    disease_result = await loop.run_in_executor(
                        None, inference.run_disease_prediction, emb_file, age, gender
                    )

        return schemas.PredictResponse(
            status="success",
            sleep_staging=staging_result,
            disease_prediction=disease_result,
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
