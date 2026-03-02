import os
import asyncio
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger

from api import schemas, inference
from api.utils import temp_directory, save_upload_file, download_file_from_url

router = APIRouter(prefix="/api/v1", tags=["inference"])

_gpu_lock = asyncio.Lock()


async def _resolve_edf_file(
    tmp_dir: str,
    file: Optional[UploadFile],
    file_url: Optional[str],
) -> str:
    """Resolve EDF file from upload or URL, return local path."""
    if file and file.filename:
        if not file.filename.lower().endswith(".edf"):
            raise HTTPException(status_code=400, detail="Only .edf files are accepted")
        edf_path = os.path.join(tmp_dir, file.filename)
        await save_upload_file(file, edf_path)
        return edf_path

    if file_url:
        loop = asyncio.get_event_loop()
        try:
            edf_path = await loop.run_in_executor(
                None, download_file_from_url, file_url, tmp_dir
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download file from URL: {e}",
            )
        if not edf_path.lower().endswith(".edf"):
            raise HTTPException(
                status_code=400,
                detail=f"Downloaded file is not .edf format: {os.path.basename(edf_path)}",
            )
        return edf_path

    raise HTTPException(
        status_code=400,
        detail="Either 'file' (upload) or 'file_url' (URL) must be provided",
    )


@router.post("/preprocess", response_model=schemas.PreprocessResponse)
async def preprocess(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
):
    with temp_directory() as tmp_dir:
        edf_path = await _resolve_edf_file(tmp_dir, file, file_url)
        hdf5_path = os.path.join(tmp_dir, os.path.basename(edf_path).rsplit(".", 1)[0] + ".hdf5")

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, inference.preprocess_edf, edf_path, hdf5_path)

    return schemas.PreprocessResponse(
        status="success",
        channels=schemas.ChannelInfo(**info["channels"]),
        duration_seconds=info["duration_seconds"],
        sample_rate=info["sample_rate"],
    )


@router.post("/embed", response_model=schemas.EmbedResponse)
async def embed(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
):
    try:
        with temp_directory() as tmp_dir:
            edf_path = await _resolve_edf_file(tmp_dir, file, file_url)
            basename = os.path.basename(edf_path).rsplit(".", 1)[0]
            hdf5_path = os.path.join(tmp_dir, basename + ".hdf5")
            emb_dir = os.path.join(tmp_dir, "emb")

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
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    age: Optional[float] = Form(None),
    gender: Optional[int] = Form(None),
    tasks: str = Form("sleep_staging,disease_prediction"),
    min_hazard_score: Optional[float] = Form(None),
):
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
            edf_path = await _resolve_edf_file(tmp_dir, file, file_url)
            basename = os.path.basename(edf_path).rsplit(".", 1)[0]
            hdf5_path = os.path.join(tmp_dir, basename + ".hdf5")
            emb_dir = os.path.join(tmp_dir, "emb")

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

        if disease_result and min_hazard_score is not None:
            filtered = [r for r in disease_result.top_risks if r.hazard_score >= min_hazard_score]
            for i, item in enumerate(filtered, start=1):
                item.rank = i
            disease_result.top_risks = filtered

        return schemas.PredictResponse(
            status="success",
            sleep_staging=staging_result,
            disease_prediction=disease_result,
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
