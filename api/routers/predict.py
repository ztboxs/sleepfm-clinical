import os
import json
import asyncio
import shutil
import tempfile
import uuid
import time
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from api import schemas, inference
from api.utils import temp_directory, save_upload_file, download_file_from_url
from api.task_status import tracker
from api.config import EXPORT_DIR, EXPORT_MAX_AGE_HOURS

router = APIRouter(prefix="/api/v1", tags=["inference"])

_gpu_lock = asyncio.Lock()


def _export_file(src_path: str, prefix: str, ext: str = ".hdf5") -> str:
    """Copy a file to the export directory with a unique ID. Returns the file_id."""
    file_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    dest = os.path.join(EXPORT_DIR, file_id + ext)
    shutil.copy2(src_path, dest)
    return file_id


def _export_json(data: dict, prefix: str) -> str:
    """Save JSON data to the export directory. Returns the file_id."""
    file_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    dest = os.path.join(EXPORT_DIR, file_id + ".json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return file_id


def _cleanup_old_exports():
    """Remove export files older than EXPORT_MAX_AGE_HOURS."""
    cutoff = time.time() - EXPORT_MAX_AGE_HOURS * 3600
    for fname in os.listdir(EXPORT_DIR):
        fpath = os.path.join(EXPORT_DIR, fname)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            os.remove(fpath)


async def _resolve_edf_file(
    tmp_dir: str,
    file: Optional[UploadFile],
    file_url: Optional[str],
) -> str:
    """Resolve EDF file from upload or URL, return local path."""
    if file and file.filename:
        if not file.filename.lower().endswith(".edf"):
            raise HTTPException(status_code=400, detail="Only .edf files are accepted")
        tracker.set_stage("uploading", "正在接收文件上传…", 3)
        edf_path = os.path.join(tmp_dir, file.filename)
        await save_upload_file(file, edf_path)
        file_size_mb = os.path.getsize(edf_path) / 1024 / 1024
        tracker.start(file.filename, file_size_mb)
        return edf_path

    if file_url:
        tracker.set_stage("downloading", "正在从 URL 下载文件…", 3)
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
        file_size_mb = os.path.getsize(edf_path) / 1024 / 1024
        tracker.start(os.path.basename(edf_path), file_size_mb)
        return edf_path

    raise HTTPException(
        status_code=400,
        detail="Either 'file' (upload) or 'file_url' (URL) must be provided",
    )


async def _resolve_input_file(
    tmp_dir: str,
    file: Optional[UploadFile],
    file_url: Optional[str],
    allowed_exts: tuple[str, ...] = (".edf", ".hdf5", ".h5"),
) -> str:
    """Resolve uploaded file (EDF or HDF5) from upload or URL."""
    if file and file.filename:
        fname_lower = file.filename.lower()
        if not any(fname_lower.endswith(ext) for ext in allowed_exts):
            raise HTTPException(status_code=400, detail=f"仅支持以下格式: {', '.join(allowed_exts)}")
        dest = os.path.join(tmp_dir, file.filename)
        await save_upload_file(file, dest)
        return dest

    if file_url:
        loop = asyncio.get_event_loop()
        try:
            dest = await loop.run_in_executor(None, download_file_from_url, file_url, tmp_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"下载文件失败: {e}")
        return dest

    raise HTTPException(status_code=400, detail="必须提供 file（上传）或 file_url（URL）")


# ==================== Download exported files ====================

@router.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download an exported file by file_id."""
    _cleanup_old_exports()
    for ext in (".hdf5", ".json"):
        fpath = os.path.join(EXPORT_DIR, file_id + ext)
        if os.path.isfile(fpath):
            media_type = "application/x-hdf5" if ext == ".hdf5" else "application/json"
            return FileResponse(
                fpath,
                media_type=media_type,
                filename=file_id + ext,
            )
    raise HTTPException(status_code=404, detail="文件不存在或已过期")


# ==================== Preprocess (EDF → HDF5, with export) ====================

@router.post("/preprocess")
async def preprocess(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
):
    with temp_directory() as tmp_dir:
        edf_path = await _resolve_edf_file(tmp_dir, file, file_url)
        hdf5_path = os.path.join(tmp_dir, os.path.basename(edf_path).rsplit(".", 1)[0] + ".hdf5")

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, inference.preprocess_edf, edf_path, hdf5_path)

        preprocessed_file_id = _export_file(hdf5_path, "preproc")

    return {
        "status": "success",
        "channels": info["channels"],
        "duration_seconds": info["duration_seconds"],
        "sample_rate": info["sample_rate"],
        "preprocessed_file_id": preprocessed_file_id,
        "download_url": f"/api/v1/download/{preprocessed_file_id}",
    }


# ==================== Embed (HDF5/EDF → embeddings, with export) ====================

@router.post("/embed")
async def embed(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    preprocessed_file_id: Optional[str] = Form(None),
):
    try:
        with temp_directory() as tmp_dir:
            if preprocessed_file_id:
                src = os.path.join(EXPORT_DIR, preprocessed_file_id + ".hdf5")
                if not os.path.isfile(src):
                    raise HTTPException(status_code=404, detail="预处理文件不存在或已过期")
                hdf5_path = os.path.join(tmp_dir, preprocessed_file_id + ".hdf5")
                shutil.copy2(src, hdf5_path)
            else:
                input_path = await _resolve_input_file(tmp_dir, file, file_url)
                if input_path.lower().endswith((".hdf5", ".h5")):
                    hdf5_path = input_path
                else:
                    basename = os.path.basename(input_path).rsplit(".", 1)[0]
                    hdf5_path = os.path.join(tmp_dir, basename + ".hdf5")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, inference.preprocess_edf, input_path, hdf5_path)

            emb_dir = os.path.join(tmp_dir, "emb")
            loop = asyncio.get_event_loop()
            async with _gpu_lock:
                emb_dict = await loop.run_in_executor(
                    None, inference.generate_embeddings, hdf5_path, emb_dir
                )

            emb_file = os.path.join(
                emb_dir, os.path.splitext(os.path.basename(hdf5_path))[0] + ".hdf5"
            )
            embedding_file_id = _export_file(emb_file, "embed")

        embeddings_info = {}
        for mod, arr in emb_dict.items():
            embeddings_info[mod] = {"shape": list(arr.shape)}

        first_shape = next(iter(emb_dict.values())).shape if emb_dict else (0, 0)
        return {
            "status": "success",
            "embeddings": embeddings_info,
            "num_5min_chunks": first_shape[0] // 60 if len(first_shape) > 0 else 0,
            "embed_dim": int(first_shape[1]) if len(first_shape) > 1 else 0,
            "embedding_file_id": embedding_file_id,
            "download_url": f"/api/v1/download/{embedding_file_id}",
        }
    except (ValueError, RuntimeError) as e:
        logger.error(f"Embed error: {e}")
        raise HTTPException(status_code=422, detail=str(e))


# ==================== Sleep Staging (standalone) ====================

@router.post("/sleep_staging")
async def sleep_staging(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    embedding_file_id: Optional[str] = Form(None),
):
    """Standalone sleep staging: accepts embedding file or EDF for full pipeline."""
    try:
        with temp_directory() as tmp_dir:
            emb_file = await _resolve_embedding_file(tmp_dir, file, file_url, embedding_file_id)

            loop = asyncio.get_event_loop()
            async with _gpu_lock:
                staging_result = await loop.run_in_executor(
                    None, inference.run_sleep_staging, emb_file
                )

        result_data = staging_result.model_dump()
        result_file_id = _export_json(result_data, "staging")

        return {
            "status": "success",
            "sleep_staging": result_data,
            "result_file_id": result_file_id,
            "download_url": f"/api/v1/download/{result_file_id}",
        }
    except (ValueError, RuntimeError) as e:
        logger.error(f"Sleep staging error: {e}")
        raise HTTPException(status_code=422, detail=str(e))


# ==================== Disease Prediction (standalone) ====================

@router.post("/disease_prediction")
async def disease_prediction(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    embedding_file_id: Optional[str] = Form(None),
    age: float = Form(...),
    gender: int = Form(...),
    min_hazard_score: Optional[float] = Form(None),
):
    """Standalone disease prediction: accepts embedding file + demographics."""
    if age < 0 or age > 150:
        raise HTTPException(status_code=400, detail="age must be between 0 and 150")
    age_normalized = age / 100.0

    try:
        with temp_directory() as tmp_dir:
            emb_file = await _resolve_embedding_file(tmp_dir, file, file_url, embedding_file_id)

            loop = asyncio.get_event_loop()
            async with _gpu_lock:
                disease_result = await loop.run_in_executor(
                    None, inference.run_disease_prediction, emb_file, age_normalized, gender
                )

        if min_hazard_score is not None:
            filtered = [r for r in disease_result.top_risks if r.hazard_score >= min_hazard_score]
            for i, item in enumerate(filtered, start=1):
                item.rank = i
            disease_result.top_risks = filtered

        result_data = disease_result.model_dump()
        result_file_id = _export_json(result_data, "disease")

        return {
            "status": "success",
            "disease_prediction": result_data,
            "result_file_id": result_file_id,
            "download_url": f"/api/v1/download/{result_file_id}",
        }
    except (ValueError, RuntimeError) as e:
        logger.error(f"Disease prediction error: {e}")
        raise HTTPException(status_code=422, detail=str(e))


async def _resolve_embedding_file(
    tmp_dir: str,
    file: Optional[UploadFile],
    file_url: Optional[str],
    embedding_file_id: Optional[str],
) -> str:
    """Resolve embedding HDF5 file from various sources."""
    if embedding_file_id:
        src = os.path.join(EXPORT_DIR, embedding_file_id + ".hdf5")
        if not os.path.isfile(src):
            raise HTTPException(status_code=404, detail="嵌入文件不存在或已过期")
        dest = os.path.join(tmp_dir, embedding_file_id + ".hdf5")
        shutil.copy2(src, dest)
        return dest

    if file and file.filename:
        fname_lower = file.filename.lower()
        if not fname_lower.endswith((".hdf5", ".h5")):
            if fname_lower.endswith(".edf"):
                return await _full_pipeline_to_embedding(tmp_dir, file=file)
            raise HTTPException(status_code=400, detail="请上传嵌入 HDF5 文件或 EDF 文件")
        dest = os.path.join(tmp_dir, file.filename)
        await save_upload_file(file, dest)
        return dest

    if file_url:
        loop = asyncio.get_event_loop()
        try:
            dest = await loop.run_in_executor(None, download_file_from_url, file_url, tmp_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"下载文件失败: {e}")
        if dest.lower().endswith(".edf"):
            return await _full_pipeline_to_embedding_from_edf(tmp_dir, dest)
        return dest

    raise HTTPException(status_code=400, detail="必须提供 embedding_file_id、file（上传）或 file_url（URL）")


async def _full_pipeline_to_embedding(tmp_dir: str, file: UploadFile) -> str:
    """EDF upload → preprocess → embed, return embedding path."""
    edf_path = os.path.join(tmp_dir, file.filename)
    await save_upload_file(file, edf_path)
    return await _full_pipeline_to_embedding_from_edf(tmp_dir, edf_path)


async def _full_pipeline_to_embedding_from_edf(tmp_dir: str, edf_path: str) -> str:
    """EDF file → preprocess → embed, return embedding path."""
    basename = os.path.basename(edf_path).rsplit(".", 1)[0]
    hdf5_path = os.path.join(tmp_dir, basename + ".hdf5")
    emb_dir = os.path.join(tmp_dir, "emb")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, inference.preprocess_edf, edf_path, hdf5_path)

    async with _gpu_lock:
        await loop.run_in_executor(None, inference.generate_embeddings, hdf5_path, emb_dir)

    return os.path.join(emb_dir, basename + ".hdf5")


# ==================== Full Predict Pipeline ====================

async def _predict_background(
    tmp_dir: str,
    edf_path: str,
    do_staging: bool,
    do_disease: bool,
    age: Optional[float],
    gender: Optional[int],
    min_hazard_score: Optional[float],
):
    """Run the full predict pipeline in the background."""
    try:
        basename = os.path.basename(edf_path).rsplit(".", 1)[0]
        hdf5_path = os.path.join(tmp_dir, basename + ".hdf5")
        emb_dir = os.path.join(tmp_dir, "emb")

        loop = asyncio.get_event_loop()

        tracker.set_stage("preprocessing", "数据预处理 — 读取 EDF / 重采样 / 滤波", 15)
        logger.info("Step 0: Preprocessing EDF …")
        await loop.run_in_executor(None, inference.preprocess_edf, edf_path, hdf5_path)

        preprocessed_file_id = _export_file(hdf5_path, "preproc")

        async with _gpu_lock:
            tracker.set_stage("embedding", "生成嵌入向量 (GPU)", 40)
            logger.info("Step 1: Generating embeddings …")
            await loop.run_in_executor(
                None, inference.generate_embeddings, hdf5_path, emb_dir
            )

            emb_file = os.path.join(
                emb_dir, os.path.splitext(os.path.basename(hdf5_path))[0] + ".hdf5"
            )

            embedding_file_id = _export_file(emb_file, "embed")

            staging_result = None
            disease_result = None

            if do_staging:
                tracker.set_stage("sleep_staging", "睡眠分期推理 (GPU)", 65)
                logger.info("Step 2a: Sleep staging …")
                staging_result = await loop.run_in_executor(
                    None, inference.run_sleep_staging, emb_file
                )

            if do_disease:
                tracker.set_stage("disease_prediction", "疾病风险预测 (GPU)", 85)
                logger.info("Step 2b: Disease prediction …")
                disease_result = await loop.run_in_executor(
                    None, inference.run_disease_prediction, emb_file, age, gender
                )

        if disease_result and min_hazard_score is not None:
            filtered = [r for r in disease_result.top_risks if r.hazard_score >= min_hazard_score]
            for i, item in enumerate(filtered, start=1):
                item.rank = i
            disease_result.top_risks = filtered

        export_ids = {
            "preprocessed_file_id": preprocessed_file_id,
            "embedding_file_id": embedding_file_id,
        }

        if staging_result:
            export_ids["staging_result_file_id"] = _export_json(
                staging_result.model_dump(), "staging"
            )
        if disease_result:
            export_ids["disease_result_file_id"] = _export_json(
                disease_result.model_dump(), "disease"
            )

        result = schemas.PredictResponse(
            status="success",
            sleep_staging=staging_result,
            disease_prediction=disease_result,
        )

        result_dict = result.model_dump()
        result_dict["export_ids"] = export_ids
        tracker.finish(result=result_dict)
        logger.info("Predict pipeline finished successfully.")

    except (ValueError, RuntimeError) as e:
        logger.error(f"Inference error: {e}")
        tracker.fail(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in predict pipeline: {e}")
        tracker.fail(str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/predict")
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

    if age is not None:
        if age < 0 or age > 150:
            raise HTTPException(status_code=400, detail="age must be between 0 and 150")
        age = age / 100.0

    if tracker.active:
        return {
            "status": "busy",
            "message": "当前有一个推理任务正在处理中，请等待完成后再提交新任务。您可以通过页面上的进度条查看当前任务状态。",
        }

    tmp_dir = tempfile.mkdtemp(prefix="sleepfm_")
    try:
        edf_path = await _resolve_edf_file(tmp_dir, file, file_url)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    asyncio.create_task(
        _predict_background(
            tmp_dir, edf_path,
            do_staging, do_disease,
            age, gender, min_hazard_score,
        )
    )

    return {"status": "accepted", "message": "推理任务已提交，请通过 /api/v1/task_status 查询进度"}
