"""
Core inference logic: preprocessing, embedding generation, sleep staging, and disease prediction.
Mirrors the demo notebook pipeline but packaged for single-file API calls.
"""

import os
import sys
import numpy as np
import torch
import h5py
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sleepfm"))
from preprocessing.preprocessing import EDFToHDF5Converter
from models.dataset import SetTransformerDataset, collate_fn

from api.config import RESAMPLE_RATE, LABEL_MAPPING_PATH
from api.models_manager import manager
from api import schemas

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def preprocess_edf(edf_path: str, hdf5_path: str) -> dict:
    """Convert EDF -> HDF5 and return channel-modality mapping info."""
    converter = EDFToHDF5Converter(
        root_dir=os.path.dirname(edf_path),
        target_dir=os.path.dirname(hdf5_path),
        resample_rate=RESAMPLE_RATE,
    )
    signals, sample_rates, channel_names = converter.read_edf(edf_path)
    duration_seconds = len(signals[0]) / sample_rates[0]

    converter.convert(edf_path, hdf5_path)

    channel_groups = manager.channel_groups
    channel_map = {"BAS": [], "RESP": [], "EKG": [], "EMG": []}
    for ch in channel_names:
        for modality in channel_map:
            if ch in channel_groups[modality]:
                channel_map[modality].append(ch)

    return {
        "channels": channel_map,
        "duration_seconds": float(duration_seconds),
        "sample_rate": RESAMPLE_RATE,
    }


def generate_embeddings(hdf5_path: str, emb_dir: str) -> dict[str, np.ndarray]:
    """Run the base model to produce per-modality 5-second-level embeddings.

    Returns a dict mapping modality name -> numpy array of shape (T, embed_dim).
    """
    cfg = manager.base_config
    channel_groups = manager.channel_groups
    model = manager.base_model
    device = manager.device

    embed_dim = cfg["embed_dim"]
    modality_types = cfg["modality_types"]

    dataset = SetTransformerDataset(cfg, channel_groups, hdf5_paths=[hdf5_path], split="test")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, num_workers=0, shuffle=False, collate_fn=collate_fn,
    )

    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, os.path.splitext(os.path.basename(hdf5_path))[0] + ".hdf5")

    if len(dataset) == 0:
        with h5py.File(hdf5_path, "r") as hf:
            file_channels = list(hf.keys())
        matched = {mod: [] for mod in modality_types}
        for ch in file_channels:
            for mod in modality_types:
                if ch in channel_groups.get(mod, []):
                    matched[mod].append(ch)
        missing = [mod for mod, chs in matched.items() if not chs]
        raise ValueError(
            f"EDF 文件的通道无法匹配所有必需的模态。"
            f"文件通道: {file_channels}，"
            f"缺失模态: {missing}，"
            f"各模态匹配: {matched}。"
            f"请确保 EDF 文件包含 BAS(脑电/眼电)、RESP(呼吸)、EKG(心电)、EMG(肌电) 四种模态的通道。"
        )

    with torch.no_grad():
        for batch in dataloader:
            batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
            data_tensors = [d.to(device, dtype=torch.float) for d in batch_data]
            masks = [m.to(device, dtype=torch.bool) for m in mask_list]

            embeddings = [model(data_tensors[i], masks[i]) for i in range(len(data_tensors))]

            embeddings_5s = [e[1] for e in embeddings]

            for i in range(len(file_paths)):
                chunk_start = chunk_starts[i]
                with h5py.File(emb_path, "a") as hf:
                    for mod_idx, mod_type in enumerate(modality_types):
                        emb_slice = embeddings_5s[mod_idx][i].cpu().numpy()
                        chunk_start_idx = chunk_start // (embed_dim * 5)
                        chunk_end_idx = chunk_start_idx + emb_slice.shape[0]
                        if mod_type in hf:
                            dset = hf[mod_type]
                            if dset.shape[0] < chunk_end_idx:
                                dset.resize((chunk_end_idx,) + emb_slice.shape[1:])
                            dset[chunk_start_idx:chunk_end_idx] = emb_slice
                        else:
                            hf.create_dataset(
                                mod_type,
                                data=emb_slice,
                                chunks=(embed_dim,) + emb_slice.shape[1:],
                                maxshape=(None,) + emb_slice.shape[1:],
                            )

    if not os.path.exists(emb_path):
        raise RuntimeError("嵌入生成失败：未产生嵌入文件，请检查 EDF 数据完整性。")

    result = {}
    with h5py.File(emb_path, "r") as hf:
        for mod in modality_types:
            if mod in hf:
                result[mod] = hf[mod][:]
    return result


def run_sleep_staging(emb_path: str) -> schemas.SleepStagingResult:
    """Run sleep staging model on precomputed embeddings."""
    cfg = manager.sleep_staging_config
    model = manager.sleep_staging_model
    device = manager.device
    channel_like = cfg.get("channel_like", cfg.get("modality_types"))
    max_seq_len = cfg["model_params"]["max_seq_length"]
    max_channels = cfg.get("max_channels", 4)

    x_data = []
    with h5py.File(emb_path, "r") as hf:
        for key in hf.keys():
            if key in channel_like:
                x_data.append(hf[key][:])

    if not x_data:
        raise ValueError("No matching modality embeddings found for sleep staging")

    x_data = np.array(x_data)
    x_tensor = torch.tensor(x_data, dtype=torch.float32)

    c, s, e = x_tensor.shape
    num_ch = min(c, max_channels)
    seq_len = min(s, max_seq_len)

    padded = torch.zeros((1, max_channels, max_seq_len, e))
    mask = torch.ones((1, max_channels, max_seq_len))

    padded[0, :num_ch, :seq_len, :] = x_tensor[:num_ch, :seq_len, :]
    mask[0, :num_ch, :seq_len] = 0

    padded = padded.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        outputs, out_mask = model(padded, mask)
        probs = torch.softmax(outputs, dim=-1).cpu().numpy()[0]  # (seq_len, 5)
        out_mask_np = out_mask.cpu().numpy()[0]

    valid_mask = out_mask_np == 0
    valid_probs = probs[valid_mask]

    epochs = []
    stage_counts = {name: 0 for name in STAGE_NAMES.values()}

    for i, prob in enumerate(valid_probs):
        predicted_idx = int(np.argmax(prob))
        predicted_stage = STAGE_NAMES[predicted_idx]
        stage_counts[predicted_stage] += 1
        epochs.append(schemas.SleepEpoch(
            index=i,
            start_sec=i * 5.0,
            end_sec=(i + 1) * 5.0,
            predicted_stage=predicted_stage,
            probabilities=schemas.StageProbabilities(
                Wake=round(float(prob[0]), 4),
                N1=round(float(prob[1]), 4),
                N2=round(float(prob[2]), 4),
                N3=round(float(prob[3]), 4),
                REM=round(float(prob[4]), 4),
            ),
        ))

    total = len(valid_probs)
    summary = {}
    for name, count in stage_counts.items():
        summary[name] = schemas.StageSummaryItem(
            count=count,
            percentage=round(count / total * 100, 2) if total > 0 else 0.0,
        )

    return schemas.SleepStagingResult(total_epochs=total, epochs=epochs, summary=summary)


def run_disease_prediction(
    emb_path: str, age: float, gender: int
) -> schemas.DiseasePredictionResult:
    """Run disease prediction model on precomputed embeddings + demographics."""
    cfg = manager.disease_config
    model = manager.disease_model
    device = manager.device
    modality_types = cfg.get("modality_types", ["BAS", "RESP", "EKG", "EMG"])
    max_seq_len = cfg["model_params"]["max_seq_length"]
    max_channels = cfg.get("max_channels", 4)

    x_data = []
    with h5py.File(emb_path, "r") as hf:
        for key in hf.keys():
            if key in modality_types:
                x_data.append(hf[key][:])

    if not x_data:
        raise ValueError("No matching modality embeddings found for disease prediction")

    x_data = np.array(x_data)
    x_tensor = torch.tensor(x_data, dtype=torch.float32)

    c, s, e = x_tensor.shape
    num_ch = min(c, max_channels)
    seq_len = min(s, max_seq_len)

    padded = torch.zeros((1, max_channels, max_seq_len, e))
    mask = torch.ones((1, max_channels, max_seq_len))

    padded[0, :num_ch, :seq_len, :] = x_tensor[:num_ch, :seq_len, :]
    mask[0, :num_ch, :seq_len] = 0

    demo = torch.tensor([[age, float(gender)]], dtype=torch.float32)

    padded = padded.to(device)
    mask = mask.to(device)
    demo = demo.to(device)

    with torch.no_grad():
        hazards = model(padded, mask, demo).cpu().numpy()[0]  # (1065,)

    label_df = pd.read_csv(LABEL_MAPPING_PATH)

    sorted_indices = np.argsort(-hazards)
    all_risks = []
    for rank, idx in enumerate(sorted_indices, start=1):
        row = label_df[label_df["label_idx"] == int(idx)]
        phecode = str(row["phecode"].values[0]) if len(row) > 0 else "unknown"
        phenotype = str(row["phenotype"].values[0]) if len(row) > 0 else "unknown"
        all_risks.append(schemas.DiseaseRiskItem(
            rank=rank,
            label_idx=int(idx),
            phecode=phecode,
            phenotype=phenotype,
            hazard_score=round(float(hazards[idx]), 6),
        ))

    return schemas.DiseasePredictionResult(
        top_risks=all_risks,
        all_hazards_count=len(hazards),
    )
