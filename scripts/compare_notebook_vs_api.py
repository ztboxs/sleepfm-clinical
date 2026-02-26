"""
对比脚本：原始 notebook 流程 vs API 推理流程
确保两者输出一致
"""
import os
import sys
import json
import requests
import numpy as np
import torch
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sleepfm"))
from preprocessing.preprocessing import EDFToHDF5Converter
from models.dataset import SetTransformerDataset, collate_fn
from models.models import SetTransformer, SleepEventLSTMClassifier, DiagnosisFinetuneFullLSTMCOXPHWithDemo
from utils import load_config, load_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLEEPFM_ROOT = os.path.join(PROJECT_ROOT, "sleepfm")
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebooks")
API_BASE = "http://localhost:6006"

EDF_PATH = os.path.join(NOTEBOOK_DIR, "demo_data", "demo_psg.edf")
DEMO_AGE = 0.030612244897959183
DEMO_GENDER = 0


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def strip_module_prefix(state_dict):
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def run_notebook_pipeline():
    """复现 notebook 完整推理流程"""
    print("=" * 60)
    print("原始 Notebook 推理流程")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="compare_test_")

    # Part 0: 预处理
    print("\n[Step 0] 预处理 EDF -> HDF5")
    hdf5_path = os.path.join(tmp_dir, "demo_psg.hdf5")
    converter = EDFToHDF5Converter(root_dir=os.path.dirname(EDF_PATH), target_dir=tmp_dir, resample_rate=128)
    converter.convert(EDF_PATH, hdf5_path)
    print(f"  HDF5 已保存: {hdf5_path}")

    # Part 1: 生成 embeddings
    print("\n[Step 1] 生成 embeddings (SetTransformer)")
    model_path = os.path.join(SLEEPFM_ROOT, "checkpoints", "model_base")
    channel_groups_path = os.path.join(SLEEPFM_ROOT, "configs", "channel_groups.json")

    config = load_config(os.path.join(model_path, "config.json"))
    channel_groups = load_data(channel_groups_path)

    model = SetTransformer(
        in_channels=config["in_channels"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        pooling_head=config["pooling_head"],
        dropout=0.0,
    )
    ckpt = torch.load(os.path.join(model_path, "best.pt"), map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device).eval()

    dataset = SetTransformerDataset(config, channel_groups, hdf5_paths=[hdf5_path], split="test")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False, collate_fn=collate_fn)

    emb_dir = os.path.join(tmp_dir, "emb")
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, "demo_psg.hdf5")

    embed_dim = config["embed_dim"]
    modality_types = config["modality_types"]

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
                                mod_type, data=emb_slice,
                                chunks=(embed_dim,) + emb_slice.shape[1:],
                                maxshape=(None,) + emb_slice.shape[1:],
                            )

    nb_embeddings = {}
    with h5py.File(emb_path, "r") as hf:
        for mod in modality_types:
            if mod in hf:
                nb_embeddings[mod] = hf[mod][:]
                print(f"  {mod}: shape={hf[mod][:].shape}")

    # Part 2: 睡眠分期
    print("\n[Step 2] 睡眠分期 (SleepEventLSTMClassifier)")
    ss_model_path = os.path.join(SLEEPFM_ROOT, "checkpoints", "model_sleep_staging")
    ss_config = load_json(os.path.join(ss_model_path, "config.json"))
    ss_model = SleepEventLSTMClassifier(**ss_config["model_params"])
    ss_ckpt = torch.load(os.path.join(ss_model_path, "best.pth"), map_location=device)
    ss_model.load_state_dict(strip_module_prefix(ss_ckpt))
    ss_model.to(device).eval()

    channel_like = ss_config.get("channel_like", ss_config.get("modality_types"))
    max_seq_len = ss_config["model_params"]["max_seq_length"]
    max_channels = ss_config.get("max_channels", 4)

    x_data = []
    with h5py.File(emb_path, "r") as hf:
        for key in hf.keys():
            if key in channel_like:
                x_data.append(hf[key][:])

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
        outputs, out_mask = ss_model(padded, mask)
        nb_ss_probs = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
        nb_ss_mask = out_mask.cpu().numpy()[0]

    valid_mask = nb_ss_mask == 0
    nb_ss_valid_probs = nb_ss_probs[valid_mask]
    nb_ss_predictions = np.argmax(nb_ss_valid_probs, axis=1)

    stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    from collections import Counter
    counts = Counter(nb_ss_predictions)
    print(f"  总 epochs: {len(nb_ss_predictions)}")
    for idx in sorted(counts.keys()):
        pct = counts[idx] / len(nb_ss_predictions) * 100
        print(f"  {stage_names[idx]}: {counts[idx]} ({pct:.1f}%)")

    # Part 3: 疾病预测
    print("\n[Step 3] 疾病预测 (DiagnosisFinetuneFullLSTMCOXPHWithDemo)")
    disease_model_path = os.path.join(SLEEPFM_ROOT, "checkpoints", "model_diagnosis")
    disease_config = load_json(os.path.join(disease_model_path, "config.json"))
    disease_params = dict(disease_config["model_params"])
    disease_params["dropout"] = 0.0
    disease_model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(**disease_params)
    disease_ckpt = torch.load(os.path.join(disease_model_path, "best.pth"), map_location=device)
    disease_model.load_state_dict(strip_module_prefix(disease_ckpt))
    disease_model.to(device).eval()

    d_modality_types = disease_config.get("modality_types", ["BAS", "RESP", "EKG", "EMG"])
    d_max_seq_len = disease_config["model_params"]["max_seq_length"]
    d_max_channels = disease_config.get("max_channels", 4)

    x_data_d = []
    with h5py.File(emb_path, "r") as hf:
        for key in hf.keys():
            if key in d_modality_types:
                x_data_d.append(hf[key][:])

    x_data_d = np.array(x_data_d)
    x_tensor_d = torch.tensor(x_data_d, dtype=torch.float32)
    c, s, e = x_tensor_d.shape
    num_ch = min(c, d_max_channels)
    seq_len = min(s, d_max_seq_len)

    padded_d = torch.zeros((1, d_max_channels, d_max_seq_len, e))
    mask_d = torch.ones((1, d_max_channels, d_max_seq_len))
    padded_d[0, :num_ch, :seq_len, :] = x_tensor_d[:num_ch, :seq_len, :]
    mask_d[0, :num_ch, :seq_len] = 0

    demo = torch.tensor([[DEMO_AGE, float(DEMO_GENDER)]], dtype=torch.float32)
    padded_d = padded_d.to(device)
    mask_d = mask_d.to(device)
    demo = demo.to(device)

    with torch.no_grad():
        nb_hazards = disease_model(padded_d, mask_d, demo).cpu().numpy()[0]

    sorted_indices = np.argsort(-nb_hazards)
    print(f"  Top 5 疾病风险:")
    import pandas as pd
    label_df = pd.read_csv(os.path.join(SLEEPFM_ROOT, "configs", "label_mapping.csv"))
    for rank, idx in enumerate(sorted_indices[:5], start=1):
        row = label_df[label_df["label_idx"] == int(idx)]
        name = row["phenotype"].values[0] if len(row) > 0 else "unknown"
        print(f"    #{rank} {name} (hazard={nb_hazards[idx]:.6f})")

    return nb_embeddings, nb_ss_valid_probs, nb_hazards


def run_api_pipeline():
    """调用 API 的完整推理流程"""
    print("\n" + "=" * 60)
    print("API 推理流程")
    print("=" * 60)

    print("\n[API] 调用 /api/v1/predict")
    with open(EDF_PATH, "rb") as f:
        r = requests.post(
            f"{API_BASE}/api/v1/predict",
            files={"file": ("demo_psg.edf", f, "application/octet-stream")},
            data={
                "age": DEMO_AGE,
                "gender": DEMO_GENDER,
                "tasks": "sleep_staging,disease_prediction",
            },
        )

    result = r.json()
    assert result["status"] == "success", f"API 返回错误: {result}"

    ss = result["sleep_staging"]
    dp = result["disease_prediction"]

    print(f"  睡眠分期总 epochs: {ss['total_epochs']}")
    for stage, info in ss["summary"].items():
        print(f"    {stage}: {info['count']} ({info['percentage']}%)")

    print(f"\n  Top 5 疾病风险:")
    for risk in dp["top_risks"][:5]:
        print(f"    #{risk['rank']} {risk['phenotype']} (hazard={risk['hazard_score']:.6f})")

    api_ss_probs = np.array([
        [e["probabilities"]["Wake"], e["probabilities"]["N1"],
         e["probabilities"]["N2"], e["probabilities"]["N3"],
         e["probabilities"]["REM"]]
        for e in ss["epochs"]
    ])

    api_hazards = np.zeros(dp["all_hazards_count"])
    for risk in dp["top_risks"]:
        api_hazards[risk["label_idx"]] = risk["hazard_score"]

    return api_ss_probs, api_hazards, dp["top_risks"]


def compare_results(nb_embeddings, nb_ss_probs, nb_hazards, api_ss_probs, api_hazards, api_top_risks):
    """对比结果"""
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    # 1. 睡眠分期对比
    print("\n--- 睡眠分期概率对比 ---")
    assert nb_ss_probs.shape == api_ss_probs.shape, \
        f"Shape 不一致: notebook={nb_ss_probs.shape}, API={api_ss_probs.shape}"
    print(f"  Shape 一致: {nb_ss_probs.shape}")

    max_diff = np.max(np.abs(nb_ss_probs - api_ss_probs))
    mean_diff = np.mean(np.abs(nb_ss_probs - api_ss_probs))
    print(f"  概率最大绝对差: {max_diff:.8f}")
    print(f"  概率平均绝对差: {mean_diff:.8f}")

    nb_pred = np.argmax(nb_ss_probs, axis=1)
    api_pred = np.argmax(api_ss_probs, axis=1)
    match_rate = np.mean(nb_pred == api_pred) * 100
    print(f"  预测一致率: {match_rate:.2f}%")

    if max_diff < 1e-4:
        print("  ✓ 睡眠分期结果一致 (差异 < 1e-4)")
    else:
        print(f"  ✗ 睡眠分期结果存在差异 (最大差 = {max_diff:.8f})")

    # 2. 疾病预测对比
    print("\n--- 疾病预测 hazard 对比 (Top 20) ---")
    nb_sorted = np.argsort(-nb_hazards)[:20]
    api_sorted_indices = [r["label_idx"] for r in api_top_risks]

    rank_match = sum(1 for a, b in zip(nb_sorted, api_sorted_indices) if a == b)
    print(f"  Top 20 排名一致数: {rank_match}/20")

    import pandas as pd
    label_df = pd.read_csv(os.path.join(SLEEPFM_ROOT, "configs", "label_mapping.csv"))

    print(f"\n  {'排名':<6} {'Notebook hazard':<18} {'API hazard':<18} {'差值':<14} {'一致?':<6} {'疾病'}")
    print(f"  {'-'*90}")
    for rank, nb_idx in enumerate(nb_sorted, start=1):
        nb_h = nb_hazards[nb_idx]
        api_risk = next((r for r in api_top_risks if r["label_idx"] == nb_idx), None)
        api_h = api_risk["hazard_score"] if api_risk else 0.0
        diff = abs(nb_h - api_h)
        match = "✓" if diff < 1e-4 else "✗"
        row = label_df[label_df["label_idx"] == int(nb_idx)]
        name = row["phenotype"].values[0] if len(row) > 0 else "unknown"
        print(f"  #{rank:<5} {nb_h:<18.6f} {api_h:<18.6f} {diff:<14.8f} {match:<6} {name[:40]}")

    # 总体 hazard 对比 (只比较 top 20 的值)
    top_nb = nb_hazards[nb_sorted]
    top_api = np.array([next((r["hazard_score"] for r in api_top_risks if r["label_idx"] == idx), 0.0)
                        for idx in nb_sorted])
    hazard_max_diff = np.max(np.abs(top_nb - top_api))
    hazard_mean_diff = np.mean(np.abs(top_nb - top_api))
    print(f"\n  Top 20 hazard 最大绝对差: {hazard_max_diff:.8f}")
    print(f"  Top 20 hazard 平均绝对差: {hazard_mean_diff:.8f}")

    if hazard_max_diff < 1e-4:
        print("  ✓ 疾病预测结果一致 (差异 < 1e-4)")
    else:
        print(f"  ✗ 疾病预测结果存在差异 (最大差 = {hazard_max_diff:.8f})")

    print("\n" + "=" * 60)
    all_ok = max_diff < 1e-4 and hazard_max_diff < 1e-4
    if all_ok:
        print("总结: 所有对比结果一致，API 输出与原始 notebook 流程完全匹配")
    else:
        print("总结: 存在差异，请检查上述详情")
    print("=" * 60)


if __name__ == "__main__":
    nb_emb, nb_ss, nb_hz = run_notebook_pipeline()
    api_ss, api_hz, api_risks = run_api_pipeline()
    compare_results(nb_emb, nb_ss, nb_hz, api_ss, api_hz, api_risks)
