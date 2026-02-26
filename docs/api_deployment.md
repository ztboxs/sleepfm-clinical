# SleepFM-Clinical HTTP API 服务部署文档

## 1. 项目概述

基于 [SleepFM-Clinical](https://github.com/zou-group/sleepfm-clinical) 开源项目，将其多模态睡眠基础模型封装为标准 HTTP API 服务，部署在云服务器上，支持公网访问。

### 1.1 模型说明

| 模型 | 类名 | 检查点 | 参数量 | 功能 |
|------|------|--------|--------|------|
| 基座模型 | SetTransformer | `model_base/best.pt` | 4.44M | 从原始 PSG 信号生成嵌入向量 |
| 睡眠分期模型 | SleepEventLSTMClassifier | `model_sleep_staging/best.pth` | 1.19M | 5 类睡眠阶段分类 (Wake/N1/N2/N3/REM) |
| 疾病预测模型 | DiagnosisFinetuneFullLSTMCOXPHWithDemo | `model_diagnosis/best.pth` | 0.91M | 1065 种疾病 Cox 风险预测 |

### 1.2 服务器环境

- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Python: 3.10 (conda env: sleepfm_env)
- PyTorch: 2.0.1
- CUDA: 可用
- 公网 IP: 116.172.96.31

## 2. 推理流程

### 2.1 端到端推理管线

```
EDF 文件上传
    │
    ▼
[Step 0] 预处理 (EDF → HDF5)
    │  - 128Hz 重采样
    │  - 低通滤波 (Butterworth, order 4)
    │  - Z-score 归一化 (per-channel)
    │  - 通道按模态分组 (BAS/RESP/EKG/EMG)
    ▼
[Step 1] 嵌入生成 (SetTransformer)
    │  - 按 5 分钟分块 (38400 samples)
    │  - 每个模态独立前向传播
    │  - 输出: 5 秒级嵌入 + 5 分钟聚合嵌入
    ▼
[Step 2a] 睡眠分期            [Step 2b] 疾病预测
    │                              │
    │  输入: 5 秒级嵌入            │  输入: 5 秒级嵌入 + 人口学特征(age, gender)
    │  模型: SleepEventLSTM        │  模型: DiagnosisCOXPH
    │  输出: 每 5 秒的 5 类概率    │  输出: 1065 种疾病风险值
    ▼                              ▼
返回 JSON 结果
```

### 2.2 数据格式

#### 输入
- **EDF 文件**: 多导睡眠图 (PSG) 标准格式，包含 EEG/EOG/ECG/EMG/呼吸等通道
- **人口学信息**: 年龄 (归一化 0~1)、性别 (0/1)

#### 输出 - 睡眠分期
- 每 5 秒一个 epoch 的睡眠阶段概率分布
- 5 类: Wake(0), N1(1), N2(2), N3(3), REM(4)

#### 输出 - 疾病预测
- 1065 种疾病的 Cox 比例风险值
- 疾病映射: `sleepfm/configs/label_mapping.csv`

## 3. HTTP API 接口规范

### 3.1 技术栈

- **Web 框架**: FastAPI
- **ASGI 服务器**: Uvicorn
- **数据校验**: Pydantic v2
- **文件上传**: python-multipart

### 3.2 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/label_mapping` | 获取疾病标签映射 |
| POST | `/api/v1/preprocess` | EDF 预处理 |
| POST | `/api/v1/embed` | 生成嵌入向量 |
| POST | `/api/v1/predict` | 完整推理 |

### 3.3 接口详细说明

#### GET /api/v1/health

健康检查接口，返回服务状态和 GPU 信息。

**响应示例:**
```json
{
  "status": "healthy",
  "gpu": {
    "name": "NVIDIA GeForce RTX 4090",
    "memory_total_mb": 24564,
    "memory_used_mb": 2048
  },
  "models_loaded": {
    "base_model": true,
    "sleep_staging_model": true,
    "disease_prediction_model": true
  }
}
```

#### GET /api/v1/label_mapping

返回 1065 种疾病的 Phecode 和表型名称映射。

**响应示例:**
```json
{
  "total": 1065,
  "mappings": [
    {"label_idx": 0, "phecode": "8.0", "phenotype": "Intestinal infection"},
    {"label_idx": 1, "phecode": "8.5", "phenotype": "Bacterial enteritis"},
    ...
  ]
}
```

#### POST /api/v1/preprocess

上传 EDF 文件进行预处理，返回通道信息和数据统计。

**请求**: `multipart/form-data`
- `file`: EDF 文件 (必须)

**响应示例:**
```json
{
  "status": "success",
  "channels": {
    "BAS": ["EEG C3-A2", "EEG C4-A1", ...],
    "RESP": ["Nasal Pressure", "Chest", ...],
    "EKG": ["ECG"],
    "EMG": ["Chin EMG"]
  },
  "duration_seconds": 28800,
  "sample_rate": 128
}
```

#### POST /api/v1/embed

上传 EDF 文件生成嵌入向量。

**请求**: `multipart/form-data`
- `file`: EDF 文件 (必须)

**响应示例:**
```json
{
  "status": "success",
  "embeddings": {
    "BAS": {"shape": [1740, 128], "data_url": "..."},
    "RESP": {"shape": [1740, 128], "data_url": "..."},
    "EKG": {"shape": [1740, 128], "data_url": "..."},
    "EMG": {"shape": [1740, 128], "data_url": "..."}
  },
  "num_chunks": 29,
  "embed_dim": 128
}
```

#### POST /api/v1/predict

完整推理接口，上传 EDF 文件和人口学信息，返回睡眠分期和/或疾病预测结果。

**请求**: `multipart/form-data`
- `file`: EDF 文件 (必须)
- `age`: float, 归一化年龄 0~1 (疾病预测必须)
- `gender`: int, 0 或 1 (疾病预测必须)
- `tasks`: str, 逗号分隔的任务列表 (可选, 默认 "sleep_staging,disease_prediction")

**响应示例:**
```json
{
  "status": "success",
  "sleep_staging": {
    "total_epochs": 1740,
    "epochs": [
      {
        "index": 0,
        "start_sec": 0.0,
        "end_sec": 5.0,
        "predicted_stage": "Wake",
        "probabilities": {
          "Wake": 0.82,
          "N1": 0.05,
          "N2": 0.08,
          "N3": 0.03,
          "REM": 0.02
        }
      }
    ],
    "summary": {
      "Wake": {"count": 640, "percentage": 36.8},
      "N1": {"count": 263, "percentage": 15.1},
      "N2": {"count": 276, "percentage": 15.9},
      "N3": {"count": 268, "percentage": 15.4},
      "REM": {"count": 293, "percentage": 16.8}
    }
  },
  "disease_prediction": {
    "top_risks": [
      {
        "rank": 1,
        "label_idx": 512,
        "phecode": "290.1",
        "phenotype": "Dementia",
        "hazard_score": 0.85
      }
    ],
    "all_hazards_count": 1065
  }
}
```

## 4. 项目结构

```
sleepfm-clinical/
├── api/
│   ├── __init__.py               # 包初始化
│   ├── main.py                   # FastAPI 应用入口 + Uvicorn 启动
│   ├── config.py                 # 服务配置 (路径、端口、限制等)
│   ├── models_manager.py         # 模型加载与管理 (单例模式)
│   ├── inference.py              # 推理逻辑封装
│   ├── schemas.py                # Pydantic 请求/响应模型定义
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── predict.py            # 推理路由 (/predict, /preprocess, /embed)
│   │   └── health.py             # 健康检查和标签映射路由
│   └── utils.py                  # 工具函数 (临时文件管理等)
├── docs/
│   └── api_deployment.md         # 本文档
├── sleepfm/                      # 原项目代码 (不修改)
│   ├── checkpoints/
│   ├── configs/
│   ├── models/
│   ├── pipeline/
│   ├── preprocessing/
│   └── utils.py
├── notebooks/
├── requirements.txt
└── env.yml
```

## 5. 关键实现细节

### 5.1 模型加载策略

- 应用启动时 (lifespan event) 一次性加载三个模型到 GPU
- 单例模式管理，避免重复加载
- 推理时使用 `torch.no_grad()` 减少显存占用
- 无需 DataParallel（单 GPU），但检查点中的权重 key 带有 `module.` 前缀需要处理

### 5.2 DataParallel 权重处理

原项目用 `nn.DataParallel` 包装模型后保存检查点，权重 key 带有 `module.` 前缀。
在 API 服务中直接用单 GPU，需要在加载时去除前缀:

```python
state_dict = checkpoint.get("state_dict", checkpoint)
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
```

### 5.3 文件处理策略

- 上传的 EDF 文件保存到临时目录
- 中间产物 (HDF5, embedding HDF5) 也存临时目录
- 推理完成后清理所有临时文件
- 使用 Python `tempfile` 模块管理

### 5.4 并发控制

- 使用 `asyncio.Lock` 对 GPU 推理加锁
- 同一时间只允许一个推理请求使用 GPU
- 避免 OOM 和资源竞争

### 5.5 CORS 配置

允许所有来源跨域访问:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 6. 部署和运行

### 6.1 安装额外依赖

```bash
conda activate sleepfm_env
pip install fastapi uvicorn[standard] python-multipart
```

### 6.2 启动服务

```bash
cd /root/autodl-tmp/work/sleepfm-clinical
conda activate sleepfm_env
python -m api.main
```

服务启动后:
- 本地访问: `http://localhost:6006`
- 公网访问: `http://<公网IP>:6006`
- API 文档: `http://<公网IP>:6006/docs`

### 6.3 测试

```bash
# 健康检查
curl http://localhost:6006/api/v1/health

# 完整推理 (使用 demo 数据)
curl -X POST http://localhost:6006/api/v1/predict \
  -F "file=@notebooks/demo_data/demo_psg.edf" \
  -F "age=0.03" \
  -F "gender=0" \
  -F "tasks=sleep_staging,disease_prediction"
```

## 7. 注意事项

1. `channel_groups.json` 中定义了通道到模态的映射，用户上传的 EDF 文件通道名必须能匹配
2. 疾病预测需要提供年龄和性别信息，不提供则跳过此任务
3. EDF 文件大小建议不超过 500MB
4. 单次推理耗时取决于 PSG 录制时长，典型 8 小时录制约需 30-60 秒
