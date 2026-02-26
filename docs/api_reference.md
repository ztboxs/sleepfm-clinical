# SleepFM-Clinical API 接口文档

> **版本**: v1.0.0  
> **基础路径**: `http://<服务器IP>:6006`  
> **协议**: HTTP  
> **数据格式**: JSON（响应）/ multipart/form-data（文件上传）  
> **Swagger 文档**: `http://<服务器IP>:6006/docs`  
> **ReDoc 文档**: `http://<服务器IP>:6006/redoc`

---

## 目录

- [1. 概述](#1-概述)
- [2. 通用说明](#2-通用说明)
- [3. 接口详情](#3-接口详情)
  - [3.1 健康检查](#31-健康检查)
  - [3.2 疾病标签映射](#32-疾病标签映射)
  - [3.3 EDF 数据预处理](#33-edf-数据预处理)
  - [3.4 生成嵌入向量](#34-生成嵌入向量)
  - [3.5 完整推理](#35-完整推理)
- [4. 数据模型定义](#4-数据模型定义)
- [5. 错误码说明](#5-错误码说明)
- [6. 调用示例](#6-调用示例)
- [7. 附录：EDF 文件兼容性要求](#7-附录edf-文件兼容性要求)

---

## 1. 概述

SleepFM-Clinical API 基于 SleepFM 多模态睡眠基础模型，提供多导睡眠图 (PSG) 的自动化分析服务，支持以下核心能力：

| 能力 | 说明 |
|------|------|
| **睡眠分期** | 将 PSG 记录逐 5 秒时段分类为 Wake / N1 / N2 / N3 / REM 五种睡眠阶段 |
| **疾病风险预测** | 基于 PSG 嵌入特征 + 人口学信息，预测 1065 种疾病的 Cox 比例风险评分 |

推理流水线：`EDF 文件上传 → 预处理(重采样/滤波/标准化) → HDF5 → SetTransformer 嵌入 → 下游任务推理`

---

## 2. 通用说明

### 2.1 请求规范

- 文件上传接口使用 `multipart/form-data` 编码
- 仅接受 `.edf` 扩展名的文件
- 文件大小限制：500 MB

### 2.2 响应规范

- 所有响应均为 `application/json` 格式
- 成功响应包含 `"status": "success"` 字段
- 错误响应包含 `"detail"` 字段描述错误原因

### 2.3 并发限制

- GPU 推理任务通过全局锁串行执行，同一时刻仅处理一个推理请求
- 非 GPU 任务（如健康检查、标签查询）不受此限制

---

## 3. 接口详情

### 3.1 健康检查

检查服务运行状态、GPU 信息和模型加载情况。

| 项目 | 说明 |
|------|------|
| **请求方式** | `GET` |
| **请求路径** | `/api/v1/health` |
| **认证** | 无 |

#### 请求参数

无。

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `string` | 服务状态，正常时为 `"healthy"` |
| `gpu` | `object` | GPU 信息对象 |
| `gpu.name` | `string` | GPU 设备名称，如 `"NVIDIA GeForce RTX 4090"`；无 GPU 时为 `"CPU only"` |
| `gpu.memory_total_mb` | `integer` | GPU 总显存（MB） |
| `gpu.memory_used_mb` | `integer` | GPU 已使用显存（MB） |
| `models_loaded` | `object` | 模型加载状态对象 |
| `models_loaded.base_model` | `boolean` | 基础嵌入模型 (SetTransformer) 是否已加载 |
| `models_loaded.sleep_staging_model` | `boolean` | 睡眠分期模型 (SleepEventLSTMClassifier) 是否已加载 |
| `models_loaded.disease_prediction_model` | `boolean` | 疾病预测模型 (DiagnosisFinetuneFullLSTMCOXPHWithDemo) 是否已加载 |

#### 响应示例

```json
{
  "status": "healthy",
  "gpu": {
    "name": "NVIDIA GeForce RTX 4090",
    "memory_total_mb": 24217,
    "memory_used_mb": 34
  },
  "models_loaded": {
    "base_model": true,
    "sleep_staging_model": true,
    "disease_prediction_model": true
  }
}
```

---

### 3.2 疾病标签映射

获取模型支持的全部 1065 种疾病标签映射表（疾病索引、Phecode 编码、表现型名称）。

| 项目 | 说明 |
|------|------|
| **请求方式** | `GET` |
| **请求路径** | `/api/v1/label_mapping` |
| **认证** | 无 |

#### 请求参数

无。

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | `integer` | 疾病标签总数 |
| `mappings` | `array[LabelMappingItem]` | 标签映射列表 |

**LabelMappingItem 对象：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `label_idx` | `integer` | 疾病索引编号（0~1064） |
| `phecode` | `string` | Phecode 疾病分类编码，如 `"290.1"` |
| `phenotype` | `string` | 疾病表现型名称（英文），如 `"Dementia"` |

#### 响应示例

```json
{
  "total": 1065,
  "mappings": [
    {"label_idx": 0, "phecode": "008.5", "phenotype": "Intestinal infection due to C. difficile"},
    {"label_idx": 1, "phecode": "008.7", "phenotype": "Intestinal infection"},
    ...
  ]
}
```

---

### 3.3 EDF 数据预处理

上传 EDF 文件，执行预处理（读取通道、重采样至 128Hz、滤波、标准化），返回通道匹配信息和录制统计数据。

| 项目 | 说明 |
|------|------|
| **请求方式** | `POST` |
| **请求路径** | `/api/v1/preprocess` |
| **Content-Type** | `multipart/form-data` |
| **认证** | 无 |

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | `file` | 是 | PSG 记录文件，仅接受 `.edf` 格式 |

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `string` | 处理状态，成功时为 `"success"` |
| `channels` | `ChannelInfo` | 按模态分组的通道匹配结果 |
| `channels.BAS` | `array[string]` | 匹配到 BAS（脑电/眼电）模态的通道名列表 |
| `channels.RESP` | `array[string]` | 匹配到 RESP（呼吸）模态的通道名列表 |
| `channels.EKG` | `array[string]` | 匹配到 EKG（心电）模态的通道名列表 |
| `channels.EMG` | `array[string]` | 匹配到 EMG（肌电）模态的通道名列表 |
| `duration_seconds` | `float` | 原始录制时长（秒） |
| `sample_rate` | `integer` | 重采样后的采样率（固定为 128 Hz） |

#### 响应示例

```json
{
  "status": "success",
  "channels": {
    "BAS": ["C3-A2"],
    "RESP": ["Airflow"],
    "EKG": ["EKG"],
    "EMG": ["Arm EMG"]
  },
  "duration_seconds": 8729.0,
  "sample_rate": 128
}
```

---

### 3.4 生成嵌入向量

上传 EDF 文件，执行预处理后通过 SetTransformer 基础模型生成各模态的 5 秒粒度嵌入向量。

| 项目 | 说明 |
|------|------|
| **请求方式** | `POST` |
| **请求路径** | `/api/v1/embed` |
| **Content-Type** | `multipart/form-data` |
| **认证** | 无 |

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | `file` | 是 | PSG 记录文件，仅接受 `.edf` 格式 |

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `string` | 处理状态，成功时为 `"success"` |
| `embeddings` | `object` | 各模态嵌入信息，key 为模态名称 (`BAS`/`RESP`/`EKG`/`EMG`) |
| `embeddings.<MOD>` | `EmbeddingInfo` | 单个模态的嵌入信息 |
| `embeddings.<MOD>.shape` | `array[integer]` | 嵌入张量维度，格式为 `[时间步数, 嵌入维度]`，如 `[1740, 128]` |
| `num_5min_chunks` | `integer` | 5 分钟分片数量 |
| `embed_dim` | `integer` | 嵌入向量维度（固定为 128） |

#### 响应示例

```json
{
  "status": "success",
  "embeddings": {
    "BAS": {"shape": [1740, 128]},
    "RESP": {"shape": [1740, 128]},
    "EKG": {"shape": [1740, 128]},
    "EMG": {"shape": [1740, 128]}
  },
  "num_5min_chunks": 29,
  "embed_dim": 128
}
```

---

### 3.5 完整推理

上传 EDF 文件，执行完整的推理流水线：预处理 → 嵌入生成 → 睡眠分期和/或疾病风险预测。这是最常用的接口，一次调用即可获取所有分析结果。

| 项目 | 说明 |
|------|------|
| **请求方式** | `POST` |
| **请求路径** | `/api/v1/predict` |
| **Content-Type** | `multipart/form-data` |
| **认证** | 无 |

#### 请求参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | `file` | 是 | — | PSG 记录文件，仅接受 `.edf` 格式 |
| `age` | `float` | 条件必填 | `null` | 归一化年龄值，范围 `[0, 1]`。当 `tasks` 包含 `disease_prediction` 时必填 |
| `gender` | `integer` | 条件必填 | `null` | 性别编码：`0` = 女性，`1` = 男性。当 `tasks` 包含 `disease_prediction` 时必填 |
| `tasks` | `string` | 否 | `"sleep_staging,disease_prediction"` | 要执行的推理任务，多个任务用逗号分隔。可选值：`sleep_staging`、`disease_prediction` |

#### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `string` | 处理状态，成功时为 `"success"` |
| `sleep_staging` | `SleepStagingResult \| null` | 睡眠分期结果。仅当 `tasks` 包含 `sleep_staging` 时返回 |
| `disease_prediction` | `DiseasePredictionResult \| null` | 疾病预测结果。仅当 `tasks` 包含 `disease_prediction` 时返回 |

**SleepStagingResult 对象：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_epochs` | `integer` | 有效 epoch 总数（每个 epoch 为 5 秒） |
| `epochs` | `array[SleepEpoch]` | 逐 epoch 的分期详情列表 |
| `summary` | `object` | 各睡眠阶段的统计摘要，key 为阶段名称 |

**SleepEpoch 对象：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `index` | `integer` | epoch 序号（从 0 开始） |
| `start_sec` | `float` | epoch 起始时间（秒） |
| `end_sec` | `float` | epoch 结束时间（秒） |
| `predicted_stage` | `string` | 预测的睡眠阶段。可选值：`Wake`、`N1`、`N2`、`N3`、`REM` |
| `probabilities` | `StageProbabilities` | 各阶段的预测概率 |

**StageProbabilities 对象：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `Wake` | `float` | 清醒阶段的预测概率 |
| `N1` | `float` | N1 浅睡阶段的预测概率 |
| `N2` | `float` | N2 浅睡阶段的预测概率 |
| `N3` | `float` | N3 深睡阶段的预测概率 |
| `REM` | `float` | REM 快速眼动阶段的预测概率 |

**StageSummaryItem 对象**（summary 中各 key 对应的值）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `count` | `integer` | 该阶段的 epoch 数量 |
| `percentage` | `float` | 该阶段占总 epoch 数的百分比 |

**DiseasePredictionResult 对象：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `top_risks` | `array[DiseaseRiskItem]` | 按风险评分降序排列的 Top 20 疾病风险列表 |
| `all_hazards_count` | `integer` | 模型输出的疾病总数（固定为 1065） |

**DiseaseRiskItem 对象：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `rank` | `integer` | 风险排名（从 1 开始） |
| `label_idx` | `integer` | 疾病索引编号（对应 label_mapping 中的 `label_idx`） |
| `phecode` | `string` | Phecode 疾病分类编码 |
| `phenotype` | `string` | 疾病表现型名称（英文） |
| `hazard_score` | `float` | Cox 比例风险评分，值越大表示风险越高 |

#### 响应示例

**请求：睡眠分期 + 疾病预测**

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
          "Wake": 0.9512,
          "N1": 0.0203,
          "N2": 0.0148,
          "N3": 0.0065,
          "REM": 0.0072
        }
      },
      {
        "index": 1,
        "start_sec": 5.0,
        "end_sec": 10.0,
        "predicted_stage": "Wake",
        "probabilities": {
          "Wake": 0.9487,
          "N1": 0.0211,
          "N2": 0.0153,
          "N3": 0.0076,
          "REM": 0.0073
        }
      }
    ],
    "summary": {
      "Wake": {"count": 1740, "percentage": 100.0},
      "N1": {"count": 0, "percentage": 0.0},
      "N2": {"count": 0, "percentage": 0.0},
      "N3": {"count": 0, "percentage": 0.0},
      "REM": {"count": 0, "percentage": 0.0}
    }
  },
  "disease_prediction": {
    "top_risks": [
      {
        "rank": 1,
        "label_idx": 746,
        "phecode": "440.3",
        "phenotype": "Atherosclerosis of native arteries of the extremities with ulceration or gangrene",
        "hazard_score": 8.890937
      },
      {
        "rank": 2,
        "label_idx": 730,
        "phecode": "427.8",
        "phenotype": "Cardiac defibrillator in situ",
        "hazard_score": 8.884488
      }
    ],
    "all_hazards_count": 1065
  }
}
```

**请求：仅睡眠分期**

当 `tasks=sleep_staging` 时，响应中 `disease_prediction` 为 `null`。

**请求：仅疾病预测**

当 `tasks=disease_prediction` 时，响应中 `sleep_staging` 为 `null`，且必须提供 `age` 和 `gender` 参数。

---

## 4. 数据模型定义

以下为所有响应中涉及的数据模型的完整定义。

### 4.1 HealthResponse

```
{
  "status":        string,     // 服务状态
  "gpu":           GpuInfo,    // GPU 信息
  "models_loaded": object      // 模型加载状态 (key: 模型名, value: boolean)
}
```

### 4.2 GpuInfo

```
{
  "name":            string,   // GPU 设备名称
  "memory_total_mb": integer,  // 总显存 (MB)
  "memory_used_mb":  integer   // 已使用显存 (MB)
}
```

### 4.3 LabelMappingResponse

```
{
  "total":    integer,                // 标签总数
  "mappings": [LabelMappingItem]      // 标签列表
}
```

### 4.4 LabelMappingItem

```
{
  "label_idx": integer,  // 疾病索引编号 (0~1064)
  "phecode":   string,   // Phecode 疾病分类编码
  "phenotype": string    // 疾病表现型名称
}
```

### 4.5 PreprocessResponse

```
{
  "status":           string,       // 处理状态
  "channels":         ChannelInfo,  // 通道匹配结果
  "duration_seconds": float,        // 录制时长 (秒)
  "sample_rate":      integer       // 重采样率 (Hz)
}
```

### 4.6 ChannelInfo

```
{
  "BAS":  [string],  // 脑电/眼电通道列表
  "RESP": [string],  // 呼吸通道列表
  "EKG":  [string],  // 心电通道列表
  "EMG":  [string]   // 肌电通道列表
}
```

### 4.7 EmbedResponse

```
{
  "status":          string,                     // 处理状态
  "embeddings":      {string: EmbeddingInfo},    // 各模态嵌入信息
  "num_5min_chunks": integer,                    // 5分钟分片数
  "embed_dim":       integer                     // 嵌入维度
}
```

### 4.8 EmbeddingInfo

```
{
  "shape": [integer]  // 嵌入张量维度 [时间步数, 嵌入维度]
}
```

### 4.9 PredictResponse

```
{
  "status":             string,                        // 处理状态
  "sleep_staging":      SleepStagingResult | null,     // 睡眠分期结果
  "disease_prediction": DiseasePredictionResult | null  // 疾病预测结果
}
```

### 4.10 SleepStagingResult

```
{
  "total_epochs": integer,                         // 有效 epoch 总数
  "epochs":       [SleepEpoch],                    // 逐 epoch 详情
  "summary":      {string: StageSummaryItem}       // 各阶段统计 (key: 阶段名)
}
```

### 4.11 SleepEpoch

```
{
  "index":           integer,             // epoch 序号
  "start_sec":       float,               // 起始时间 (秒)
  "end_sec":         float,               // 结束时间 (秒)
  "predicted_stage": string,              // 预测阶段: Wake/N1/N2/N3/REM
  "probabilities":   StageProbabilities   // 各阶段概率
}
```

### 4.12 StageProbabilities

```
{
  "Wake": float,  // 清醒概率
  "N1":   float,  // N1 浅睡概率
  "N2":   float,  // N2 浅睡概率
  "N3":   float,  // N3 深睡概率
  "REM":  float   // REM 快速眼动概率
}
```

### 4.13 StageSummaryItem

```
{
  "count":      integer,  // epoch 数量
  "percentage": float     // 占比 (%)
}
```

### 4.14 DiseasePredictionResult

```
{
  "top_risks":         [DiseaseRiskItem],  // Top 20 疾病风险
  "all_hazards_count": integer             // 疾病总数 (1065)
}
```

### 4.15 DiseaseRiskItem

```
{
  "rank":         integer,  // 风险排名 (从1开始)
  "label_idx":    integer,  // 疾病索引编号
  "phecode":      string,   // Phecode 编码
  "phenotype":    string,   // 疾病名称
  "hazard_score": float     // Cox 风险评分
}
```

### 4.16 ErrorResponse

```
{
  "detail": string  // 错误描述信息
}
```

---

## 5. 错误码说明

| HTTP 状态码 | 含义 | 常见场景 |
|-------------|------|----------|
| `200` | 成功 | 请求正常处理 |
| `400` | 请求参数错误 | 上传非 `.edf` 文件；疾病预测缺少 `age` 或 `gender` |
| `422` | 数据处理错误 | EDF 文件通道不匹配所需模态；嵌入生成失败 |
| `500` | 服务器内部错误 | 未预期的异常 |

#### 错误响应示例

**400 - 文件格式错误：**

```json
{
  "detail": "Only .edf files are accepted"
}
```

**400 - 缺少必填参数：**

```json
{
  "detail": "age and gender are required for disease_prediction task"
}
```

**422 - 通道不匹配：**

```json
{
  "detail": "EDF 文件的通道无法匹配所有必需的模态。文件通道: ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']，缺失模态: ['BAS', 'EKG']，各模态匹配: {'BAS': [], 'RESP': [], 'EKG': [], 'EMG': []}。请确保 EDF 文件包含 BAS(脑电/眼电)、RESP(呼吸)、EKG(心电)、EMG(肌电) 四种模态的通道。"
}
```

---

## 6. 调用示例

### 6.1 curl

**健康检查：**

```bash
curl -X GET http://<服务器IP>:6006/api/v1/health
```

**疾病标签映射：**

```bash
curl -X GET http://<服务器IP>:6006/api/v1/label_mapping
```

**预处理：**

```bash
curl -X POST http://<服务器IP>:6006/api/v1/preprocess \
  -F "file=@/path/to/recording.edf"
```

**生成嵌入：**

```bash
curl -X POST http://<服务器IP>:6006/api/v1/embed \
  -F "file=@/path/to/recording.edf"
```

**完整推理（睡眠分期 + 疾病预测）：**

```bash
curl -X POST http://<服务器IP>:6006/api/v1/predict \
  -F "file=@/path/to/recording.edf" \
  -F "age=0.45" \
  -F "gender=1" \
  -F "tasks=sleep_staging,disease_prediction"
```

**仅睡眠分期：**

```bash
curl -X POST http://<服务器IP>:6006/api/v1/predict \
  -F "file=@/path/to/recording.edf" \
  -F "tasks=sleep_staging"
```

**仅疾病预测：**

```bash
curl -X POST http://<服务器IP>:6006/api/v1/predict \
  -F "file=@/path/to/recording.edf" \
  -F "age=0.45" \
  -F "gender=0" \
  -F "tasks=disease_prediction"
```

### 6.2 Python (requests)

```python
import requests

BASE_URL = "http://<服务器IP>:6006"

# 健康检查
resp = requests.get(f"{BASE_URL}/api/v1/health")
print(resp.json())

# 完整推理
with open("/path/to/recording.edf", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/api/v1/predict",
        files={"file": ("recording.edf", f, "application/octet-stream")},
        data={
            "age": 0.45,
            "gender": 1,
            "tasks": "sleep_staging,disease_prediction",
        },
    )

result = resp.json()

# 读取睡眠分期结果
if result.get("sleep_staging"):
    staging = result["sleep_staging"]
    print(f"总 epochs: {staging['total_epochs']}")
    for stage, info in staging["summary"].items():
        print(f"  {stage}: {info['count']} ({info['percentage']}%)")

# 读取疾病预测结果
if result.get("disease_prediction"):
    prediction = result["disease_prediction"]
    for risk in prediction["top_risks"][:5]:
        print(f"  #{risk['rank']} {risk['phenotype']}: {risk['hazard_score']:.4f}")
```

### 6.3 JavaScript (fetch)

```javascript
// 健康检查
const healthResp = await fetch('http://<服务器IP>:6006/api/v1/health');
const health = await healthResp.json();

// 完整推理
const formData = new FormData();
formData.append('file', edfFileBlob, 'recording.edf');
formData.append('age', '0.45');
formData.append('gender', '1');
formData.append('tasks', 'sleep_staging,disease_prediction');

const resp = await fetch('http://<服务器IP>:6006/api/v1/predict', {
  method: 'POST',
  body: formData,
});
const result = await resp.json();
```

---

## 7. 附录：EDF 文件兼容性要求

### 通道匹配规则

模型要求 EDF 文件包含 **全部四种模态** 的通道，每种模态至少匹配 1 个通道：

| 模态 | 说明 | 常见匹配通道名示例 |
|------|------|-------------------|
| **BAS** | 脑电 (EEG) / 眼电 (EOG) | `C3-A2`, `C4-A1`, `F3-M2`, `O1-M2`, `EOG(L)`, `EOG(R)`, `EEG`, `E1`, `E2`, `Fp1`, `Fp2` |
| **RESP** | 呼吸相关 | `Airflow`, `SpO2`, `Chest`, `Abd`, `Nasal`, `Pulse`, `PPG`, `SaO2` |
| **EKG** | 心电 | `EKG`, `ECG`, `ECG II`, `ECG1`, `ECG2` |
| **EMG** | 肌电 | `EMG`, `Arm EMG`, `Chin`, `Chin EMG`, `LEG(L)`, `LEG(R)`, `Leg L`, `Leg R` |

通道名匹配区分大小写，完整的支持通道列表见项目文件 `sleepfm/configs/channel_groups.json`。

### 已知不兼容的公开数据集

| 数据集 | 原因 |
|--------|------|
| Sleep-EDF (Cassette/Telemetry) | 通道名格式不同（如 `EEG Fpz-Cz`）不在支持列表中 |
| 部分 PhysioNet 公开数据集 | 通道命名规范不同 |

### 推荐测试文件

项目自带演示文件 `notebooks/demo_data/demo_psg.edf`：
- 通道：`C3-A2`, `Airflow`, `Arm EMG`, `EKG`（四种模态齐全）
- 时长：约 8729 秒（~2.4 小时）
- 对应人口学参数：`age=0.0306`, `gender=0`
