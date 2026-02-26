# SleepFM-Clinical API 测试指南

## 1. 前置准备

### 1.1 确认服务已启动

```bash
conda activate sleepfm_env
cd /root/autodl-tmp/work/sleepfm-clinical
python -m api.main
```

服务启动后会输出：
```
Models loaded. API is ready.
Uvicorn running on http://0.0.0.0:6006
```

### 1.2 测试数据

项目自带演示数据，路径为：

```
notebooks/demo_data/
├── demo_psg.edf          # 演示 PSG 记录文件 (18MB)
├── demo_age_gender.csv   # 人口学信息 (age=0.0306, gender=0)
├── demo_psg.csv          # PSG 信号元数据
├── is_event.csv          # 疾病事件标注
└── time_to_event.csv     # 事件时间标注
```

**`demo_psg.edf` 文件信息：**

| 属性     | 值                              |
|----------|--------------------------------|
| 通道     | C3-A2, Airflow, Arm EMG, EKG   |
| 模态覆盖 | BAS, RESP, EMG, EKG（四种齐全） |
| 采样率   | 256 Hz                          |
| 时长     | 约 8729 秒（~2.4 小时）         |
| 文件大小 | 18 MB                           |

**人口学参数（用于疾病预测）：**

| 参数   | 值       | 说明                    |
|--------|----------|------------------------|
| age    | 0.0306   | 归一化年龄（范围 0~1）  |
| gender | 0        | 0 = 女性，1 = 男性      |

---

## 2. 通过 Web 界面测试

浏览器访问：`http://<服务器IP>:6006/`

### 2.1 检查系统状态

1. 点击 **系统状态** 标签页
2. 确认三个模型均显示 **已加载**
3. 确认 GPU 信息正常

### 2.2 完整推理测试

1. 点击 **完整推理** 标签页
2. 点击上传区域，选择 `notebooks/demo_data/demo_psg.edf`
3. 填写参数：
   - 年龄：`0.0306`
   - 性别：`女性 (0)`
4. 勾选任务：**睡眠分期** + **疾病风险预测**
5. 点击 **开始推理**
6. 等待处理完成（约 30-60 秒），查看结果：
   - **睡眠分期** 标签：查看各睡眠阶段分布和逐时段详情
   - **疾病风险** 标签：查看 Top 20 疾病风险排名
   - **原始 JSON** 标签：查看完整响应数据

### 2.3 仅测试预处理

1. 点击 **数据预处理** 标签页
2. 上传 `demo_psg.edf`
3. 点击 **开始预处理**
4. 查看通道分组结果：应看到 BAS、RESP、EKG、EMG 各有匹配通道

### 2.4 仅测试嵌入生成

1. 点击 **生成嵌入** 标签页
2. 上传 `demo_psg.edf`
3. 点击 **生成嵌入**
4. 查看四种模态的嵌入维度信息

### 2.5 查看疾病标签映射

1. 点击 **疾病标签映射** 标签页
2. 浏览 1065 种疾病编码
3. 使用搜索框过滤，例如输入 `diabetes` 或 `290`

---

## 3. 通过命令行 (curl) 测试

以下命令均在服务器上执行。如需远程测试，将 `localhost` 替换为服务器公网 IP。

### 3.1 健康检查

```bash
curl http://localhost:6006/api/v1/health
```

期望响应：
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

### 3.2 获取疾病标签映射

```bash
curl http://localhost:6006/api/v1/label_mapping | python3 -m json.tool | head -30
```

### 3.3 预处理

```bash
curl -X POST http://localhost:6006/api/v1/preprocess \
  -F "file=@notebooks/demo_data/demo_psg.edf"
```

期望响应：
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

### 3.4 生成嵌入

```bash
curl -X POST http://localhost:6006/api/v1/embed \
  -F "file=@notebooks/demo_data/demo_psg.edf"
```

期望响应包含四种模态的嵌入维度信息。

### 3.5 仅睡眠分期

```bash
curl -X POST http://localhost:6006/api/v1/predict \
  -F "file=@notebooks/demo_data/demo_psg.edf" \
  -F "tasks=sleep_staging"
```

### 3.6 仅疾病预测

```bash
curl -X POST http://localhost:6006/api/v1/predict \
  -F "file=@notebooks/demo_data/demo_psg.edf" \
  -F "age=0.0306" \
  -F "gender=0" \
  -F "tasks=disease_prediction"
```

### 3.7 完整推理（睡眠分期 + 疾病预测）

```bash
curl -X POST http://localhost:6006/api/v1/predict \
  -F "file=@notebooks/demo_data/demo_psg.edf" \
  -F "age=0.0306" \
  -F "gender=0" \
  -F "tasks=sleep_staging,disease_prediction"
```

期望响应（简化）：
```json
{
  "status": "success",
  "sleep_staging": {
    "total_epochs": 1745,
    "epochs": [...],
    "summary": {
      "Wake": {"count": ..., "percentage": ...},
      "N1": {...},
      "N2": {...},
      "N3": {...},
      "REM": {...}
    }
  },
  "disease_prediction": {
    "top_risks": [
      {"rank": 1, "phecode": "...", "phenotype": "...", "hazard_score": ...},
      ...
    ],
    "all_hazards_count": 1065
  }
}
```

---

## 4. 通过 Python 脚本测试

```python
import requests

BASE_URL = "http://localhost:6006"
EDF_PATH = "notebooks/demo_data/demo_psg.edf"

# 健康检查
r = requests.get(f"{BASE_URL}/api/v1/health")
print("Health:", r.json())

# 完整推理
with open(EDF_PATH, "rb") as f:
    r = requests.post(
        f"{BASE_URL}/api/v1/predict",
        files={"file": ("demo_psg.edf", f, "application/octet-stream")},
        data={"age": 0.0306, "gender": 0, "tasks": "sleep_staging,disease_prediction"},
    )

result = r.json()
print("Status:", result["status"])

if result.get("sleep_staging"):
    ss = result["sleep_staging"]
    print(f"Sleep staging: {ss['total_epochs']} epochs")
    for stage, info in ss["summary"].items():
        print(f"  {stage}: {info['count']} ({info['percentage']}%)")

if result.get("disease_prediction"):
    dp = result["disease_prediction"]
    print(f"\nTop 5 disease risks:")
    for risk in dp["top_risks"][:5]:
        print(f"  #{risk['rank']} {risk['phenotype']} (phecode={risk['phecode']}, hazard={risk['hazard_score']:.4f})")
```

---

## 5. EDF 文件兼容性要求

### 5.1 通道要求

模型要求 EDF 文件包含以下**全部四种模态**的通道，每种模态至少 1 个通道：

| 模态 | 说明     | 常见通道名示例                                         |
|------|----------|-------------------------------------------------------|
| BAS  | 脑电/眼电 | C3-A2, C4-A1, F3-M2, O1-M2, EOG(L), EOG(R), EEG, E1, E2 |
| RESP | 呼吸相关 | Airflow, SpO2, Chest, Abd, Nasal, Pulse, PPG           |
| EKG  | 心电     | EKG, ECG, ECG II, ECG1                                  |
| EMG  | 肌电     | Arm EMG, EMG, Chin, Leg(L), Leg(R), Chin EMG            |

完整的支持通道列表见 `sleepfm/configs/channel_groups.json`。

### 5.2 不兼容文件的常见症状

如果 EDF 文件的通道名不在支持列表中，将收到类似以下错误：

```
EDF 文件的通道无法匹配所有必需的模态。
文件通道: ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', ...],
缺失模态: ['RESP', 'EKG', 'EMG']
```

### 5.3 已知不兼容的数据集

| 数据集                      | 原因                                      |
|-----------------------------|------------------------------------------|
| Sleep-EDF (Cassette/Telemetry) | 通道名如 `EEG Fpz-Cz` 不在支持列表中       |
| PhysioNet 部分公开数据集      | 通道命名规范不同                           |

建议使用项目自带的 `demo_psg.edf` 进行首次测试验证。

---

## 6. API 文档

FastAPI 自动生成的交互式 API 文档：

- Swagger UI：`http://<服务器IP>:6006/docs`
- ReDoc：`http://<服务器IP>:6006/redoc`

可在 Swagger UI 中直接上传文件并测试接口。

---

## 7. 常见问题

### Q1: 推理需要多长时间？

以 `demo_psg.edf`（~2.4 小时记录）为例：
- 预处理：~10 秒
- 嵌入生成：~15 秒
- 睡眠分期：<1 秒
- 疾病预测：<1 秒
- **总计：约 30 秒**

实际耗时与 EDF 文件时长和 GPU 性能相关。

### Q2: 端口被占用怎么办？

```bash
fuser -k 6006/tcp
```

### Q3: 如何查看服务日志？

服务日志直接输出到启动终端。如需后台运行：

```bash
nohup python -m api.main > logs/api.log 2>&1 &
```

### Q4: age 参数如何归一化？

项目使用的归一化方式为线性缩放到 0~1 区间。demo 数据中 `age=0.0306` 对应较年轻的受试者。具体的归一化公式需参考原始训练数据的处理方式。
