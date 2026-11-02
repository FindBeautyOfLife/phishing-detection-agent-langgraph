# 钓鱼网站检测智能体

这是一个基于 URL 结构特征、机器学习模型和 LangGraph 工作流的钓鱼网站检测智能体。项目支持命令行调用、FastAPI 接口和可视化网页工作台。

## 功能

- 单个 URL 检测
- 批量 URL 检测
- 风险原因解释和误判分析
- 生成 Markdown / TXT 检测报告
- 导出 CSV / Excel 检测结果
- 生成风险图表和防钓鱼安全提示图
- 通过大语言模型进行意图识别和结果问答

## 数据集来源

训练数据来自 Kaggle 数据集：

[65 Lakh URLs Labelled as 1 and 0](https://www.kaggle.com/datasets/somanshumahajan/65-lakh-urls-labelled-as-1-and-0/data)

本仓库不包含完整数据集。原始数据规模较大，下载、使用和再分发时请遵守 Kaggle 页面及数据集许可证要求。

首次使用时，需要先准备数据并训练模型：

1. 从上述 Kaggle 页面下载数据集。
2. 将 CSV 文件放入项目根目录的 `data/` 文件夹。
3. 确认数据包含 URL 列和标签列。
4. 安装项目依赖。
5. 运行训练脚本：

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

训练脚本会在项目根目录生成 `phishing_model.pkl`、`feature_names.pkl` 和 `model_comparison_results.csv`。检测智能体依赖这些训练产物，训练完成后才能进行 URL 检测。

## 环境安装和启动顺序

建议使用与模型训练一致的 Python 环境：

```powershell
python -m pip install -r requirements.txt
```

完整启动顺序：

```powershell
# 1. 下载 Kaggle 数据集并放入 data/
# 2. 训练模型
python train_model.py

# 3. 配置 .env 后启动智能体
python -m uvicorn api_service:app --reload
```

如果没有先运行 `train_model.py`，启动服务本身仍可加载，但执行 URL 检测时会提示找不到模型文件。

## 配置大语言模型

请先复制配置模板：

```powershell
copy .env.example .env
```

然后编辑项目根目录下的 `.env` 文件，填写阿里云百炼配置：

```env
OPENAI_API_KEY=你的阿里云百炼API_Key
OPENAI_MODEL=qwen-plus
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 启动方式

### 可视化网页和 FastAPI

```powershell
python -m uvicorn api_service:app --reload
```

打开：

```text
http://127.0.0.1:8000/
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

### 命令行

```powershell
python cli_app.py
```

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/api/v1/workflow` | 通用工作流 |
| POST | `/api/v1/detect` | 单个 URL 检测 |
| POST | `/api/v1/batch-detect` | 批量 URL 检测 |

## 项目结构

```text
api_service.py       FastAPI 服务和网页入口
cli_app.py           命令行入口
workflow_graph.py    LangGraph 工作流
workflow_nodes.py    工作流节点和意图识别
feature_extractor.py URL 特征提取
model_service.py    模型加载和预测
train_model.py      模型训练脚本
generators/          报告、表格、图表和海报生成器
frontend/            可视化网页前端
```

## 模型局限性

模型主要依据 URL 的结构特征进行判断，不会直接访问或解析网页内容、JavaScript、证书详情和完整跳转链。结果不能替代人工安全审查，尤其是短链接、重定向链接和新型攻击样本。
