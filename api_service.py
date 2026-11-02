from __future__ import annotations

from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from workflow_graph import run_workflow


class WorkflowRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="自然语言任务")
    last_detection_results: Any | None = Field(
        default=None,
        description="上一次检测结果，用于支持解释和报告等多轮任务",
    )


class DetectRequest(BaseModel):
    url: str = Field(..., min_length=1, description="待检测的 URL，可带或不带协议")
    last_detection_results: Any | None = Field(default=None)


class BatchDetectRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, description="待检测的 URL 列表")
    last_detection_results: Any | None = Field(default=None)


app = FastAPI(
    title="钓鱼网站检测智能体服务",
    description="基于机器学习模型和 LangGraph 工作流的 URL 风险检测服务。",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


def _invoke_workflow(
    user_input: str, last_detection_results: Any | None = None
) -> dict[str, Any]:
    try:
        return run_workflow(
            user_input=user_input,
            last_detection_results=last_detection_results,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/workflow")
def workflow(request: WorkflowRequest) -> dict[str, Any]:
    return _invoke_workflow(
        request.user_input,
        request.last_detection_results,
    )


@app.post("/api/v1/detect")
def detect(request: DetectRequest) -> dict[str, Any]:
    return _invoke_workflow(
        f"检测这个网址：{request.url}",
        request.last_detection_results,
    )


@app.post("/api/v1/batch-detect")
def batch_detect(request: BatchDetectRequest) -> dict[str, Any]:
    urls = [url.strip() for url in request.urls if url.strip()]
    if not urls:
        raise HTTPException(status_code=422, detail="urls 不能是空列表。")

    return _invoke_workflow(
        "批量检测以下网址：\n" + "\n".join(urls),
        request.last_detection_results,
    )
