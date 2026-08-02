from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from furniture_ai.config import settings
from furniture_ai.infer.furnish import run_furnish

app = FastAPI(title="Floorplan Furnisher Pro", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.allow_origins,
    allow_credentials=settings.app.allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

REQ_COUNTER = Counter("http_requests_total", "Total HTTP requests", ["path", "method", "code"])
LATENCY = Histogram("http_latency_seconds", "HTTP request latency", ["path", "method"])
auth_scheme = HTTPBearer(auto_error=False)

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class TokenRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=128)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    LATENCY.labels(path, request.method).observe(time.perf_counter() - started)
    REQ_COUNTER.labels(path, request.method, response.status_code).inc()
    return response


def _jwt_secret() -> str:
    if settings.app.jwt_secret is None:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    secret = settings.app.jwt_secret.get_secret_value()
    if len(secret) < 16:
        raise HTTPException(status_code=503, detail="Authentication secret is too short")
    return secret


def require_jwt(
    credentials: HTTPAuthorizationCredentials | None = Depends(auth_scheme),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        return jwt.decode(
            credentials.credentials,
            _jwt_secret(),
            algorithms=["HS256"],
            audience=settings.app.jwt_audience,
            issuer=settings.app.jwt_issuer,
            options={"require_exp": True, "require_sub": True},
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


async def _read_image(file: UploadFile) -> Image.Image:
    if file.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are accepted")

    payload = await file.read(settings.app.max_upload_bytes + 1)
    if len(payload) > settings.app.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit")
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        with Image.open(BytesIO(payload)) as probe:
            probe.verify()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image") from exc

    if image.width * image.height > settings.app.max_image_pixels:
        raise HTTPException(status_code=413, detail="Image dimensions exceed the safe pixel limit")
    return image


def _run_pipeline(image: Image.Image) -> dict:
    try:
        return run_furnish(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, object]:
    weights = Path(settings.segmenter.weights_path).expanduser()
    return {
        "ok": True,
        "environment": settings.app.environment,
        "auth_configured": settings.app.jwt_secret is not None,
        "segmenter_weights_ready": weights.is_file(),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/token")
def token(request: TokenRequest) -> dict[str, str]:
    if not settings.app.issue_demo_tokens:
        raise HTTPException(status_code=404, detail="Demo token issuance is disabled")
    now = datetime.now(timezone.utc)
    payload = {
        "sub": request.subject,
        "iss": settings.app.jwt_issuer,
        "aud": settings.app.jwt_audience,
        "iat": now,
        "exp": now + timedelta(minutes=settings.app.token_ttl_minutes),
    }
    encoded = jwt.encode(payload, _jwt_secret(), algorithm="HS256")
    return {"access_token": encoded, "token_type": "bearer"}


@app.post("/furnish", dependencies=[Depends(require_jwt)])
async def furnish(file: UploadFile = File(...)) -> dict:
    image = await _read_image(file)
    output = _run_pipeline(image)
    return {"layout": output["layout"]}


@app.post("/furnish/overlay", dependencies=[Depends(require_jwt)])
async def furnish_overlay(file: UploadFile = File(...)) -> Response:
    image = await _read_image(file)
    output = _run_pipeline(image)
    return Response(content=output["overlay_png"], media_type="image/png")
