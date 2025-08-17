
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime, timedelta
import time
from PIL import Image

from furniture_ai.config import settings
from furniture_ai.infer.furnish import run_furnish

app = FastAPI(title="Floorplan Furnisher Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQ_COUNTER = Counter("http_requests_total", "Total HTTP requests", ["path","method","code"])
LATENCY = Histogram("http_latency_seconds", "HTTP request latency", ["path","method"])
auth_scheme = HTTPBearer(auto_error=False)

@app.middleware("http")
async def metrics_mw(request, call_next):
    start = time.time()
    response = await call_next(request)
    LATENCY.labels(request.url.path, request.method).observe(time.time()-start)
    REQ_COUNTER.labels(request.url.path, request.method, response.status_code).inc()
    return response

def require_jwt(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not creds:
        raise HTTPException(401, "Missing token")
    try:
        jwt.decode(creds.credentials, settings.app.jwt_secret, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(401, "Invalid token")

@app.get("/health")
def health(): return {"ok": True}

@app.get("/metrics")
def metrics(): return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/token")
def token():
    payload = {"sub":"user","exp": datetime.utcnow() + timedelta(hours=1)}
    t = jwt.encode(payload, settings.app.jwt_secret, algorithm="HS256")
    return {"access_token": t, "token_type":"bearer"}

@app.post("/furnish", dependencies=[Depends(require_jwt)])
async def furnish(file: UploadFile = File(...), weights_path: str | None = None):
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")
    out = run_furnish(img, weights_path=weights_path)
    return {"layout": out["layout"]}

@app.post("/furnish/overlay", dependencies=[Depends(require_jwt)])
async def furnish_overlay(file: UploadFile = File(...), weights_path: str | None = None):
    img = Image.open(file.file).convert("RGB")
    out = run_furnish(img, weights_path=weights_path)
    return Response(content=out["overlay_png"], media_type="image/png")
