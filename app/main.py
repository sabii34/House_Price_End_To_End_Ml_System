from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import time

from src.config import load_settings
from src.logger import get_logger

from .schemas import HouseFeatures, PredictResponse
from .service import model_service

settings = load_settings()
log = get_logger("api", settings.log_level)

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="Industry-style ML inference service (preprocess + model pipeline)",
)

# --- CORS (standard for production; tighten allow_origins later) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request logging middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    log.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f} ms)")
    return response


@app.on_event("startup")
def startup_event():
    try:
        model_service.load()
        log.info("[bold green]✅ Model loaded[/bold green]")
        log.info(f"Artifact: {model_service.artifact_path}")
    except Exception as e:
        log.info(f"[bold red]❌ Failed to load model[/bold red] {e}")
        raise


@app.get("/health")
def health():
    return {"status": "ok", "artifact": model_service.artifact_path}


@app.get("/version")
def version():
    return {
        "service": "house-price-api",
        "version": "1.0.0",
        "artifact": model_service.artifact_path,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: HouseFeatures):
    try:
        pred = model_service.predict_one(payload.model_dump())
        return PredictResponse(
            prediction=pred,
            model_artifact=model_service.artifact_path,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

