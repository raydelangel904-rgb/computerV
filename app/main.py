import os
import sqlite3
import time
import base64
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from deepface import DeepFace
except ImportError:  # pragma: no cover
    DeepFace = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"
DB_PATH = DATA_DIR / "secure_access.db"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="SecureFace Access",
    description="Sistema seguro de acceso con reconocimiento facial biométrico."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if TEMPLATES_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")


class MoodAnalysisRequest(BaseModel):
    image: str


class RegisterUserRequest(BaseModel):
    full_name: str
    image: str


class LoginUserRequest(BaseModel):
    full_name: str
    image: str


def validate_runtime_dependencies() -> List[str]:
    messages = []
    if cv2 is None:
        messages.append("OpenCV no está instalado en el entorno. Ejecuta: pip install opencv-python-headless==4.10.0.84")
    elif not hasattr(cv2, "CascadeClassifier"):
        messages.append("OpenCV está instalado pero no tiene CascadeClassifier. Ejecuta: pip install --force-reinstall opencv-python-headless==4.10.0.84")
    elif not has_opencv_haar_cascade():
        messages.append("OpenCV está instalado pero falta la cascada Haar de cara. Ejecuta: pip install --force-reinstall opencv-python-headless==4.10.0.84")
    if DeepFace is None:
        messages.append("DeepFace no está instalado. Ejecuta: pip install -r requirements.txt")
    return messages


def has_opencv_haar_cascade() -> bool:
    try:
        if cv2 is None:
            return False
        data_dir = getattr(cv2, "data", None)
        haar_dir = getattr(data_dir, "haarcascades", None)
        if haar_dir:
            cascade_path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                return True
        data_root = Path(cv2.__file__).resolve().parent / "data"
        cascade_path = data_root / "haarcascade_frontalface_default.xml"
        return cascade_path.exists()
    except Exception:
        return False


def get_face_detector_backend() -> str:
    if has_opencv_haar_cascade():
        return "opencv"
    try:
        import mtcnn  # type: ignore
        return "mtcnn"
    except Exception:
        try:
            import retinaface  # type: ignore
            return "retinaface"
        except Exception:
            return "opencv"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL UNIQUE,
            image_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_base64_image(base64_str: str, file_path: str) -> bool:
    try:
        if base64_str.startswith("data:image") and "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_data = base64.b64decode(base64_str)
        with open(file_path, "wb") as f:
            f.write(img_data)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Error al decodificar imagen Base64: {exc}")
        return False


def normalize_name(value: str) -> str:
    return " ".join(value.strip().split())


def user_folder_for(name: str) -> Path:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    if not safe:
        safe = "user"
    return USERS_DIR / safe


def detect_face_in_image(image_path: str):
    detector_backend = get_face_detector_backend()
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=True,
            detector_backend=detector_backend,
            silent=True,
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        if not analysis:
            raise ValueError("No se detectó ninguna cara válida.")
        return analysis
    except Exception as exc:
        msg = str(exc).lower()
        if "face could not be detected" in msg or "no face" in msg or "faces" in msg and "not found" in msg:
            raise ValueError("No se detectó ningún rostro en la imagen.")
        raise


def compare_faces(reference_path: str, candidate_path: str):
    detector_backend = get_face_detector_backend()
    try:
        result = DeepFace.verify(
            img1_path=reference_path,
            img2_path=candidate_path,
            detector_backend=detector_backend,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=True,
            silent=True,
        )
        if isinstance(result, dict):
            verified = bool(result.get("verified", False))
            distance = float(result.get("distance", 1.0))
            similarity = max(0.0, 1.0 - distance)
            return {"verified": verified, "distance": distance, "similarity": similarity}
    except Exception:
        pass
    return {"verified": False, "distance": 1.0, "similarity": 0.0}


def get_all_users() -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, full_name, image_path, created_at FROM users ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(
        content="<h1>SecureFace Access</h1><p>Frontend no encontrado.</p>",
        status_code=404,
    )


@app.get("/api/users")
async def list_users():
    return {"success": True, "users": get_all_users()}


@app.post("/api/register")
async def register_user(payload: RegisterUserRequest):
    runtime_errors = validate_runtime_dependencies()
    if runtime_errors:
        raise HTTPException(status_code=503, detail="; ".join(runtime_errors))

    full_name = normalize_name(payload.full_name)
    if not full_name:
        raise HTTPException(status_code=400, detail="El nombre es obligatorio.")
    if not payload.image:
        raise HTTPException(status_code=400, detail="Debes enviar una foto facial.")

    try:
        folder = user_folder_for(full_name)
        folder.mkdir(parents=True, exist_ok=True)

        image_path = folder / f"{int(time.time() * 1000)}.jpg"
        if not save_base64_image(payload.image, str(image_path)):
            raise HTTPException(status_code=400, detail="La imagen enviada no tiene un formato válido.")

        detect_face_in_image(str(image_path))

        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                "INSERT INTO users (full_name, image_path, created_at) VALUES (?, ?, ?)",
                (full_name, str(image_path), time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            if image_path.exists():
                image_path.unlink()
            raise HTTPException(status_code=409, detail="Ese nombre ya está registrado.")
        finally:
            conn.close()

        return {
            "success": True,
            "message": "Usuario registrado con éxito.",
            "user": {"full_name": full_name, "image_path": str(image_path)},
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo registrar la persona: {str(exc)}")


@app.post("/api/login")
async def login_user(payload: LoginUserRequest):
    runtime_errors = validate_runtime_dependencies()
    if runtime_errors:
        raise HTTPException(status_code=503, detail="; ".join(runtime_errors))

    full_name = normalize_name(payload.full_name)
    if not full_name:
        raise HTTPException(status_code=400, detail="Debes indicar tu nombre.")
    if not payload.image:
        raise HTTPException(status_code=400, detail="No se recibió la imagen para autenticación.")

    temp_dir = DATA_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"login_{int(time.time() * 1000)}.jpg"

    if not save_base64_image(payload.image, str(temp_path)):
        raise HTTPException(status_code=400, detail="Formato de imagen inválido.")

    try:
        detect_face_in_image(str(temp_path))

        user_candidates = get_all_users()
        best_match = None
        best_score = 0.0

        for user in user_candidates:
            if user["full_name"].lower() != full_name.lower():
                continue
            ref_path = user["image_path"]
            if not os.path.exists(ref_path):
                continue
            comparison = compare_faces(ref_path, str(temp_path))
            score = comparison.get("similarity", 0.0)
            if score > best_score:
                best_match = user
                best_score = score

        if best_match is None:
            raise HTTPException(status_code=404, detail="No existe un usuario registrado con ese nombre.")
        if best_score < 0.82:
            raise HTTPException(status_code=401, detail="Acceso denegado: la identidad facial no coincide.")

        return {
            "success": True,
            "message": "Acceso concedido por reconocimiento facial.",
            "user": {"full_name": best_match["full_name"], "id": best_match["id"]},
            "similarity": round(best_score, 4),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en autenticación: {str(exc)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/analyze-mood")
async def analyze_mood(payload: MoodAnalysisRequest):
    runtime_errors = validate_runtime_dependencies()
    if runtime_errors:
        raise HTTPException(status_code=503, detail="; ".join(runtime_errors))

    temp_dir = DATA_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"mood_{int(time.time() * 1000)}.jpg"

    if not save_base64_image(payload.image, str(temp_path)):
        raise HTTPException(status_code=400, detail="La imagen provista no tiene un formato Base64 decodificable válido.")

    start_time = time.time()
    try:
        analysis = DeepFace.analyze(
            img_path=str(temp_path),
            actions=["emotion"],
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )
        inference_time = round(time.time() - start_time, 3)

        if isinstance(analysis, list):
            result = analysis[0]
        else:
            result = analysis

        emotions_raw = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "neutral")
        region = result.get("region", {"x": 0, "y": 0, "w": 0, "h": 0})

        if not emotions_raw:
            raise ValueError("DeepFace no devolvió resultados válidos de emoción.")

        return {
            "success": True,
            "dominant_emotion": dominant_emotion,
            "emotions": {k: float(round(v, 2)) for k, v in emotions_raw.items()},
            "box": {
                "x": int(region.get("x", 0)),
                "y": int(region.get("y", 0)),
                "w": int(region.get("w", 0)),
                "h": int(region.get("h", 0)),
            },
            "inference_time_seconds": float(inference_time),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento neuronal: {str(exc)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()
