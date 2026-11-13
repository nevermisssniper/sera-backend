import hmac
import os
from uuid import UUID
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from tasks import generate_images_task
from generation_security import callback_config

app = FastAPI()


def require_service_token(authorization: Optional[str] = Header(default=None)):
    token = os.environ.get("GENERATION_SERVICE_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="Generation service is not configured")
    if not authorization or not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


class GenerateRequest(BaseModel):
    userId: str = Field(min_length=1)
    jobId: UUID
    gender: Optional[str] = None
    age: Optional[int] = None
    hairColor: Optional[str] = None
    hairLength: Optional[str] = None
    ethnicity: Optional[str] = None
    bodyType: Optional[str] = None
    attire: Optional[str] = None
    backgrounds: Optional[str] = None
    glasses: bool = False
    instanceImages: List[UUID] = Field(min_length=1, max_length=100)

    class Config:
        extra = "forbid"  # In particular, do not accept caller-controlled callback URLs.


@app.post("/generate", dependencies=[Depends(require_service_token)])
def generate_images(req: GenerateRequest):
    try:
        callback_config()  # Fail before enqueueing if callbacks cannot be delivered.
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Callback service is not configured")
    import json
    payload = json.loads(req.json())  # UUIDs must be strings for Celery's JSON serializer.
    task = generate_images_task.apply_async(args=[payload], task_id=str(req.jobId))
    return {"message": "Generation request enqueued.", "task_id": task.id}
