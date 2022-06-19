from typing import Dict

from fastapi import APIRouter
from src.ml.prediction import Data, classifier

router = APIRouter()

@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}

@router.get("/test")
def test() -> Dict[str, str]:
    return {"test": "ok"}

@router.post("/predict")
def predict(data: Data) -> Dict[str, str]:
    predict = classifier.predict_bert(data.data)
    return {"predict": int(predict)}
