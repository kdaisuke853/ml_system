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

@router.get("/label_news")
def label() -> Dict[int, str]:
    return classifier.news_label

@router.post("/predict_news")
def predict_news(data: Data) -> Dict[str, str]:
    predict = classifier.predict_news_bert(data.data)
    return {"predict": str(predict)}

@router.get("/label_rectum")
def label() -> Dict[int, str]:
    return classifier.rectum_label

@router.post("/predict_rectum")
def predict_rectum(data: Data) -> Dict[str, str]:
    predict = classifier.predict_rectum_bert(data.data)
    return {"predict": str(predict)}