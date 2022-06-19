import json
from logging import getLogger
import numpy as np
from pydantic import BaseModel
from typing import List

from transformers import BertJapaneseTokenizer
import onnxruntime as ort

logger = getLogger(__name__)

class Data(BaseModel):
    data: List[str]

class Bert_model_cl(object):
    def __init__(self, 
            model_path, 
            config_path,
        ):
        self.model_filepath = model_path
        self.config = json.load(open(config_path))
        self.classifier = None
        self.encoding = None

        self.load_model()

    def load_model(self):
        logger.info("Loading model...")
        logger.info(f"Model filepath: {self.model_filepath}")
        self.classifier = ort.InferenceSession(self.model_filepath)
        self.input_ids = self.classifier.get_inputs()[0].name
        self.attention_mask = self.classifier.get_inputs()[1].name
        self.token_type_ids = self.classifier.get_inputs()[2].name
        self.output_0 = self.classifier.get_outputs()[0].name
        logger.info("Model loaded.")

    def encoding_text(self, text: List[str]):
        logger.info("Encoding text...")
        tokenizer = BertJapaneseTokenizer.from_pretrained(self.config['model_name'])
        max_len = self.config["max_length"]
        encoding = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True
            )
        encoding["input_ids"] = np.array(encoding["input_ids"], dtype='int')
        encoding["attention_mask"] = np.array(encoding["attention_mask"], dtype='int')
        encoding["token_type_ids"] = np.array(encoding["token_type_ids"], dtype='int')
        encoding["input_ids"] = encoding["input_ids"].reshape(1, 128)
        encoding["attention_mask"] = encoding["attention_mask"].reshape(1, 128)
        encoding["token_type_ids"] = encoding["token_type_ids"].reshape(1, 128)

        logger.info("Text encoded.")
        return encoding

    def predict_bert(self, text: List[str]):
        self.encoding = self.encoding_text(text)
        pred = self.classifier.run(None, input_feed=dict(self.encoding))
        pred_logits = pred[0][0].sum(axis=0)
        pred_label = pred_logits.argmax()
        logger.info(f"Predicted label: {pred_label}")
        return pred_label

classifier = Bert_model_cl(
    model_path="./models/model.onnx",
    config_path="./config/bert_config.json",
)