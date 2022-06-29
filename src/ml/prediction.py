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
            config_path,
        ):
        self.config = json.load(open(config_path))
        self.news_model_filepath = self.config["news_model_path"]
        self.rectum_model_filepath = self.config["rectum_model_path"]
        self.classifier = None
        self.encoding = None

        self.load_rectum_model()
        self.load_news_model()

    def load_rectum_model(self):
        logger.info("Loading rectum model...")
        logger.info(f"Model filepath: {self.rectum_model_filepath}")
        self.rectum_classifier = ort.InferenceSession(self.rectum_model_filepath)
        # self.input_ids = self.rectum_classifier.get_inputs()[0].name
        # self.output = self.rectum_classifier.get_outputs()[0].name
        logger.info("Model rectum loaded.")

    def load_news_model(self):
        logger.info("Loading news model...")
        logger.info(f"Model filepath: {self.news_model_filepath}")
        self.news_classifier = ort.InferenceSession(self.news_model_filepath)
        # self.input_ids = self.rectum_classifier.get_inputs()[0].name
        # self.output = self.rectum_classifier.get_outputs()[0].name
        logger.info("Model news loaded.")

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

    def predict_news_bert(self, text: List[str]):
        self.encoding = self.encoding_text(text)
        logger.info("Predicting...")
        dict_input = {"input_ids": self.encoding["input_ids"]}
        pred = self.news_classifier.run(None, input_feed=dict_input)
        pred_label = pred[0].argmax()
        logger.info(f"Predicted label: {pred_label}")
        return pred_label

    def predict_rectum_bert(self, text: List[str]):
        self.encoding = self.encoding_text(text)
        logger.info("Predicting...")
        dict_input = {"input_ids": self.encoding["input_ids"]}
        pred = self.rectum_classifier.run(None, input_feed=dict_input)
        pred_label = pred[0].argmax()
        logger.info(f"Predicted label: {pred_label}")
        return pred_label

classifier = Bert_model_cl(
    config_path="./config/bert_config.json",
)