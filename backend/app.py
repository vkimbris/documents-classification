import codecs
import csv
import pandas as pd

from fastapi import FastAPI, File, UploadFile

from src.document_classifier import SklearnDocumentClassifier
from src.document_parser import TikaDocumentParser

from src.constants import *


app = FastAPI()


document_classifier = SklearnDocumentClassifier(
    model_path=MODEL_PATH, path_to_old_train_data=PATH_TO_OLD_DATA
)

document_parser = TikaDocumentParser(
    tika_server=TIKA_SERVER
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/labels")
def get_labels():
    labels = document_classifier.pipeline.classes_
    labels = list(labels)
    
    return {"labels": labels}

@app.post("/updateModel")
def train(train_data: UploadFile = File(...)):
    new_train_data = pd.read_csv(train_data.file)

    document_classifier.train(new_train_data)

    return {"status": "Model trained succesfully."}

@app.post("/classify-doc")
async def classify_document(files: list[UploadFile]):
    predictions = []
    for file in files:
        file_name = file.filename
        
        text = await file.read()
        text = document_parser.parse(text)

        label = document_classifier.predict([text])[0]

        predictions.append({
            "file": file_name, "label": label
        })  
    
    return predictions