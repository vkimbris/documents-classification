import os

from fastapi import FastAPI, File, UploadFile

from src.document_classifier import SklearnDocumentClassifier
from src.document_parser import parse_document
from src.constants import *


app = FastAPI()


document_classifier = SklearnDocumentClassifier(
    model_path=MODEL_PATH
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/classify-doc")
async def classify_document(file: UploadFile = File(...)):
    print(type(file))
    
    return {"filename": file.filename}