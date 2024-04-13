from fastapi import FastAPI, File, UploadFile

from src.document_classifier import SklearnDocumentClassifier
from src.document_parser import TikaDocumentParser

from src.constants import *


app = FastAPI()


document_classifier = SklearnDocumentClassifier(
    model_path=MODEL_PATH
)

document_parser = TikaDocumentParser(
    tika_server=TIKA_SERVER
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


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