import codecs
import csv
import pandas as pd

from fastapi import FastAPI, File, UploadFile

from src.document_classifier import SklearnDocumentClassifier
from src.document_parser import TikaDocumentParser

from src.regs import SparsePhiRegularizer, SparseThetaRegularizer
from src.plsi import PLSI

from octis.dataset.dataset import Dataset

from src.constants import *


app = FastAPI()


document_classifier = SklearnDocumentClassifier(
    model_path=CLASSIFIER_PATH, 
    path_to_old_train_data=CLASSIFIER_PATH_PATH_TO_OLD_DATA, 
    random_state=CLASSIFIER_RANDOM_STATE, 
    test_size=CLASSIFIER_TEST_SIZE
)

document_parser = TikaDocumentParser(
    tika_server=PARSER_TIKA_SERVER
)

topic_modeling_dataset = Dataset()
topic_modeling_dataset.load_custom_dataset_from_folder(TOPIC_MODELER_PATH_TO_DATASET)


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

    metrics = document_classifier.train(new_train_data)

    return {"status": "Model trained succesfully.", "classificationReport": metrics}

@app.post("/classifyDocuments")
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

@app.post("/topicModeling")
def topic_modeling(n_topics: int, seed: int):
    topic_modeler = PLSI(
        dataset=topic_modeling_dataset,
        num_topics=n_topics,
        seed=seed,
        regularizers={
            "phi": [
                SparsePhiRegularizer(alpha=TOPIC_MODELER_PHI)
            ],

            "theta": [
                SparseThetaRegularizer(alpha=TOPIC_MODELER_THETA)
            ]
        }
    )

    topics = topic_modeler.train(max_iter=100, verbose=True)["topics"]
    representative_docs = topic_modeler.get_representative_docs(TOPIC_MODELER_PASS_TO_ORIGINAL_CORPUS, top_n_docs=3)

    results = []
    for k in range(n_topics):
        results.append({
            "topicID": k,
            "topicKeyWords": topics[k],
            "topicRepresentativeDocuments": representative_docs[k]
        })

    return results