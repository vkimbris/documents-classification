from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from src.databases import LocalFolderDB
from src.types import *
from src.constants import *

app = FastAPI()


database = LocalFolderDB(
    local_folder_path=DB_LOCAL_FOLDER_PATH, documents_types=DB_DOCUMENTS_TYPES
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/addUser")
def add_user(index: str, name: str, surname: str):
    database.add_user(User(
        index=index, name=name, surname=surname
    ))


@app.post("/addDocument")
async def add_document(user_id: str, document_type: str, document: UploadFile = File(...)):
    database.add_document(
        user_id, document_type, document
    )
