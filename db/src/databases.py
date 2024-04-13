import os


from .types import User, Document

from abc import ABC, abstractmethod
from typing import Any, List


class BaseDB(ABC):
    """
    A base class for database logic.

    Attributes
    ----------
    model_path : str
        the location of the model in the local folder

    Methods
    -------
    add_user(self, user: User) -> None:
        Creates user instance.

    add_document(self, user: User, document: Document) -> None:
        Adds document associated with user.

    get_documents(self, user_id: str) -> List[str]:
        Returns all documents paths for user with index=user_id.
    """

    def __init__(self, documents_types: List[str]) -> None:
        super().__init__()

        self.documents_types = documents_types

    @abstractmethod
    def add_user(self, user: User) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_document(self, user_id: str, document_type: str, document: Any) -> None:
        raise NotImplementedError
    
    def get_documents(self, user_id: str) -> List[str]:
        raise NotImplementedError
    

class LocalFolderDB(BaseDB):
    """
    A class for local folder database logic.

    Attributes
    ----------
    model_path : str
        the location of the model in the local folder

    Methods
    -------
    add_user(self, user: User) -> None:
        Creates user instance.

    add_document(self, user: User, document: Document) -> None:
        Adds document associated with user.

    get_documents(self, user_id: str) -> List[str]:
        Returns all documents paths for user with index=user_id.
    """

    def __init__(self, local_folder_path: str, documents_types) -> None:
        super().__init__(documents_types)

        self.local_folder_path = local_folder_path

        self.path_to_config_file = local_folder_path + "/config.txt"

    def add_user(self, user: User) -> None:
        with open(self.path_to_config_file, "a") as f:
            f.write(f"{user.index} | {user.name} | {user.surname}\n")

        user_path = f"{self.local_folder_path}/{user.index}"
        
        os.mkdir(user_path)
        for document_type in self.documents_types:
            os.mkdir(user_path + "/" + document_type)
    
    def add_document(self, user_id: str, document_type: str, document: Any) -> None:
        file_location = f"{self.local_folder_path}/{user_id}/{document_type}/{document.filename}"
    
        with open(file_location, "wb+") as file_object:
            file_object.write(document.file.read())

        return {"info": f"file '{document.filename}' saved at '{file_location}'"}
    
    def get_documents(self, user_id: str) -> List[str]:
        pass
