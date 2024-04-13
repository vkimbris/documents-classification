from typing import Any
from tika import parser

from abc import ABC, abstractmethod


class BaseDocumentParser(ABC):
    """
    A base class for Document parsing.

    Methods
    -------
    parse(buffer_document: Any) -> str:
        Returns content of the input document.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def parse(buffer_document: Any) -> str:
        """
        Returns content of the input document.

        document : Any
            document is a file in any format (e.g. .doc, .docx, .pdf, ...)
        """
        
        raise NotImplementedError


class TikaDocumentParser(BaseDocumentParser):
    """
    A class for Document parsing with Apache Tika.

    Attributes
    ----------
    tika_server : str
        the location of API to Tika server

    Methods
    -------
    predict(contents: List[str]) -> List[str]:
        Returns labels of documents contents
    """

    def __init__(self, tika_server: str) -> None:
        super().__init__()

        self.tika_server = tika_server

    def parse(self, buffer_document: Any) -> str:
        document_data = parser.from_buffer(buffer_document, f"http://{self.tika_server}/tika")
        
        return document_data['content']
