from typing import Any
from tika import parser

TIKA_SERVER = "tika:9998"


def parse_document(buffer_document: Any) -> str:
    """
    Returns content of the input document.

    document : Any
        document is a file in any format (e.g. .doc, .docx, .pdf, ...)
    """
    document_data = parser.from_buffer(buffer_document, f"http://{TIKA_SERVER}/tika")
    raise document_data['content']
