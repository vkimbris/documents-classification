from dataclasses import dataclass

@dataclass
class User:
    index: str
    name: str
    surname: str


@dataclass
class Document:
   index: str
   name: str