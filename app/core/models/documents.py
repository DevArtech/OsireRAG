"""
Module: documents.py

Contains the Document class, which is used to represent a document in the system.

Classes:
- Document: A Pydantic model representing a document.

Functions:
- None

Usage:
- Import the Document class from this module into other modules that require document representation.

Author: Adam Haile
Date: 9/25/2024
"""

import os
from pypdf import PdfReader
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr, field_validator


class Document(BaseModel):
    """
    A Pydantic model representing a document.

    Attributes:
    - directory: str: The path to the document file.
    - _content: Optional[str]: The content of the document.
    - _metadata: Optional[Dict[str, Any]]: The metadata of the document.

    Methods:
    - __init__: Custom initialization method for the Document class. Reads the content of the document file.
    - validate_directory: Validates the directory attribute.

    Usage:
    - document = Document(directory="/path/to/file.txt")

    Author: Adam Haile
    Date: 10/16/2024
    """

    directory: str = Field(default="/path/to/file.txt")
    _content: Optional[str] = PrivateAttr(default=None)
    _metadata: Optional[Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        """
        Custom initialization method for the Document class. Reads the content of the document file

        Args:
        - data: Dict: The data to initialize the Document object with.

        Returns:
        - None

        Raises:
        - ValueError: If the file type is not supported.

        Usage:
        - document = Document(directory="/path/to/file.txt")

        Author: Adam Haile
        Date: 10/16/2024
        """
        super().__init__(**data)
        self.directory = self.directory.replace("\\", "/")

        if self.directory.endswith(".txt") or self.directory.endswith(".html"):
            with open(self.directory, "r", encoding="utf-8") as file:
                self._content = file.read()
        elif self.directory.endswith(".pdf"):
            reader = PdfReader(self.directory)
            self._content = "\n<PAGE>\n".join(
                [page.extract_text() for page in reader.pages]
            )
            self._metadata = {"pages": len(reader.pages)}
        else:
            raise ValueError(
                f"File {os.path.splitext(self.directory)[1]} type not supported"
            )

    @field_validator("directory")
    def validate_directory(cls, value, info):
        if ".." in value:
            raise ValueError('Path traversal ("..") is not allowed')

        if not os.path.exists(value):
            raise ValueError(f'"{value}" directory does not exist')

        if value.endswith(".txt") or value.endswith(".html") or value.endswith(".pdf"):
            return value

        raise ValueError(f"File {os.path.splitext(value)[1]} type not supported")

    @property
    def content(self) -> Optional[str]:
        return self._content

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata
