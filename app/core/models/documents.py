import os
from pypdf import PdfReader
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr, field_validator


class Document(BaseModel):
    directory: str = Field(default="/path/to/file.txt")
    _content: Optional[str] = PrivateAttr(default=None)
    _metadata: Optional[Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self.directory = self.directory.replace("\\", "/")

        if self.directory.endswith(".txt") or self.directory.endswith(".html"):
            with open(self.directory, "r") as file:
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
