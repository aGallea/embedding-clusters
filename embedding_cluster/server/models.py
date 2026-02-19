from __future__ import annotations

from pydantic import BaseModel


class CollectionInfo(BaseModel):
    name: str
    count: int


class CollectionDetail(BaseModel):
    name: str
    count: int
    metadata_fields: list[str]


class MessageResponse(BaseModel):
    message: str
