"""Data models for the Medical Coding Agent."""

from pydantic import BaseModel, Field


class Question(BaseModel):
    """A multiple-choice exam question."""
    id: int
    text: str
    options: dict[str, str] = Field(description="Options A, B, C, D")
