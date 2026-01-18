from pydantic import BaseModel, Field


class Question(BaseModel):
    id: int
    text: str
    options: dict[str, str] = Field(description="Dictionary of options (A, B, C, D)")
