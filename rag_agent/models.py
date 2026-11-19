from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Document:
    """Raw source document before chunking."""

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        if not self.text.strip():
            raise ValueError("document text must be non-empty")


@dataclass(frozen=True)
class Chunk:
    """Indexed passage slice traceable back to a parent Document."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_len(self) -> int:
        return self.end_char - self.start_char


@dataclass(frozen=True)
class ScoredChunk:
    """Retrieved chunk paired with a relevance score and source retriever tag."""

    chunk: Chunk
    score: float
    retriever: str = "unknown"
    rank: int = 0
