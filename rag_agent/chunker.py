from typing import Iterable, List, Sequence
from rag_agent.models import Chunk, Document


class RecursiveChunker:
    """Hierarchical semantic text splitter with configurable chunk_size and sliding overlap."""

    DEFAULT_SEPARATORS: Sequence[str] = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ")

    def __init__(
        self,
        chunk_size: int = 320,
        chunk_overlap: int = 64,
        separators: Sequence[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must satisfy 0 <= chunk_overlap < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators) if separators is not None else self.DEFAULT_SEPARATORS

    def chunk_document(self, doc: Document) -> List[Chunk]:
        atomic_pieces = self._split_recursive(doc.text, list(self.separators))
        windows = self._merge_with_overlap(atomic_pieces)

        chunks: List[Chunk] = []
        search_cursor = 0
        for idx, text_slice in enumerate(windows):
            stripped = text_slice.strip()
            if not stripped:
                continue
            pos = doc.text.find(stripped, search_cursor)
            if pos == -1:
                pos = doc.text.find(stripped)
            start_char = max(0, pos)
            end_char = start_char + len(stripped)
            search_cursor = max(search_cursor, start_char + 1)

            meta = dict(doc.metadata)
            meta["chunk_index"] = idx
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{idx}",
                    doc_id=doc.doc_id,
                    text=stripped,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=meta,
                )
            )
        return chunks

    def chunk_documents(self, docs: Iterable[Document]) -> List[Chunk]:
        out: List[Chunk] = []
        for d in docs:
            out.extend(self.chunk_document(d))
        return out

    def _split_recursive(self, text: str, seps: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not seps:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        sep = seps[0]
        rest_seps = seps[1:]
        if sep not in text:
            return self._split_recursive(text, rest_seps)

        raw_parts = text.split(sep)
        pieces: List[str] = []
        for i, part in enumerate(raw_parts):
            reconstructed = part if i == 0 else (sep + part if sep.startswith("\n#") else part)
            if not reconstructed:
                continue
            if len(reconstructed) <= self.chunk_size:
                pieces.append(reconstructed)
            else:
                pieces.extend(self._split_recursive(reconstructed, rest_seps))
        return pieces

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for piece in pieces:
            p = piece.strip()
            if not p:
                continue
            join_len = len(p) + (1 if current else 0)
            if current and current_len + join_len > self.chunk_size:
                chunks.append(" ".join(current))
                # Retain trailing pieces that fit within chunk_overlap
                overlap_buf: List[str] = []
                overlap_len = 0
                for prev in reversed(current):
                    add_l = len(prev) + (1 if overlap_buf else 0)
                    if overlap_len + add_l > self.chunk_overlap:
                        break
                    overlap_buf.insert(0, prev)
                    overlap_len += add_l
                current = overlap_buf
                current_len = overlap_len

            current.append(p)
            current_len += len(p) + (1 if len(current) > 1 else 0)

        if current:
            chunks.append(" ".join(current))
        return chunks
