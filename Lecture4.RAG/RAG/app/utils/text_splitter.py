import re
from typing import List, Optional
from abc import ABC, abstractmethod


class BaseTextSplitter(ABC):
    """Base class for splitting text into chunks"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: chunk size in characters
            chunk_overlap: overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Splits text into chunks"""
        pass

    def _merge_splits(self, splits: List[str], separator: str = " ") -> List[str]:
        """Combines small parts into chunks of the desired size"""
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length > self.chunk_size and current_chunk:
                chunk_text = separator.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    current_chunk = self._get_overlap_splits(
                        current_chunk, separator)
                    current_length = len(separator.join(current_chunk))
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(split)
            current_length += split_length + len(separator)

        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        return chunks

    def _get_overlap_splits(self, splits: List[str], separator: str) -> List[str]:
        """Gets parts for overlap between chunks"""
        overlap_length = 0
        overlap_splits = []

        # Take splits from the end until reaching the desired overlap
        for split in reversed(splits):
            if overlap_length + len(split) + len(separator) <= self.chunk_overlap:
                overlap_splits.insert(0, split)
                overlap_length += len(split) + len(separator)
            else:
                break

        return overlap_splits


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    Recursively splits text, trying to preserve semantic boundaries
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or [
            "\n\n", "\n", ".", "!", "?", ";", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Recursively splits text"""
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting function"""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # If a part is too large, split it recursively
                if remaining_separators:
                    sub_splits = self._split_text_recursive(
                        split, remaining_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(split)

        return self._merge_splits(good_splits, separator)


class SentenceTextSplitter(BaseTextSplitter):
    """
    Splits text by sentences
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def split_text(self, text: str) -> List[str]:
        """Splits text by sentences"""
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return self._merge_splits(sentences, " ")


class ParagraphTextSplitter(BaseTextSplitter):
    """
    Splits text by paragraphs
    """

    def split_text(self, text: str) -> List[str]:
        """Splits text by paragraphs"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return self._merge_splits(paragraphs, '\n\n')


class FixedSizeTextSplitter(BaseTextSplitter):
    """
    Splits text into fixed-size chunks
    """

    def split_text(self, text: str) -> List[str]:
        """Splits text into fixed-size chunks"""
        chunks = []

        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks


class MarkdownTextSplitter(BaseTextSplitter):
    """
    Special splitter for Markdown documents
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.md_separators = [
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n###### ",
            "\n\n",
            "\n",
            " ",
            ""
        ]

    def split_text(self, text: str) -> List[str]:
        """Splits Markdown text preserving structure"""
        return self._split_text_recursive(text, self.md_separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting respecting Markdown structure"""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        # Rejoin parts with separator (except the first one)
        formatted_splits = []
        for i, split in enumerate(splits):
            if i > 0 and separator != "":
                split = separator + split
            formatted_splits.append(split)

        good_splits = []
        for split in formatted_splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if remaining_separators:
                    sub_splits = self._split_text_recursive(
                        split, remaining_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(split)

        return self._merge_splits(good_splits, "")


class TextSplitter:
    """
    Factory for creating text splitters
    """

    def __init__(self,
                 splitter_type: str = "recursive",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Args:
            splitter_type: type of splitter (recursive, sentence, paragraph, fixed, markdown)
            chunk_size: chunk size
            chunk_overlap: overlap between chunks
        """
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = self._create_splitter()

    def _create_splitter(self) -> BaseTextSplitter:
        """Creates the appropriate splitter type"""
        if self.splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(self.chunk_size, self.chunk_overlap)
        elif self.splitter_type == "sentence":
            return SentenceTextSplitter(self.chunk_size, self.chunk_overlap)
        elif self.splitter_type == "paragraph":
            return ParagraphTextSplitter(self.chunk_size, self.chunk_overlap)
        elif self.splitter_type == "fixed":
            return FixedSizeTextSplitter(self.chunk_size, self.chunk_overlap)
        elif self.splitter_type == "markdown":
            return MarkdownTextSplitter(self.chunk_size, self.chunk_overlap)
        else:
            raise ValueError(
                f"Unsupported splitter type: {self.splitter_type}")

    def split_text(self, text: str) -> List[str]:
        """Splits text into chunks"""
        return self.splitter.split_text(text)

    def get_chunks_info(self, text: str) -> dict:
        """Returns information about the text splitting"""
        chunks = self.split_text(text)

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            "min_chunk_size": min(len(chunk) for chunk in chunks) if chunks else 0,
            "max_chunk_size": max(len(chunk) for chunk in chunks) if chunks else 0,
            "total_characters": len(text),
            "chunks_preview": chunks[:3] if len(chunks) > 3 else chunks
        }
