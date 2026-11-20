import pickle
import regex
from collections.abc import Iterator, Iterable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab.copy()
        self.merges = merges

        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens

        if not self.vocab:
            next_id = 0
        else:
            next_id = max(self.vocab.keys()) + 1

        for token in self.special_tokens:
            if token.encode("utf-8") not in self.vocab.values():
                self.vocab[next_id] = token.encode("utf-8")
                next_id += 1

        self.token_id_by_bytes = {byte: id for id, byte in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    def decode(self, ids: list[int]) -> str:
        byte_parts = []
        for id in ids:
            byte_parts.append(self.vocab[id])
        combined_bytes = b"".join(byte_parts)
        return combined_bytes.decode("utf-8", errors="replace")

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None,
                   ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        pieces = []
        pretoken_bytes = pretoken.encode("utf-8")
        for byte in pretoken_bytes:
            pieces.append(bytes([byte]))

        i = 0
        while True:
            best_pair = None
            min_rank = float("inf")
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair
            if best_pair is None:
                break

            new_pieces = []
            i = 0
            while i < len(pieces):
                if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == best_pair:
                    new_pieces.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1

            pieces = new_pieces

        token_ids = []
        for piece in pieces:
            id = self.token_id_by_bytes[piece]
            token_ids.append(id)

        return token_ids

    def encode(self, text: str) -> list[int]:
        token_ids = []

        if self.special_tokens:
            sorted_tokens = sorted([regex.escape(token) for token in self.special_tokens],
                                   key=len, reverse=True)
            pattern = "(" + "|".join(sorted_tokens) + ")"
        else:
            pattern = None

        if pattern:
            text_parts = regex.split(pattern, text)
        else:
            text_parts = [text]

        for part in text_parts:
            if not part: continue
            if part in self.special_tokens:
                token_ids.append(self.token_id_by_bytes[part.encode("utf-8")])
            else:
                for match in regex.finditer(PAT, part):
                    token_ids.extend(self._encode_pretoken(match.group()))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            # 1. 检查尾巴是否是不完整的特殊 token 前缀
            is_unsafe = False
            for token in self.special_tokens:
                for k in range(1, len(token)):
                    if buffer.endswith(token[:k]):
                        is_unsafe = True
                        break
                if is_unsafe:
                    break
            if is_unsafe:
                continue

            # 2. 用 regex 的 \s 统一找空白段的起点
            whitespace_matches = list(regex.finditer(r"\s+", buffer))
            if not whitespace_matches:
                continue

            # 最后一个连续空白段的起始位置
            whitespace_start = whitespace_matches[-1].start()

            # 3. 检查特殊 token 内部截断 (把你原代码里的 pos 换成 whitespace_start)
            is_internal = False
            for token in self.special_tokens:
                for m in regex.finditer(regex.escape(token), buffer):
                    if m.start() < whitespace_start < m.end():
                        is_internal = True
                        break
                if is_internal:
                    break
            if is_internal:
                continue

            if whitespace_start > 0:
                safe_text = buffer[:whitespace_start]
                buffer = buffer[whitespace_start:]
                for id in self.encode(safe_text):
                    yield id

        if buffer:
            for id in self.encode(buffer):
                yield id
