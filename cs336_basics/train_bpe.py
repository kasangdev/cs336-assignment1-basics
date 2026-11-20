import regex
import multiprocessing
import time
from typing import Dict, List, Tuple

import os
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_text_chunk(
        text_chunk: list
):
    local_counts = {}

    for chunk in text_chunk:
        for match in regex.finditer(PAT, chunk):
            local_counts[match.group()] = local_counts.get(match.group(), 0) + 1

    return local_counts


def process_file_chunk(file_chunk: tuple[str, int, int, list[str]]

                       ) -> dict[str, int]:
    input_path, chunk_start, chunk_end, special_tokens = file_chunk

    with open(input_path, "rb") as f:
        f.seek(chunk_start)
        length = chunk_end - chunk_start
        chunk_bytes = f.read(length)
        text = chunk_bytes.decode("utf-8")

        special_tokens_list = []
        for special_token in special_tokens:
            special_tokens_list.append(regex.escape(special_token))
        pattern = "|".join(special_tokens_list)
        if pattern:
            chunk_list = regex.split(pattern, text)
        else:
            chunk_list = [text]

    return process_text_chunk(chunk_list)


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> Tuple[Dict[int, bytes], list[Tuple[bytes, bytes]]]:
    start1 = time.perf_counter()

    token_bytes_by_id = {i: bytes([i]) for i in range(256)}
    i = 256
    for special_token in special_tokens:
        token_bytes_by_id[i] = special_token.encode("utf-8")
        i += 1

    file_chunks = []
    desired_num_chunks = 512
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file_chunks.append((input_path, start, end, special_tokens))

    end1 = time.perf_counter()
    time_read_split = end1 - start1

    start2 = time.perf_counter()

    merges = []
    token_ids_by_pretoken = {}
    pretoken_counts = {}
    if len(file_chunks) <= 1:
        if file_chunks:
            single_result = process_file_chunk(file_chunks[0])
            for token, count in single_result.items():
                pretoken_counts[token] = pretoken_counts.get(token, 0) + count
    else:
        with multiprocessing.Pool(processes=20) as pool:
            results_iterator = pool.imap_unordered(process_file_chunk, file_chunks, chunksize=1)
            for local_dict in results_iterator:
                for token, count in local_dict.items():
                    pretoken_counts[token] = pretoken_counts.get(token, 0) + count

    end2 = time.perf_counter()

    time_tokenize = end2 - start2

    start3 = time.perf_counter()

    for token in pretoken_counts:
        token_ids_by_pretoken[token] = list(token.encode("utf-8"))

    pair_to_tokens = {}
    for token, sequence in token_ids_by_pretoken.items():
        for position in range(len(sequence) - 1):
            pair = (sequence[position], sequence[position + 1])
            if pair not in pair_to_tokens:
                pair_to_tokens[pair] = set()
            pair_to_tokens[pair].add(token)

    id_pair_counts = {}
    for token, sequence in token_ids_by_pretoken.items():
        for position in range(len(sequence) - 1):
            pair = (sequence[position], sequence[position + 1])
            id_pair_counts[pair] = id_pair_counts.get(pair, 0) + pretoken_counts[token]

    end3 = time.perf_counter()

    time_build_index = end3 - start3

    start4 = time.perf_counter()

    while len(token_bytes_by_id) < vocab_size:

        most_frequent_id_pairs = []
        if not len(id_pair_counts):
            break
        max_count = max(list(id_pair_counts.values()))
        if max_count == 0:
            break
        for pair, count in id_pair_counts.items():
            if count == max_count:
                most_frequent_id_pairs.append(pair)

        id_1, id_2 = most_frequent_id_pairs[0]
        best_pair = (id_1, id_2)
        best_pair_bytes = (token_bytes_by_id[id_1], token_bytes_by_id[id_2])
        for pair in most_frequent_id_pairs[1:]:
            id1, id2 = pair
            candidate_pairs_bytes = (token_bytes_by_id[id1], token_bytes_by_id[id2])
            if candidate_pairs_bytes > best_pair_bytes:
                best_pair_bytes = candidate_pairs_bytes
                best_pair = (id1, id2)

        merges.append(best_pair_bytes)

        newtoken = best_pair_bytes[0] + best_pair_bytes[1]
        new_id = len(token_bytes_by_id)
        token_bytes_by_id[new_id] = newtoken

        for token in list(pair_to_tokens[best_pair]):
            seq = token_ids_by_pretoken[token]
            for position in range(len(seq) - 1):
                pair = (seq[position], seq[position + 1])
                if pair in pair_to_tokens:
                    pair_to_tokens[pair].discard(token)
                    id_pair_counts[pair] -= pretoken_counts[token]

            new_seq = []
            i = 0
            n = len(seq)
            while i < n:
                if seq[i] == best_pair[0] and i + 1 < n and seq[i + 1] == best_pair[1]:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            token_ids_by_pretoken[token] = new_seq

            for position in range(len(new_seq) - 1):
                pair = (new_seq[position], new_seq[position + 1])
                if pair not in pair_to_tokens:
                    pair_to_tokens[pair] = set()
                pair_to_tokens[pair].add(token)
                id_pair_counts[pair] = id_pair_counts.get(pair, 0) + pretoken_counts[token]

    end4 = time.perf_counter()
    time_bpe_merge = end4 - start4

    print(f"1. 读文件切分: {time_read_split:.2f}s")
    print(f"2. 预分词统计: {time_tokenize:.2f}s")
    print(f"3. 建立初始索引: {time_build_index:.2f}s")
    print(f"4. BPE合并循环: {time_bpe_merge:.2f}s")
    total_time = time_read_split + time_tokenize + time_build_index + time_bpe_merge
    print(f"总耗时: {total_time:.2f}s")
    return token_bytes_by_id, merges
