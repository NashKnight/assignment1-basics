import regex as re
from collections import Counter
import os
from typing import BinaryIO
from multiprocessing import Pool

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

def process_chunk(input_path, start, end, special_tokens):
    """
    并行对每一个 chunk 做以下操作：
    1. special tokens 切分
    2. 提取 pre-tokens
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # step3: 处理 special tokens
        pattern = "|".join(re.escape(spec) for spec in special_tokens)  # 先转义再用 | 连接，否则 | 也会被转义
        parts = re.split(pattern, chunk)  # 按照 special tokens 切割文本

        # step4: pre-tokenize 边统计初始频率
        local_counts = Counter()
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in parts:  # 取 special tokens 划分的每一个模块
            for match in re.finditer(PAT, part):  # finditer 返回 match 对象
                pre_token = match.group().encode("utf-8")  # pre-token 的原始 utf-8 表示, b'low'
                pre_token = tuple(bytes([x]) for x in pre_token)  # 拆成元组, (b'l', b'o', b'w')
                local_counts[pre_token] += 1
        return local_counts

def train_bpe(input_path, vocab_size, special_tokens):
    """
    输入:
        input_path: str | os.PathLike
            训练语料文件路径，例如 "tests/fixtures/corpus.en"
        vocab_size: int
            最终词表大小（含 256 个 byte + special tokens + merge 出的 token）
        special_tokens: list[str]
            特殊 token 字符串列表，例如 ["<|endoftext|>"]

    输出:
        vocab: dict[int, bytes]
            token id -> token bytes
        merges: list[tuple[bytes, bytes]]
            按时间顺序的 merge 规则列表，每项是 (left, right)

    流程:
        1. 初始化词表
        2. 读取文件，切分即将并行处理的 chunks
        3. 处理 special tokens（先按它们切开 / 去掉，不当普通文本去计 merge）
        4. 用 GPT-2 那条 regex 做 pre-tokenization，每个 pre-token → 转成 UTF-8 bytes 序列，并统计出现次数（频率表）
        5. 在频率表上做 BPE 循环：反复找最高频相邻 pair → merge → 记入 merges、更新 vocab
        返回 (vocab, merges)

    优化(相比于 naive bpe):
        1. 并行处理 special tokens 和 pre_tokens
        2. 只需要扫描一遍构建 counts，之后在 counts 上逐循环处理，而非每轮构建 pair_counts
    """
    # step1: 初始化词表
    vocab = {x: bytes([x]) for x in range(256)}  # 初始化256个字符
    for idx, spec in enumerate(special_tokens):  # 初始化 special tokens
        vocab[256+idx] = spec.encode("utf-8")
    vocab_len = len(vocab)

    # step2: 读取文件，利用给定的 find_chunk_boundaries 切分即将并行的 chunks
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        counts = Counter()  # 统计 pre-token 的频率表
    # 对每个 chunk 并行调用 process_chunk 处理
    jobs = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        # pool.map 只能应用一个参数，pool.starmap 可以让每个 worker 解包序列而接收多个参数(这里 jobs 每个元素是一个参数元组)
        results = pool.starmap(process_chunk, jobs)  # results 是多个 workers 返回的 local_counts 组成的列表
    # 把统计结果填入 counts
    counts = Counter()  # 统计 pre-token 的频率表
    for result in results:
        counts.update(result)  # 用局部字典更新全局字典
    
    # step5: bpe merge 循环
    merges = []
    while vocab_len < vocab_size:
        pair_counts = Counter()
        # 统计
        for pre_token in counts:
            for idx in range(len(pre_token)-1):
                pair_counts[(pre_token[idx], pre_token[idx+1])] += counts[pre_token]
        # 先比 pair_counts，计数相同时比 token 本身，选择字典序较大的
        best_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))
        # pre_tokens 中所有 best_pair 合并，构建新的 counts
        new_counts = Counter()
        for pre_token in counts:
            i = 0
            new_pre_token = []  # 新的合并后的 pre_token
            while i < len(pre_token):
                if i < len(pre_token)-1 and (pre_token[i], pre_token[i+1]) == best_pair:  # 遇到 best_pair
                    new_pre_token.append(best_pair[0] + best_pair[1])  # 添加 best_pair 的 bytes 形式(bytes 相加等价于字符拼接)
                    i += 2
                else:
                    new_pre_token.append(pre_token[i])
                    i += 1
            new_counts[tuple(new_pre_token)] += counts[pre_token]  # 频率复制
        counts = new_counts
        # 更新 vocab 和 merged
        vocab[vocab_len] = best_pair[0] + best_pair[1]
        vocab_len += 1
        merges.append(best_pair)

    return vocab, merges