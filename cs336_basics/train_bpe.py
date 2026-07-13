import regex as re
from collections import Counter

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
        1. 读文件 → text (str)
        2. 处理 special tokens（先按它们切开 / 去掉，不当普通文本去计 merge）
        3. 用 GPT-2 那条 regex 做 pre-tokenization
        4. 每个 pre-token → 转成 UTF-8 bytes 序列，并统计出现次数（频率表）
        5. 在频率表上做 BPE 循环：反复找最高频相邻 pair → merge → 记入 merges、更新 vocab
        6. 返回 (vocab, merges)
    """
    # step1: 读取文本并初始化词表
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    vocab = {x: bytes([x]) for x in range(256)}  # 初始化256个字符
    for idx, spec in enumerate(special_tokens):  # 初始化 special tokens
        vocab[256+idx] = spec.encode("utf-8")
    
    # step2: 处理 special tokens
    pattern = "|".join(re.escape(spec) for spec in special_tokens)  # 先转义再用 | 连接，否则 | 也会被转义
    parts = re.split(pattern, text)  # 按照 special tokens 切割文本

    # step3: pre-tokenize 边统计初始频率
    counts = Counter()  # 统计 pre-token 的频率表
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:  # 取 special tokens 划分的每一个模块
        for match in re.finditer(PAT, part):  # finditer 返回 match 对象
            pre_token = match.group().encode("utf-8")  # pre-token 的原始 utf-8 表示, b'low'
            pre_token = tuple(bytes([x]) for x in pre_token)  # 拆成元组, (b'l', b'o', b'w')
            counts[pre_token] += 1
    
    # step4: bpe merge 循环
