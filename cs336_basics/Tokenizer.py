import regex as re
import base64
import json

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        # encode 需要 vocab 和 merges 的反映射，否则每次遍历查表太慢
        self.token_rank = {v: k for k, v in self.vocab.items()}  # key=token, value=id
        self.merge_rank = {merge: i for i, merge in enumerate(self.merges)}  # key=merges, value=idx

    @classmethod  # from_files 要在没有实例的时候创建一个实例，不能依赖 self，因此需要 @classmethod 自动把 Tokenizer 类传给第一个参数 cls
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        输入：
            vocab_filepath: str，保存 vocab 的文件路径
            merges_filepath: str，保存 merges 的文件路径
            special_tokens: list[str]，需要整体保留的特殊 token
        输出：
            return: Tokenizer，根据文件中保存的 vocab 和 merges 构造出的 tokenizer

        流程：
            1. 读取 vocab 文件，还原 token_id -> bytes 的映射
            2. 读取 merges 文件，还原按 rank 排序的 (bytes, bytes) merge 列表
            3. 调用 Tokenizer(vocab, merges, special_tokens) 构造实例
        """
        # step1: 从 vocab 文件读取 token_id -> bytes
        vocab = {}
        with open(vocab_filepath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():  # 空行
                    continue
                record = json.loads(line)
                if record["record_type"] == "vocab":
                    # json 保存时不能保存任意 bytes 对象，因此训练时保存了 base64 编码 (注意 bpe merge 以后是可能产生不完整的 utf-8 片段的，这些片段无法直接解码成 Unicode 写入 json 中)
                    vocab[int(record["token_id"])] = base64.b64decode(record["bytes_base64"])

        # step2: 从 merges 文件读取按训练顺序排列的 merge pair
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
            for record in sorted(records, key=lambda x: int(x["rank"])):
                if record["record_type"] == "merge":
                    merges.append(
                        (
                            base64.b64decode(record["left_base64"]),
                            base64.b64decode(record["right_base64"]),
                        )
                    )

        # step3: 返回用文件内容构造出的 Tokenizer 实例
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        """
        输入：
            text: str
        输出：
            return: list，包含转化后 token id 的列表
        """
        # step1: text 按 special token 切分
        # 先转义再用 | 连接，否则 | 也会被转义，然后 pattern 要带捕获组
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(spec) for spec in self.special_tokens) + ")"
            parts = re.split(pattern, text)  # 按照 special tokens 切割文本
        else:  # 如果没有 speical tokens，把 text 视为单独的普通文本段落
            parts = [text]

        # step2: 用 GPT-2 regex 进行 pre-tokenize
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = []
        for part in parts:  # 取 special tokens 划分的每一个模块
            if part in self.special_tokens:  # special tokens 本身直接添加
                tokens.append(part.encode("utf-8"))
                continue
            # 对于一般情况的文本，再使用 merge 逻辑
            for match in re.finditer(PAT, part):  # finditer 返回 match 对象
                pre_token = match.group().encode("utf-8")  # pre-token 的原始 utf-8 表示, b'low'
                pre_token = list(bytes([x]) for x in pre_token)  # 拆成列表, [b'l', b'o', b'w']
                # 开始根据 merges 进行 bytes 合并
                while True:
                    merge_pair = None
                    length = len(pre_token)
                    for i in range(length-1):
                        pair = (pre_token[i], pre_token[i+1])
                        if pair in self.merge_rank and (merge_pair is None or self.merge_rank[pair] < self.merge_rank[merge_pair]):
                            merge_pair = pair
                    if not merge_pair:  # 没有新的可合并 pair
                        break
                    # 对本轮需要合并的 pair 再遍历 pre_token 拼装成新的 pre_token
                    new_pre_token = []
                    i = 0
                    while i < length:
                        if i < length - 1 and (pre_token[i], pre_token[i+1]) == merge_pair:
                            new_pre_token.append(merge_pair[0] + merge_pair[1])
                            i += 2
                        else:
                            new_pre_token.append(pre_token[i])
                            i += 1
                    pre_token = new_pre_token
                tokens.extend(pre_token)
        
        input_ids = []
        for token in tokens:
            input_ids.append(self.token_rank[token])

        return input_ids
    
    def encode_iterable(self, iterable):
        """
        输入：
            iterable: 可迭代对象，元素是 str（常见用法是打开的文本文件，按行迭代）
        输出：
            return: generator，逐个 yield token id（int），而不是一次性返回完整 list

        目的：
            对大文本做流式 encode，避免把整个文件读进内存后再调用 encode。

        流程：
            1. 遍历 iterable 中的每一段文本（例如每一行）
            2. 对这段文本调用已有的 encode，得到 id 列表
            3. 再把局部的 id 逐个 yield
        """
        # step1: for chunk in iterable
        # step2: ids = self.encode(chunk)
        # step3: for token_id in ids: yield token_id
        for chunk in iterable:
            ids = self.encode(chunk)
            for id in ids:
                yield id
                

    def decode(self, ids):
        """
        输入：
            ids: list[int]，token id 序列
        输出：
            return: str，解码后的文本字符串

        流程：
            1. 用 vocab 把每个 id 映射回对应的 bytes token
            2. 将所有 bytes 按顺序拼接成一整段 bytes
            3. 再按 UTF-8 解码成 str（注意处理非法字节序列，通常 errors="replace"）
        """
        # step1: id -> bytes（查 self.vocab）
        # step2: 拼接所有 bytes
        # step3: bytes.decode("utf-8", errors="replace") 得到文本
        text = b""
        for id in ids:
            text += self.vocab[id]
        return text.decode("utf-8", errors="replace")