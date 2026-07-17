import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # encode 需要 vocab 的反映射
        self.token_to_id = {v: k for k, v in self.vocab.items()}

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 自己训练 llm 的时候才会用到
        pass

    def encode(self, text):
        """
        输入：
            text: str
        输出：
            return: list，包含转化后 token id 的列表
        """
        # step1: text 按 special token 切分
        pattern = "|".join(re.escape(spec) for spec in self.special_tokens)  # 先转义再用 | 连接，否则 | 也会被转义
        parts = re.split(pattern, text)  # 按照 special tokens 切割文本

        # step2: 用 GPT-2 regex 进行 pre-tokenize
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = []
        for part in parts:  # 取 special tokens 划分的每一个模块
            for match in re.finditer(PAT, part):  # finditer 返回 match 对象
                pre_token = match.group().encode("utf-8")  # pre-token 的原始 utf-8 表示, b'low'
                pre_token = tuple(bytes([x]) for x in pre_token)  # 拆成元组, (b'l', b'o', b'w')
                pieces = []
                # 开始根据 merges 进行 bytes 合并
                i = 0
                for i in range(len(pre_token))
                    # 如果已合并的部分 pre 可以和当前 bytes 合并，则合并添加到 pieces
                    if pieces and (pieces[-1], pre_token[i]) in self.merges:
                        pre = pieces[-1]
                        pieces.append(pre + pre_token[i])
                    # 如果不能合并，单独添加当前 bytes
                    else:
                        pieces.append(pre_token[i])
            tokens.expand(pieces)
        
        input_ids = []
        for token in tokens:
            input_ids.append(self.token_to_id[token])

        return input_ids

    def decode(self, ids)