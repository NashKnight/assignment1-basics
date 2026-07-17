有几个关键问题：

re.split(pattern, text) 会把 special token 丢掉
但 encode 要保留 special token，并输出它自己的 id。你现在会忽略所有 special token。

BPE merge 逻辑不对
你现在只看 pieces[-1] 和当前 byte 能不能 merge，这只做了一遍从左到右的贪心合并。真正 BPE encode 要按 merges 的 rank 顺序反复应用，直到没有可合并 pair。

self.merges 如果是 list，in self.merges 很慢
应该先在 __init__ 里建：

self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
代码语法/方法问题
for i in range(len(pre_token)) 少冒号；tokens.expand(pieces) 应该是 tokens.extend(pieces)。

缩进有问题
tokens.extend(pieces) 应该在每个 match 处理完之后执行，而不是在 for match 外面，否则只会加入最后一个 pieces。

这个不是 encode_iterable
encode(text) 是处理一个完整字符串并返回 list[int]。
encode_iterable(iterable) 是处理很多字符串片段并 yield token id。

special tokens 的处理确实应该用 finditer 找位置，而不是普通 split 丢掉它们。大概结构是：

def encode(self, text):
    token_ids = []
    for chunk, is_special in self._split_on_special_tokens(text):
        if is_special:
            token_ids.append(self.token_to_id[chunk.encode("utf-8")])
        else:
            token_ids.extend(self._encode_normal_text(chunk))
    return token_ids
普通文本部分：

def _encode_normal_text(self, text):
    token_ids = []
    for match in re.finditer(PAT, text):
        pieces = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        pieces = self._apply_merges(pieces)
        token_ids.extend(self.token_to_id[piece] for piece in pieces)
    return token_ids
_apply_merges 应该每次找当前 pieces 中 rank 最小的 pair，然后 merge 它：

while True:
    pairs = [
        (self.merge_ranks[(pieces[i], pieces[i + 1])], i)
        for i in range(len(pieces) - 1)
        if (pieces[i], pieces[i + 1]) in self.merge_ranks
    ]
    if not pairs:
        break
    _, i = min(pairs)
    pieces = pieces[:i] + (pieces[i] + pieces[i + 1],) + pieces[i + 2:]
所以：你的方向对，但现在还不是正确的 BPE encode。