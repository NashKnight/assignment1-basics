# Assignment 1 实验报告

## 可复现说明

所有已经序列化的 BPE 训练结果都保存在 `artifacts/bpe_training_results.jsonl` 中。`scripts/train_bpe_datasets.py` 只负责训练数据集并写入 JSONL 记录；本报告是根据这些记录和对应日志手写整理的，不由脚本自动生成。

训练 TinyStories tokenizer 的命令：

```bash
nohup bash -lc 'source /root/.local/bin/env && uv run python scripts/train_bpe_datasets.py train --dataset tinystories --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --results artifacts/bpe_training_results.jsonl' > logs/train_bpe_tinystories_10000.log 2>&1 < /dev/null &
```

训练 OpenWebText tokenizer 的命令：

```bash
nohup bash -lc 'source /root/.local/bin/env && uv run python scripts/train_bpe_datasets.py train --dataset owt --input data/owt_train.txt --vocab-size 32000 --results artifacts/bpe_training_results.jsonl' > logs/train_bpe_owt_32000.log 2>&1 < /dev/null &
```

## Problem: unicode1

### (a)

`chr(0)` 返回 Unicode 空字符，也就是 `U+0000`。

### (b)

它的 `repr` 表示是转义字符串 `\x00`，而直接 `print` 时会输出一个不可见的空控制字符。

### (c)

当空字符出现在文本中时，它仍然是字符串的一部分，也会计入字符串长度；只是通常没有可见的打印字形。

## Problem: unicode2

### (a)

在 byte-level tokenizer 中更适合使用 UTF-8，因为它是互联网上最主流的文本编码，对以 ASCII 为主的英文文本也更紧凑，同时仍然可以把任意 Unicode 字符串表示成字节序列。相比之下，UTF-16 和 UTF-32 会在常见英文文本中引入更多额外的零字节或固定宽度模式，浪费序列长度，也会让基于网页语料学到的 byte merge 不那么自然。

### (b)

例如，`"こんにちは".encode("utf-8")` 会让题目中的错误函数失败，因为这个函数试图逐个字节单独解码；但每个日文字符在 UTF-8 中都由多个字节共同表示，单独拆开的字节并不是合法的独立 UTF-8 字符。

### (c)

`b"\xff\xff"` 是一个无法解码成 Unicode 字符的两字节序列，因为 `0xff` 既不是合法的 UTF-8 起始字节，也不是合法的 UTF-8 continuation byte。

## Problem: train_bpe

我在 `cs336_basics/train_bpe.py` 中实现了 `train_bpe(input_path, vocab_size, special_tokens)`：它返回 `dict[int, bytes]` 形式的 vocab 和按生成顺序排列的 `list[tuple[bytes, bytes]]` merges；训练时会把 special tokens 当作硬边界处理，使用 GPT-2 风格的正则表达式做 pre-tokenization，并支持给实验脚本使用的进度回调。当前针对 `tests/test_train_bpe.py` 的结果是：`test_train_bpe` 和 `test_train_bpe_special_tokens` 通过，`test_train_bpe_speed` 还需要继续优化（小 fixture 上约 `2.85s`，测试阈值是 `1.5s`）。

## Problem: train_bpe_tinystories

### (a)

我在 TinyStories 上训练了一个 byte-level BPE tokenizer，最大词表大小为 10,000，并把 `<|endoftext|>` 加入为 special token。词表和 merges 已经序列化到 `artifacts/bpe_training_results.jsonl`；训练总耗时为 `25:36.48`，峰值常驻内存为 `3458572 KB`（约 `3.3 GiB`），词表中最长的 token 是 `b' accomplishment'`，长度为 15 bytes，解码后是 `' accomplishment'`。这个结果是合理的，因为 TinyStories 中常见英文词片段，尤其是带前导空格的词，会因为频率较高而被 BPE 合并成单个 token。

### (b)

当前 tokenizer 训练中最慢的部分是 BPE merge loop。具体来说，每一轮 merge 都会在当前所有 pre-token 上重新统计相邻 pair 的频率，并重建 `counts` 表；这种重复扫描主导了整体运行时间。
