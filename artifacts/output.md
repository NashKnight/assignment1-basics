bpe

```
root@notebook-job-fdb2-hzl-143411-6cxzj:/user/hongchenye/CS336-assignments/assignment1-b
asics# uv run pytest tests/test_train_bpe.pyn_bpe.py
================================= test session starts ==================================
platform linux -- Python 3.12.13, pytest-9.0.2, pluggy-1.6.0
rootdir: /user/hongchenye/CS336-assignments/assignment1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 3 items                                                                      

tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

================================== 3 passed in 3.56s ===================================
```