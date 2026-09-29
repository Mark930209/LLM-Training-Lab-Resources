# 底座模型（权重不入库）

本目录存放 22 篇实验用的开源底座权重，988MB，不进 git（见仓库根 .gitignore）。

拉取方式（huggingface_hub，国内可加 `HF_ENDPOINT=https://hf-mirror.com`）：

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B", local_dir="model/Qwen2.5-0.5B")
```

模型信息（Apache 2.0）：Qwen2ForCausalLM，494M 参数，hidden 896 / 24 层 /
14 头 / GQA 2 KV 头 / vocab 151936（tokenizer 151643，293 行余量）/
tied embedding / max_pos 32768 / RoPE theta 1e6。
