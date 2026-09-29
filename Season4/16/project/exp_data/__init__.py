"""__init__.py —— exp_data：16 篇 Data Pipeline Lab。

从 Common Crawl WARC 到可训练 token 分片的全链路：
采样解析 → 正文抽取 → boilerplate 去除 → 质量过滤 → 语种识别
→ 去重 → PII 处理 → 分片 → 并行 tokenize → manifest 血缘。
"""
