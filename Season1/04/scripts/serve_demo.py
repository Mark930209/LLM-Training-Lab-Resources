#!/usr/bin/env python3
"""serve_demo.py —— 模型对比演示 WebUI（零依赖，只用 Python 标准库）。

用途：切换不同规模的模型，输入同一个提示词，并排看续写结果有什么不同。
这是 04 篇"加大语料会发生什么"的直观演示——语料和步数相同，只换模型或语料。

设计取舍：
    不用 Gradio/Flask（需要额外安装依赖，读者环境未必有）；
    只用 http.server + json，读者 clone 下来就能跑。

用法（WSL2 内，工程根目录）：
    python scripts/serve_demo.py
    # 然后浏览器打开 http://localhost:9981

    # 也可指定端口与默认温度
    python scripts/serve_demo.py --port 9981 --temperature 0.8

说明：提示词可自由修改，改完回车或点"生成"就会让四个模型全部重新续写。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_scale.data import load_corpus  # noqa: E402
from exp_scale.model import SuperMiniGPT  # noqa: E402

RUNS = Path.home() / "llm-training-lab/runs"
DATA = Path.home() / "llm-training-lab/exp_scale/data"

# 模型清单：标签 → (run 目录名, 语料档, 说明)
MODELS = {
    "10m_small": {
        "label": "12.36M · 小语料",
        "run": "scale_main_scale_10m_20260916_090305",
        "corpus": "small",
        "note": "西游记单本 72.3 万字，best val 4.3391，1200 步过拟合",
    },
    "10m_large": {
        "label": "12.36M · 大语料",
        "run": "scale_main_scale_10m_large_20260916_103221",
        "corpus": "large",
        "note": "四大名著 309.1 万字，best val 3.6464，3000 步未过拟合",
    },
    "30m_large": {
        "label": "35.32M · 大语料",
        "run": "scale_main_scale_30m_20260916_090558",
        "corpus": "large",
        "note": "四大名著，best val 3.6163，2750 步过拟合",
    },
    "100m_large": {
        "label": "89.57M · 大语料",
        "run": "scale_main_scale_100m_20260916_091537",
        "corpus": "large",
        "note": "四大名著，best val 3.7649，1750 步过拟合",
    },
}

_cache: dict[str, dict] = {}
_lock = threading.Lock()


def load_model(key: str) -> dict:
    """按需加载模型并缓存。

    推理模式（eval）下不需要优化器状态，四档模型合计约 600MB 权重，
    8GB 卡完全装得下，因此全部缓存、不逐出——换提示词重新生成时无需重载，
    响应从十几秒降到一秒内。
    """
    with _lock:
        if key in _cache:
            return _cache[key]
        spec = MODELS[key]
        run_dir = RUNS / spec["run"]
        ckpt = run_dir / "ckpt_best.pt"
        if not ckpt.exists():
            ckpt = run_dir / "ckpt_last.pt"
        ck = torch.load(ckpt, map_location="cuda", weights_only=False)
        exp = ck["config"]["experiment"]
        tok, _, _ = load_corpus(DATA, exp["seq_len"], corpus=spec["corpus"])
        model = SuperMiniGPT(tok.vocab_size, exp["hidden"], exp["layers"],
                             exp["heads"], exp["seq_len"]).cuda()
        model.load_state_dict(ck["model"])
        model.eval()
        entry = {"model": model, "tok": tok, "seq_len": exp["seq_len"],
                 "meta": {**spec, "vocab": tok.vocab_size,
                          "best_val": ck.get("best_val")}}
        _cache[key] = entry
        print(f"[loaded] {spec['label']} "
              f"(显存 {torch.cuda.memory_allocated()/1e6:.0f} MB)")
        return entry


def generate(model_key: str, prompt: str, max_tokens: int,
             temperature: float, top_k: int) -> dict:
    e = load_model(model_key)
    ids = e["tok"].encode(prompt) or [0]
    ctx = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = e["model"].generate(ctx, max_tokens, temperature=temperature,
                                  top_k=top_k)
    text = e["tok"].decode(out[0].tolist())
    return {"text": text, "meta": e["meta"], "prompt": prompt}


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>模型对比演示 · LLM Training Lab</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:"Noto Sans SC","Microsoft YaHei",-apple-system,sans-serif;
       background:#F4F7FB; color:#1E293B; padding:28px; }
h1 { font-size:26px; color:#0F2A6B; margin-bottom:6px; }
.sub { font-size:14px; color:#64748B; margin-bottom:20px; }
.bar { background:#fff; border:2px solid #CBD5E1; border-radius:14px;
       padding:16px 20px; margin-bottom:18px; display:flex; gap:14px;
       align-items:center; flex-wrap:wrap; }
input[type=text] { flex:1; min-width:220px; padding:11px 15px; font-size:15px;
       border:1.5px solid #CBD5E1; border-radius:9px; font-family:inherit; }
input[type=text]:focus { outline:none; border-color:#3B82F6; }
button { padding:11px 26px; font-size:15px; font-weight:700; color:#fff;
       background:#3B82F6; border:none; border-radius:9px; cursor:pointer;
       font-family:inherit; }
button:hover { background:#2563EB; }
button:disabled { background:#94A3B8; cursor:wait; }
.opts { display:flex; gap:16px; align-items:center; font-size:13.5px; color:#475569; }
.opts label { display:flex; gap:6px; align-items:center; }
.grid { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }
.card { background:#fff; border:2px solid #CBD5E1; border-radius:14px;
        padding:18px 20px; display:flex; flex-direction:column; }
.card.loading { opacity:.65; }
.card h3 { font-size:17px; color:#0F2A6B; margin-bottom:4px; }
.card .note { font-size:12.5px; color:#64748B; margin-bottom:10px;
        padding-bottom:9px; border-bottom:1px solid #E2E8F0; }
.out { font-size:15px; line-height:1.85; color:#1E293B; white-space:pre-wrap;
       word-break:break-all; flex:1; min-height:150px; }
.out .hl { background:#FEF3C7; }
.tag { display:inline-block; font-size:11.5px; font-weight:700; padding:2px 9px;
       border-radius:20px; background:#EFF6FF; color:#1D4ED8; margin-left:8px; }
.footer { margin-top:20px; font-size:13px; color:#94A3B8; text-align:center; }
</style>
</head>
<body>
<h1>模型对比演示：同一个提示词，不同模型续写</h1>
<div class="sub">语料和步数固定，切换模型规模与语料档位，看续写结果有什么变化（base 模型是续写器，不是问答助手）</div>

<div class="bar">
  <input type="text" id="prompt" value="却说那" placeholder="输入开头，模型会往下续写（改完按回车即可重新生成）">
  <div class="opts">
    <label>长度 <input type="number" id="maxTok" value="180" min="20" max="400" style="width:64px;padding:6px;"></label>
    <label>温度 <input type="number" id="temp" value="0.8" min="0.1" max="1.5" step="0.1" style="width:58px;padding:6px;"></label>
  </div>
  <button id="go">生成</button>
</div>

<div class="grid" id="grid"></div>
<div class="footer" id="footer">输出为真实推理结果，temperature 0.8 / top-k 40 ｜ 首次生成需加载模型，请稍候</div>

<script>
const MODELS = __MODELS__;
const grid = document.getElementById('grid');

// 初始化卡片
Object.entries(MODELS).forEach(([key, m]) => {
  const d = document.createElement('div');
  d.className = 'card'; d.id = 'card-' + key;
  d.innerHTML = `<h3>${m.label}<span class="tag">待生成</span></h3>
                 <div class="note">${m.note}</div>
                 <div class="out" id="out-${key}"></div>`;
  grid.appendChild(d);
});

const btn = document.getElementById('go');
const promptEl = document.getElementById('prompt');
const footer = document.getElementById('footer');

async function generateAll() {
  const prompt = promptEl.value;
  const max_tokens = +document.getElementById('maxTok').value;
  const temperature = +document.getElementById('temp').value;
  btn.disabled = true; btn.textContent = '生成中…';
  footer.textContent = '生成中：四个模型并行推理同一提示词…';

  const t0 = performance.now();
  await Promise.all(Object.keys(MODELS).map(async (key) => {
    const card = document.getElementById('card-' + key);
    const out = document.getElementById('out-' + key);
    card.classList.add('loading');
    out.textContent = '…';
    card.querySelector('.tag').textContent = '生成中';
    try {
      const r = await fetch('/api/generate', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({model:key, prompt, max_tokens, temperature, top_k:40})
      });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      const p = j.prompt;
      const rest = j.text.startsWith(p) ? j.text.slice(p.length) : j.text;
      out.innerHTML = `<span class="hl">${p}</span>${rest}`;
      card.querySelector('.tag').textContent = 'val ' + (j.meta.best_val||0).toFixed(3);
    } catch(e) {
      out.textContent = '出错: ' + e;
      card.querySelector('.tag').textContent = '失败';
    }
    card.classList.remove('loading');
  }));
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  footer.textContent = `本次生成耗时 ${dt}s ｜ 改提示词后按回车或点"生成"可重新续写 ｜ `
    + `输出为真实推理结果，temperature ${temperature} / top-k 40`;
  btn.disabled = false; btn.textContent = '生成';
}

btn.onclick = generateAll;
promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !btn.disabled) generateAll();
});

// 页面打开后自动预热模型（后台加载，不阻塞界面）
fetch('/api/preload', { method:'POST' }).then(() => {
  footer.textContent = '模型已就绪 ｜ 改提示词后按回车或点"生成"即可重新续写';
}).catch(() => {});

// 首屏自动跑一次，方便直接看到效果
window.addEventListener('load', () => setTimeout(generateAll, 400));
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静音默认日志
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path in ("/", "/index.html"):
            html = PAGE.replace("__MODELS__", json.dumps(
                {k: {"label": v["label"], "note": v["note"]}
                 for k, v in MODELS.items()}, ensure_ascii=False))
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/preload":
            try:
                for k in MODELS:
                    load_model(k)
                self._send(200, json.dumps({"ok": True}).encode("utf-8"),
                           "application/json")
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"error": str(e)}).encode("utf-8"),
                           "application/json")
            return
        if path != "/api/generate":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            res = generate(req.get("model", "10m_large"),
                           req.get("prompt", "却说那"),
                           int(req.get("max_tokens", 180)),
                           float(req.get("temperature", 0.8)),
                           int(req.get("top_k", 40)))
            self._send(200, json.dumps(res, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"),
                       "application/json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9981)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"演示服务已启动: http://localhost:{args.port}")
    print("可用模型:")
    for k, v in MODELS.items():
        print(f"  {k:12s} {v['label']:18s} {v['note']}")
    srv.serve_forever()


if __name__ == "__main__":
    main()