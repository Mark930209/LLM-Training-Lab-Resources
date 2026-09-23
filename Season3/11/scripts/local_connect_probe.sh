#!/usr/bin/env bash
# 本机 WSL 主动连远端监听端口，验证 mirrored 模式下 WSL→WSL 入站直连。
set -u
python3 - <<'PYEOF'
import socket
s = socket.socket()
s.settimeout(8)
try:
    s.connect(('192.168.0.126', 29801))
    s.send(b'PING')
    print('recv:', s.recv(64))
    print('RESULT: WSL_TO_WSL_DIRECT_OK')
except Exception as e:
    print('RESULT: FAILED', type(e).__name__, e)
finally:
    s.close()
PYEOF
