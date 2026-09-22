#!/usr/bin/env bash
# 远端 WSL 主动连本机监听端口，验证 mirrored 模式下反向入站。
set -u
python3 - <<'PYEOF'
import socket
s = socket.socket()
s.settimeout(8)
try:
    s.connect(('10.66.8.189', 29802))
    s.send(b'PING-FROM-4090')
    print('recv:', s.recv(64))
    print('RESULT: REMOTE_TO_LOCAL_WSL_DIRECT_OK')
except Exception as e:
    print('RESULT: FAILED', type(e).__name__, e)
finally:
    s.close()
PYEOF
