#!/usr/bin/env bash
# 远端 WSL 监听探针：在 29801 上起 TCP 监听，验证 mirrored 模式下入站是否可达。
# 用法： ssh <host> 'bash -s' < remote_listen_probe.sh
set -u
PORT=29801
LOG=/tmp/listen${PORT}.log
rm -f "$LOG"
nohup timeout 90 python3 - "$PORT" > "$LOG" 2>&1 <<'PYEOF' &
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', port))
s.listen(1)
print('listening on', port, flush=True)
c, a = s.accept()
print('accepted from', a, flush=True)
data = c.recv(64)
print('received', data, flush=True)
c.send(b'PONG-' + data)
c.close()
PYEOF
sleep 1
cat "$LOG"
