"""LAN TCP 吞吐探测：一端 listen，一端 connect 并推数据。

用法：
  服务端（gpu_office）:  python lan_throughput.py server --port 29600
  客户端（本机 WSL）  :  python lan_throughput.py client --host 10.66.8.91 --port 29600
"""

import argparse
import socket
import time

CHUNK = 4 * 1024 * 1024  # 4 MB


def server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    print(f"[server] listening on 0.0.0.0:{port}", flush=True)
    conn, addr = s.accept()
    print(f"[server] accepted from {addr}", flush=True)
    total = 0
    t0 = time.perf_counter()
    while True:
        buf = conn.recv(CHUNK)
        if not buf:
            break
        total += len(buf)
    dt = time.perf_counter() - t0
    gbps = total / dt / 1e9
    mbps = total / dt / 1e6 * 8
    print(f"[server] received {total/1e6:.1f} MB in {dt:.3f}s", flush=True)
    print(f"[server] throughput = {mbps:.1f} Mbps = {gbps:.3f} GB/s", flush=True)
    conn.close()
    s.close()


def client(host, port, mb):
    payload = b"\x01" * CHUNK
    n_chunks = max(1, mb * 1024 * 1024 // CHUNK)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    t_connect0 = time.perf_counter()
    s.connect((host, port))
    print(f"[client] connected to {host}:{port} in {(time.perf_counter()-t_connect0)*1000:.2f} ms", flush=True)

    # 延迟探测：小包往返
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t0 = time.perf_counter()
    s.sendall(b"ping")
    dt_ping = (time.perf_counter() - t0) * 1000
    print(f"[client] first-send latency ~{dt_ping:.3f} ms", flush=True)

    t0 = time.perf_counter()
    for _ in range(n_chunks):
        s.sendall(payload)
    s.shutdown(socket.SHUT_WR)
    dt = time.perf_counter() - t0
    total = n_chunks * CHUNK
    mbps = total / dt / 1e6 * 8
    gbps = total / dt / 1e9
    print(f"[client] sent {total/1e6:.1f} MB in {dt:.3f}s", flush=True)
    print(f"[client] throughput = {mbps:.1f} Mbps = {gbps:.3f} GB/s", flush=True)
    s.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=["server", "client"])
    p.add_argument("--host", default="10.66.8.91")
    p.add_argument("--port", type=int, default=29600)
    p.add_argument("--mb", type=int, default=512)
    a = p.parse_args()
    if a.role == "server":
        server(a.port)
    else:
        client(a.host, a.port, a.mb)
