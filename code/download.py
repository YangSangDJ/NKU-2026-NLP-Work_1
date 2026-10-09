"""带重试与断点续传的下载脚本（解决间歇性 DNS/网络失败）。

策略：
  1. 正常 urllib 下载（含重试与断点续传）；
  2. 若系统 DNS 解析失败，改用 UDP 直连 8.8.8.8 查询 A 记录，
     按 IP 直连并携带 SNI（server_hostname）完成 HTTPS 下载。

用法: python download.py <url> <dest> [max_attempts]
"""
import http.client
import socket
import ssl
import struct
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import urllib.request

DOH_SERVER = ("8.8.8.8", 53)


def _dns_query_a(host: str, timeout: float = 5.0) -> list[str]:
    """通过 UDP 向 8.8.8.8 查询 A 记录，返回 IPv4 地址列表。"""
    import random
    tid = random.randint(0, 0xFFFF)
    header = struct.pack(">HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    qname = b"".join(bytes([len(p)]) + p.encode() for p in host.split(".")) + b"\x00"
    question = qname + struct.pack(">HH", 1, 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(header + question, DOH_SERVER)
        data, _ = sock.recvfrom(4096)
    finally:
        sock.close()
    if len(data) < 12:
        return []
    ancount = struct.unpack(">H", data[6:8])[0]
    if ancount == 0:
        return []
    # 跳过 question 部分
    offset = 12
    while offset < len(data):
        if data[offset] == 0:
            offset += 1
            break
        offset += data[offset] + 1
    offset += 4  # QTYPE + QCLASS
    ips = []
    for _ in range(ancount):
        # 跳过应答的 NAME 字段（多数为 0xC00C 压缩指针，占 2 字节）
        if offset < len(data) and (data[offset] & 0xC0) == 0xC0:
            offset += 2
        else:
            while offset < len(data) and data[offset] != 0:
                offset += data[offset] + 1
            offset += 1
        if offset + 10 > len(data):
            break
        atype, aclass, ttl, rdlen = struct.unpack(">HHIH", data[offset:offset + 10])
        offset += 10
        if atype == 1 and rdlen == 4 and offset + 4 <= len(data):
            ips.append(socket.inet_ntoa(data[offset:offset + 4]))
        offset += rdlen
    return ips


def _download_by_ip(url: str, dest: Path, ip: str, redirects: int = 5) -> int:
    """按 IP 直连下载（显式 SNI），跟随重定向，返回已下载字节数。"""
    u = urlparse(url)
    port = u.port or 443
    ctx = ssl.create_default_context()
    raw = socket.create_connection((ip, port), timeout=120)
    tls = ctx.wrap_socket(raw, server_hostname=u.hostname)  # SNI 用真实域名
    conn = http.client.HTTPSConnection(ip, port, timeout=120)
    conn.sock = tls
    headers = {"User-Agent": "Mozilla/5.0", "Host": u.hostname}
    existing = dest.stat().st_size if dest.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
    path = u.path + (f"?{u.query}" if u.query else "")
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    if resp.status in (301, 302, 303, 307, 308):
        location = resp.getheader("Location")
        conn.close()
        if not location or redirects <= 0:
            raise RuntimeError(f"too many redirects: {url}")
        next_url = location if location.startswith("http") else (
            f"{u.scheme}://{u.hostname}{location}"
        )
        return _download_follow_dns(next_url, dest, redirects - 1)
    if resp.status not in (200, 206):
        conn.close()
        raise RuntimeError(f"HTTP {resp.status} {resp.reason}")
    mode = "ab" if (existing and resp.status == 206) else "wb"
    with open(dest, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    conn.close()
    return dest.stat().st_size


def _download_follow_dns(url: str, dest: Path, redirects: int = 5) -> int:
    """解析（系统 DNS，失败则 UDP 8.8.8.8）后按 IP 直连下载。"""
    host = urlparse(url).hostname
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        ips = sorted(set(i[4][0] for i in infos if ":" not in i[4][0]))
    except Exception:
        ips = _dns_query_a(host)
    if not ips:
        raise RuntimeError(f"cannot resolve {host}")
    last = None
    for ip in ips:
        try:
            return _download_by_ip(url, dest, ip, redirects)
        except Exception as e:
            last = e
    raise last or RuntimeError("all IPs failed")


def download(url: str, dest: Path, max_attempts: int = 8):
    dest.parent.mkdir(parents=True, exist_ok=True)
    host = urlparse(url).hostname
    for attempt in range(1, max_attempts + 1):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            existing = dest.stat().st_size if dest.exists() else 0
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                with open(dest, "ab" if existing else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            print(f"OK(urllib) {dest} size={dest.stat().st_size}", flush=True)
            return True
        except Exception as e:
            print(f"attempt {attempt} urllib failed: {type(e).__name__}: {e}", flush=True)
            # DNS 解析失败时走固定 DNS + IP 直连
            if isinstance(e, (socket.gaierror, urllib.error.URLError)) and host:
                try:
                    ips = _dns_query_a(host)
                    for ip in ips:
                        try:
                            size = _download_by_ip(url, dest, ip)
                            print(f"OK(by-ip {ip}) {dest} size={size}", flush=True)
                            return True
                        except Exception as e2:
                            print(f"  by-ip {ip} failed: {type(e2).__name__}: {e2}", flush=True)
                except Exception as e3:
                    print(f"  dns-query failed: {type(e3).__name__}: {e3}", flush=True)
            if attempt < max_attempts:
                time.sleep(3 * attempt)
    return False


if __name__ == "__main__":
    url, dest = sys.argv[1], Path(sys.argv[2])
    attempts = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ok = download(url, dest, attempts)
    sys.exit(0 if ok else 1)
