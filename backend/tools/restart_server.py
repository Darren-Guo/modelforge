"""重启 8000 端口的后端服务（供审查工作流的测试门控使用）。

用法：python backend/tools/restart_server.py
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # backend/


def kill_port(port: int) -> None:
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    pids = set()
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            parts = line.split()
            if parts:
                pids.add(parts[-1])
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)


def wait_port(port: int, timeout: float = 25.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


if __name__ == "__main__":
    kill_port(8000)
    time.sleep(1.0)
    log = open(ROOT / "server.log", "w", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    ok = wait_port(8000)
    print("server restarted on :8000" if ok else "server failed to start")
    sys.exit(0 if ok else 1)
