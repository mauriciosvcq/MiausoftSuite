from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import msvcrt  # type: ignore
except Exception:
    msvcrt = None  # type: ignore

SUITE_NAME = "MiausoftSuite"
QUEUE_DIR = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / SUITE_NAME / "dispatcher"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_FILE = QUEUE_DIR / "queue.jsonl"
LOCK_FILE = QUEUE_DIR / "lock"

APP_EXE = {
    "txt_full": "MiausoftSuite_ConversorTXT_Completo.exe",
    "txt_chapters": "MiausoftSuite_ConversorTXT_PorCapitulos.exe",
    "split_merge": "MiausoftSuite_DivisorFusionador.exe",
    "replace_chars": "MiausoftSuite_ReemplazadorCaracteres.exe",
}

def _append_line(obj: dict) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    if msvcrt is None:
        with open(QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "
")
        return
    with open(QUEUE_FILE, "a", encoding="utf-8") as f:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        except Exception:
            pass
        f.write(json.dumps(obj, ensure_ascii=False) + "
")
        f.flush()
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass

def _try_acquire_lock() -> bool:
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False

def _release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass

def _drain_queue(app_id: str) -> list[str]:
    if not QUEUE_FILE.exists():
        return []
    try:
        data = QUEUE_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    take: list[str] = []
    keep: list[str] = []
    for ln in data:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if obj.get("app_id") == app_id:
            for p in obj.get("paths", []):
                if isinstance(p, str) and p:
                    take.append(p)
        else:
            keep.append(ln)
    try:
        if keep:
            QUEUE_FILE.write_text("
".join(keep) + "
", encoding="utf-8")
        else:
            QUEUE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    seen = set()
    out: list[str] = []
    for p in take:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def main() -> int:
    args = sys.argv[1:]
    app_id = None
    if "--app" in args:
        i = args.index("--app")
        if i + 1 < len(args):
            app_id = args[i + 1]
            args = args[:i] + args[i + 2:]
    if not app_id or app_id not in APP_EXE:
        return 2

    paths = [a for a in args if not a.startswith("--") and a.strip()]
    _append_line({"app_id": app_id, "paths": paths})

    if not _try_acquire_lock():
        return 0

    try:
        time.sleep(0.35)
        all_paths = _drain_queue(app_id)

        exe = Path(sys.executable).resolve()
        install_dir = exe.parent
        target = install_dir / APP_EXE[app_id]
        if not target.exists():
            target = Path.cwd() / APP_EXE[app_id]

        subprocess.Popen([str(target)] + all_paths, cwd=str(target.parent))
        return 0
    finally:
        _release_lock()

if __name__ == "__main__":
    raise SystemExit(main())
