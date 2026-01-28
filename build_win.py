# build_win.py (robusto, desde 0)
# MiausoftSuite 3.0 — Builder robusto para Windows (Python 3.14+)
#
# - Compila 4 apps (.pyw) + un dispatcher (para menú contextual multi-selección en una sola ventana).
# - Genera payload zip para el instalador.
# - Lanza el builder del instalador (installer_win.py).
#
from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SUITE_NAME = "MiausoftSuite"
SUITE_VERSION = "3.0"

# Requisito por Python 3.14 (PyInstaller agregó soporte en 6.15.0)
MIN_PYINSTALLER = (6, 15, 0)

# Entradas del usuario (ahora son .pyw)
APP_SCRIPTS: Dict[str, str] = {
    "txt_full": "conversoratxtcompleto.pyw",
    "txt_chapters": "conversoratxtporcapitulos.pyw",
    "split_merge": "divideyfusionatxt.pyw",
    "replace_chars": "reemplazacaracteresentxt.pyw",
}

# Nombres de EXE estables (sin espacios para evitar edge-cases en registry / cmd)
APP_EXE_BASE: Dict[str, str] = {
    "txt_full": "MiausoftSuite_ConversorTXT_Completo",
    "txt_chapters": "MiausoftSuite_ConversorTXT_PorCapitulos",
    "split_merge": "MiausoftSuite_DivisorFusionador",
    "replace_chars": "MiausoftSuite_ReemplazadorCaracteres",
    "dispatcher": "MiausoftSuite_Dispatcher",
}


@dataclass(frozen=True)
class BuildPaths:
    project_dir: Path
    release_dir: Path
    dist_dir: Path
    work_dir: Path
    spec_dir: Path
    payload_zip: Path


def _die(msg: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _is_windows() -> bool:
    return os.name == "nt"


def _parse_version_tuple(s: str) -> Tuple[int, int, int]:
    parts: List[int] = []
    buf = ""
    for ch in s:
        if ch.isdigit():
            buf += ch
        else:
            if buf:
                parts.append(int(buf))
                buf = ""
            if len(parts) >= 3:
                break
    if buf and len(parts) < 3:
        parts.append(int(buf))
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _ensure_pyinstaller() -> None:
    try:
        v = importlib.metadata.version("pyinstaller")
    except Exception as e:
        _die(
            "PyInstaller no está instalado.\n"
            "Instala/actualiza con:\n"
            "  python -m pip install -U pyinstaller\n"
            f"Detalle: {e}"
        )
    vt = _parse_version_tuple(v)
    if vt < MIN_PYINSTALLER:
        _die(
            f"PyInstaller {v} no soporta bien Python {sys.version_info.major}.{sys.version_info.minor}.\n"
            f"Necesitas PyInstaller >= {MIN_PYINSTALLER[0]}.{MIN_PYINSTALLER[1]}.{MIN_PYINSTALLER[2]}.\n"
            "Actualiza con:\n"
            "  python -m pip install -U pyinstaller"
        )


def _ensure_tk_available() -> None:
    try:
        import tkinter  # noqa: F401
    except Exception as e:
        _die(
            "Este Python no puede importar tkinter.\n"
            "En Windows, instala Python con el componente 'tcl/tk and IDLE'.\n"
            f"Detalle: {e}"
        )


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    print("\n>> " + " ".join(map(str, cmd)))
    p = subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _maybe(project_dir: Path, name: str) -> Optional[Path]:
    p = project_dir / name
    return p if p.exists() else None


def _resolve_paths(project_dir: Path, release_dir: Path) -> BuildPaths:
    release_dir = release_dir.resolve()
    return BuildPaths(
        project_dir=project_dir.resolve(),
        release_dir=release_dir,
        dist_dir=release_dir / "dist",
        work_dir=release_dir / "build",
        spec_dir=release_dir / "spec",
        payload_zip=release_dir / "payload" / "miausoftsuite_payload.zip",
    )


def _write_dispatcher_source(tmp_dir: Path) -> Path:
    # Nota: usamos triple comillas DOBLES dentro para no romper el archivo externo.
    code = """    from __future__ import annotations

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
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return
    with open(QUEUE_FILE, "a", encoding="utf-8") as f:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        except Exception:
            pass
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
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
            QUEUE_FILE.write_text("\n".join(keep) + "\n", encoding="utf-8")
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
"""
    out = tmp_dir / "dispatcher_tmp.py"
    out.write_text(code.strip() + "\n", encoding="utf-8", newline="\n")
    return out


def _pyinstaller_cmd(
    *,
    name: str,
    script: Path,
    dist_dir: Path,
    work_dir: Path,
    spec_dir: Path,
    icon: Optional[Path],
    windowed: bool,
    onefile: bool,
    add_data: List[Tuple[Path, str]],
) -> List[str]:
    args: List[str] = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--noupx",
        "--name", name,
        "--distpath", str(dist_dir),
        "--workpath", str(work_dir),
        "--specpath", str(spec_dir),
        "--log-level", "WARN",
    ]
    args.append("--onefile" if onefile else "--onedir")
    if windowed:
        args.append("--windowed")
    if icon is not None:
        args += ["--icon", str(icon)]
    for src, dest in add_data:
        args += ["--add-data", f"{src}{os.pathsep}{dest}"]
    args.append(str(script))
    return args


def _build_exe(
    *,
    bp: BuildPaths,
    name: str,
    script: Path,
    icon: Optional[Path],
    onefile: bool,
    windowed: bool,
    add_data: List[Tuple[Path, str]],
) -> Path:
    work = bp.work_dir / name
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    args = _pyinstaller_cmd(
        name=name,
        script=script,
        dist_dir=bp.dist_dir,
        work_dir=work,
        spec_dir=bp.spec_dir,
        icon=icon,
        windowed=windowed,
        onefile=onefile,
        add_data=add_data,
    )
    _run(args, cwd=bp.project_dir)

    exe = bp.dist_dir / (name + ".exe")
    if exe.exists():
        return exe
    cand = bp.dist_dir / name / (name + ".exe")
    if cand.exists():
        return cand
    _die(f"No se encontró el ejecutable resultante: {name}")
    raise AssertionError("unreachable")


def _zip_payload(payload_zip: Path, items: List[Tuple[Path, str]]) -> None:
    payload_zip.parent.mkdir(parents=True, exist_ok=True)
    if payload_zip.exists():
        payload_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(payload_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in items:
            zf.write(src, arcname=arc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, default=Path.cwd())
    ap.add_argument("--release-dir", type=Path, default=Path.cwd() / "release")
    ap.add_argument("--apps-mode", choices=["onefile", "onedir"], default="onefile")
    ap.add_argument("--setup-mode", choices=["onefile", "onedir"], default="onedir")
    ap.add_argument("--also-setup-onefile", action="store_true")
    ap.add_argument("--console-setup", action="store_true")
    ap.add_argument("--uac-admin-setup", action="store_true")
    args = ap.parse_args()

    if not _is_windows():
        _die("Este builder es para Windows (os.name == 'nt').")
    if sys.version_info < (3, 14):
        _die("Requiere Python 3.14+.")

    _ensure_pyinstaller()
    _ensure_tk_available()

    bp = _resolve_paths(args.project_dir, args.release_dir)
    for d in [bp.dist_dir, bp.work_dir, bp.spec_dir, bp.payload_zip.parent]:
        d.mkdir(parents=True, exist_ok=True)

    icon = _maybe(bp.project_dir, "Miausoft.ico")
    png = _maybe(bp.project_dir, "Miausoft.png")
    cfg = _maybe(bp.project_dir, "config.py")

    add_data: List[Tuple[Path, str]] = []
    for res in [icon, png, cfg]:
        if res is not None:
            add_data.append((res, "."))

    apps_onefile = (args.apps_mode == "onefile")
    windowed = True

    built: Dict[str, Path] = {}
    for app_id, script_name in APP_SCRIPTS.items():
        script = bp.project_dir / script_name
        if not script.exists():
            _die(f"No se encontró: {script}")

        built[app_id] = _build_exe(
            bp=bp,
            name=APP_EXE_BASE[app_id],
            script=script,
            icon=icon,
            onefile=apps_onefile,
            windowed=windowed,
            add_data=add_data,
        )

    tmp_dir = bp.release_dir / "_tmp_dispatcher"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dispatcher_src = _write_dispatcher_source(tmp_dir)

    built["dispatcher"] = _build_exe(
        bp=bp,
        name=APP_EXE_BASE["dispatcher"],
        script=dispatcher_src,
        icon=icon,
        onefile=apps_onefile,
        windowed=windowed,
        add_data=add_data,
    )

    payload_items: List[Tuple[Path, str]] = []
    for _, p in built.items():
        payload_items.append((p, p.name))
    for res in [icon, png, cfg]:
        if res is not None:
            payload_items.append((res, res.name))

    _zip_payload(bp.payload_zip, payload_items)
    print(f"\nOK payload: {bp.payload_zip}")

    installer_py = bp.project_dir / "installer_win.py"
    if not installer_py.exists():
        _die(f"No se encontró installer_win.py en: {installer_py}")

    def _build_setup(mode: str) -> None:
        cmd = [
            sys.executable, str(installer_py),
            "--project-dir", str(bp.project_dir),
            "--payload", str(bp.payload_zip),
            "--out-dir", str(bp.release_dir),
            "--mode", mode,
        ]
        if args.console_setup:
            cmd.append("--console")
        if args.uac_admin_setup:
            cmd.append("--uac-admin")
        _run(cmd, cwd=bp.project_dir)

    _build_setup(args.setup_mode)
    if args.also_setup_onefile and args.setup_mode != "onefile":
        _build_setup("onefile")


if __name__ == "__main__":
    main()
