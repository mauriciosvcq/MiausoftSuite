# build_installer.py
# =============================================================================
# MiausoftSuite: BUILD (4 apps) + INSTALLER (mismo .py)
#
# Objetivo:
# - Ejecutar este archivo como .py (desde PyCharm) => BUILD:
#     1) Compila 4 apps a .\apps\ (única carpeta de build esperada)
#     2) Compila el instalador "MiuasoftSuite 3 for Windows.exe" junto a .\apps\
#     3) En consola: SOLO barra de progreso
#
# - Ejecutar este archivo como .exe (PyInstaller onefile, frozen) => INSTALLER:
#     1) PRIMER PASO SIEMPRE: instala una copia funcional en SendTo (prioridad)
#     2) Integra Inicio + Menú contextual + .miau apuntando a la copia de SendTo
#     3) SEGUNDO PASO (best-effort): copia adicional al destino elegido (ej. Program Files)
#        Si falla por permisos u otra razón: NO se colapsa, NO aborta; SendTo queda listo.
#
# Requisitos explícitos:
# - Sin binarios/base64 embebidos: el instalador copia EXEs desde la carpeta hermana .\apps
# - Fuente Comfortaa embebida en cada exe (add-data fonts\Comfortaa.ttf)
# - Ícono Miausoft.ico embebido (add-data + --icon)
# - Registro y shortcuts: HKCU (no admin), con refresh de shell.
# =============================================================================

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Constantes de suite
# =============================================================================

SUITE_VERSION_STR = "3.0"
SUITE_YEAR_STR = "2026"

INSTALLER_EXE_NAME = "MiuasoftSuite 3 for Windows"  # PyInstaller agrega .exe

INSTALLER_TITLE_LABEL = "Bienvenido al instalador de MiasoftSuite"
INSTALLER_SUBTITLE_LABEL = f"Gracias. Elija dónde instalar la versión {SUITE_VERSION_STR} de MiasoftSuite ({SUITE_YEAR_STR})"

START_MENU_FOLDER = "MiausoftSuite"
CONTEXT_MENU_ROOT_NAME = "MiausoftSuite"

# IDs internos (estables) para mapear nombre -> exe -> accesos/registro
APP_ID_TXT_FULL = "txt_full"
APP_ID_TXT_CHAPTERS = "txt_chapters"
APP_ID_SPLIT_MERGE = "split_merge"
APP_ID_REPLACE_CHARS = "replace_chars"

PRESET_EXT = ".miau"
PRESET_PROGID = "MiausoftSuite.Preset"
PRESET_TARGET_APP_ID = APP_ID_TXT_FULL

DISPLAY_NAMES: Dict[str, str] = {
    APP_ID_TXT_FULL: "MiausoftSuite Conversor a txt (completo)",
    APP_ID_TXT_CHAPTERS: "MiausoftSuite Conversor a txt (por capítulos)",
    APP_ID_SPLIT_MERGE: "MiausoftSuite Divisor y Fusionador",
    APP_ID_REPLACE_CHARS: "MiausoftSuite Reemplazador de caracteres",
}

# Nombres de archivo a instalar en SendTo (sin subcarpetas y sin guiones bajos).
# El payload original en .\apps\ conserva EXE_FILENAMES; aquí solo renombramos la copia instalada.
SENDTO_EXE_NAMES: Dict[str, str] = {
    APP_ID_TXT_FULL: "MiausoftSuite Conversor a txt (completo).exe",
    APP_ID_TXT_CHAPTERS: "MiausoftSuite Conversor a txt (por capítulos).exe",
    APP_ID_SPLIT_MERGE: "MiausoftSuite Divisor y Fusionador.exe",
    APP_ID_REPLACE_CHARS: "MiausoftSuite Reemplazador de caracteres.exe",
}

EXE_FILENAMES: Dict[str, str] = {
    APP_ID_TXT_FULL: "MiausoftSuite_ConversorTxtCompleto.exe",
    APP_ID_TXT_CHAPTERS: "MiausoftSuite_ConversorTxtPorCapitulos.exe",
    APP_ID_SPLIT_MERGE: "MiausoftSuite_DivideYFusionaTxt.exe",
    APP_ID_REPLACE_CHARS: "MiausoftSuite_ReemplazaCaracteresEnTxt.exe",
}

SCRIPT_FILENAMES: Dict[str, str] = {
    APP_ID_TXT_FULL: "conversoratxtcompleto.py",
    APP_ID_TXT_CHAPTERS: "conversoratxtporcapitulos.py",
    APP_ID_SPLIT_MERGE: "divideyfusionatxt.py",
    APP_ID_REPLACE_CHARS: "reemplazacaracteresentxt.py",
}

TARGET_EXTS_CONTEXT = [".pdf", ".epub", ".docx", ".doc", ".pptx", ".ppt", ".txt"]

# =============================================================================
# Logging (archivo + buffer para ventana de error)
# =============================================================================

_LOG_LOCK = threading.Lock()
_LOG_LINES: List[str] = []

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))

def _base_dir() -> Path:
    # Carpeta del .py o del .exe
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

def _apps_payload_dir() -> Path:
    return _base_dir() / "apps"

def _log_file_path() -> Path:
    # Log persistente (sin ensuciar root del proyecto):
    # - build: .\apps\_logs\build_installer.log
    # - installer: %TEMP%\MiausoftSuite_installer.log
    if _is_frozen():
        base = Path(os.environ.get("TEMP") or tempfile.gettempdir())
        return base / "MiausoftSuite_installer.log"
    d = _apps_payload_dir() / "_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "build_installer.log"

def _log(msg: str) -> None:
    s = f"[{_now_str()}] {msg}"
    with _LOG_LOCK:
        _LOG_LINES.append(s)
    try:
        lp = _log_file_path()
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "a", encoding="utf-8", errors="replace") as f:
            f.write(s + "\n")
    except OSError:
        # Logging es best-effort: nunca rompe el flujo
        pass

def _log_exc(prefix: str) -> None:
    _log(prefix + ":\n" + traceback.format_exc())

def _log_dump_text(limit: int = 6000) -> str:
    with _LOG_LOCK:
        txt = "\n".join(_LOG_LINES)
    return txt[-limit:] if len(txt) > limit else txt

# =============================================================================
# Consola: barra de progreso (builder)
# =============================================================================

class _ConsoleProgress:
    def __init__(self, total_steps: int) -> None:
        self.total = max(1, int(total_steps))
        self.done = 0
        self._last = ""

    def set(self, done: int) -> None:
        self.done = max(0, min(self.total, int(done)))
        self._draw()

    def inc(self, delta: int = 1) -> None:
        self.set(self.done + int(delta))

    def _draw(self) -> None:
        width = 42
        pct = self.done / self.total
        fill = int(round(width * pct))
        bar = "[" + ("#" * fill) + ("." * (width - fill)) + f"] {int(round(pct*100)):3d}%"
        if bar == self._last:
            return
        self._last = bar
        sys.stdout.write("\r" + bar)
        sys.stdout.flush()

    def finish(self) -> None:
        self.set(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()

# =============================================================================
# Builder: PyInstaller
# =============================================================================

def _pyi_data_sep() -> str:
    return ";" if os.name == "nt" else ":"

def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    # Capturamos salida para no ensuciar consola (solo barra)
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        raise RuntimeError("Subprocess failed rc=" + str(p.returncode) + "\n" + out)

def _resolve_icon_path() -> Path:
    # Prioridad: junto al proyecto -> ruta absoluta
    candidates = [
        _base_dir() / "Miausoft.ico",
        Path(r"E:\MiausoftSuite\Miausoft.ico"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No se encontró Miausoft.ico (proyecto o E:\\MiausoftSuite\\Miausoft.ico)")

def _resolve_font_path() -> Path:
    candidates = [
        _base_dir() / "fonts" / "Comfortaa.ttf",
        Path(r"E:\MiausoftSuite\fonts\Comfortaa.ttf"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No se encontró Comfortaa.ttf (.\\fonts o E:\\MiausoftSuite\\fonts\\Comfortaa.ttf)")

def _resolve_script_path(app_id: str) -> Path:
    p = _base_dir() / SCRIPT_FILENAMES[app_id]
    if not p.exists():
        raise FileNotFoundError("No se encontró script fuente: " + str(p))
    return p

def _build_one_app(app_id: str, *, dist_apps_dir: Path, icon_path: Path, font_path: Path, tmp_root: Path) -> None:
    # Compila un .py -> .exe onefile en dist_apps_dir sin crear basura en root
    script = _resolve_script_path(app_id)
    name = Path(EXE_FILENAMES[app_id]).stem

    add_data_font = f"{font_path}{_pyi_data_sep()}fonts"
    add_data_icon = f"{icon_path}{_pyi_data_sep()}."

    workpath = tmp_root / f"pyi_work_{name}"
    specpath = tmp_root / "specs"
    workpath.mkdir(parents=True, exist_ok=True)
    specpath.mkdir(parents=True, exist_ok=True)
    dist_apps_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--noconsole",
        "--distpath", str(dist_apps_dir),
        "--workpath", str(workpath),
        "--specpath", str(specpath),
        "--name", name,
        "--icon", str(icon_path),
        "--add-data", add_data_font,
        "--add-data", add_data_icon,
        str(script),
    ]
    _run(cmd)

def _build_installer_exe(*, icon_path: Path, font_path: Path, tmp_root: Path) -> Path:
    # Compila ESTE MISMO archivo a instalador .exe y lo deja junto a apps/
    this_script = Path(__file__).resolve()
    name = INSTALLER_EXE_NAME

    add_data_font = f"{font_path}{_pyi_data_sep()}fonts"
    add_data_icon = f"{icon_path}{_pyi_data_sep()}."

    distpath = _base_dir()
    workpath = tmp_root / "pyi_work_installer"
    specpath = tmp_root / "specs"
    workpath.mkdir(parents=True, exist_ok=True)
    specpath.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--noconsole",
        "--distpath", str(distpath),
        "--workpath", str(workpath),
        "--specpath", str(specpath),
        "--name", name,
        "--icon", str(icon_path),
        "--add-data", add_data_font,
        "--add-data", add_data_icon,
        str(this_script),
    ]
    _run(cmd)

    out = distpath / (name + ".exe")
    if not out.exists():
        raise FileNotFoundError("PyInstaller terminó sin producir: " + str(out))
    return out

def run_build_chain() -> None:
    if os.name != "nt":
        raise RuntimeError("Este build está diseñado para Windows.")

    icon_path = _resolve_icon_path()
    font_path = _resolve_font_path()

    apps_dir = _apps_payload_dir()
    apps_dir.mkdir(parents=True, exist_ok=True)

    # Limpieza mínima (solo nuestros exes)
    for exe in EXE_FILENAMES.values():
        p = apps_dir / exe
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    total_steps = len(EXE_FILENAMES) + 1  # 4 apps + instalador
    pb = _ConsoleProgress(total_steps=total_steps)
    pb.set(0)

    tmp_root = Path(tempfile.mkdtemp(prefix="miausoft_build_"))
    _log("Temp build dir: " + str(tmp_root))

    try:
        step = 0
        for app_id in (APP_ID_TXT_FULL, APP_ID_TXT_CHAPTERS, APP_ID_SPLIT_MERGE, APP_ID_REPLACE_CHARS):
            _log("Building app: " + app_id)
            _build_one_app(app_id, dist_apps_dir=apps_dir, icon_path=icon_path, font_path=font_path, tmp_root=tmp_root)
            step += 1
            pb.set(step)

        _log("Building installer EXE")
        _build_installer_exe(icon_path=icon_path, font_path=font_path, tmp_root=tmp_root)
        pb.finish()

    finally:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except OSError:
            pass

# =============================================================================
# Installer: rutas + utilidades
# =============================================================================

def _resource_path(rel: str) -> Path:
    # Si exe es onefile, los datos van a sys._MEIPASS
    if _is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / rel
    return _base_dir() / rel

def _sendto_dir() -> Path:
    # %APPDATA%\Microsoft\Windows\SendTo
    appdata = os.environ.get("APPDATA", "")
    return Path(appdata) / "Microsoft" / "Windows" / "SendTo"

def _startmenu_dir() -> Path:
    # %APPDATA%\Microsoft\Windows\Start Menu\Programs
    appdata = os.environ.get("APPDATA", "")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"

def _default_install_dir() -> Path:
    # Preferencia: Program Files (solo best-effort; puede fallar sin admin)
    pf = os.environ.get("ProgramFiles", "")
    if pf:
        return Path(pf) / "MiausoftSuite"
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        return Path(local) / "Programs" / "MiausoftSuite"
    return Path.home() / "AppData" / "Local" / "Programs" / "MiausoftSuite"


def _program_files_roots() -> List[Path]:
    roots: List[Path] = []
    for envk in ("ProgramFiles", "ProgramFiles(x86)"):
        v = os.environ.get(envk, "")
        if v:
            try:
                roots.append(Path(v).resolve())
            except Exception:
                roots.append(Path(v))
    # Dedup manteniendo orden
    out: List[Path] = []
    seen: set[str] = set()
    for r in roots:
        k = str(r).lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _normalize_secondary_target_dir(tgt: Path) -> Path:
    """Si el usuario elige la raíz de Program Files, forzar una carpeta propia.

    - Ej: C:\Program Files  -> C:\Program Files\MiausoftSuite
    - Para cualquier otra ruta: se respeta.
    """
    try:
        tr = tgt.resolve()
    except Exception:
        tr = tgt
    for pf in _program_files_roots():
        try:
            if tr == pf:
                return pf / "MiausoftSuite"
        except Exception:
            continue
    return tgt

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _copy_file_atomic(src: Path, dst: Path) -> None:
    # Copia a tmp y luego reemplaza: evita ejecutable incompleto
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    shutil.copy2(src, tmp)
    try:
        if dst.exists():
            dst.unlink()
    except OSError:
        pass
    tmp.replace(dst)

def _payload_validate(payload_dir: Path) -> None:
    if not payload_dir.exists():
        raise FileNotFoundError("No existe la carpeta .\\apps junto al instalador: " + str(payload_dir))
    missing = []
    for exe in EXE_FILENAMES.values():
        if not (payload_dir / exe).exists():
            missing.append(exe)
    if missing:
        raise FileNotFoundError("Faltan EXEs en .\\apps: " + ", ".join(missing))

# =============================================================================
# Shell integrations (HKCU) + shortcuts
# =============================================================================

def _run_powershell(script: str) -> None:
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
    _run(cmd)
def _icon_location_spec(icon: Path) -> str:
    """Devuelve una especificación de icono robusta para Windows.

    - Para .ico/.cur: usar la ruta directa (sin índice) porque algunos entornos muestran "hoja en blanco" con ",0".
    - Para .exe/.dll: usar "ruta,0".
    """
    s = str(icon)
    if "," in s:
        return s
    try:
        ext = Path(s).suffix.lower()
    except Exception:
        ext = ""
    if ext in (".ico", ".cur"):
        return s
    return s + ",0"


def _powershell_create_shortcut(lnk_path: Path, target_path: Path, *, working_dir: Optional[Path] = None, icon_path: Optional[Path] = None) -> None:
    wd = str(working_dir if working_dir else target_path.parent)
    ic = _icon_location_spec(icon_path) if icon_path else ""
    lnk = str(lnk_path)
    tgt = str(target_path)

    ps = [
        '$WshShell = New-Object -ComObject WScript.Shell',
        f'$Shortcut = $WshShell.CreateShortcut("{lnk}")',
        f'$Shortcut.TargetPath = "{tgt}"',
        f'$Shortcut.WorkingDirectory = "{wd}"',
    ]
    if ic:
        ps.append(f'$Shortcut.IconLocation = "{ic}"')
    ps.append('$Shortcut.Save()')
    _run_powershell("\n".join(ps))

def _refresh_shell() -> None:
    # SHChangeNotify(ASSOCCHANGED) refresca asociaciones/context menu
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        fn = getattr(shell32, "SHChangeNotify", None)
        if fn is None:
            return
        # void SHChangeNotify(LONG wEventId, UINT uFlags, LPCVOID dwItem1, LPCVOID dwItem2);
        fn.argtypes = [ctypes.c_long, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = None
        SHCNE_ASSOCCHANGED = 0x08000000
        SHCNF_IDLIST = 0x0000
        fn(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)

        # Refresco de iconos (Start/Explorer) best-effort
        try:
            _run(["ie4uinit.exe", "-show"])
        except Exception:
            pass
    except OSError:
        pass
    except Exception:
        # Best-effort: nunca rompe
        pass

def _install_start_menu_shortcuts(installed: Dict[str, Path], *, icon_path: Optional[Path]) -> None:
    folder = _startmenu_dir() / START_MENU_FOLDER
    _ensure_dir(folder)
    for app_id, exe_path in installed.items():
        name = DISPLAY_NAMES.get(app_id, exe_path.stem)
        lnk = folder / (name + ".lnk")
        _powershell_create_shortcut(lnk, exe_path, working_dir=exe_path.parent, icon_path=exe_path)

def _install_sendto_shortcuts(installed: Dict[str, Path], *, icon_path: Optional[Path]) -> None:
    # Además de copiar EXEs a SendTo\MiausoftSuite\, creamos .lnk en SendTo\ (root) para fácil acceso
    root_sendto = _sendto_dir()
    _ensure_dir(root_sendto)
    for app_id, exe_path in installed.items():
        name = DISPLAY_NAMES.get(app_id, exe_path.stem)
        lnk = root_sendto / (name + ".lnk")
        _powershell_create_shortcut(lnk, exe_path, working_dir=exe_path.parent, icon_path=exe_path)

def _install_preset_association(preset_exe: Path, *, icon_path: Optional[Path]) -> None:
    import winreg  # type: ignore
    icon = _icon_location_spec(icon_path or preset_exe)
    # HKCU\Software\Classes\.miau -> ProgID
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\\" + PRESET_EXT) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, PRESET_PROGID)

    base = r"Software\Classes\\" + PRESET_PROGID
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "MiausoftSuite Preset")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + r"\DefaultIcon") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, icon)
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + r"\shell\open\command") as k:
        cmd = f"\"{preset_exe}\" \"%1\""
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, cmd)

def _install_context_menu_for_extension(ext: str, installed: Dict[str, Path], *, icon_path: Optional[Path]) -> None:
    # Submenú "MiausoftSuite" en SystemFileAssociations\<ext>
    import winreg  # type: ignore

    root_key = r"Software\Classes\SystemFileAssociations\\" + ext + r"\shell\\" + CONTEXT_MENU_ROOT_NAME
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, root_key) as k:
        winreg.SetValueEx(k, "MUIVerb", 0, winreg.REG_SZ, CONTEXT_MENU_ROOT_NAME)
        winreg.SetValueEx(k, "SubCommands", 0, winreg.REG_SZ, "")
        winreg.SetValueEx(
            k,
            "ExtendedSubCommandsKey",
            0,
            winreg.REG_SZ,
            r"" + CONTEXT_MENU_ROOT_NAME + r"\shell\\" + CONTEXT_MENU_ROOT_NAME,
        )
        icon_for_menu = icon_path
        if (not icon_for_menu) and installed:
            try:
                icon_for_menu = next(iter(installed.values()))
            except Exception:
                icon_for_menu = None
        if icon_for_menu:
            winreg.SetValueEx(k, "Icon", 0, winreg.REG_SZ, _icon_location_spec(icon_for_menu))

    verbs_root = r"Software\Classes\\" + CONTEXT_MENU_ROOT_NAME + r"\shell\\" + CONTEXT_MENU_ROOT_NAME + r"\shell"
    for app_id, exe_path in installed.items():
        verb = app_id
        verb_key = verbs_root + "\\" + verb
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, verb_key) as k:
            winreg.SetValueEx(k, "MUIVerb", 0, winreg.REG_SZ, DISPLAY_NAMES.get(app_id, exe_path.stem))
            # Player permite multi-selección (Explorer puede imponer límites en algunos casos)
            winreg.SetValueEx(k, "MultiSelectModel", 0, winreg.REG_SZ, "Player")
            icon_for_item = icon_path or exe_path
            winreg.SetValueEx(k, "Icon", 0, winreg.REG_SZ, _icon_location_spec(icon_for_item))
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, verb_key + r"\command") as k:
            cmd = f"\"{exe_path}\" %*"
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, cmd)

def _integrate_shell(installed: Dict[str, Path], *, icon_path: Optional[Path]) -> None:
    # Inicio + SendTo (lnk) + Context menu + preset .miau
    try:
        _install_start_menu_shortcuts(installed, icon_path=icon_path)
    except Exception:
        _log_exc("StartMenu shortcuts failed")

    for ext in TARGET_EXTS_CONTEXT:
        try:
            _install_context_menu_for_extension(ext, installed, icon_path=icon_path)
        except Exception:
            _log_exc("Context menu failed ext=" + ext)

    try:
        preset_exe = installed.get(PRESET_TARGET_APP_ID)
        if preset_exe:
            _install_preset_association(preset_exe, icon_path=icon_path)
    except Exception:
        _log_exc("Preset association failed")

    _refresh_shell()

# =============================================================================
# Instalación (SendTo primero, luego best-effort destino)
# =============================================================================

def _copy_payload_exes(payload_dir: Path, target_dir: Path, *, dst_name_map: Optional[Dict[str, str]] = None, status_cb=None, progress_cb=None) -> Dict[str, Path]:
    _ensure_dir(target_dir)
    installed: Dict[str, Path] = {}
    app_ids = [APP_ID_TXT_FULL, APP_ID_TXT_CHAPTERS, APP_ID_SPLIT_MERGE, APP_ID_REPLACE_CHARS]
    total = max(1, len(app_ids))

    for idx, app_id in enumerate(app_ids, start=1):
        src = payload_dir / EXE_FILENAMES[app_id]
        dst_name = (dst_name_map or {}).get(app_id, EXE_FILENAMES[app_id])
        dst = target_dir / dst_name

        if status_cb:
            status_cb("Copiando: " + DISPLAY_NAMES.get(app_id, app_id))

        _copy_file_atomic(src, dst)
        installed[app_id] = dst

        if progress_cb:
            progress_cb(idx / float(total))

    return installed

def perform_install(sendto_priority: bool, target_dir: Path, *, status_cb=None, progress_cb=None) -> Tuple[Path, Dict[str, Path]]:
    payload = _apps_payload_dir()
    _payload_validate(payload)

    # NOTA CRÍTICA:
    # - En onefile, _resource_path("Miausoft.ico") vive en _MEIPASS (temporal).
    # - Por eso copiamos el icono a una ruta persistente (SendTo\Miausoft.ico)
    #   y usamos esa ruta para shortcuts/registro.
    icon_tmp = _resource_path("Miausoft.ico")


    # 1) PRIMER PASO (siempre): SendTo (sin subcarpetas)
    sendto_root = _sendto_dir()
    _ensure_dir(sendto_root)

    # Limpieza de versiones previas que instalaban en subcarpeta SendTo\MiausoftSuite
    legacy_dir = sendto_root / START_MENU_FOLDER
    if legacy_dir.exists() and legacy_dir.is_dir():
        try:
            shutil.rmtree(legacy_dir, ignore_errors=True)
        except Exception:
            pass

    # Remover nombres antiguos en SendTo (best-effort) para evitar duplicados
    for _app_id in [APP_ID_TXT_FULL, APP_ID_TXT_CHAPTERS, APP_ID_SPLIT_MERGE, APP_ID_REPLACE_CHARS]:
        _lnk = ((DISPLAY_NAMES.get(_app_id) or "").strip() + ".lnk")
        for _name in (EXE_FILENAMES.get(_app_id), SENDTO_EXE_NAMES.get(_app_id), _lnk):
            if not _name:
                continue
            try:
                (sendto_root / _name).unlink()
            except Exception:
                pass

    # Icono persistente (best-effort)
    # - Preferimos copiar el icono embebido (MEIPASS) a SendTo\Miausoft.ico.
    # - Si no existe, intentamos usar E:\MiausoftSuite\Miausoft.ico (si está disponible).
    icon_path: Optional[Path] = None
    try:
        if icon_tmp.exists():
            icon_persist = sendto_root / "Miausoft.ico"
            _copy_file_atomic(icon_tmp, icon_persist)
            icon_path = icon_persist
        else:
            ext_ico = Path(r"E:\MiausoftSuite\Miausoft.ico")
            if ext_ico.exists():
                icon_path = ext_ico
    except Exception:
        icon_path = None

    if status_cb:
        status_cb("Instalando (1/2): SendTo (prioridad)...")

    def prog1(p: float) -> None:
        if progress_cb:
            # 0..0.80 reservado para SendTo (copia) + integración
            progress_cb(min(0.80, max(0.0, p) * 0.80))

    installed_sendto = _copy_payload_exes(payload, sendto_root, dst_name_map=SENDTO_EXE_NAMES, status_cb=status_cb, progress_cb=prog1)

    if status_cb:
        status_cb("Instalando (1/2): integrando Inicio, SendTo, menú contextual y .miau...")

    try:
        _integrate_shell(installed_sendto, icon_path=icon_path)
    except Exception:
        # No fatal: la copia principal SendTo ya existe
        _log_exc("Shell integration failed (non-fatal)")

    if progress_cb:
        progress_cb(0.80)

    # 2) SEGUNDO PASO (best-effort): copia adicional al destino elegido
    # Si falla: se ignora, sin colapso.
    try:
        tgt = Path(target_dir)
    except Exception:
        tgt = _default_install_dir()
    tgt = _normalize_secondary_target_dir(tgt)

    try:
        if status_cb:
            status_cb("Instalando (2/2): copia adicional (best-effort)...")

        def prog2(p: float) -> None:
            if progress_cb:
                progress_cb(0.80 + min(1.0, max(0.0, p)) * 0.20)

        # Copia adicional (best-effort)
        installed_tgt = _copy_payload_exes(payload, tgt, status_cb=status_cb, progress_cb=prog2)

        # Si esta copia existe, apuntar el Inicio a esta ruta (Program Files / destino elegido).
        # Esto no afecta SendTo (que sigue siendo la copia prioritaria para menú contextual/registro).
        try:
            _install_start_menu_shortcuts(installed_tgt, icon_path=None)
            _refresh_shell()
        except Exception:
            _log_exc('StartMenu repoint to target failed')

        # Copiar ícono (opcional)
        if icon_path:
            try:
                _copy_file_atomic(icon_path, tgt / "Miausoft.ico")
            except Exception:
                pass

    except Exception:
        _log_exc("Secondary copy failed (ignored)")
        # Mantener progreso en 100% de todos modos
        if progress_cb:
            progress_cb(1.0)

    return sendto_root, installed_sendto

# =============================================================================
# UI Installer (Tkinter): robusto (no colapsa)
# =============================================================================

class InstallerApp:
    """
    Instalador UI (Tk):
    - Misma estética que ProgressDialog de Miausoft (config.py)
    - Sin auto-resize tardío (evita "ventana en blanco" o alargado)
    - Toda la UI se actualiza SOLO desde el hilo principal (after + cola)
    """

    def __init__(self) -> None:
        self._q: "Queue[Tuple[str, object]]" = Queue()
        self._install_thread: Optional[threading.Thread] = None
        self._installing = False

        # Tk solo en modo installer
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import filedialog, messagebox

        self.tk = tk
        self.tkfont = tkfont
        self.filedialog = filedialog
        self.messagebox = messagebox

        self.root = tk.Tk()

        # Tema y geometría central (config.py), usando pantalla REAL
        self._apply_theme_and_geometry()

        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # UI (ProgressDialog + selector de ruta + botones)
        self._build_ui()

        # Poll de mensajes (thread -> UI)
        self.root.after(40, self._poll_queue)

    def _apply_theme_and_geometry(self) -> None:
        try:
            import config  # type: ignore

            try:
                config.ensure_font_registered("Comfortaa")
            except Exception:
                pass

            # Geometría φ-relativa del installer, basada en pantalla REAL
            try:
                sw = int(self.root.winfo_screenwidth() or 0)
                sh = int(self.root.winfo_screenheight() or 0)
            except Exception:
                sw, sh = 0, 0

            try:
                w, h = config.window_geometry("installer", screen_w=sw or None, screen_h=sh or None)
            except Exception:
                w, h = (700, 340)

            # Centrado
            try:
                x = max(0, int((sw - w) // 2)) if sw else 0
                y = max(0, int((sh - h) // 2)) if sh else 0
                self.root.geometry(f"{int(w)}x{int(h)}+{x}+{y}")
            except Exception:
                self.root.geometry(f"{int(w)}x{int(h)}")

            self.root.resizable(False, False)

            # Title
            try:
                self.root.title(str((config.get_config("installer").get("window") or {}).get("title", "Instalador de MiausoftSuite")))
            except Exception:
                self.root.title("Instalador de MiausoftSuite")

            # Theme (NO positional args; apply_theme es keyword-only)
            try:
                config.apply_theme(self.root, app="installer", title="Instalador de MiausoftSuite", icon="Miausoft.ico")
            except Exception:
                pass

            # Icono de ventana (taskbar/titlebar): best-effort
            ico = _resource_path("Miausoft.ico")
            if ico.exists():
                try:
                    self.root.iconbitmap(str(ico))
                except Exception:
                    pass
                # iconphoto suele ayudar en Windows cuando el icono del EXE no se refleja en la ventana
                try:
                    from PIL import Image, ImageTk  # type: ignore
                    img = Image.open(ico)
                    self._iconphoto = ImageTk.PhotoImage(img)
                    self.root.iconphoto(True, self._iconphoto)
                except Exception:
                    pass
        except Exception:
            # Fallback
            self.root.geometry("700x340")
            self.root.resizable(False, False)

    def _build_ui(self) -> None:
        tk = self.tk
        tkfont = self.tkfont

        # Forzar a que ProgressDialog use tokens del installer (misma estética)
        try:
            import config  # type: ignore
            config.PROGRESS_UI_CONFIG = config.get_config("installer")
        except Exception:
            pass

        # Crear ProgressDialog (misma plantilla de ventanas de progreso)
        try:
            import config  # type: ignore
            ProgressDialog = config.ProgressDialog
            cfg = config.get_config("installer")
            colors = (cfg.get("colors") or {})
            bg = str(colors.get("app_bg", "#f3f3f3"))
            box_bg = str(colors.get("box_bg", "#ffffff"))
            text_fg = str(colors.get("text", "#2a2a2a"))
            btn_face = str(colors.get("btn_face", "#fafafa"))
            btn_hover = str(colors.get("btn_face_hover", "#f0f0f0"))
            btn_down = str(colors.get("btn_face_down", "#f0f0f0"))
        except Exception:
            cfg = {}
            bg, box_bg, text_fg = "#f3f3f3", "#ffffff", "#2a2a2a"
            btn_face, btn_hover, btn_down = "#fafafa", "#f0f0f0", "#f0f0f0"
            ProgressDialog = None  # type: ignore

        self.root.configure(bg=bg)

        if ProgressDialog is None:
            # Si config no está disponible, no colapsar (fallback mínimo)
            class _Stub:
                def set_subtitle(self, _t: str) -> None:
                    return
                def set_progress(self, _p: float) -> None:
                    return
            self.pdlg = _Stub()

            frame = tk.Frame(self.root, bg=bg, bd=0, highlightthickness=0)
            frame.place(x=0, y=0, relwidth=1, relheight=1)
            lbl = tk.Label(frame, text=INSTALLER_TITLE_LABEL, bg=bg, fg=text_fg, anchor="w")
            lbl.pack(fill="x", padx=16, pady=16)
            return

        self.pdlg = ProgressDialog(self.root, icon_search_dir=str(_base_dir()))
        self.pdlg.show()
        self.pdlg.set_title(INSTALLER_TITLE_LABEL)
        self.pdlg.set_subtitle(INSTALLER_SUBTITLE_LABEL)
        self.pdlg.set_progress(0.0)

        # Asegurar layout 1 vez y congelar coordenadas (ventana no es resizable)
        # Coordenadas de la barra (donde irá el Entry antes de instalar)
        # Evitar update_idletasks() / _apply_metrics() aquí: en Tk 9 (Python 3.14) puede provocar crash nativo (0xC000041D).
        # Medición best-effort; el overlay se re-sincroniza más abajo.
        try:
            bar_x = int(self.pdlg.progress.winfo_x())
            bar_y = int(self.pdlg.progress.winfo_y())
            bar_w = int(self.pdlg.progress.winfo_width())
            bar_h = int(self.pdlg.progress.winfo_height())
            if bar_w <= 0 or bar_h <= 0:
                raise ValueError('bar geom not ready')
        except Exception:
            try:
                ww = int(self.root.winfo_width() or cfg['window']['min_width'])
                hh = int(self.root.winfo_height() or cfg['window']['min_height'])
                lay = cfg.get('layout') or {}
                bar_w = int(ww * float(lay.get('bar_w_rel_w', 1/1.6180339887)))
                bar_h = int(max(int(lay.get('bar_min_h_px', 24)), int(hh * float(lay.get('bar_h_rel_h', 1/12)))))
            except Exception:
                bar_w, bar_h = 360, 26
            bar_x = 12
            bar_y = 12
        self._bar_geom = (bar_x, bar_y, bar_w, bar_h)

        # Mantener la barra colocada para evitar que un relayout interno la "re-aparezca";
        # el selector de ruta queda por encima.
        try:
            self.pdlg.progress.lower()
        except Exception:
            pass

        # Fuente para el Entry: misma escala que "row" del config (Comfortaa)
        try:
            import config  # type: ignore
            sw = int(self.root.winfo_screenwidth() or 0)
            sh = int(self.root.winfo_screenheight() or 0)
            tokens = config.design_tokens("installer", screen_w=sw or None, screen_h=sh or None)
            fam, px, bold, italic = (tokens.get("fonts_px") or {}).get("row") or ("Comfortaa", 11, False, False)
            config.ensure_font_registered(str(fam))

            # El Entry tiene altura fija (bar_h). Limitar la fuente para que nunca
            # quede más alta que la caja (en Windows, la métrica real puede exceder el "size").
            _cfg_i = config.get_config("installer")
            _min_px = int((_cfg_i.get("fonts") or {}).get("min_px", 8))
            _pad_y = max(1, int(bar_h * 0.22))
            _avail = max(8, int(bar_h - (2 * _pad_y)))
            _cap_px = max(_min_px, int(_avail * 0.80))
            _safe_px = max(6, min(int(px), int(_cap_px)))

            entry_font = config.font_tuple(int(_safe_px), bold=bool(bold), italic=bool(italic), family=str(fam), min_px=_min_px)

            # Botones
            btn_px = config.shared_button_font_px(self.root.winfo_width(), self.root.winfo_height(), _cfg_i)
        except Exception:
            entry_font = ("Segoe UI", 10)
            btn_px = 10

        # Caja de ruta (sin borde)
        self.path_var = tk.StringVar(value=str(_default_install_dir()))
        self.path_box = tk.Frame(self.pdlg.info_box, bg=box_bg, bd=0, highlightthickness=0)
        self.path_entry = tk.Entry(
            self.path_box,
            textvariable=self.path_var,
            bd=0,
            highlightthickness=0,
            relief="flat",
            bg=box_bg,
            fg=text_fg,
            insertbackground=text_fg,
            font=entry_font,
        )
        self.path_entry.pack(fill="both", expand=True, padx=(10, 10), pady=max(1, int(bar_h * 0.22)))

        # Botones tipo "píldora" (misma familia Miausoft)
        class _PillButton(tk.Canvas):
            def __init__(self, parent, text: str, command, *, face: str, hover: str, down: str, fg: str, font, radius: int = 9999):
                super().__init__(parent, bd=0, highlightthickness=0, relief="flat", bg=bg)
                self._text = text
                self._cmd = command
                self._face = face
                self._hover = hover
                self._down = down
                self._fg = fg
                self._font = font
                self._radius = max(6, int(radius))
                self._state = "face"
                self._enabled = True
                self.bind("<Configure>", lambda e: self._redraw())
                self.bind("<Enter>", lambda e: self._set_state("hover"))
                self.bind("<Leave>", lambda e: self._set_state("face"))
                self.bind("<ButtonPress-1>", lambda e: self._set_state("down"))
                self.bind("<ButtonRelease-1>", self._on_release)
                self.configure(cursor="hand2")

            def set_enabled(self, enabled: bool) -> None:
                self._enabled = bool(enabled)
                try:
                    self.configure(cursor="hand2" if self._enabled else "arrow")
                except Exception:
                    pass
                self._redraw()

            def _set_state(self, st: str) -> None:
                if not self._enabled:
                    st = "face"
                self._state = st
                self._redraw()

            def _on_release(self, _e) -> None:
                if not self._enabled:
                    return
                # Si el mouse sigue dentro, ejecutar
                try:
                    x = self.winfo_pointerx() - self.winfo_rootx()
                    y = self.winfo_pointery() - self.winfo_rooty()
                    inside = 0 <= x <= self.winfo_width() and 0 <= y <= self.winfo_height()
                except Exception:
                    inside = True
                self._set_state("hover" if inside else "face")
                if inside:
                    try:
                        self._cmd()
                    except Exception:
                        _log_exc("Button command failed")

            def _redraw(self) -> None:
                self.delete("all")
                w = max(1, int(self.winfo_width()))
                h = max(1, int(self.winfo_height()))
                r = int(min(h // 2, w // 2, self._radius))
                fill = self._face if self._state == "face" else (self._hover if self._state == "hover" else self._down)

                # Píldora
                self.create_rectangle(r, 0, w - r, h, fill=fill, outline="")
                self.create_oval(0, 0, 2 * r, h, fill=fill, outline="")
                self.create_oval(w - 2 * r, 0, w, h, fill=fill, outline="")

                # Texto
                self.create_text(w // 2, h // 2, text=self._text, fill=self._fg, font=self._font)

        # Botón browse pequeño "..."
        browse_w = max(34, int(bar_h * 1.10))
        gap = max(6, int(bar_h * 0.22))
        entry_w = max(80, bar_w - browse_w - gap)

        self.btn_browse = _PillButton(
            self.pdlg.info_box,
            "...",
            self._on_browse,
            face=btn_face,
            hover=btn_hover,
            down=btn_down,
            fg=text_fg,
            font=("Comfortaa", max(9, int(btn_px))),
            radius=9999,
        )

        # Place selector (en el lugar exacto de la barra)
        self.path_box.place(x=bar_x, y=bar_y, width=entry_w, height=bar_h)
        self.btn_browse.place(x=bar_x + entry_w + gap, y=bar_y, width=browse_w, height=bar_h)

        # Botones inferiores (Cancelar / Instalar)
        try:
            import config  # type: ignore
            bw, bh = config.shared_button_box_px(self.root.winfo_width(), self.root.winfo_height(), config.get_config("installer"))
            btn_font_px = config.shared_button_font_px(self.root.winfo_width(), self.root.winfo_height(), config.get_config("installer"))
            fam = (config.get_config("installer").get("fonts") or {}).get("family", "Comfortaa")
            config.ensure_font_registered(str(fam))
            btn_font = config.font_tuple(int(btn_font_px), bold=False, italic=False, family=str(fam))
            lay = (config.get_config("installer").get("layout") or {})
            pad = max(10, int(self.pdlg.info_box.winfo_height() * float(lay.get("boxes_pad_rel_h", 1/8))))
        except Exception:
            bw, bh = 160, 40
            btn_font = ("Comfortaa", 10)
            pad = 12

        info_h = int(self.pdlg.info_box.winfo_height())
        y_btn = max(0, info_h - pad - int(bh))

        self.btn_cancel = _PillButton(
            self.pdlg.info_box,
            "Cancelar",
            self._on_cancel,
            face=btn_face,
            hover=btn_hover,
            down=btn_down,
            fg=text_fg,
            font=btn_font,
            radius=9999,
        )
        self.btn_install = _PillButton(
            self.pdlg.info_box,
            "Instalar",
            self._on_install_clicked,
            face=btn_face,
            hover=btn_hover,
            down=btn_down,
            fg=text_fg,
            font=btn_font,
            radius=9999,
        )

        self.btn_cancel.place(x=bar_x, y=y_btn, width=int(bw), height=int(bh))
        self.btn_install.place(x=max(bar_x, bar_x + bar_w - int(bw)), y=y_btn, width=int(bw), height=int(bh))

        # Sincronizar overlays (Entry/Browse/Buttons) con el layout real del ProgressDialog.
        # Esto evita desalineaciones si Tk recalcula métricas (DPI/fuentes) durante el arranque.
        self._overlay_pending = False

        def _schedule_overlay(_e=None) -> None:
            if getattr(self, "_installing", False):
                return
            if getattr(self, "_overlay_pending", False):
                return
            self._overlay_pending = True

            def _run() -> None:
                self._overlay_pending = False
                if getattr(self, "_installing", False):
                    return
                try:
                    bx = int(self.pdlg.progress.winfo_x())
                    by = int(self.pdlg.progress.winfo_y())
                    bw0 = int(self.pdlg.progress.winfo_width())
                    bh0 = int(self.pdlg.progress.winfo_height())
                    if bw0 <= 0 or bh0 <= 0:
                        return

                    browse_w0 = max(34, int(bh0 * 1.10))
                    gap0 = max(6, int(bh0 * 0.22))
                    entry_w0 = max(80, bw0 - browse_w0 - gap0)

                    # Selector (en el área de la barra)
                    try:
                        self.path_box.place(x=bx, y=by, width=entry_w0, height=bh0)
                        self.btn_browse.place(x=bx + entry_w0 + gap0, y=by, width=browse_w0, height=bh0)
                        self.path_box.lift()
                        self.btn_browse.lift()
                    except Exception:
                        pass

                    # Botones inferiores (alineados al ancho de la barra)
                    try:
                        import config  # type: ignore
                        _cfg = config.get_config("installer")
                        bb_w, bb_h = config.shared_button_box_px(self.root.winfo_width(), self.root.winfo_height(), _cfg)
                        info_h0 = int(self.pdlg.info_box.winfo_height())
                        lay0 = (_cfg.get("layout") or {})
                        pad0 = max(10, int(info_h0 * float(lay0.get("boxes_pad_rel_h", 1/8))))
                        y0 = max(0, info_h0 - pad0 - int(bb_h))
                        self.btn_cancel.place(x=bx, y=y0, width=int(bb_w), height=int(bb_h))
                        self.btn_install.place(x=max(bx, bx + bw0 - int(bb_w)), y=y0, width=int(bb_w), height=int(bb_h))
                    except Exception:
                        pass
                except Exception:
                    _log_exc("Overlay sync failed")

            try:
                self.root.after_idle(_run)
            except Exception:
                self._overlay_pending = False

        try:
            self.pdlg.info_box.bind("<Configure>", _schedule_overlay, add="+")
            self.root.after_idle(_schedule_overlay)
            self.root.after(150, _schedule_overlay)  # una segunda pasada por DPI/fuentes
        except Exception:
            pass

    def _on_browse(self) -> None:
        try:
            d = self.filedialog.askdirectory(initialdir=self.path_var.get() or str(_default_install_dir()))
            if d:
                self.path_var.set(d)
        except Exception:
            _log_exc("Browse failed")

    def _on_cancel(self) -> None:
        if self._installing:
            # No matamos hilo (evita corrupción); solo dejamos terminar el paso actual
            try:
                self.pdlg.set_subtitle("Cancelación solicitada. Finalizando paso actual...")
            except Exception:
                pass
            return
        try:
            self.root.destroy()
        except Exception:
            pass

    def _on_install_clicked(self) -> None:
        if self._installing:
            return
        self._installing = True

        # UI: ocultar selector y mostrar barra de progreso (misma posición)
        try:
            self.path_box.place_forget()
        except Exception:
            pass
        try:
            self.btn_browse.place_forget()
        except Exception:
            pass

        try:
            # La barra ya está colocada por ProgressDialog; solo la traemos al frente.
            self.pdlg.progress.lift()
            self.pdlg.set_progress(0.0)
        except Exception:
            pass

        try:
            self.btn_install.set_enabled(False)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Hilo de instalación
        target_dir = Path(self.path_var.get() or str(_default_install_dir()))
        self._install_thread = threading.Thread(target=self._install_worker, args=(target_dir,), daemon=True)
        self._install_thread.start()

    def _install_worker(self, target_dir: Path) -> None:
        def status_cb(msg: str) -> None:
            self._q.put(("status", msg))

        def progress_cb(p: float) -> None:
            self._q.put(("progress", float(p)))

        try:
            status_cb("Validando paquete...")
            perform_install(True, target_dir, status_cb=status_cb, progress_cb=progress_cb)
            self._q.put(("done", None))
        except Exception:
            _log_exc("Installer worker failed")
            self._q.put(("error", _log_dump_text()))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    try:
                        # El subtítulo es el mismo canal de estado que usa el ProgressDialog en apps
                        self.pdlg.set_subtitle(str(payload))
                    except Exception:
                        pass
                elif kind == "progress":
                    try:
                        self.pdlg.set_progress(float(payload))
                    except Exception:
                        pass
                elif kind == "done":
                    try:
                        self.pdlg.set_progress(1.0)
                        self.pdlg.set_subtitle("Instalación completada.")
                    except Exception:
                        pass
                    # Cerrar 2s después (requisito)
                    try:
                        import config  # type: ignore
                        delay = int((config.get_config("installer").get("app") or {}).get("close_delay_ms", 2000))
                    except Exception:
                        delay = 2000
                    self.root.after(max(0, delay), self.root.destroy)
                elif kind == "error":
                    # Ventana de error está permitida; NO colapsar
                    try:
                        self.messagebox.showerror("Error del instalador", str(payload))
                    except Exception:
                        pass
                    self._installing = False
                    try:
                        self.btn_install.set_enabled(True)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Empty:
            pass
        finally:
            try:
                self.root.after(60, self._poll_queue)
            except Exception:
                pass

    def run(self) -> None:
        self.root.mainloop()

# =============================================================================
# Args y entrypoint
# =============================================================================

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--build", action="store_true", help="Forzar modo build (solo .py)")
    p.add_argument("--run-installer", action="store_true", help="Ejecutar UI installer desde .py (debug)")
    p.add_argument("--help", action="store_true")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)

    if args.help:
        print("build_installer.py\n  - Ejecuta sin args para build (en .py) o installer (en .exe).")
        return 0

    try:
        # Si es ejecutable frozen => instalador UI
        if _is_frozen():
            InstallerApp().run()
            return 0

        # Si es .py:
        if args.run_installer:
            InstallerApp().run()
            return 0

        # Por defecto: BUILD encadenado
        run_build_chain()
        return 0

    except Exception:
        # Build: si falla, al menos deja un mensaje minimal + log en apps\_logs
        _log_exc("FATAL main")
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.stderr.write("FATAL: fallo en build/installer. Log: " + str(_log_file_path()) + "\n")
        sys.stderr.flush()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
