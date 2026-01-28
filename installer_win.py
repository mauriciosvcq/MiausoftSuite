# installer_win.py (robusto, desde 0)
# MiausoftSuite 3.0 — Builder de instalador para Windows (Python 3.14+)
#
# Anti-colapso:
# - Rechaza PyInstaller viejo (>= 6.15.0 por Python 3.14).
# - Runtime del setup NO importa tkinter al inicio; cae a modo headless si Tk falla.
# - Logging en %TEMP% y MessageBox nativo.
# - onefile: fija --runtime-tmpdir a carpeta estable.
# - SendTo: .lnk por PowerShell; si falla -> .cmd.
# - Menú contextual: best-effort con SubCommands + CommandStore; si falla, NO aborta.
#
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


SUITE_NAME = "MiausoftSuite"
SUITE_VERSION = "3.0"
STABLE_TAG = "3.0 (2026.1)"

MIN_PYINSTALLER = (6, 15, 0)

APP_EXE_NAMES: Dict[str, str] = {
    "txt_full": "MiausoftSuite_ConversorTXT_Completo.exe",
    "txt_chapters": "MiausoftSuite_ConversorTXT_PorCapitulos.exe",
    "split_merge": "MiausoftSuite_DivisorFusionador.exe",
    "replace_chars": "MiausoftSuite_ReemplazadorCaracteres.exe",
    "dispatcher": "MiausoftSuite_Dispatcher.exe",
}

DISPLAY_NAMES: Dict[str, str] = {
    "txt_full": "MiausoftSuite Conversor a txt (completo)",
    "txt_chapters": "MiausoftSuite Conversor a txt (por capítulos)",
    "split_merge": "MiausoftSuite Divisor y Fusionador",
    "replace_chars": "MiausoftSuite Reemplazador de caracteres",
}


def _die(msg: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _is_windows() -> bool:
    return os.name == "nt"


def _parse_version_tuple(s: str) -> Tuple[int, int, int]:
    parts = []
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
            f"PyInstaller {v} es demasiado viejo para Python {sys.version_info.major}.{sys.version_info.minor}.\n"
            f"Necesitas PyInstaller >= {MIN_PYINSTALLER[0]}.{MIN_PYINSTALLER[1]}.{MIN_PYINSTALLER[2]}.\n"
            "Actualiza con:\n"
            "  python -m pip install -U pyinstaller"
        )


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    print("\n>> " + " ".join(map(str, cmd)))
    p = subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _load_module_from_path(py_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _maybe(project_dir: Path, name: str) -> Optional[Path]:
    p = project_dir / name
    return p if p.exists() else None


def _render_setup_runtime(cfg_base: dict, cfg_progress: dict, cfg_installer: dict) -> str:
    # Template grande: usamos marcadores y replace para evitar choques con llaves.
    tpl = """
# setup_runtime.py (generado) — stdlib-only
from __future__ import annotations

import base64
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import winreg  # type: ignore
except Exception:
    winreg = None  # type: ignore

SUITE_NAME = "MiausoftSuite"
SUITE_VERSION = "3.0"
STABLE_TAG = "3.0 (2026.1)"

DISPLAY_NAMES: Dict[str, str] = json.loads("__DISPLAY_NAMES_JSON__")
APP_EXE_NAMES: Dict[str, str] = json.loads("__APP_EXE_NAMES_JSON__")

CFG_BASE: Dict[str, Any] = json.loads("__CFG_BASE_JSON__")
CFG_PROGRESS: Dict[str, Any] = json.loads("__CFG_PROGRESS_JSON__")
CFG_INSTALLER: Dict[str, Any] = json.loads("__CFG_INSTALLER_JSON__")

PHI = (1 + 5 ** 0.5) / 2

def _temp_log_path() -> Path:
    t = os.environ.get("TEMP") or os.environ.get("TMP") or str(Path.home())
    return Path(t) / f"{SUITE_NAME}_Setup.log"

LOG_PATH = _temp_log_path()

def _log(msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

def _msgbox(title: str, text: str, flags: int = 0x40) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, text, title, flags)
    except Exception:
        pass

def _install_excepthook() -> None:
    def hook(exc_type, exc, tb):
        _log("EXCEPCION NO CONTROLADA:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
        _msgbox(f"{SUITE_NAME} Setup", f"Error inesperado:\n{exc}\n\nLog:\n{LOG_PATH}", 0x10)
    sys.excepthook = hook

    if hasattr(threading, "excepthook"):
        def th_hook(args):  # type: ignore
            _log("EXCEPCION THREAD:\n" + "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
            _msgbox(f"{SUITE_NAME} Setup", f"Error en segundo plano:\n{args.exc_value}\n\nLog:\n{LOG_PATH}", 0x10)
        threading.excepthook = th_hook  # type: ignore

def _resource_path(name: str) -> Path:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / name
    return Path(__file__).resolve().parent / name

def _sendto_dir() -> Path:
    return Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "SendTo"

def _ps_encoded(script: str) -> str:
    return base64.b64encode(script.encode("utf-16le")).decode("ascii")

def _run_powershell(script: str) -> None:
    enc = _ps_encoded(script)
    p = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", enc],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"PowerShell code={p.returncode}")

def _try_create_lnk(lnk: Path, target: Path, icon: Optional[Path] = None) -> bool:
    try:
        icon_loc = f"{icon},0" if icon else ""
        script_lines = [
            "$WshShell = New-Object -ComObject WScript.Shell",
            f"$Shortcut = $WshShell.CreateShortcut('{str(lnk)}')",
            f"$Shortcut.TargetPath = '{str(target)}'",
            f"$Shortcut.WorkingDirectory = '{str(target.parent)}'",
        ]
        if icon_loc:
            script_lines.append(f"$Shortcut.IconLocation = '{icon_loc}'")
        script_lines.append("$Shortcut.Save()")
        _run_powershell("\n".join(script_lines))
        return True
    except Exception as e:
        _log(f"No se pudo crear .lnk ({lnk.name}): {e}")
        return False

def _create_sendto_cmd(cmd_path: Path, target: Path) -> None:
    content = "\n".join([
        "@echo off",
        "setlocal",
        f"set EXE={str(target)}",
        'start "" "%EXE%" %*',
        "endlocal",
    ]) + "\n"
    cmd_path.write_text(content, encoding="utf-8", newline="\r\n")

def _install_sendto(install_dir: Path) -> None:
    sd = _sendto_dir()
    sd.mkdir(parents=True, exist_ok=True)

    for app_id, display in DISPLAY_NAMES.items():
        exe = install_dir / APP_EXE_NAMES[app_id]
        if not exe.exists():
            raise FileNotFoundError(f"No se encontró {exe.name} en {install_dir}")

        lnk = sd / f"{display}.lnk"
        if _try_create_lnk(lnk, exe, icon=exe):
            continue

        cmd = sd / f"{display}.cmd"
        _create_sendto_cmd(cmd, exe)

def _write_context_menu(install_dir: Path) -> Tuple[bool, str]:
    if winreg is None:
        return False, "winreg no disponible."
    try:
        root = "Software\\Classes\\*\\shell\\" + SUITE_NAME
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, root) as k:
            winreg.SetValueEx(k, "MUIVerb", 0, winreg.REG_SZ, f"{SUITE_NAME} {SUITE_VERSION}")
            verbs = []
            for app_id in DISPLAY_NAMES:
                verbs.append(f"{SUITE_NAME}.{app_id}")
            winreg.SetValueEx(k, "SubCommands", 0, winreg.REG_SZ, ";".join(verbs))
            any_exe = install_dir / APP_EXE_NAMES["txt_full"]
            if any_exe.exists():
                winreg.SetValueEx(k, "Icon", 0, winreg.REG_SZ, str(any_exe))

        cs = "Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\CommandStore\\shell"
        dispatcher = install_dir / APP_EXE_NAMES.get("dispatcher", "MiausoftSuite_Dispatcher.exe")

        for app_id, display in DISPLAY_NAMES.items():
            verb = f"{SUITE_NAME}.{app_id}"
            vkey = cs + "\\" + verb
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, vkey) as vk:
                winreg.SetValueEx(vk, "", 0, winreg.REG_SZ, display)
                winreg.SetValueEx(vk, "MultiSelectModel", 0, winreg.REG_SZ, "Player")
            cmd_key = vkey + "\\command"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, cmd_key) as ck:
                if dispatcher.exists():
                    winreg.SetValueEx(ck, "", 0, winreg.REG_SZ, f'"{dispatcher}" --app {app_id} "%1"')
                else:
                    exe = install_dir / APP_EXE_NAMES[app_id]
                    winreg.SetValueEx(ck, "", 0, winreg.REG_SZ, f'"{exe}" "%1"')

        return True, "OK"
    except Exception as e:
        _log("Fallo menú contextual:\n" + traceback.format_exc())
        return False, str(e)

def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".__miausoft_write_test__"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def _default_install_dir() -> Path:
    pf = os.environ.get("ProgramFiles") or "C:\\Program Files"
    return Path(pf) / f"{SUITE_NAME} {SUITE_VERSION}"

def _fallback_user_dir() -> Path:
    lad = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(lad) / SUITE_NAME / "Apps"

def _extract_payload(payload: Path, target: Path, cb=None) -> None:
    with zipfile.ZipFile(payload, "r") as zf:
        names = zf.namelist()
        total = max(1, len(names))
        for i, n in enumerate(names, start=1):
            zf.extract(n, path=target)
            if cb:
                cb(i / total)

def _install(desired: Path, cb=None) -> Tuple[Path, str]:
    warnings = []
    payload = _resource_path("payload.zip")
    if not payload.exists():
        raise FileNotFoundError("No se encontró payload.zip dentro del instalador.")

    install_dir = desired
    if not _is_writable_dir(install_dir):
        warnings.append("No se pudo escribir en la ruta elegida/Program Files. Se usará una ruta de usuario.")
        install_dir = _fallback_user_dir()
        if not _is_writable_dir(install_dir):
            install_dir = _sendto_dir() / f"{SUITE_NAME}_Apps"
            install_dir.mkdir(parents=True, exist_ok=True)

    if cb: cb(0.05)
    _extract_payload(payload, install_dir, cb=(lambda f: cb(0.05 + 0.70 * f) if cb else None))
    if cb: cb(0.80)

    _install_sendto(install_dir)
    if cb: cb(0.90)

    ok, msg = _write_context_menu(install_dir)
    if not ok:
        warnings.append("No se pudo crear el menú contextual. Se dejó instalado SendTo.")
        warnings.append(f"Detalle: {msg}")

    if cb: cb(1.0)
    return install_dir, "\n".join(warnings).strip()

def _run_headless() -> int:
    try:
        desired = _default_install_dir()
        install_dir, warn = _install(desired, cb=None)
        msg = "Instalación completada (modo compatibilidad).\n\n"
        if warn:
            msg += "Advertencias:\n" + warn + "\n\n"
        msg += f"Ruta:\n{install_dir}\n\nLog:\n{LOG_PATH}"
        _msgbox(f"{SUITE_NAME} Setup", msg, 0x40)
        return 0
    except Exception as e:
        _log("Fallo headless:\n" + traceback.format_exc())
        _msgbox(f"{SUITE_NAME} Setup", f"Fallo crítico.\n\nError:\n{e}\n\nLog:\n{LOG_PATH}", 0x10)
        return 1

def _run_gui() -> int:
    try:
        import tkinter as tk
    except Exception as e:
        _log("Tk no pudo importarse: " + str(e))
        return _run_headless()

    colors = (CFG_PROGRESS.get("colors") or CFG_INSTALLER.get("colors") or {}) or {}
    utilities = (CFG_PROGRESS.get("utilities") or CFG_INSTALLER.get("utilities") or {}) or {}
    btn_cfg = (CFG_BASE.get("buttons") or {}) or {}

    bg = colors.get("bg", "#121212")
    title_fg = colors.get("title", "#ffffff")
    sub_fg = colors.get("subtitle", "#cfcfcf")
    entry_bg = colors.get("entry_bg", "#1f1f1f")
    entry_fg = colors.get("entry_fg", "#ffffff")

    win = (CFG_PROGRESS.get("window") or {}) or {}
    W = int(win.get("w", 640))
    H = int(win.get("h", 360))
    PAD = int(round(18 * PHI))
    Y0 = int(round(22 * PHI))

    def _button_colors(kind: str, hover: bool, pressed: bool):
        cols = btn_cfg.get("colors", {}) or {}
        if kind == "secondary":
            normal = cols.get("secondary_bg", "#2b2b2b")
            hov = cols.get("secondary_hover", "#343434")
        else:
            normal = cols.get("primary_bg", "#2f6fed")
            hov = cols.get("primary_hover", "#3c7cff")
        fg = cols.get("text", "#ffffff")
        return (hov, fg) if (hover or pressed) else (normal, fg)

    class RoundedButton(tk.Canvas):
        def __init__(self, master, text: str, command, kind: str):
            super().__init__(master, highlightthickness=0, bd=0, bg=master.cget("bg"))
            self._text = text
            self._cmd = command
            self._kind = kind
            self._hover = False
            self._pressed = False
            self._w = int(btn_cfg.get("w", 170))
            self._h = int(btn_cfg.get("h", 44))
            self._r = int(btn_cfg.get("radius", 12))
            self._font = (btn_cfg.get("font", "Segoe UI"), int(btn_cfg.get("font_size", 11)), "bold")
            self.configure(width=self._w, height=self._h)
            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
            self.bind("<Button-1>", self._on_press)
            self.bind("<ButtonRelease-1>", self._on_release)
            self._redraw()

        def _rr(self, x1, y1, x2, y2, r, **kw):
            pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2, x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1]
            return self.create_polygon(pts, smooth=True, **kw)

        def _redraw(self):
            self.delete("all")
            bgc, fgc = _button_colors(self._kind, self._hover, self._pressed)
            self._rr(2, 2, self._w-2, self._h-2, self._r, fill=bgc, outline="")
            self.create_text(self._w//2, self._h//2, text=self._text, fill=fgc, font=self._font)

        def _on_enter(self, _e=None):
            self._hover = True
            self._redraw()

        def _on_leave(self, _e=None):
            self._hover = False
            self._pressed = False
            self._redraw()

        def _on_press(self, _e=None):
            self._pressed = True
            self._redraw()

        def _on_release(self, _e=None):
            was = self._pressed
            self._pressed = False
            self._redraw()
            if was and callable(self._cmd):
                self._cmd()

    class ProgressBar(tk.Canvas):
        def __init__(self, master):
            super().__init__(master, highlightthickness=0, bd=0, bg=master.cget("bg"))
            self._cfg = (CFG_PROGRESS.get("progressbar") or {}) or {}
            self._frac = 0.0
            self.bind("<Configure>", lambda e: self._redraw())
            self.after(0, self._redraw)

        def set(self, f: float):
            self._frac = max(0.0, min(1.0, float(f)))
            self._redraw()

        def _redraw(self):
            self.delete("all")
            w = max(1, int(self.winfo_width()))
            h = max(1, int(self.winfo_height()))
            cols = self._cfg.get("colors", {}) or {}
            track = cols.get("track", "#2b2b2b")
            fill = cols.get("fill", "#2f6fed")
            self.create_rectangle(0, 0, w, h, fill=track, outline="")
            fw = int(w * self._frac)
            if fw > 0:
                self.create_rectangle(0, 0, fw, h, fill=fill, outline="")

    root = tk.Tk()
    root.title(f"{SUITE_NAME} Setup")
    root.configure(bg=bg)
    root.geometry(f"{W}x{H}")
    root.resizable(False, False)

    try:
        ico = _resource_path("Miausoft.ico")
        if ico.exists():
            root.iconbitmap(str(ico))
    except Exception:
        pass

    def tk_exc(exc, val, tb):
        _log("Tk callback exception:\n" + "".join(traceback.format_exception(exc, val, tb)))
        _msgbox(f"{SUITE_NAME} Setup", f"Error de interfaz:\n{val}\n\nLog:\n{LOG_PATH}", 0x10)
    root.report_callback_exception = tk_exc  # type: ignore

    welcome = tk.Frame(root, bg=bg)
    welcome.place(relx=0, rely=0, relwidth=1, relheight=1)

    title = tk.Label(
        welcome, text=f"Gracias por instalar {SUITE_NAME} {SUITE_VERSION}",
        bg=bg, fg=title_fg,
        font=(utilities.get("title_family", "Segoe UI"), int(utilities.get("title_size", 18)), "bold"),
    )
    subtitle = tk.Label(
        welcome, text=f"Última versión estable {SUITE_VERSION} ({STABLE_TAG}). Elija donde instalar:",
        bg=bg, fg=sub_fg,
        font=(utilities.get("subtitle_family", "Segoe UI"), int(utilities.get("subtitle_size", 11))),
    )
    path_var = tk.StringVar(value=str(_default_install_dir()))
    entry = tk.Entry(
        welcome, textvariable=path_var,
        bg=entry_bg, fg=entry_fg, insertbackground=entry_fg,
        relief="flat",
        font=(utilities.get("entry_family", "Segoe UI"), int(utilities.get("entry_size", 11))),
    )

    prog = tk.Frame(root, bg=bg)
    status_var = tk.StringVar(value="Preparando…")
    prog_title = tk.Label(
        prog, text=f"Instalando {SUITE_NAME} {SUITE_VERSION}…",
        bg=bg, fg=title_fg,
        font=(utilities.get("title_family", "Segoe UI"), int(utilities.get("title_size", 18)), "bold"),
    )
    status = tk.Label(
        prog, textvariable=status_var,
        bg=bg, fg=sub_fg,
        font=(utilities.get("subtitle_family", "Segoe UI"), int(utilities.get("subtitle_size", 11))),
    )
    pb = ProgressBar(prog)

    worker = {"done": False, "ok": False, "msg": ""}

    def do_cancel():
        try:
            root.destroy()
        except Exception:
            pass

    def place_two(btn_r: tk.Canvas, btn_l: tk.Canvas):
        gap = int(round(12 * PHI))
        bw = int(btn_r.cget("width"))
        bh = int(btn_r.cget("height"))
        x2 = W - PAD - bw
        x1 = x2 - gap - bw
        y = H - PAD - bh
        btn_l.place(x=x1, y=y)
        btn_r.place(x=x2, y=y)

    def place_one(btn: tk.Canvas):
        bw = int(btn.cget("width"))
        bh = int(btn.cget("height"))
        x = W - PAD - bw
        y = H - PAD - bh
        btn.place(x=x, y=y)

    btn_install = RoundedButton(welcome, "Instalar", lambda: None, "primary")
    btn_cancel = RoundedButton(welcome, "Cancelar", do_cancel, "secondary")
    btn_cancel_p = RoundedButton(prog, "Cancelar", do_cancel, "secondary")

    def switch_to_progress():
        welcome.place_forget()
        prog.place(relx=0, rely=0, relwidth=1, relheight=1)

    def do_install(desired: Path):
        try:
            def cb(f: float):
                pb.set(f)
                root.update_idletasks()

            status_var.set("Extrayendo archivos…")
            pb.set(0.02)

            install_dir, warn = _install(desired, cb=cb)

            msg = "Instalación completada.\n\n"
            if warn:
                msg += "Advertencias:\n" + warn + "\n\n"
            msg += f"Ruta:\n{install_dir}\n\nLog:\n{LOG_PATH}"

            worker["ok"] = True
            worker["msg"] = msg
        except Exception as e:
            _log("Fallo instalación:\n" + traceback.format_exc())
            worker["ok"] = False
            worker["msg"] = f"No se pudo completar la instalación.\n\nError:\n{e}\n\nLog:\n{LOG_PATH}"
        finally:
            worker["done"] = True

    def poll():
        if worker["done"]:
            _msgbox(f"{SUITE_NAME} Setup", worker["msg"], 0x40 if worker["ok"] else 0x10)
            try:
                root.destroy()
            except Exception:
                pass
            return
        root.after(120, poll)

    def on_install():
        desired = Path(path_var.get().strip() or "")
        if not str(desired):
            _msgbox(f"{SUITE_NAME} Setup", "Ruta inválida.", 0x10)
            return
        switch_to_progress()

        prog_title.place(x=PAD, y=Y0)
        status.place(x=PAD, y=Y0 + int(round(44 * PHI)))
        pb.place(x=PAD, y=Y0 + int(round(84 * PHI)), width=W - 2*PAD, height=int(round(18 * PHI)))
        place_one(btn_cancel_p)

        t = threading.Thread(target=do_install, args=(desired,), daemon=True)
        t.start()
        root.after(80, poll)

    btn_install.bind("<ButtonRelease-1>", lambda e: on_install())

    title.place(x=PAD, y=Y0)
    subtitle.place(x=PAD, y=Y0 + int(round(44 * PHI)))
    entry.place(x=PAD, y=Y0 + int(round(84 * PHI)), width=W - 2*PAD, height=int(round(26 * PHI)))
    place_two(btn_install, btn_cancel)

    root.mainloop()
    return 0

def main() -> int:
    _install_excepthook()
    _log("Inicio Setup")
    if "--headless" in sys.argv:
        return _run_headless()
    return _run_gui()

if __name__ == "__main__":
    raise SystemExit(main())
"""
    def j(x: Any) -> str:
        return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    out = tpl
    out = out.replace("__DISPLAY_NAMES_JSON__", j(DISPLAY_NAMES).replace("\\", "\\\\"))
    out = out.replace("__APP_EXE_NAMES_JSON__", j(APP_EXE_NAMES).replace("\\", "\\\\"))
    out = out.replace("__CFG_BASE_JSON__", j(cfg_base).replace("\\", "\\\\"))
    out = out.replace("__CFG_PROGRESS_JSON__", j(cfg_progress).replace("\\", "\\\\"))
    out = out.replace("__CFG_INSTALLER_JSON__", j(cfg_installer).replace("\\", "\\\\"))
    return out.strip() + "\n"


def build_setup(
    project_dir: Path,
    payload_zip: Path,
    out_dir: Path,
    *,
    mode: str,
    console: bool,
    uac_admin: bool,
) -> Path:
    if not _is_windows():
        _die("Este builder es para Windows.")
    _ensure_pyinstaller()

    if not payload_zip.exists():
        _die(f"No existe payload.zip: {payload_zip}")

    cfg_py = project_dir / "config.py"
    if not cfg_py.exists():
        _die(f"No existe config.py en: {cfg_py}")

    cfg_mod = _load_module_from_path(cfg_py, "miausoft_config")
    cfg_base = getattr(cfg_mod, "CFG_BASE", {}) or {}
    cfg_progress = getattr(cfg_mod, "CFG_PROGRESS", {}) or {}
    cfg_installer = getattr(cfg_mod, "CFG_INSTALLER", {}) or {}

    out_dir.mkdir(parents=True, exist_ok=True)

    build_dir = out_dir / "_setup_build"
    dist_dir = out_dir / "_setup_dist"
    spec_dir = out_dir / "_setup_spec"
    for d in [build_dir, dist_dir, spec_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    runtime_py = out_dir / "_setup_runtime.py"
    runtime_py.write_text(_render_setup_runtime(cfg_base, cfg_progress, cfg_installer), encoding="utf-8", newline="\n")

    ico = _maybe(project_dir, "Miausoft.ico")
    png = _maybe(project_dir, "Miausoft.png")

    runtime_tmp = "C:\\Users\\Public\\Documents\\MiausoftSuiteSetupTmp"

    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--noupx",
        "--name", f"{SUITE_NAME}Setup",
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir),
        "--specpath", str(spec_dir),
        "--log-level", "WARN",
    ]

    if mode == "onefile":
        args.append("--onefile")
        args += ["--runtime-tmpdir", runtime_tmp]
    else:
        args.append("--onedir")

    args.append("--console" if console else "--windowed")
    if uac_admin:
        args.append("--uac-admin")

    if ico is not None:
        args += ["--icon", str(ico)]

    args += ["--collect-all", "tkinter"]
    args += ["--hidden-import", "_tkinter"]

    args += ["--add-data", f"{payload_zip}{os.pathsep}payload.zip"]

    if ico is not None:
        args += ["--add-data", f"{ico}{os.pathsep}Miausoft.ico"]
    if png is not None:
        args += ["--add-data", f"{png}{os.pathsep}Miausoft.png"]

    args.append(str(runtime_py))
    _run(args, cwd=project_dir)

    if mode == "onefile":
        exe = dist_dir / f"{SUITE_NAME}Setup.exe"
        if not exe.exists():
            _die(f"No se generó: {exe}")
        final = out_dir / f"{SUITE_NAME}Setup.exe"
        if final.exists():
            final.unlink(missing_ok=True)
        shutil.move(str(exe), str(final))
        return final

    folder = dist_dir / f"{SUITE_NAME}Setup"
    if not folder.exists():
        _die(f"No se generó carpeta onedir: {folder}")
    final = out_dir / f"{SUITE_NAME}Setup_onedir"
    if final.exists():
        shutil.rmtree(final, ignore_errors=True)
    shutil.move(str(folder), str(final))
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, default=Path.cwd())
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path.cwd() / "release")
    ap.add_argument("--mode", choices=["onefile", "onedir"], default="onedir")
    ap.add_argument("--console", action="store_true")
    ap.add_argument("--uac-admin", action="store_true")
    args = ap.parse_args()

    built = build_setup(
        args.project_dir.resolve(),
        args.payload.resolve(),
        args.out_dir.resolve(),
        mode=args.mode,
        console=args.console,
        uac_admin=args.uac_admin,
    )
    print(f"OK: {built}")


if __name__ == "__main__":
    main()
