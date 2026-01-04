# config.py
# Config centralizado para toda la familia Miausoft.
# (Incluye utilidades visuales; visual.py y progress_ui.py debe vivir aquí.)
from __future__ import annotations

from math import sqrt
from pathlib import Path
import os
import copy
from typing import Any, Optional, Dict

# ──────────────────────────────────────────────────────────────────────────────
# Tipografía: soporte Comfortaa en todos los textos
# - Si la fuente no está instalada, en Windows puede cargarse "privada" para el proceso
#   desde un .ttf local (sin privilegios de administrador).
# - Si no se encuentra, se usa fallback

_FONTS_REGISTERED: set[str] = set()

def _is_windows() -> bool:
    return os.name == "nt"

def _font_candidate_paths(family: str) -> list[Path]:
    family_norm = (family or "").strip().lower()
    base_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    env_dir = (os.getenv("MIAUSOFT_FONTS_DIR") or "").strip()
    dirs: list[Path] = []
    if env_dir:
        try:
            dirs.append(Path(env_dir))
        except Exception:
            pass

    dirs += [
        base_dir / "fonts",
        Path.cwd() / "fonts",
    ]

    if _is_windows():
        win_dir = Path(os.getenv("WINDIR", r"C:\Windows"))
        dirs.append(win_dir / "Fonts")

    if family_norm == "comfortaa":
        names = [
            "Comfortaa-Regular.ttf",
            "Comfortaa-VariableFont_wght.ttf",
            "comfortaa.ttf",
        ]
    else:
        # intento genérico para otras familias
        fam = (family or "").strip()
        names = [f"{fam}.ttf", f"{fam}-Regular.ttf", f"{family_norm}.ttf"]

    for d in dirs:
        for n in names:
            p = d / n
            if p.exists():
                candidates.append(p)

    # dedup preservando orden
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(Path(rp))
    return out

def _win_register_font_private(ttf_path: Path) -> bool:
    """Registra un .ttf para uso privado del proceso (Windows)."""
    try:
        import ctypes
        from ctypes import wintypes

        FR_PRIVATE = 0x10
        n = ctypes.windll.gdi32.AddFontResourceExW(
            wintypes.LPCWSTR(str(ttf_path)),
            wintypes.DWORD(FR_PRIVATE),
            0,
        )
        if int(n) > 0:
            try:
                HWND_BROADCAST = 0xFFFF
                WM_FONTCHANGE = 0x001D
                ctypes.windll.user32.SendMessageW(HWND_BROADCAST, WM_FONTCHANGE, 0, 0)
            except Exception:
                pass
            return True
    except Exception:
        return False
    return False

def ensure_font_registered(family: str) -> None:
    """Best-effort: hace visible la familia (si hay .ttf disponible) para este proceso."""
    fam = (family or "").strip()
    if not fam:
        return
    key = fam.lower()
    if key in _FONTS_REGISTERED:
        return
    _FONTS_REGISTERED.add(key)

    if not _is_windows():
        return

    for p in _font_candidate_paths(fam):
        _win_register_font_private(p)


PHI = (1 + sqrt(5)) / 2  # φ ≈ 1.6180339887

# Referencias "base" (solo para convertir fallback -> px de forma consistente).
# No se usan como "diseño local"; únicamente permiten que incluso el fallback
# (cuando el SO/WM reporta 0, o antes de que Tk calcule geometría) siga siendo φ-relativo.
REF_SCREEN_W = 1366
REF_SCREEN_H = 768


def phi_rel(n: int) -> float:
    """Relativo a φ en la forma acordada: 1/(PHI*N). N SIEMPRE entero."""
    return 1 / (PHI * int(n))


def phi_px(ref: int, n: int, *, min_px: int = 1) -> int:
    """Convierte un 1/(PHI*N) a px usando un ref (W/H). Siempre φ-relativo (incluso fallback)."""
    try:
        v = int(ref * phi_rel(int(n)))
    except Exception:
        v = int(min_px)
    return max(int(min_px), int(v))


def phi_px_w(n: int, *, min_px: int = 1) -> int:
    return phi_px(REF_SCREEN_W, int(n), min_px=min_px)


def phi_px_h(n: int, *, min_px: int = 1) -> int:
    return phi_px(REF_SCREEN_H, int(n), min_px=min_px)


# ──────────────────────────────────────────────────────────────────────────────
# Base (tokens compartidos)
BASE_CONFIG: Dict[str, Any] = {
    "global": {
        "tk_scaling": 1.0,
        # Más compacto (pero legible). N entero.
        "ui_scale": phi_rel(1),  # ≈ 0.618
    },

    "app": {
        "safe_extract_timeout": 30,
        "pdf_workers": 0,  # 0 => auto (usa ~1/2 CPU, cap 4)
        "pdf_prefetch": 0, # 0 => auto (workers*2)
        "close_delay_ms": 2000,
        # IPC multi-selección (Explorer puede ejecutar 1 proceso por archivo)
        "multiselect": {
            "settle_ms": 7000,
            "stable_ms": 650,
            "poll_ms": 60,
        },
        "pdf_text_workers": 4,
        "pdf_text_prefetch": 0,
                "ocr": {
            "enable": True,
            "force": False,
            "timeout": 60,
            "timeout_sec": 60,
            "skip_text": True,
            "min_chars": 20,
            # Nuevo disparador: OCR cuando > trigger_ratio del documento tiene < min_chars.
            "trigger_ratio": 0.10,
            "dpi": 75,
            # 0 => auto (se calcula por CPU y #páginas a OCR).
            "jobs": 0,
            "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            # Limpieza barata post-OCR: elimina NBSP y deja solo whitelist + whitespace.
            "sanitize_ocr_output": True,
        },

        # FadePopover (Fusionador) — sin fallbacks locales
        "popover_frame_ms": int(PHI * 25),  # φ * entero
        "popover_alpha_step": phi_rel(10),
    },

    "window": {
        "use_relative": True,

        # Horizontales y compactas. (N entero)
        "rel_width":  phi_rel(2),
        "rel_height": phi_rel(3),

        # Fallback px (mínimos) — también φ-relativo
        "min_width":  phi_px_w(2),
        "min_height": phi_px_h(3),

        # Compatibilidad (no limitar en la práctica)
        "max_width_rel_screen": 1.0,
        "max_height_rel_screen": 0.95,

        "title": "Miausoft",
        "icon_name": "Miausoft.ico",
        "dpi_scale_policy": "auto",
        "resizable": True,
        "topmost": False,
    },

    "win11": {
        "backdrop": "acrylic",          # mica | acrylic | tabbed | none
        "use_accent_from_dwm": True,
        "prefer_dark_mode": True,
        "enable_backdrop": False,
        "enable_on_progress": True,
    },

    "colors": {
        "app_bg":   "#f3f3f3",
        "box_bg":   "#FFFFFF",
        "text":     "#2a2a2a",
        "muted":    "#767676",
        "subtext":  "#767676",  # alias para subtítulos (progress_ui)

        "btn_base":       "#fafafa",
        "btn_face":       "#fafafa",
        "btn_face_hover": "#f0f0f0",
        "btn_face_down":  "#f0f0f0",

        "progress_track": "#FFFFFF",

        "scroll_trough":  "#F6F7FB",
        "scroll_arrow":   "#222222",
    },

    "fonts": {
        "family": "Comfortaa",
        "fallback_family": "Segoe UI",

        # Relativos + fallback px (ambos φ-relativos, N entero)
        # Nota: el "px" sirve para cuando Tk aún no reporta altura/pantalla (fallback determinista).
        "title":   {"rel_px": 1/(PHI*25), "px": phi_px_h(25, min_px=15), "bold": True,  "italic": False},
        "sub":     {"rel_px": 1/(PHI*28), "px": phi_px_h(28, min_px=15), "bold": False, "italic": True},
        "head":    {"rel_px": 1/(PHI*30), "px": phi_px_h(30, min_px=15), "bold": True,  "italic": False},
        "row":     {"rel_px": 1/(PHI*25), "px": phi_px_h(25, min_px=15),  "bold": False, "italic": False},
        "btn":     {"rel_px": 1/(PHI*28), "px": phi_px_h(28, min_px=15),  "bold": False, "italic": False},
        "err":     {"rel_px": 1/(PHI*30), "px": phi_px_h(30, min_px=15),  "bold": False, "italic": True},
        "preview": {"rel_px": 1/(PHI*25), "px": phi_px_h(25, min_px=9),  "bold": False, "italic": False},

        # Mínimo global (fallback) φ-relativo
        "min_px": phi_px_h(60, min_px=8),
    },

    "layout": {
        # px (todo φ-relativo, N entero)
        "divider_px": phi_px_h(475, min_px=1),
        "scrollbar_fallback_w_px": phi_px_w(120, min_px=6),

        # Más compacto, pero sin "aplastar" (N entero)
        "table_pad_px": phi_px_h(70, min_px=6),
        "row_pad_min_px": phi_px_h(80, min_px=5),

        "drag_factor": 5,
        "drag_min_px": phi_px_h(30, min_px=12),
        "row_min_px": phi_px_h(55, min_px=10),
        "row_max_px": phi_px_h(10, min_px=25),  # cap duro: 25px

        "divider_alpha": 0.10,
        "zebra_alpha": 0.04,
        "btn_edge_alpha": 1,

        # Filas cómodas (N entero)
        "row_height_rel": 1/(PHI*5),
        "row_sep_rel":    1/(PHI*15),

        # Botones: más anchos para evitar que un módulo baje más la fuente que otro.
        "btn_w_rel_w":    1/(PHI*5),
        "btn_h_rel_h":    1/(PHI*19),

        # Máximos (evita botones gigantes al agrandar demasiado)
        "btn_w_max_rel_w": 1/(PHI*4),
        "btn_h_max_rel_h": 1/(PHI*18),

        # Mínimos (N entero)
        "btn_min_w_px": phi_px_w(6, min_px=int(PHI*7)),
        "btn_min_h_px": phi_px_h(20, min_px=int(PHI)),

        # Padding interno (N entero)
        "btn_text_pad_px": phi_px_h(70, min_px=3),
        "btn_text_pad_rel_h": 1/(PHI*7),

        # Radios para píldoras (φ * entero)
        "btn_radius_px": int(PHI * 5000),

        "preview_rel_w":  1/PHI,
        "preview_rel_h":  1.0,
    },

    # Columnas genéricas
    "columns": {
        "weights_raw": {
            "archivo": 1/(PHI*5),
            "sufijo":  1/(PHI*5),
            "inicio":  1/(PHI*5),
            "fin":     1/(PHI*5),
            "caps":    1/(PHI*5),
        },
        "base_min_px": phi_px_w(11, min_px=80),
        "min_px": {
            "archivo": phi_px_w(6, min_px=120),
            "sufijo":  phi_px_w(6, min_px=120),
            "inicio":  phi_px_w(16, min_px=50),
            "fin":     phi_px_w(16, min_px=50),
            "caps":    phi_px_w(6, min_px=120),
        },
    },

    # Reemplazador: columnas (incluye Regex)
    "replacements": {
        "columns": {
            "weights_raw": {"pattern": 0.48, "replace": 0.48},
            "min_px": {
                "pattern": phi_px_w(8, min_px=120),
                "replace": phi_px_w(25, min_px=5),
                "regex":   phi_px_w(30, min_px=10),
            },
            "regex_rel_w": 1/(PHI*30),
        }
    },

    # Controles: SlideButton (base)
    "controls": {
        "slide": {
            "radius_px": phi_px_h(2, min_px=40),
            "pad_px": phi_px_h(300, min_px=2),
            "track_off": "#e9eaec",
            "track_on":  "#e6e7e9",
            "knob": "#ffffff",
            "text_on":  "#2a2a2a",
            "text_off": "#767676",
            "border": "#dedede",
            "shadow_light": "#ffffff",
            "shadow_dark":  "#c9c9c9",
            "shadow_offset_px": phi_px_h(250, min_px=2),
            "shadow_inner_offset_px": phi_px_h(475, min_px=1),
            "anim_ms": 120,
            "frame_ms": 30,
            "notify_ms": int(PHI * 50),  # φ * entero
        }
    },

    "progress_logic": {
        "enable_ocr": False,
        "ocr_min_chars": 20,
        "ocr_trigger_ratio": 0.10,
        "ocr_whitelist": 'ABCDEFGHIJKLMNÑOPQRSTUWXYZabcdefghijklmnñopqrstuwxyz1234567890!"#%/()=\'?ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψως',
        "ocr_sanitize_output": True,
        "ocr_dpi": 75,
        "ocr_min_consecutive_pages": 3,
        "ocr_jobs": 1,
        "ocr_force": False,
        "ocr_timeout": 60,
        "ocr_skip_text": True,
    },

    # Capítulos
    "chapters": {
        "prefer_bookmarks_only": False,
        "heuristic_pdf_max_scan": None,
        "heuristic_pdf_max_nohit": 400,
        "heuristic_lines_head": 10,
        "max_bookmark_level": 2,
        "min_gap_pages": 3,
        "merge_similarity_threshold": 0.60,
        "merge_max_gap": 3,
    },
    "chapters_fastmenu": {
        "show_full_doc": True,
        "show_pdf_bookmarks": True,
        "show_epub_toc": True,
        "show_quick_scan": True,
        "show_full_scan": True,
    },
    "chapters_runtime": {
        "time_budget_ms": 3600,
        "fast_scan_pages": 60,
        "full_scan_pages": None,
    },
    "chapters_popup": {
        "use_relative": True,
        "rel_w_tree": 1/(PHI*1),
        "rel_h_tree": 1.0,

        # fallbacks φ-relativos (N entero)
        "min_w_px": phi_px_w(2, min_px=320),
        "max_h_px": phi_px_h(1, min_px=420),
        "fixed_w_px": phi_px_w(2, min_px=360),
        "fixed_h_px": phi_px_h(1, min_px=320),
    },

    "persistence": {
        "enable_cache": True,
        "cache_file": str(Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "Miausoft" / "cap_index_v3.json"),
    },

    "performance": {
        "threads_for_chapters": 4,
        "pdf_detect_max_scan": None,
        "pdf_detect_max_nohit": 800,
        "fitz_raster_scale": 0.8,
        "chapter_cache_size": 1024,
        "prefer_pymupdf_text": True,
        "use_pypdf_outlines_fallback": False,
        "max_workers_for_detection": 2,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Overrides por aplicación (SOLO tokens; sin lógica)
#
# IMPORTANTE:
# - "installer" DEBE existir al mismo nivel que el resto de apps (no anidado).
# - La geometría del instalador se controla por variables globales, gobernadas aquí
#   (sin “diseños locales” en el instalador).
#
# Especificación pedida (V3):
#   ancho  = phi_rel(INSTALLER_PHI_ITER_W)
#   alto   = phi_rel(INSTALLER_PHI_ITER_H)
INSTALLER_PHI_ITER_W: int = 2
INSTALLER_PHI_ITER_H: int = 3
INSTALLER_REL_WIDTH: float  = phi_rel(INSTALLER_PHI_ITER_W)
INSTALLER_REL_HEIGHT: float = phi_rel(INSTALLER_PHI_ITER_H)

APP_OVERRIDES: Dict[str, Any] = {
    "progress": {
        "window": {
            "title": "Procesando…",
            "use_relative": True,

            # Larga y baja (horizontal) — N entero
            "rel_width":  phi_rel(2),
            "rel_height": phi_rel(3),

            # fallbacks φ-relativos
            "min_width":  phi_px_w(2, min_px=420),
            "min_height": phi_px_h(4, min_px=180),

            "icon_name": "Miausoft.ico",
            "resizable": False,
            "topmost": True,
            "transient_to_parent": False,
        },
        "fonts": {
            "title": {"rel_px": 1/(PHI*25), "px": phi_px_h(32, min_px=12), "bold": True,  "italic": False},
            "sub":   {"rel_px": 1/(PHI*30), "px": phi_px_h(46, min_px=9),  "bold": False, "italic": True},
        },
        "layout": {
            "icon_box_width_rel_w": 1/(PHI*2),

            "boxes_pad_rel_h": 1/(PHI*15),
            "boxes_pad_min_px": 0,
            "merge_middle_padding": True,

            "bar_w_rel_w": 1/(PHI),

            "bar_h_rel_h": 1/(PHI*8),
            "bar_min_h_px": phi_px_h(40, min_px=int(PHI*6)),
            "bar_radius_px": int(PHI * 5000),

            "bar_top_rel_inner":       1/(PHI*9),
            "gap_bar_title_rel_inner": 1/(PHI*12),
            "gap_title_sub_rel_inner": 1/(PHI*18),
        },
    },

    "installer": {
        "window": {
            "title": "Instalador de MiausoftSuite",
            "use_relative": True,

            "rel_width":  INSTALLER_REL_WIDTH,
            "rel_height": INSTALLER_REL_HEIGHT,

            "min_width":  phi_px_w(2, min_px=420),
            "min_height": phi_px_h(3, min_px=220),

            "icon_name": "Miausoft.ico",
            "resizable": False,
            "topmost": False,
            "transient_to_parent": False,
        },
        "fonts": {
            "title": {"rel_px": 1/(PHI*20), "px": phi_px_h(32, min_px=12), "bold": True,  "italic": False},
            "sub":   {"rel_px": 1/(PHI*30), "px": phi_px_h(46, min_px=9),  "bold": False, "italic": True},
        },
        "layout": {
            "icon_box_width_rel_w": 1/(PHI*2),
            "boxes_pad_rel_h": 1/(PHI*15),
            "boxes_pad_min_px": 0,
            "merge_middle_padding": True,

            "bar_w_rel_w": 1/(PHI),

            "bar_h_rel_h": 1/(PHI*8),
            "bar_min_h_px": phi_px_h(40, min_px=int(PHI*6)),
            "bar_radius_px": int(PHI * 5000),

            "bar_top_rel_inner":       1/(PHI*9),
            "gap_bar_title_rel_inner": 1/(PHI*12),
            "gap_title_sub_rel_inner": 1/(PHI*18),
        },
    },

    "conversor": {"window": {"title": "Miausoft Conversor"}},
    "conversor_capitulos": {"window": {"title": "Miausoft Conversor de Capítulos"}},
    "reemplazos": {"window": {"title": "Miausoft Reemplazos de Caracteres"}},

    "fusionador": {
        "window": {"title": "Miausoft Fusionador"},
        "columns": {"min_px": {"archivo": phi_px_w(6, min_px=120), "modo": phi_px_w(9, min_px=70), "partes": phi_px_w(13, min_px=55)}},
        "controls": {"slide": {"notify_ms": int(PHI * 50)}},
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Merge / loader
def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def get_config(app: str | None = None) -> Dict[str, Any]:
    if app and app in APP_OVERRIDES:
        return deep_merge(BASE_CONFIG, APP_OVERRIDES[app])
    return copy.deepcopy(BASE_CONFIG)

# Alias por compatibilidad: módulos legados esperan CONFIG
CONFIG = get_config("conversor_capitulos")

def design_tokens(app: str | None = None, *, screen_w: int | None = None, screen_h: int | None = None) -> Dict[str, Any]:
    cfg = get_config(app)
    sh = int(screen_h or REF_SCREEN_H)

    def rel_px(rel, ref, *, px_fallback: int | None = None):
        min_px = int((cfg.get("fonts", {}) or {}).get("min_px", phi_px_h(60, min_px=8)))
        try:
            r = float(rel)
            if 0 < r <= 1:
                base = int(ref * r) if ref and ref > 0 else int(px_fallback or min_px)
                return max(min_px, base)
            return max(min_px, int(r))
        except Exception:
            return max(min_px, int(px_fallback or min_px))

    f = cfg.get("fonts", {}) or {}
    fam = f.get("family", f.get("fallback_family", "Segoe UI"))
    fonts_px = {
        "title": (fam, rel_px((f.get("title", {}) or {}).get("rel_px", 1/(PHI*28)), sh, px_fallback=(f.get("title", {}) or {}).get("px")),
                  bool((f.get("title", {}) or {}).get("bold", True)),
                  bool((f.get("title", {}) or {}).get("italic", False))),
        "head":  (fam, rel_px((f.get("head", {}) or {}).get("rel_px",  1/(PHI*34)), sh, px_fallback=(f.get("head", {}) or {}).get("px")),
                  bool((f.get("head", {}) or {}).get("bold", True)),
                  bool((f.get("head", {}) or {}).get("italic", False))),
        "row":   (fam, rel_px((f.get("row", {}) or {}).get("rel_px",   1/(PHI*40)), sh, px_fallback=(f.get("row", {}) or {}).get("px")),
                  bool((f.get("row", {}) or {}).get("bold", False)),
                  bool((f.get("row", {}) or {}).get("italic", False))),
        "btn":   (fam, rel_px((f.get("btn", {}) or {}).get("rel_px",   1/(PHI*40)), sh, px_fallback=(f.get("btn", {}) or {}).get("px")),
                  bool((f.get("btn", {}) or {}).get("bold", False)),
                  bool((f.get("btn", {}) or {}).get("italic", False))),
    }
    return {
        "fonts_px": fonts_px,
        "colors": (cfg.get("colors", {}) or {}).copy(),
        "layout": (cfg.get("layout", {}) or {}).copy(),
        "window": (cfg.get("window", {}) or {}).copy(),
        "win11":  (cfg.get("win11",  {}) or {}).copy(),
    }

def window_geometry(app: str | None = None, screen_w: int | None = None, screen_h: int | None = None) -> tuple[int, int]:
    cfg = get_config(app)
    sw = int(screen_w or REF_SCREEN_W)
    sh = int(screen_h or REF_SCREEN_H)
    win = cfg.get("window", {}) or {}
    if bool(win.get("use_relative", True)):
        ww = int(max(int(win.get("min_width", phi_px_w(2))),  sw * float(win.get("rel_width",  phi_rel(2)))))
        wh = int(max(int(win.get("min_height", phi_px_h(3))), sh * float(win.get("rel_height", phi_rel(3)))))
    else:
        ww = int(win.get("min_width", phi_px_w(2)))
        wh = int(win.get("min_height", phi_px_h(3)))
    return ww, wh

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades visuales
_DWMWA_SYSTEMBACKDROP_TYPE = 38
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMSBT_NONE = 0
_DWMSBT_MAINWINDOW = 2
_DWMSBT_TRANSIENT = 3
_DWMSBT_TABBED = 4

def system_accent(cfg: Optional[Dict[str, Any]] = None, default: str = "#0078d4") -> str:
    r"""Devuelve el color de acento del sistema.

    Prioridad:
      1) Registro (Explorer\\Accent\\AccentColorMenu) — coincide con el "resalte" real de Win11.
      2) DWM (DwmGetColorizationColor) — fallback.
      3) Registro DWM (DWM\\ColorizationColor) — último recurso.
    """
    cfg = cfg or BASE_CONFIG
    try:
        if not bool((cfg.get("win11", {}) or {}).get("use_accent_from_dwm", True)):
            return default
    except Exception:
        return default

    # 1) UI Accent (Explorer)
    try:
        import winreg  # type: ignore
        k = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Accent",
        )
        val, _ = winreg.QueryValueEx(k, "AccentColorMenu")
        r = val & 0xFF
        g = (val >> 8) & 0xFF
        b = (val >> 16) & 0xFF
        if (r, g, b) != (0, 0, 0):
            return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        pass

    # 2) DWM
    try:
        import ctypes
        from ctypes import wintypes, byref
        try:
            dwm = ctypes.WinDLL("dwmapi")
            # Prototipos para evitar llamadas inseguras
            try:
                dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
                dwm.DwmSetWindowAttribute.restype = wintypes.HRESULT
            except Exception:
                pass
        except Exception:
            dwm = None
        fn = getattr(dwm, "DwmGetColorizationColor", None) if dwm is not None else None
        if fn is None:
            raise RuntimeError("DwmGetColorizationColor no disponible")
        color = wintypes.DWORD()
        opaque = wintypes.BOOL()
        fn(byref(color), byref(opaque))
        c = color.value & 0x00FFFFFF
        r = (c >> 16) & 0xFF
        g = (c >> 8) & 0xFF
        b = c & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        pass

    # 3) Registro DWM
    try:
        import winreg  # type: ignore
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\DWM")
        val, _ = winreg.QueryValueEx(k, "ColorizationColor")
        r = (val >> 16) & 0xFF
        g = (val >> 8) & 0xFF
        b = val & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return default



def set_win11_backdrop(hwnd: int, mode: str = "mica", prefer_dark: bool = True) -> None:
    """Aplica Mica/Acrylic/Tabbed (Win11). No-op fuera de Windows.

    Nota: DWM + ctypes puede causar crashes nativos en algunas combinaciones de Windows/temas.
    Por máxima estabilidad, permite deshabilitar:
      - MIAUSOFT_DISABLE_DWM=1
    """
    try:
        import os as _os
        if str(_os.environ.get("MIAUSOFT_DISABLE_DWM", "")).strip() == "1":
            return
    except Exception:
        pass

    try:
        import ctypes
        from ctypes import wintypes, byref

        try:
            dwm = ctypes.WinDLL("dwmapi")
        except Exception:
            return

        fn_set = getattr(dwm, "DwmSetWindowAttribute", None)
        if fn_set is None:
            return

        # Prototipos (evita llamadas inseguras)
        try:
            fn_set.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            fn_set.restype = wintypes.HRESULT
        except Exception:
            pass

        # Dark mode (best-effort)
        pv = ctypes.c_int(1 if prefer_dark else 0)
        try:
            fn_set(
                wintypes.HWND(int(hwnd)),
                ctypes.c_uint(_DWMWA_USE_IMMERSIVE_DARK_MODE),
                byref(pv),
                ctypes.sizeof(pv),
            )
        except Exception:
            pass

        m = (mode or "none").strip().lower()
        if m == "mica":
            typ = ctypes.c_int(_DWMSBT_MAINWINDOW)
        elif m == "acrylic":
            typ = ctypes.c_int(_DWMSBT_TRANSIENT)
        elif m == "tabbed":
            typ = ctypes.c_int(_DWMSBT_TABBED)
        else:
            typ = ctypes.c_int(_DWMSBT_NONE)

        try:
            fn_set(
                wintypes.HWND(int(hwnd)),
                ctypes.c_uint(_DWMWA_SYSTEMBACKDROP_TYPE),
                byref(typ),
                ctypes.sizeof(typ),
            )
        except Exception:
            pass
    except Exception:
        return


def font_tuple(base_px: int, *, bold: bool = False, italic: bool = False, family: str = "Segoe UI", min_px: int = 8):
    style = []
    if bold:
        style.append("bold")
    if italic:
        style.append("italic")
    return (family, max(min_px, int(base_px)), " ".join(style) if style else "")

class Theme:
    """Tema ligero para Tk (opcional)."""
    def __init__(self, root):
        self.root = root
        self.accent = system_accent(get_config())
        h = int(root.winfo_screenheight() or REF_SCREEN_H)
        f = (get_config().get("fonts", {}) or {})
        min_px = int(f.get("min_px", phi_px_h(60, min_px=8)))
        fam = f.get("family", f.get("fallback_family", "Segoe UI"))
        self.fonts = {
            "title": font_tuple(int(h * (f.get("title", {}).get("rel_px", 1/(PHI*28)))), bold=True, family=fam, min_px=min_px),
            "sub":   font_tuple(int(h * (f.get("sub",   {}).get("rel_px", 1/(PHI*42)))), italic=True, family=fam, min_px=min_px),
            "text":  font_tuple(int(h * (f.get("row",   {}).get("rel_px", 1/(PHI*40)))), family=fam, min_px=min_px),
            "mono":  ("Consolas", max(min_px, int(h * (1/(PHI*36))))),
        }

    def apply(self, *, app: str | None = None, title: str | None = None, icon_name: str | None = None):
        cfg = get_config(app)
        if title:
            try:
                self.root.title(title)
            except Exception:
                pass

        # Ícono de ventana (barra de tareas / Alt-Tab). Todos los .exe llevan el ícono embebido,
        # pero Tk puede mostrar un ícono genérico si no se fija explícitamente.
        if icon_name:
            try:
                from pathlib import Path as _Path
                import sys as _sys
                import os as _os

                candidates = []
                try:
                    p = _Path(icon_name)
                    if p.is_absolute():
                        candidates.append(p)
                except Exception:
                    pass

                env_p = str(_os.environ.get('MIAUSOFT_ICON_PATH', '')).strip()
                if env_p:
                    try:
                        candidates.append(_Path(env_p))
                    except Exception:
                        pass

                # PyInstaller onefile: datos en _MEIPASS
                if getattr(_sys, 'frozen', False) and hasattr(_sys, '_MEIPASS'):
                    try:
                        candidates.append(_Path(getattr(_sys, '_MEIPASS')) / str(icon_name))
                    except Exception:
                        pass

                # En desarrollo (.py): junto al proyecto / cwd
                try:
                    candidates.append(_Path(__file__).resolve().parent / str(icon_name))
                except Exception:
                    pass
                try:
                    candidates.append(_Path.cwd() / str(icon_name))
                except Exception:
                    pass

                # Fallback configurable (evita rutas locales hardcodeadas)
                env_root = str(_os.environ.get('MIAUSOFT_PROJECT_ROOT', '')).strip()
                if env_root:
                    try:
                        candidates.append(_Path(env_root) / str(icon_name))
                    except Exception:
                        pass

                ico_path = None
                for c in candidates:
                    try:
                        if c and c.exists():
                            ico_path = c
                            break
                    except Exception:
                        continue

                if ico_path:
                    # iconbitmap (Windows) + iconphoto (ayuda cuando el icono del EXE no refleja en la ventana)
                    try:
                        self.root.iconbitmap(str(ico_path))
                    except Exception:
                        pass

                    try:
                        from PIL import Image, ImageTk  # type: ignore
                        img = Image.open(str(ico_path))
                        self._tk_icon = ImageTk.PhotoImage(img)
                        self.root.iconphoto(True, self._tk_icon)
                    except Exception:
                        pass
            except Exception:
                pass

        # Backdrop Win11 (opcional; se puede desactivar para máxima estabilidad)
        try:
            win11 = (cfg.get("win11", {}) or {})
            import os as _os
            if bool(win11.get("enable_backdrop", False)) or (str(_os.environ.get("MIAUSOFT_ENABLE_DWM", "")).strip() == "1"):
                set_win11_backdrop(
                    self.root.winfo_id(),
                    win11.get("backdrop", "acrylic"),
                    bool(win11.get("prefer_dark_mode", True)),
                )
        except Exception:
            pass

        # Fondo
        try:
            self.root.configure(bg=(cfg.get("colors", {}) or {}).get("app_bg", "#f3f3f3"))
        except Exception:
            pass

        # Tipografías globales (no rompe nada si falla)
        try:
            import tkinter.font as tkfont
            from tkinter import ttk

            f = cfg.get("fonts", {}) or {}
            fam = f.get("family", f.get("fallback_family", "Segoe UI"))
            fb_fam = f.get("fallback_family", "Segoe UI")
            min_px = int(f.get("min_px", phi_px_h(60, min_px=8)))

            # Último intento: registrar el .ttf local (Windows) para esta sesión/proceso
            try:
                ensure_font_registered(str(fam))
            except Exception:
                pass

            # Si la familia aún no existe en Tk, usa fallback
            try:
                fams = {str(x).lower() for x in tkfont.families(self.root)}
                if str(fam).lower() not in fams:
                    fam = fb_fam
            except Exception:
                pass

            def _rel(style: dict, fallback_rel: float):
                try:
                    r = float(style.get("rel_px", fallback_rel))
                except Exception:
                    r = float(fallback_rel)
                px_fb = style.get("px", None)
                hh = int(self.root.winfo_screenheight() or 0)
                if hh <= 0:
                    return max(min_px, int(px_fb or min_px))
                base = int(hh * r) if 0 < r <= 1 else int(r)
                if base <= 0 and px_fb is not None:
                    base = int(px_fb)
                return max(min_px, base)

            title_px = _rel((f.get("title", {}) or {}), 1/(PHI*28))
            row_px   = _rel((f.get("row", {}) or {}),   1/(PHI*40))
            btn_px   = _rel((f.get("btn", {}) or {}),   1/(PHI*40))

            title_f = tkfont.Font(family=fam, size=title_px, weight="bold")
            text_f  = tkfont.Font(family=fam, size=row_px)
            btn_f   = tkfont.Font(family=fam, size=btn_px)

            # Option database (Tk): cubre widgets tk "clásicos"
            self.root.option_add("*Font", text_f)
            self.root.option_add("*Label.Font", text_f)
            self.root.option_add("*Message.Font", text_f)
            self.root.option_add("*Button.Font", btn_f)
            self.root.option_add("*Menubutton.Font", btn_f)
            self.root.option_add("*Checkbutton.Font", text_f)
            self.root.option_add("*Radiobutton.Font", text_f)
            self.root.option_add("*Entry.Font", text_f)
            self.root.option_add("*Text.Font", text_f)
            self.root.option_add("*Listbox.Font", text_f)
            self.root.option_add("*Menu.Font", text_f)

            # Option database (TTK)
            self.root.option_add("*TLabel.Font", text_f)
            self.root.option_add("*TButton.Font", btn_f)
            self.root.option_add("*TMenubutton.Font", btn_f)
            self.root.option_add("*TEntry.Font", text_f)
            self.root.option_add("*Treeview.Font", text_f)
            self.root.option_add("*Treeview.Heading.Font", title_f)

            # Configuración explícita TTK (no rompe estilos específicos; solo defaults)
            try:
                st = ttk.Style(self.root)
                st.configure("TLabel", font=text_f)
                st.configure("TButton", font=btn_f)
                st.configure("TMenubutton", font=btn_f)
                st.configure("TEntry", font=text_f)
                st.configure("Treeview", font=text_f)
                st.configure("Treeview.Heading", font=title_f)
            except Exception:
                pass

            # Named fonts: cambia la tipografía “por defecto” de Tk (afecta menús, etc.)
            def _safe_named(name: str, *, size: int, weight: str = "normal", slant: str = "roman"):
                try:
                    nf = tkfont.nametofont(name)
                    nf.configure(family=fam, size=int(size), weight=weight, slant=slant)
                except Exception:
                    pass

            _safe_named("TkDefaultFont", size=row_px)
            _safe_named("TkTextFont", size=row_px)
            _safe_named("TkMenuFont", size=row_px)
            _safe_named("TkFixedFont", size=row_px)
            _safe_named("TkHeadingFont", size=title_px, weight="bold")
            _safe_named("TkCaptionFont", size=title_px, weight="bold")
            _safe_named("TkSmallCaptionFont", size=row_px)
            _safe_named("TkIconFont", size=row_px)
            _safe_named("TkTooltipFont", size=row_px)
        except Exception:
            pass


def attach_activation(root, *, app: str | None = None) -> None:
    """Corrige el comportamiento de Z-order: al restaurar/activar la ventana, sube al frente
    sin quedar siempre-on-top.

    pass  # update_idletasks deshabilitado por estabilidad (Tk 9 / Python 3.14)
    de foco/mapeo para prevenir re-entrancia que puede terminar en crash nativo en Windows.
    """
    cfg = get_config(app)
    win = (cfg.get("window") or {})
    keep_topmost = bool(win.get("topmost", False))

    _pending = {"flag": False}
    _cycling = {"flag": False}

    def _topmost_cycle():
        if keep_topmost:
            return
        if _cycling["flag"]:
            return
        try:
            if not root.winfo_exists():
                return
        except Exception:
            return

        _cycling["flag"] = True
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        def _unset():
            try:
                if root.winfo_exists():
                    root.attributes("-topmost", False)
            except Exception:
                pass
            _cycling["flag"] = False

        # Diferir el unset un tick; no forzar idle tasks aquí.
        try:
            root.after(1, _unset)
        except Exception:
            _unset()

    def _run_raise():
        _pending["flag"] = False
        try:
            if not root.winfo_exists():
                return
        except Exception:
            return

        try:
            if str(root.state()) == "iconic":
                return
        except Exception:
            pass

        try:
            root.lift()
        except Exception:
            pass

        if not keep_topmost:
            try:
                root.after_idle(_topmost_cycle)
            except Exception:
                _topmost_cycle()

    def _raise(_evt=None):
        if _pending["flag"]:
            return
        _pending["flag"] = True
        try:
            root.after_idle(_run_raise)
        except Exception:
            _run_raise()

    try:
        root.bind("<Map>", _raise, add="+")
        root.bind("<FocusIn>", _raise, add="+")
    except Exception:
        pass

    try:
        root.after_idle(_raise)
    except Exception:
        pass


def apply_theme(root, *, app: str | None = None, title: str | None = None, icon: str | None = None) -> Theme:
    th = Theme(root)
    th.apply(app=app, title=title, icon_name=icon)
    try:
        attach_activation(root, app=app)
    except Exception:
        pass
    return th


# ─────────────────────────────────────────
# Helpers compartidos (botones) — 100% gobernado desde config.py
def shared_button_box_px(win_w: int, win_h: int, cfg: Optional[Dict[str, Any]] = None):
    """
    Devuelve (bw, bh) en px para los botones principales, de manera idéntica
    para TODAS las apps (Fusionador / Reemplazos / Conversor de capítulos).
    Todo se gobierna por BASE_CONFIG["layout"].
    """
    c = cfg or (globals().get("CONFIG") or get_config(None))
    lay = (c.get("layout") or {})
    W = max(1, int(win_w))
    H = max(1, int(win_h))

    bw = int(W * float(lay.get("btn_w_rel_w", 1/(PHI*5))))
    bh = int(H * float(lay.get("btn_h_rel_h", 1/(PHI*20))))

    min_w = int(lay.get("btn_min_w_px", 1))
    min_h = int(lay.get("btn_min_h_px", 1))

    max_w = int(W * float(lay.get("btn_w_max_rel_w", 1/(PHI*4))))
    max_h = int(H * float(lay.get("btn_h_max_rel_h", 1/(PHI*15))))

    bw = max(min_w, min(bw, max_w))
    bh = max(min_h, min(bh, max_h))
    return int(bw), int(bh)

def shared_button_font_px(win_w: int, win_h: int, cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    Tamaño de fuente responsivo para botones principales (idéntico entre apps).
    Se basa en fonts.btn.rel_px y se limita para no exceder la altura útil del botón.
    """
    c = cfg or (globals().get("CONFIG") or get_config(None))
    f = (c.get("fonts") or {})
    btn = (f.get("btn") or {})
    rel = float(btn.get("rel_px", 1/(PHI*40)))
    min_px = int(f.get("min_px", phi_px_h(60, min_px=8)))

    px = max(min_px, int(max(1, win_h) * rel))
    bw, bh = shared_button_box_px(win_w, win_h, c)
    # margen interno φ-relativo para evitar texto “apretado”
    cap = max(min_px, int(bh * (1 - float((c.get("layout") or {}).get("btn_text_pad_rel_h", 1/(PHI*8))))))
    return int(max(min_px, min(px, cap)))

# ──────────────────────────────────────────────────────────────────────────────
# ProgressDialog (antes progress_ui.py) — integrado en config.py (sin perder API)
# ──────────────────────────────────────────────────────────────────────────────
import tkinter as tk
import subprocess
import importlib
import importlib.util

_PROBE_CACHE: dict[str, bool] = {}

def _probe_import(modname: str, *, timeout_s: float = 6.0) -> bool:
    if modname in _PROBE_CACHE:
        return _PROBE_CACHE[modname]
    try:
        if importlib.util.find_spec(modname) is None:
            _PROBE_CACHE[modname] = False
            return False
    except Exception:
        _PROBE_CACHE[modname] = False
        return False
    try:
        cmd = [sys.executable, "-c", f"import {modname}"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_s))
        ok = (p.returncode == 0)
    except Exception:
        ok = False
    _PROBE_CACHE[modname] = ok
    return ok

def _ensure_pil() -> bool:
    global _HAS_PIL, Image, ImageTk
    if _HAS_PIL is not None:
        return bool(_HAS_PIL)
    if not _probe_import("PIL"):
        _HAS_PIL = False
        Image = None  # type: ignore
        ImageTk = None  # type: ignore
        return False
    try:
        from PIL import Image as _Image, ImageTk as _ImageTk  # type: ignore
        Image = _Image  # type: ignore
        ImageTk = _ImageTk  # type: ignore
        _HAS_PIL = True
        return True
    except Exception:
        _HAS_PIL = False
        Image = None  # type: ignore
        ImageTk = None  # type: ignore
        return False

import tkinter.font as tkfont
from pathlib import Path
Image = None  # type: ignore
ImageTk = None  # type: ignore
_HAS_PIL: bool | None = None
import ctypes
from ctypes import wintypes, byref
PROGRESS_UI_CONFIG = get_config("progress")

# ======================= Win11 backdrop / accent =======================
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMSBT_MAINWINDOW = 2
DWMSBT_TRANSIENT  = 3
DWMSBT_TABBED     = 4
DWMSBT_NONE       = 0

def _accent_from_dwm() -> str:
    r"""
    Color de acento (W11):
      1) Registro (Explorer\\Accent\\AccentColorMenu) — coincide con el "resalte" real.
      2) DWM (DwmGetColorizationColor) — fallback.
      3) Azul por defecto.
    """
    # 1) Registro (UI accent). El DWORD suele venir como AABBGGRR (little-endian).
    try:
        import winreg  # type: ignore
        k = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Accent",
        )
        val, _ = winreg.QueryValueEx(k, "AccentColorMenu")
        r = val & 0xFF
        g = (val >> 8) & 0xFF
        b = (val >> 16) & 0xFF
        if (r, g, b) != (0, 0, 0):
            return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        pass

    # 2) DWM (colorization)
    try:
        import ctypes
        from ctypes import wintypes, byref
        try:
            dwm = ctypes.WinDLL("dwmapi")
        except Exception:
            dwm = None
        fn = getattr(dwm, "DwmGetColorizationColor", None) if dwm is not None else None
        if fn is None:
            raise RuntimeError("DwmGetColorizationColor no disponible")
        color = wintypes.DWORD()
        opaque = wintypes.BOOL()
        fn(byref(color), byref(opaque))
        c = color.value & 0x00FFFFFF
        r = (c >> 16) & 0xFF
        g = (c >> 8) & 0xFF
        b = c & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#0078d4"



def _wrap_ellipsis_px(text: str, font: tkfont.Font, max_width_px: int, max_lines: int = 2,
                     break_chars: str = "/\\_-.:") -> str:
    """Wrap por píxel (rompe también tokens largos) y recorta con elipsis para no desbordar."""
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s:
        return ""
    max_width_px = max(10, int(max_width_px or 0))
    max_lines = max(1, int(max_lines or 1))

    out: list[str] = []
    cur = ""
    last_break = -1
    truncated = False

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        i += 1

        if ch == "\n":
            out.append(cur.rstrip())
            cur = ""
            last_break = -1
            if len(out) >= max_lines:
                truncated = (i < n)
                break
            continue

        cur += ch
        if ch.isspace() or (ch in break_chars):
            last_break = len(cur) - 1

        if font.measure(cur) > max_width_px:
            if last_break >= 0 and last_break < (len(cur) - 1):
                out.append(cur[: last_break + 1].rstrip())
                cur = cur[last_break + 1 :].lstrip()
            else:
                # romper a nivel carácter
                out.append(cur[:-1].rstrip())
                cur = ch.lstrip()
            last_break = -1

            if len(out) >= max_lines:
                truncated = True
                cur = ""
                break

    if cur and len(out) < max_lines:
        out.append(cur.rstrip())

    if len(out) > max_lines:
        out = out[:max_lines]
        truncated = True

    if truncated and out:
        ell = "…"
        last = out[-1].rstrip()
        # quitar ya una elipsis existente
        if last.endswith(ell):
            return "\n".join(out)
        while last and font.measure(last + ell) > max_width_px:
            last = last[:-1].rstrip()
        if not last:
            out[-1] = ell
        else:
            out[-1] = last + ell

    return "\n".join(out)
def _font(family: str, fallback: str, size_px: float, *, bold=False, italic=False, min_px: int = 8):
    size = max(int(min_px), int(size_px))
    fam = family or fallback or "Segoe UI"
    style = []
    if bold: style.append("bold")
    if italic: style.append("italic")
    return (fam, size, " ".join(style) if style else "")

# ======================= Barra neumórfica =======================
class _NeumorphicProgress(tk.Canvas):
    """
    Barra de progreso "pill" (máxima redondez) dibujada con rectángulos + óvalos.

    Ventaja frente a polygon+smooth: las esquinas quedan realmente circulares
    (en Tkinter, create_polygon suaviza con splines y puede verse menos redondo).
    """
    def __init__(self, parent, width, height, track_bg, accent, radius=14):
        super().__init__(parent, width=width, height=height, bg=track_bg, highlightthickness=0, bd=0)
        self.width = int(width)
        self.height = int(height)
        self.radius = int(radius)
        self.accent = accent
        self.track_bg = track_bg
        self.progress_pct = 0.0

        # IDs: track
        self._t_left = None
        self._t_mid = None
        self._t_right = None

        # IDs: fill
        self._f_left = None
        self._f_mid = None
        self._f_right = None

        self._draw_track()
        self.set_progress(0.0)

    def _pill_r(self, w: int) -> int:
        """Radio efectivo: siempre tan redondo como permita el alto (pill completo)."""
        w = max(1, int(w))
        h2 = max(0, int(self.height // 2))
        # Si el alto lo permite, forzamos pill (r=h/2), pero no puede exceder w/2.
        return int(min(h2, w // 2))

    def _ensure_track_items(self):
        if self._t_left is not None:
            return
        # Track (tres piezas)
        self._t_left = self.create_oval(0, 0, 0, 0, fill=self.track_bg, outline="")
        self._t_mid = self.create_rectangle(0, 0, 0, 0, fill=self.track_bg, outline="")
        self._t_right = self.create_oval(0, 0, 0, 0, fill=self.track_bg, outline="")

        # Fill (tres piezas) — se pueden ocultar si el ancho es muy pequeño
        self._f_left = self.create_oval(0, 0, 0, 0, fill=self.accent, outline="")
        self._f_mid = self.create_rectangle(0, 0, 0, 0, fill=self.accent, outline="")
        self._f_right = self.create_oval(0, 0, 0, 0, fill=self.accent, outline="")

    def _draw_track(self):
        self.delete("all")
        self._t_left = self._t_mid = self._t_right = None
        self._f_left = self._f_mid = self._f_right = None
        self._ensure_track_items()
        self._layout_track()
        self._layout_fill(0)

    def _layout_track(self):
        w = max(1, int(self.width))
        h = max(1, int(self.height))
        r = self._pill_r(w)

        if r <= 0:
            # fallback rectangular
            self.coords(self._t_mid, 0, 0, w, h)
            self.itemconfigure(self._t_mid, state="normal")
            for it in (self._t_left, self._t_right):
                self.itemconfigure(it, state="hidden")
            return

        # Ojos: left [0..2r], right [w-2r..w], mid [r..w-r]
        self.coords(self._t_left, 0, 0, 2 * r, h)
        self.coords(self._t_right, w - 2 * r, 0, w, h)
        self.coords(self._t_mid, r, 0, w - r, h)

        for it in (self._t_left, self._t_mid, self._t_right):
            self.itemconfigure(it, state="normal")

    def _layout_fill(self, fill_w: int):
        w = max(1, int(self.width))
        h = max(1, int(self.height))
        fw = int(max(0, min(w, int(fill_w))))

        if fw <= 0:
            for it in (self._f_left, self._f_mid, self._f_right):
                self.itemconfigure(it, state="hidden")
            return

        r = self._pill_r(fw)
        if r <= 0:
            self.coords(self._f_mid, 0, 0, fw, h)
            self.itemconfigure(self._f_mid, state="normal")
            for it in (self._f_left, self._f_right):
                self.itemconfigure(it, state="hidden")
            return

        # Caso pequeño: fw <= 2r ⇒ un solo óvalo (elíptico) ocupa todo
        if fw <= 2 * r:
            self.coords(self._f_left, 0, 0, fw, h)
            self.itemconfigure(self._f_left, state="normal")
            for it in (self._f_mid, self._f_right):
                self.itemconfigure(it, state="hidden")
            return

        # Caso normal: pill 3 piezas
        self.coords(self._f_left, 0, 0, 2 * r, h)
        self.coords(self._f_right, fw - 2 * r, 0, fw, h)
        self.coords(self._f_mid, r, 0, fw - r, h)

        for it in (self._f_left, self._f_mid, self._f_right):
            self.itemconfigure(it, state="normal")

    def set_size(self, width, height, radius=None):
        self.width = int(width)
        self.height = int(height)
        if radius is not None:
            self.radius = int(radius)
        self.config(width=self.width, height=self.height)
        self._draw_track()
        self.set_progress(self.progress_pct)

    def set_progress(self, pct: float):
        try:
            pct = float(pct)
        except Exception:
            pct = 0.0
        pct = max(0.0, min(1.0, pct))
        self.progress_pct = pct
        fill_w = int(round(self.width * pct))
        self._layout_fill(fill_w)

class ProgressDialog(tk.Frame):
    """
    Overlay de progreso (sin ventanas hijas): se dibuja dentro del mismo `parent`.

    API:
      - set_title(text)
      - set_subtitle(text)
      - set_progress(pct in [0,1])
      - set_global_span(total_units:int)
      - set_done(done_units:int)
      - inc(delta:int=1)
      - show_counts_in_subtitle(enable:bool)
      - show()
      - close()
    """
    def __init__(self, parent, icon_search_dir: Path | None = None):
        cfg = PROGRESS_UI_CONFIG
        bg = cfg["colors"]["app_bg"]
        super().__init__(parent, bg=bg, bd=0, highlightthickness=0)

        self.parent = parent
        self._shown = False
        self._focus_prev = None

        # Contenedor interno
        self.host = tk.Frame(self, bg=bg, bd=0, highlightthickness=0)
        self.host.place(x=0, y=0, relwidth=1, relheight=1)

        # Cajas
        self.icon_box = tk.Frame(self.host, bg=bg, bd=0, highlightthickness=0)
        self.info_box = tk.Frame(self.host, bg=bg, bd=0, highlightthickness=0)
        self.icon_box.place(x=0, y=0)
        self.info_box.place(x=0, y=0)

        # Recursos UI
        app_dir = Path(icon_search_dir or Path(__file__).parent)
        self._ico_path = app_dir / cfg["window"]["icon_name"]
        self._logo_img = None
        self.lbl_icon  = tk.Label(self.icon_box, bg=bg, bd=0, highlightthickness=0)

        accent = _accent_from_dwm()
        self.progress = _NeumorphicProgress(
            self.info_box, 100, 10,
            cfg["colors"]["progress_track"],
            accent, radius=cfg["layout"]["bar_radius_px"]
        )

        self.lbl_main = tk.Label(self.info_box, text="", bg=bg, fg=cfg["colors"]["text"], bd=0)
        self.lbl_sub  = tk.Label(self.info_box, text="", bg=bg, fg=cfg["colors"]["subtext"], bd=0)

        # Texto crudo (para wrap/ellipsis sin perder el original)
        self._raw_title = ""
        self._raw_subtitle = ""
        self._font_title = None
        self._font_sub = None

        # Estado
        self._wrap_px = 300
        self._layout_pending = False
        self._max_title_lines = 2
        self._max_sub_lines = 3

        # Compat: algunas apps esperan atributos privados (_total_units/_done_units)
        # Mantener ambos nombres sincronizados.
        self._total_units = 1
        self._done_units  = 0

        self.global_total = 1
        self.done_units   = 0

        self.show_counts  = False
        # Reaccionar a resize del overlay
        self.bind("<Configure>", lambda e: self._schedule_layout())

        # Oculto al inicio
        self.place_forget()

    def _schedule_layout(self):
        """Debounce: evita recalcular layout en cada tick (páginas)."""
        if getattr(self, '_layout_pending', False):
            return
        self._layout_pending = True

        def _run():
            self._layout_pending = False
            try:
                self._apply_metrics()
            except Exception:
                return

        try:
            self.after_idle(_run)
        except Exception:
            self._layout_pending = False


    def _ensure_font(self, existing, spec):
        """Normaliza `spec` (tupla tipo Tk) a un `tkfont.Font` reutilizable.

        Evita fallos de `metrics()/measure()` cuando se guarda una tupla como `font`
        (puede terminar en alturas 0px => subtítulo invisible y layout inestable).
        """
        try:
            fam, sz, st = spec
        except Exception:
            fam, sz, st = ("Segoe UI", 10, "")
        fam = str(fam or "Segoe UI")
        try:
            sz_i = int(sz)
        except Exception:
            sz_i = 10
        st_l = str(st or "").lower()
        weight = "bold" if "bold" in st_l else "normal"
        slant = "italic" if "italic" in st_l else "roman"

        if isinstance(existing, tkfont.Font):
            try:
                existing.config(family=fam, size=sz_i, weight=weight, slant=slant)
                return existing
            except Exception:
                pass

        try:
            return tkfont.Font(family=fam, size=sz_i, weight=weight, slant=slant)
        except Exception:
            try:
                return tkfont.Font(font=self.lbl_main.cget("font"))
            except Exception:
                return tkfont.Font(size=sz_i)


    # ---- API ----
    def set_title(self, text: str):
        self._raw_title = str(text or "")
        self._render_text()
        self._schedule_layout()

    def set_subtitle(self, text: str):
        self._raw_subtitle = str(text or "")
        self._render_text()
        self._schedule_layout()

    def _display_subtitle(self) -> str:
        base = str(self._raw_subtitle or "")
        if self.show_counts and self.global_total:
            counts = f"{self.done_units}/{self.global_total}"
            return (f"{base}  ·  {counts}" if base.strip() else counts)
        return base

    def _render_text(self) -> None:
        """Aplica wrap/ellipsis a título/subtítulo con métricas actuales."""
        try:
            w = max(120, int(getattr(self, "_wrap_px", 300)))
            ft = self._font_title or tkfont.Font(font=self.lbl_main.cget("font"))
            fs = self._font_sub or tkfont.Font(font=self.lbl_sub.cget("font"))
            self.lbl_main.config(text=_wrap_ellipsis_px(self._raw_title, ft, w, max_lines=max(1, int(getattr(self, '_max_title_lines', 2)))))
            _sub = self._display_subtitle()
            ms = max(0, int(getattr(self, '_max_sub_lines', 3)))
            if ms <= 0 or not (_sub or '').strip():
                self.lbl_sub.config(text='')
            else:
                self.lbl_sub.config(text=_wrap_ellipsis_px(_sub, fs, w, max_lines=ms))
        except Exception:
            # fallback sin romper
            try:
                self.lbl_main.config(text=str(self._raw_title or ""))
                self.lbl_sub.config(text=str(self._display_subtitle() or ""))
            except Exception:
                pass

    def set_progress(self, pct: float):
        self.progress.set_progress(pct)

    def set_global_span(self, total_units: int):
        total = max(1, int(total_units))
        self.global_total = total
        self._total_units = total
        self.done_units = 0
        self._done_units = 0
        # mantener barra consistente
        self.set_progress(0.0)
        self._update_counts()

    def set_done(self, done_units: int):
        done = max(0, int(done_units))
        total = int(getattr(self, "global_total", 1) or 1)
        total = max(1, total)

        self.done_units = min(done, total)
        self._done_units = self.done_units

        # Mantener compat con _total_units
        self._total_units = total

        self.set_progress(min(1.0, self.done_units / float(total)))
        self._update_counts()

    def inc(self, delta: int = 1):
        self.set_done(self.done_units + int(delta))

    def show_counts_in_subtitle(self, enable: bool):
        self.show_counts = bool(enable)
        self._update_counts()

    def show(self):
        if self._shown:
            self.lift()
            return
        try:
            self._focus_prev = self.parent.focus_get()
        except Exception:
            self._focus_prev = None

        self.place(x=0, y=0, relwidth=1, relheight=1)
        self.lift()
        self._shown = True
        self._schedule_layout()

    def close(self):
        if not self._shown:
            return
        self.place_forget()
        self._shown = False
        try:
            if self._focus_prev and self._focus_prev.winfo_exists():
                self._focus_prev.focus_set()
        except Exception:
            pass

    # ---- Internos ----
    def _update_counts(self):
        # Renderiza siempre (activar/desactivar + cambios de done/total) sin desbordes
        self._render_text()
        self._schedule_layout()


    def _apply_metrics(self):
        if not self.winfo_exists():
            return

        cfg = PROGRESS_UI_CONFIG
        lay = cfg["layout"]

        W  = max(self.winfo_width(),  cfg["window"]["min_width"])
        H  = max(self.winfo_height(), cfg["window"]["min_height"])

        # Cajas: ícono a la izquierda con ancho 1/(PHI*1)
        icon_w = max(1, int(W * float(lay["icon_box_width_rel_w"])))
        info_w = max(1, W - icon_w)
        self.icon_box.place(x=0,      y=0, width=icon_w, height=H)
        self.info_box.place(x=icon_w, y=0, width=info_w, height=H)

        # Padding global (muy pequeño para resultar 0px normalmente)
        pad_px = max(int(H * float(lay["boxes_pad_rel_h"])), int(lay["boxes_pad_min_px"]))
        icon_pad_left = icon_pad_right = icon_pad_top = icon_pad_bottom = pad_px
        info_pad_left = info_pad_right = info_pad_top = info_pad_bottom = pad_px

        inner_icon_w = max(1, icon_w - (icon_pad_left + icon_pad_right))
        inner_icon_h = max(1, H      - (icon_pad_top  + icon_pad_bottom))
        inner_w      = max(1, info_w - (info_pad_left + info_pad_right))
        inner_h      = max(1, H      - (info_pad_top  + info_pad_bottom))

        # Ícono: ocupa 100% de alto (sin padding efectivo) dentro de su caja.
        icon_side = max(1, min(inner_icon_w, inner_icon_h))
        if self._ico_path.exists():
            # Preferir Pillow (calidad + escalado fino). Si no está disponible, usar Tk PhotoImage (fallback).
            try:
                if _HAS_PIL and Image is not None and ImageTk is not None:
                    img = Image.open(self._ico_path).resize((icon_side, icon_side), getattr(getattr(Image, "Resampling", Image), "LANCZOS"))
                    self._logo_img = ImageTk.PhotoImage(img)
                else:
                    self._logo_img = tk.PhotoImage(file=str(self._ico_path))
                    # downscale entero para que quepa; evita zoom (no inventa pixeles)
                    try:
                        w0, h0 = int(self._logo_img.width()), int(self._logo_img.height())
                        if w0 > icon_side or h0 > icon_side:
                            sx = max(1, w0 // icon_side)
                            sy = max(1, h0 // icon_side)
                            self._logo_img = self._logo_img.subsample(sx, sy)
                    except Exception:
                        pass
                self.lbl_icon.config(image=self._logo_img)
            except Exception:
                self.lbl_icon.config(image="")
        else:
            self.lbl_icon.config(image="")

        cx = icon_pad_left + inner_icon_w // 2
        cy = icon_pad_top  + inner_icon_h // 2
        self.lbl_icon.place(x=cx, y=cy, anchor="center")

        # Wrap de texto
        try:
            self._wrap_px = max(120, int(inner_w))
            self.lbl_main.config(wraplength=self._wrap_px, justify="left", anchor="nw")
            self.lbl_sub.config(wraplength=self._wrap_px, justify="left", anchor="nw")
        except Exception:
            pass

        # Fuentes (responsivas al alto)
        fam = cfg["fonts"]["family"]
        fallback = cfg["fonts"]["fallback_family"]
        min_px = cfg["fonts"]["min_px"]
        tcfg = cfg["fonts"]["title"]
        scfg = cfg["fonts"]["sub"]
        f_title = _font(fam, fallback, H * float(tcfg["rel_px"]), bold=tcfg["bold"], italic=tcfg["italic"], min_px=min_px)
        f_sub   = _font(fam, fallback, H * float(scfg["rel_px"]), bold=scfg["bold"], italic=scfg["italic"], min_px=min_px)

        # Normalizar a tkfont.Font para que existan metrics()/measure() (wrap por píxel + alturas estables)
        self._font_title = self._ensure_font(self._font_title, f_title)
        self._font_sub   = self._ensure_font(self._font_sub,   f_sub)
        self.lbl_main.config(font=self._font_title)
        self.lbl_sub.config(font=self._font_sub)

        # Barra: ancho φ-relativo al ancho total y clamp al panel derecho; alineada a la izquierda
        bar_w_target = int(W * float(lay.get("bar_w_rel_w", 1/(PHI*1))))
        bar_w = max(10, min(inner_w, bar_w_target))
        bar_h = max(int(lay["bar_min_h_px"]), int(H * float(lay["bar_h_rel_h"])))
        bar_h = min(bar_h, inner_h)

        self.progress.set_size(bar_w, bar_h, radius=lay["bar_radius_px"])

        y_bar  = info_pad_top + int(float(lay["bar_top_rel_inner"]) * inner_h)
        y_gap1 =              int(float(lay["gap_bar_title_rel_inner"]) * inner_h)
        y_gap2 =              int(float(lay["gap_title_sub_rel_inner"]) * inner_h)

        # Barra dentro de la caja derecha: alineada de izquierda a derecha (respeta padding)
        self.progress.place(x=info_pad_left, y=y_bar)  # dentro de info_box

        y_title = y_bar + self.progress.height + y_gap1
        # Texto: layout fijo para evitar jitter y GARANTIZAR el subtítulo debajo del título.
        sub_present = bool((self._display_subtitle() or "").strip())

        # 1 línea de título (suficiente para 'Procesando…') y 1 línea de subtítulo si existe.
        self._max_title_lines = 1
        self._max_sub_lines = 1 if sub_present else 0
        self._render_text()

        def _linespace(font_obj, default_px: int) -> int:
            try:
                ls = int(font_obj.metrics("linespace") or 0)
            except Exception:
                ls = 0
            if ls >= 8:
                return ls
            # Fallback robusto cuando Tk aún no tiene métricas (Tk 9 / Py 3.14):
            try:
                sz = int(font_obj.cget("size") or 0)
            except Exception:
                sz = 0
            sz = abs(int(sz or 10))
            return max(8, int(default_px or 0) or int(sz * 1.4) or 12)

        try:
            ft = self._font_title or tkfont.Font(font=self.lbl_main.cget("font"))
            fs = self._font_sub or tkfont.Font(font=self.lbl_sub.cget("font"))
        except Exception:
            ft = fs = None

        lh_t = _linespace(ft, 18) if ft is not None else 18
        lh_s = _linespace(fs, 14) if fs is not None else 14

        title_h = int(self._max_title_lines * lh_t)
        sub_h = int(self._max_sub_lines * lh_s)

        # y del subtítulo: siempre la línea inmediata debajo del título (con gap configurable).
        y_sub = y_title + title_h + (y_gap2 if sub_present else 0)

        # Anti-recorte (sin reflow): si el bloque se sale, subirlo lo máximo posible.
        try:
            bottom_limit = int(info_pad_top + inner_h)
            block_bottom = max(int(y_bar + int(getattr(self.progress, 'height', 0) or 0)),
                               int(y_sub + sub_h) if sub_h > 0 else int(y_title + title_h))
            if block_bottom > bottom_limit:
                overshoot = int(block_bottom - bottom_limit)
                avail_up = max(0, int(y_bar - info_pad_top))
                shift = min(overshoot, avail_up)
                if shift > 0:
                    y_bar -= shift
                    y_title -= shift
                    y_sub -= shift
                    try:
                        self.progress.place_configure(y=y_bar)
                    except Exception:
                        try:
                            self.progress.place(x=info_pad_left, y=y_bar)
                        except Exception:
                            pass
        except Exception:
            pass


        # Title
        self.lbl_main.place(x=info_pad_left, y=y_title, width=inner_w, height=int(title_h))

        # Subtitle (justo debajo del título)
        if sub_h <= 0:
            try:
                self.lbl_sub.place_forget()
            except Exception:
                pass
        else:
            self.lbl_sub.place(x=info_pad_left, y=y_sub, width=inner_w, height=int(sub_h))

# Pre-registro del font preferido (Windows) para que Tk lo vea aunque no esté instalado globalmente.
try:
    _pref = ((BASE_CONFIG.get("fonts") or {}).get("family") or "").strip()
    if _pref:
        ensure_font_registered(_pref)
except Exception:
    pass
