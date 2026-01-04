#reemplazacaracteresentxt.py
from __future__ import annotations

import re
import sys
import threading
import unicodedata
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

# =============================================================================
# Normalización de rutas: en algunos flujos (multi-selección de Explorer) las rutas
# pueden llegar como str; el resto del pipeline asume Path.
def _coerce_path(p) -> Path:
    s = str(p) if p is not None else ""
    try:
        s = os.path.expandvars(s)
    except Exception:
        pass
    try:
        pp = Path(s).expanduser()
    except Exception:
        pp = Path(s)
    # Solo resolver cuando existe (evita fallos en rutas aún no creadas / permisos)
    try:
        if pp.exists():
            try:
                pp = pp.resolve()
            except Exception:
                pass
    except Exception:
        pass
    return pp

# =============================================================================
# Bytes util para geométricos (idéntico comportamiento al original)
def _strip_geom_bytes(raw: bytes) -> tuple[bytes, int]:
    """Remove UTF-8 byte sequences for Box Drawing U+2500..U+257F,
    Block Elements U+2580..U+259F, and Geometric Shapes U+25A0..U+25FF
    without decoding the file. Returns (filtered_bytes, removed_count)."""
    out = bytearray()
    i = 0
    removed = 0
    n = len(raw)
    while i < n:
        b0 = raw[i]
        if i + 2 < n and b0 == 0xE2:
            b1 = raw[i + 1]
            b2 = raw[i + 2]
            # U+2500..U+257F => E2 94 80..BF and E2 95 80..BF
            if (b1 in (0x94, 0x95)) and (0x80 <= b2 <= 0xBF):
                removed += 1
                i += 3
                continue
            # U+2580..U+25FF => E2 96 80..BF and E2 97 80..BF
            if (b1 in (0x96, 0x97)) and (0x80 <= b2 <= 0xBF):
                removed += 1
                i += 3
                continue
        out.append(b0)
        i += 1
    return bytes(out), removed

# =============================================================================
# Config centralizado (sin fallbacks locales)
from config import get_config, window_geometry, PHI, apply_theme, shared_button_box_px, shared_button_font_px, ProgressDialog
CONFIG = get_config("reemplazos")
# =============================================================================
# Catálogo (solo en menú contextual, por categorías, sin ventanas extra)

def _unique_chars_no_ws(s: str) -> List[str]:
    """Dedup y NUNCA incluir whitespace (clave para Puntuación)."""
    out: List[str] = []
    seen = set()
    for ch in s:
        if not ch:
            continue
        if ch.isspace():
            continue
        if ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return out

PUNCT_CHARS = _unique_chars_no_ws(
    "§-℃©|–℉®¦—=№™℗<>⁅⁆[]‹›⌈⌉{}«»⌊⌋⟨⟩⟦⟧`´⟪⟫⟬⟭‸¸⟮⟯‖‗¨¯′‘’‛″‶“”„‟‴‷⁗*†‡•¶⁋⁎⁕‰‣⁌⁍⁑⁜‱♪⁽⁾⁂※;·․⁞⁛¬⁏‥⁚⁖⁙✓:…⁝⁘_✕‼°⁔⁀?¿⁈ª‿⁐‽⸘⁇º⁁⁄–⏑±µ⁓⌀⏓⏒‾^~⌁⏖⏔⏕⁊⁒⌂"
)
GEOM_CHARS = _unique_chars_no_ws(
    "←→↖↗↜↟↑↔↙↘↝↡↓↕↚↛↞↠↢↣↨↭↮↯↤↦↩↪↰↱↥↧↫↬↲↳↴↺↸↹⇄⇅↵↻↼⇀↿↾↶↷↽⇁⇃⇂⇆⇋⇌⇎⇑⇖⇇⇉⇍⇏⇓⇙⇈⇊⇐⇒⇔⇕⇗⇜⇞⇡⇤⇥⇘⇝⇟⇣⇦⇨⇚⇛⇠⇢⇧⇩⇪⇭⇰⇳⇹⇼⇫⇮⇱⇵⇷⇸⇬⇯⇲⇶⇺⇻⇽▢▥▨▬▭⇾▣▦▩▮▯⇿▤▧◊▪▫▰▱◀▶◁▷▲△◄►◅▻▴▵◂▸◃▹◆◇◈◌●◙▼▽○◍◓◚▾▿◉◎◒◛◖◗◜◝◤◥◐◑◟◞◣◢◕◔◠◡◧◨◲◱◶◵◸◹◳◰◷◴◺◿◫⊟◯◬◭◮⌜⌝⌏⌎◘⌞⌟⌍⌌◩■◻■□◪"
)
CURRENCY_CHARS = _unique_chars_no_ws(
    "$¥⃁₫₩֏€₹₺₪₡₦£₽₢₱฿₲₮₸₾₥៛₨₴₵₭؋૱¢﷼₣₿₼௹¤⃀₻ƒ₧৲₰৳₯₤₷৹₳₠ℳ₶৻"
)
OTHER_CHARS = _unique_chars_no_ws(
    "⨀⨃⨆⨉⨌⨏⨁⨄⨇⨊⨍⨐⨂⨅⨈⨋⨎⨑⨒⨕⨘⨛⨞⨡⨓⨖⨙⨜⨟⨢⨔⨗⨚⨝⨠⨣⨤⨧⨪⨭⨰⨳⨥⨨⨫⨮⨱⨴⨦⨩⨬⨯⨲⨵⨶⨹⨼⨿⩂⩅⨷⨺⨽⩀⩃⩆⨸⨻⨾⩁⩄⩇⩈⩋⩎⩑⩔⩗⩉⩌⩏⩒⩕⩘⩊⩍⩐⩓⩖⩙⩚⩝⩠⩣⩦⩩⩛⩞⩡⩤⩧⩪⩜⩟⩢⩥⩨⩫⩬⩯⩲⩵⩸⩻⩭⩰⩳⩶⩹⩼⩮⩱⩴⩷⩺⩽⩾⪁⪄⪇⪊⪍⩿⪂⪅⪈⪋⪎⪀⪃⪆⪉⪌⪏⪐⪓⪖⪙⪜⪟⪑⪔⪗⪚⪝⪠⪒⪕⪘⪛⪞⪡⪢⪥⪨⪫⪮⪱⪣⪦⪩⪬⪯⪲⪤⪧⪪⪭⪰⪳⪴⪷⪺⪽⫀⫃⪵⪸⪻⪾⫁⫄⪶⪹⪼⪿⫂⫅⫆⫉⫌⫏⫒⫕⫇⫊⫍⫐⫓⫖⫈⫋⫎⫑⫔⫗⫘⫛⫞⫡⫤⫧⫙⫝̸⫟⫢⫥⫨⫚⫝⫠⫣⫦⫩⫪⫭⫰⫳⫶⫹⫫⫮⫱⫴⫷⫺⫬⫯⫲⫵⫸⫻"
)
GREEK_CHARS = _unique_chars_no_ws(
    "ΑαΔδΗηΒβΕεΘθΓγΖζΙιΚκΝνΠπΛλξΞξΡρΜμΟοΣςΤτχΧσΥυΨψΦφΩω"
)

SYMBOL_CATEGORIES: Dict[str, List[str]] = {
    "Puntuación": PUNCT_CHARS,
    "Geométricos": GEOM_CHARS,
    "Moneda": CURRENCY_CHARS,
    "Otros": OTHER_CHARS,
    "Letras griegas": GREEK_CHARS,
}

def _symbol_name_natural(ch: str) -> str:
    nm = unicodedata.name(ch, "")
    if not nm:
        return f"U+{ord(ch):04X}"
    return nm.replace("_", " ").title()

# =============================================================================
# Utilidades visuales
def _font(family, size_px, *, bold=False, italic=False):
    style_bits = []
    if bold: style_bits.append("bold")
    if italic: style_bits.append("italic")
    return (family, max(int(CONFIG["fonts"]["min_px"]), int(size_px)), " ".join(style_bits) if style_bits else "")

def _metrics(win_w: int, win_h: int):
    def rw(f): return max(1, int(win_w * float(f)))
    def rh(f): return max(1, int(win_h * float(f)))
    return rw, rh

class FlatButton(tk.Canvas):
    """
    Botón plano (Canvas) con neumorfismo ligero.
    - Sin fallbacks locales: todo sale de CONFIG / PHI.
    - Auto-fit de fuente por ancho/alto.
    - Permite "shared font" (para que todos los botones usen el tamaño más pequeño requerido).
    """
    def __init__(self, parent, text, command=None, bg=None,
                 width=120, height=28, radius=None, font=None, colors=None, **kw):
        super().__init__(parent, width=int(width), height=int(height), bd=0,
                         highlightthickness=0, bg=bg or parent.cget("bg"), **kw)
        self._wpx, self._hpx = int(width), int(height)
        self._r = int(radius if radius is not None else int(CONFIG["layout"]["btn_radius_px"]))

        self._text = str(text)
        self._cmd = command
        self._bg = bg or parent.cget("bg")
        self._hover = False
        self._pressed = False

        self._base_font = font or _font(
            CONFIG["fonts"]["family"],
            int(CONFIG["fonts"]["min_px"]),
            bold=bool(CONFIG["fonts"]["btn"]["bold"]),
            italic=bool(CONFIG["fonts"]["btn"]["italic"]),
        )
        self._shared_font_obj: tkfont.Font | None = None
        self._fit_font_obj: tkfont.Font | None = None
        self._fit_cache: tuple[int, int, str, tuple] | None = None

        c = CONFIG["colors"]
        self._colors = colors or {
            "text": c["text"],
            "base": c["btn_base"],
            "face": c["btn_face"],
            "face_hover": c["btn_face_hover"],
            "face_down": c["btn_face_down"],
        }

        self.bind("<Enter>",  lambda e: self._set_hover(True))
        self.bind("<Leave>",  lambda e: self._set_hover(False))
        self.bind("<ButtonPress-1>",  lambda e: self._set_pressed(True))
        self.bind("<ButtonRelease-1>", self._on_click)
        self.bind("<Configure>", lambda e: self.after_idle(self._redraw))
        self.after_idle(self._redraw)

    def _style_from_tuple(self, ft: tuple) -> tuple[str, int, str, str]:
        fam = str(ft[0]) if len(ft) > 0 else str(CONFIG["fonts"]["family"])
        size = int(ft[1]) if len(ft) > 1 else int(CONFIG["fonts"]["min_px"])
        style = str(ft[2]) if len(ft) > 2 else ""
        weight = "bold" if "bold" in style else "normal"
        slant = "italic" if "italic" in style else "roman"
        return fam, size, weight, slant

    def set_shared_font(self, font_obj: tkfont.Font | None):
        self._shared_font_obj = font_obj
        self.after_idle(self._redraw)

    def best_fit_size(self) -> int:
        fam, base_size, weight, slant = self._style_from_tuple(self._base_font)
        min_px = int(CONFIG["fonts"]["min_px"])
        w = max(1, int(self._wpx))
        h = max(1, int(self._hpx))

        # padding φ (no hardcoded px)
        pad_x = max(int(4 * PHI), int(h / (PHI * 2.4)))
        pad_y = max(int(3 * PHI), int(h / (PHI * 3.0)))
        avail_w = max(1, w - 2 * pad_x)
        avail_h = max(1, h - 2 * pad_y)

        lo = min_px
        hi = max(min_px, int(base_size))
        best = lo

        # binary search
        while lo <= hi:
            mid = (lo + hi) // 2
            f = tkfont.Font(family=fam, size=mid, weight=weight, slant=slant)
            tw = int(f.measure(self._text))
            th = int(f.metrics("linespace"))
            if tw <= avail_w and th <= avail_h:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return int(best)

    def _fit_font(self) -> tkfont.Font:
        w = max(1, int(self._wpx))
        h = max(1, int(self._hpx))
        cache_key = (w, h, self._text, self._base_font)
        if self._fit_cache == cache_key and self._fit_font_obj is not None:
            return self._fit_font_obj

        fam, base_size, weight, slant = self._style_from_tuple(self._base_font)
        size = self.best_fit_size()
        self._fit_font_obj = tkfont.Font(family=fam, size=int(size), weight=weight, slant=slant)
        self._fit_cache = cache_key
        return self._fit_font_obj

    def config(self, **kw):
        # no tocar self._w (interno de Tk). Siempre _wpx/_hpx.
        if "width" in kw:
            self._wpx = int(kw["width"])
            super().config(width=self._wpx)
            kw.pop("width", None)
        if "height" in kw:
            self._hpx = int(kw["height"])
            super().config(height=self._hpx)
            kw.pop("height", None)
        if "font" in kw:
            self._base_font = kw["font"]
            self._fit_cache = None
            kw.pop("font", None)
        if "text" in kw:
            self._text = str(kw["text"])
            self._fit_cache = None
            kw.pop("text", None)
        if kw:
            super().config(**kw)
        self.after_idle(self._redraw)

    def _set_hover(self, v: bool):
        self._hover = bool(v)
        self.after_idle(self._redraw)

    def _set_pressed(self, v: bool):
        self._pressed = bool(v)
        self.after_idle(self._redraw)

    def _on_click(self, _event):
        self._set_pressed(False)
        if callable(self._cmd):
            try:
                self._cmd()
            except Exception:
                pass

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        r = int(max(0, r))
        if r <= 0:
            return self.create_rectangle(x1, y1, x2, y2, **kw)
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
               x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
               x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _redraw(self):
        self.delete("all")
        w = max(1, int(self._wpx))
        h = max(1, int(self._hpx))
        r = int(CONFIG["layout"]["btn_radius_px"])

        base = self._colors["base"]
        face = self._colors["face_down"] if self._pressed else (
            self._colors["face_hover"] if self._hover else self._colors["face"]
        )

        self._rounded_rect(0, 0, w, h, r, fill=base, outline="")
        self._rounded_rect(0, 0, w, h, r, fill=face, outline="")

        fnt = self._shared_font_obj if self._shared_font_obj is not None else self._fit_font()
        self.create_text(w // 2, h // 2, text=self._text, anchor="center",
                         fill=self._colors["text"], font=fnt)

# =============================================================================
# Presets
PRESET_EXT = ".miau"

def _user_documents_dir() -> Path:
    """Best-effort para obtener 'Documentos' del usuario (Windows/macOS/Linux)."""
    home = Path.home()

    # Windows: intenta USERPROFILE/Documents
    up = os.environ.get("USERPROFILE")
    if up:
        cand = Path(up) / "Documents"
        if cand.exists():
            return cand

    # macOS / Linux típicamente usan ~/Documents
    cand = home / "Documents"
    return cand

# =============================================================================
# Tabla de Reglas: Buscar | Regex | Reemplazar
class RulesTable(tk.Frame):
    COL_PATTERN = "pat"
    COL_REGEX   = "rx"
    COL_REPL    = "rep"

    def __init__(self, parent, fonts):
        super().__init__(parent, bg=CONFIG["colors"]["app_bg"], bd=0, highlightthickness=0)
        c = CONFIG["colors"]; lay = CONFIG["layout"]

        style = ttk.Style(self)
        try: style.theme_use('clam')
        except Exception: pass

        rowheight = max(int(lay["row_min_px"]), int(self.winfo_toplevel().winfo_height() * float(lay["row_height_rel"])))

        
        rowheight = min(int(lay.get("row_max_px", 50)), int(rowheight))

        style.configure("Miau.Treeview",
                        background=c["box_bg"],
                        fieldbackground=c["box_bg"],
                        foreground=c["text"],
                        borderwidth=0,
                        relief="flat",
                        font=fonts["row"],
                        rowheight=rowheight)
        style.layout("Miau.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.configure("Miau.Treeview.Heading",
                        background=c["app_bg"], foreground=c["text"],
                        font=fonts["head"], relief="flat", borderwidth=0)
        style.map("Miau.Treeview",
                  background=[("selected", c["box_bg"])],
                  foreground=[("selected", c["text"])])

        style.configure("Flat.Vertical.TScrollbar",
                        gripcount=0, background=c["app_bg"],
                        bordercolor=c["app_bg"], darkcolor=c["app_bg"], lightcolor=c["app_bg"],
                        troughcolor=c["scroll_trough"], arrowcolor=c["scroll_arrow"],
                        relief="flat", borderwidth=0)

        self.tree = ttk.Treeview(self, columns=(self.COL_PATTERN, self.COL_REGEX, self.COL_REPL), show="headings",
                                 style="Miau.Treeview", selectmode="extended")
        self.vs = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview,
                                style="Flat.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=self.vs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vs.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1); self.columnconfigure(0, weight=1)

        self.tree.heading(self.COL_PATTERN, text="Buscar")
        self.tree.heading(self.COL_REGEX,   text="Regex")
        self.tree.heading(self.COL_REPL,    text="Reemplazar por")

        self._fonts = fonts
        self._fonts_row = fonts["row"]

        self._apply_columns()

        # Estado por fila
        self._rx_state: dict[str, bool] = {}
        self._rx_locked: set[str] = set()

        # Modo por fila:
        # - "literal": reemplazo de substring literal
        # - "set":     eliminar cualquiera de los caracteres del patrón (solo usado por "Eliminar todos")
        self._rule_mode: dict[str, str] = {}

        self._editor: Optional[tk.Entry] = None
        self._edit_iid: Optional[str] = None
        self._edit_col: Optional[str] = None

        self._last_popup_xy: Optional[tuple[int, int]] = None

        self._auto_rows: dict[str, list[str]] = {}  # clave -> [iids]

        self._menu_pattern = self._build_token_menu(allow_batch=True)
        self._menu_repl = self._build_token_menu(allow_batch=False)

        self.tree.bind("<Double-1>", self._begin_edit)
        self.tree.bind("<Button-3>", self._right_click)
        self.tree.bind("<Delete>",   lambda e: self.remove_selected())
        self.tree.bind("<Button-1>", self._on_click_toggle)

        # Zebra
        self.tree.tag_configure("odd", background=self._alpha(c["box_bg"], lay.get("zebra_alpha", 0.06)))
        self.tree.tag_configure("even", background=c["box_bg"])

        # adder row always at end
        self._adder_iid: Optional[str] = None
        self._insert_adder_row()

        self.bind("<Configure>", lambda e: self.after_idle(self._apply_columns))

    # ---- helpers ----
    def _alpha(self, hex_color: str, alpha: float) -> str:
        try:
            hc = hex_color.lstrip("#")
            r = int(hc[0:2], 16)/255.0; g = int(hc[2:4], 16)/255.0; b = int(hc[4:6], 16)/255.0
            r = r + (1.0 - r) * alpha; g = g + (1.0 - g) * alpha; b = b + (1.0 - b) * alpha
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        except Exception:
            return hex_color

    def _apply_columns(self):
        lay = CONFIG["layout"]
        cols_cfg = CONFIG["replacements"]["columns"]
        w_raw  = cols_cfg["weights_raw"]
        min_px = cols_cfg["min_px"]

        w_pat = float(w_raw["pattern"])
        w_rep = float(w_raw["replace"])

        min_pat = int(min_px["pattern"])
        min_rep = int(min_px["replace"])
        min_rx  = int(min_px["regex"])

        total_w = max(int(CONFIG["window"]["min_width"]), int(self.winfo_width() or 0))
        sb  = int(lay["scrollbar_fallback_w_px"])
        pad = int(lay["table_pad_px"])
        usable = max(1, total_w - sb - 2 * pad)

        # reserva franja para "Regex"
        rx_rel = float(cols_cfg["regex_rel_w"])
        rx_w = max(min_rx, int(usable * rx_rel))
        remain = max(1, usable - rx_w)

        w1 = int(remain * w_pat)
        w2 = remain - w1

        w1 = max(min_pat, w1)
        w2 = max(min_rep, w2)

        over = (w1 + w2) - remain
        if over > 0:
            reducible1 = w1 - min_pat
            take1 = min(reducible1, over)
            w1 -= take1
            over -= take1

            reducible2 = w2 - min_rep
            take2 = min(reducible2, over)
            w2 -= take2
            over -= take2

        if w1 + w2 > remain:
            w2 = max(min_rep, remain - w1)
        if w1 + w2 > remain:
            w1 = max(min_pat, remain - w2)

        self.tree.column(self.COL_PATTERN, width=w1, minwidth=min_pat, anchor="w", stretch=True)
        self.tree.column(self.COL_REGEX,   width=rx_w, minwidth=min_rx, anchor="center", stretch=False)
        self.tree.column(self.COL_REPL,    width=w2, minwidth=min_rep, anchor="w", stretch=True)
    def refresh_style_metrics(self, rowheight_px: int, head_font, row_font):
        style = ttk.Style(self)
        # cap duro de altura de fila (evita filas gigantes al redimensionar)
        _min = int(CONFIG["layout"].get("row_min_px", 10))
        _max = int(CONFIG["layout"].get("row_max_px", 50))
        rowheight_px = max(_min, min(_max, int(rowheight_px)))
        style.configure("Miau.Treeview", rowheight=int(rowheight_px), font=row_font)
        style.configure("Miau.Treeview.Heading", font=head_font)
        self._fonts_row = row_font
        if self._editor and self._editor.winfo_exists():
            try:
                x, y, w, h = self.tree.bbox(self._edit_iid, f"#{1 if self._edit_col==self.COL_PATTERN else (2 if self._edit_col==self.COL_REGEX else 3)}")
                self._editor.place_configure(x=x+1, y=y+1, width=max(24, w-2), height=max(18, h-2))
                self._editor.configure(font=row_font)
            except Exception:
                pass
        self._apply_columns()

    def _ensure_strip_vars(self):
        if hasattr(self, "_strip_vars") and isinstance(getattr(self, "_strip_vars"), dict):
            return
        self._strip_vars = {
            'punct': tk.BooleanVar(value=False),
            'geom':  tk.BooleanVar(value=False),
            'curr':  tk.BooleanVar(value=False),
            'other': tk.BooleanVar(value=False),
            'greek': tk.BooleanVar(value=False),
        }

    # ---- CRUD ----

    def add_rule(self, pattern: str = "", repl: str = "", *, regex: bool = True, lock_regex: bool = False, mode: str = "literal"):
        # ensure adder is last by recreating it after insert
        if self._adder_iid and self._adder_iid in self.tree.get_children(""):
            try:
                self.tree.delete(self._adder_iid)
            except Exception:
                pass
            self._adder_iid = None

        iid = self.tree.insert("", "end", values=(pattern, "☑" if regex else "☐", repl))
        self._rx_state[iid] = bool(regex)
        if lock_regex:
            self._rx_locked.add(iid)

        # modo de literalidad
        self._rule_mode[iid] = ("set" if str(mode).lower() == "set" else "literal")

        self._apply_zebra()
        self._insert_adder_row()
        return iid


    def remove_selected(self):
        sel = [iid for iid in self.tree.selection() if "adder" not in (self.tree.item(iid, "tags") or ())]
        if not sel: return
        for iid in sel:
            self._rx_state.pop(iid, None)
            self._rx_locked.discard(iid)
            self._rule_mode.pop(iid, None)
            for key, ids in list(self._auto_rows.items()):
                if iid in ids:
                    ids.remove(iid)
        for iid in sel:
            try: self.tree.delete(iid)
            except Exception: pass
        self._apply_zebra()
        self._insert_adder_row()


    def export_rules(self) -> List[Dict[str, object]]:
        """Exporta reglas para aplicar en archivos.
        - regex: respeta el checkbox, pero el motor también fuerza regex si el patrón usa tokens (^p,^w,...) o si empieza con 're:'.
        - mode:
            * 'literal' => reemplazo de substring literal (ej. //, ...., McGraw Hill)
            * 'set'     => reemplaza cualquier carácter del patrón (solo lo usa 'Eliminar todos')
        """
        out: List[Dict[str, object]] = []
        for iid in self.tree.get_children(""):
            tags = tuple(self.tree.item(iid, "tags") or ())
            if "adder" in tags:
                continue
            pat, _rx_txt, rep = self.tree.item(iid, "values")
            out.append({
                "pattern": str(pat),
                "replace": str(rep),
                "regex": bool(self._rx_state.get(iid, True)),
                "mode": str(self._rule_mode.get(iid, "literal")),
            })
        return out


    def requires_regex(self) -> bool:
        return any(self._rx_state.values())

    # ---- edición de celdas ----
    def _cell_info(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell": return None, None, (0,0,0,0)
        iid = self.tree.identify_row(event.y); colid = self.tree.identify_column(event.x)
        if not iid or not colid: return None, None, (0,0,0,0)
        cols = self.tree.cget("columns")
        try:
            index = int(colid[1:]) - 1; name = cols[index]
        except Exception:
            return None, None, (0,0,0,0)
        bbox = self.tree.bbox(iid, colid)
        return iid, name, bbox if bbox else (0,0,0,0)

    def _begin_edit(self, event):
        iid, colid, bbox = self._cell_info(event)
        if not iid: return
        tags = tuple(self.tree.item(iid, "tags") or ())
        if "adder" in tags:
            self._on_click_adder()
            return
        if colid not in (self.COL_PATTERN, self.COL_REPL): return
        self._open_editor(iid, colid, bbox)

    def _open_editor(self, iid: str, colid: str, bbox):
        if self._editor is not None and self._editor.winfo_exists():
            try: self._commit_editor()
            except Exception:
                try: self._editor.destroy()
                except Exception: pass
        x, y, w, h = bbox
        cur_text = self.tree.set(iid, colid)
        self._editor = tk.Entry(self.tree, bd=0, relief="flat",
                                font=self._fonts_row,
                                bg=CONFIG["colors"]["box_bg"], fg=CONFIG["colors"]["text"],
                                insertbackground=CONFIG["colors"]["text"])
        self._editor.insert(0, cur_text)
        self._editor.place(x=x+1, y=y+1, width=max(24, w-2), height=max(18, h-2))
        self._edit_iid = iid; self._edit_col = colid
        self._editor.focus_set(); self._editor.icursor("end")
        self._editor.bind("<Return>", lambda e: self._commit_editor())
        self._editor.bind("<Escape>", lambda e: self._cancel_editor())
        # macOS: soporta Button-3 (común) y Button-2 (Tk Aqua)
        if colid in (self.COL_PATTERN, self.COL_REPL):
            try:
                self._editor.bind("<Button-3>", self._popup_token_menu)
                self._editor.bind("<Button-2>", self._popup_token_menu)
            except Exception:
                pass
        self._editor.bind("<FocusOut>", lambda e: self._commit_editor())
        if colid in (self.COL_PATTERN, self.COL_REPL):
            self._editor.bind("<Button-3>", self._popup_token_menu)

    def _commit_editor(self):
        if not self._editor or not (self._editor.winfo_exists() and self._edit_iid and self._edit_col): return
        val = self._editor.get()  # NO strip(): preserva espacios literales
        self.tree.set(self._edit_iid, self._edit_col, val)
        self._editor.destroy(); self._editor = None; self._edit_iid = None; self._edit_col = None

    def _cancel_editor(self):
        if self._editor and self._editor.winfo_exists(): self._editor.destroy()
        self._editor = None; self._edit_iid = None; self._edit_col = None

    def commit_pending_edits(self):
        try: self._commit_editor()
        except Exception: pass

    # ---- toggle checkbox & adder ----
    def _on_click_toggle(self, event):
        iid, colid, _bbox = self._cell_info(event)
        if not iid:
            return
        tags = tuple(self.tree.item(iid, "tags") or ())
        if "adder" in tags:
            self._on_click_adder()
            return
        if colid != self.COL_REGEX:
            return
        if iid in self._rx_locked:
            return
        cur = self._rx_state.get(iid, True)
        new = not cur
        self._rx_state[iid] = new
        self.tree.set(iid, self.COL_REGEX, "☑" if new else "☐")

    def _on_click_adder(self):
        # crea nueva regla vacía y abre editor en Buscar
        new_iid = self.add_rule("", "", regex=True)
        try:
            bbox_new = self.tree.bbox(new_iid, "#1")
            self._open_editor(new_iid, self.COL_PATTERN, bbox_new if bbox_new else (0,0,12,24))
        except Exception:
            pass

    # ---- presets ----
    def _documents_dir(self) -> Path:
        return _user_documents_dir()

    def save_preset_dialog(self):
        self.commit_pending_edits()
        initial = self._documents_dir()
        fn = filedialog.asksaveasfilename(
            title="Guardar configuración",
            initialdir=str(initial),
            defaultextension=PRESET_EXT,
            filetypes=[("Preset", f"*{PRESET_EXT}"), ("Todos", "*.*")]
        )
        if not fn:
            return
        try:
            payload = {
                "version": 2,
                "name": Path(fn).stem,
                "rules": self._export_preset_rules(),
            }
            Path(fn).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Guardar configuración", str(e))

    def load_preset_dialog(self):
        self.commit_pending_edits()
        initial = self._documents_dir()
        fn = filedialog.askopenfilename(
            title="Cargar configuración",
            initialdir=str(initial),
            filetypes=[("Preset", f"*{PRESET_EXT}"), ("Todos", "*.*")]
        )
        if not fn:
            return
        try:
            self.load_preset(Path(fn))
        except Exception as e:
            messagebox.showerror("Cargar configuración", str(e))

    def _clear_all_rules(self):
        # borra todas las filas (excepto el adder)
        for iid in list(self.tree.get_children("")):
            tags = tuple(self.tree.item(iid, "tags") or ())
            if "adder" in tags:
                continue
            try:
                self.tree.delete(iid)
            except Exception:
                pass
        self._rx_state.clear()
        self._rx_locked.clear()
        self._rule_mode.clear()
        self._auto_rows.clear()
        self._apply_zebra()
        self._insert_adder_row()

        self._ensure_strip_vars()
        for k in self._strip_vars:
            self._strip_vars[k].set(False)

    def _export_preset_rules(self) -> List[Dict[str, object]]:
        # mismo formato para aplicar y para guardar (incluye 'mode')
        return self.export_rules()


    def load_preset(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        rules = data.get("rules")
        if not isinstance(rules, list):
            raise ValueError("Preset inválido: falta 'rules'")

        self._clear_all_rules()

        # reconstruir filas
        self._ensure_strip_vars()
        for item in rules:
            if not isinstance(item, dict):
                continue
            pat = str(item.get("pattern", ""))
            rep = str(item.get("replace", ""))
            rx = bool(item.get("regex", True))
            mode = str(item.get("mode", item.get("kind", "literal")))
            iid = self.add_rule(pat, rep, regex=rx, mode=mode)

            # re-vincular batch si coincide con alguno de los sets
            if (not rx) and rep == "":
                # identifica clave por contenido
                for key, chars in self._lists_for_batch.items():
                    joined = "".join(chars or [])
                    if pat == joined:
                        self._strip_vars[key].set(True)
                        self._auto_rows.setdefault(key, []).append(iid)
                        self._rule_mode[iid] = "set"
                        break

        self._apply_zebra()
        self._insert_adder_row()

    # ---- menú contextual ----
    def _build_token_menu(self, allow_batch: bool = True):
        m = tk.Menu(self, tearoff=0)

        # Presets
        m.add_command(label="Cargar configuración…", command=self.load_preset_dialog)
        m.add_command(label="Guardar configuración…", command=self.save_preset_dialog)
        m.add_separator()

        # Bloque solo para columna Buscar
        if allow_batch:
            self._ensure_strip_vars()
            sm_all = tk.Menu(m, tearoff=0)
            sm_all.add_checkbutton(label='Puntuación',     variable=self._strip_vars['punct'], command=lambda: self._toggle_batch('punct'))
            sm_all.add_checkbutton(label='Geométricos',    variable=self._strip_vars['geom'],  command=lambda: self._toggle_batch('geom'))
            sm_all.add_checkbutton(label='Moneda',         variable=self._strip_vars['curr'],  command=lambda: self._toggle_batch('curr'))
            sm_all.add_checkbutton(label='Otros',          variable=self._strip_vars['other'], command=lambda: self._toggle_batch('other'))
            sm_all.add_checkbutton(label='Letras griegas', variable=self._strip_vars['greek'], command=lambda: self._toggle_batch('greek'))
            m.add_cascade(label='Eliminar todos', menu=sm_all)
            m.add_separator()

        def add_token(label, token):
            m.add_command(label=label, command=lambda t=token: self._insert_token(t))

        add_token('^p  (párrafo)', '^p')
        add_token('^l  (salto de línea)', '^l')
        add_token('^t  (tabulación)', '^t')
        add_token('^s  (espacio)', '^s')
        add_token('^w  (espacios consecutivos)', '^w')
        add_token('^?  (cualquier carácter)', '^?')
        add_token('^#  (números)', '^#')

        m.add_separator()

        # ======== un cascade por cada categoría (sin "Catálogo") ========
        for cat_name, chars in SYMBOL_CATEGORIES.items():
            sm = tk.Menu(m, tearoff=0)
            for ch in chars:
                if cat_name == "Puntuación" and ch.isspace():
                    continue
                label = f"{ch}  {_symbol_name_natural(ch)}  (U+{ord(ch):04X})"
                sm.add_command(
                    label=label,
                    command=lambda c=ch, submenu=sm: self._insert_token_keep_open(c, submenu)
                )
            m.add_cascade(label=cat_name, menu=sm)

        # Mapas usados por el batch
        self._lists_for_batch = {
            'punct': PUNCT_CHARS,
            'geom':  GEOM_CHARS,
            'curr':  CURRENCY_CHARS,
            'other': OTHER_CHARS,
            'greek': GREEK_CHARS,
        }
        return m

    def _popup_token_menu(self, event=None):
        try:
            if event is not None:
                self._last_popup_xy = (int(event.x_root), int(event.y_root))
        except Exception:
            pass

        try:
            menu = self._menu_pattern if self._edit_col == self.COL_PATTERN else self._menu_repl
        except Exception:
            menu = getattr(self, "_menu_pattern", None) or getattr(self, "_menu_repl", None)

        if menu is None:
            self._menu_pattern = self._build_token_menu(allow_batch=True)
            self._menu_repl = self._build_token_menu(allow_batch=False)
            menu = self._menu_pattern

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass

    def _right_click(self, event):
        iid, colid, bbox = self._cell_info(event)
        if iid and colid in (self.COL_PATTERN, self.COL_REPL):
            tags = tuple(self.tree.item(iid, "tags") or ())
            if "adder" in tags:
                self._on_click_adder()
                return
            self._open_editor(iid, colid, bbox)
            self._popup_token_menu(event)

    def _insert_token(self, token: str):
        if self._editor is None or not self._editor.winfo_exists(): return
        pos = self._editor.index(tk.INSERT)
        cur = self._editor.get()
        new = cur[:pos] + token + cur[pos:]
        self._editor.delete(0, tk.END); self._editor.insert(0, new); self._editor.icursor(pos + len(token))

    def _repost_menu(self, menu: tk.Menu):
        try:
            if self._last_popup_xy:
                x, y = self._last_popup_xy
            else:
                x = self.winfo_pointerx()
                y = self.winfo_pointery()
            menu.tk_popup(x, y)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass

    def _insert_token_keep_open(self, token: str, submenu_to_reopen: tk.Menu):
        # Inserta y vuelve a mostrar EL MISMO submenú para seguir clickeando símbolos
        self._insert_token(token)
        self.after(1, lambda m=submenu_to_reopen: self._repost_menu(m))

    def _apply_zebra(self):
        children = list(self.tree.get_children(""))
        # keep adder last
        if self._adder_iid in children and children[-1] != self._adder_iid:
            try:
                self.tree.move(self._adder_iid, "", "end")
            except Exception:
                pass
            children = list(self.tree.get_children(""))
        idx = 0
        for iid in children:
            tags = tuple(self.tree.item(iid, "tags") or ())
            if "adder" in tags:
                continue
            self.tree.item(iid, tags=("odd" if idx % 2 else "even",))
            idx += 1

    # ---- expansión por lotes para "Eliminar todos" ----
    def _toggle_batch(self, key: str):
        self._ensure_strip_vars()
        enable = self._strip_vars[key].get()

        # limpia filas anteriores de ese grupo
        for iid in self._auto_rows.get(key, []):
            try:
                self._rx_state.pop(iid, None)
                self._rx_locked.discard(iid)
                self._rule_mode.pop(iid, None)
                self.tree.delete(iid)
            except Exception:
                pass
        self._auto_rows[key] = []

        chars = self._lists_for_batch.get(key)
        if enable:
            joined = "".join(chars or [])
            iid = self.add_rule(joined, "", regex=False, lock_regex=False, mode="set")
            self._auto_rows[key].append(iid)
        self._apply_zebra()

    # ---- adder row helpers ----
    def _insert_adder_row(self):
        if self._adder_iid and self._adder_iid in self.tree.get_children(""):
            try:
                self.tree.delete(self._adder_iid)
            except Exception:
                pass
        self._adder_iid = self.tree.insert("", "end", values=("+", "", ""), tags=("adder",))
        try:
            fam, size = self._fonts_row[0], self._fonts_row[1]
            self.tree.tag_configure("adder", font=(fam, size, "italic"))
        except Exception:
            self.tree.tag_configure("adder")

# =============================================================================
# Tokens Word y motor de reemplazo
# ^w: SOLO espacios repetidos (NO afecta ^l ni ^p)
TOKEN_MAP_PATTERN = {
    "^p": r"(?:\r\n|\n|\r)",
    "^l": r"\n",
    "^t": r"\t",
    "^s": r" ",
    "^w": r"(?<=\S) {2,}(?=\S)",
    "^#": r"\d+",
    "^?": r".",
}
TOKEN_MAP_REPL    = {
    "^p": "\n",
    "^l": "\n",
    "^t": "\t",
    "^s": " ",
    "^w": " ",
    "^#": "",
    "^?": "",
}

def compile_rule(pattern_text: str, repl_text: str, *, regex_mode: bool):
    r"""
    Construye (regex, reemplazo) con soporte de tokens con caret en ambos lados.
    - En el patrón, ^? y ^# se convierten en grupos de captura (.) y (\d+).
    - En el reemplazo, ^? y ^# se expanden a \g<n> según el orden de aparición.
    - ^p ^l ^t ^s ^w se expanden literalmente como antes.
    - Acepta patrones literales con tokens o "re:" crudos.
    """
    text = pattern_text or ""
    is_raw_regex = False
    if text.startswith("re:"):
        text = text[3:]
        is_raw_regex = True

    groups = {"?": [], "#": []}
    group_index = 1
    i = 0
    out_pat = []

    def emit_literal(ch: str):
        out_pat.append(ch if is_raw_regex else re.escape(ch))

    while i < len(text):
        ch = text[i]
        if ch == "^" and i+1 < len(text):
            t = text[i+1]
            tok = "^" + t
            if tok == "^?":
                out_pat.append("(.)")
                groups["?"].append(group_index); group_index += 1
                i += 2
                continue
            if tok == "^#":
                out_pat.append(r"(\d+)")
                groups["#"].append(group_index); group_index += 1
                i += 2
                continue
            if tok in TOKEN_MAP_PATTERN:
                out_pat.append(TOKEN_MAP_PATTERN[tok])
                i += 2
                continue
        emit_literal(ch)
        i += 1

    pat_str = "".join(out_pat)
    pat = re.compile(pat_str, flags=re.DOTALL | re.MULTILINE)

    rep_raw = repl_text or ""
    j = 0
    out_rep = []
    cursors = {"?": 0, "#": 0}

    while j < len(rep_raw):
        ch = rep_raw[j]
        if ch == "^" and j+1 < len(rep_raw):
            t = rep_raw[j+1]
            tok = "^" + t
            if tok == "^?":
                idxs = groups["?"]
                if idxs:
                    k = min(cursors["?"], len(idxs)-1)
                    out_rep.append(r"\g<%d>" % idxs[k])
                    if cursors["?"] < len(idxs): cursors["?"] += 1
                j += 2
                continue
            if tok == "^#":
                idxs = groups["#"]
                if idxs:
                    k = min(cursors["#"], len(idxs)-1)
                    out_rep.append(r"\g<%d>" % idxs[k])
                    if cursors["#"] < len(idxs): cursors["#"] += 1
                j += 2
                continue
            if tok in TOKEN_MAP_REPL:
                out_rep.append(TOKEN_MAP_REPL[tok])
                j += 2
                continue
        out_rep.append(rep_raw[j])
        j += 1

    repl_str = "".join(out_rep)

    def _repl_func(m, s=repl_str):
        try:
            return m.expand(s)
        except Exception:
            return s

    return pat, _repl_func

def apply_on_text(text: str, rules: List[Dict[str, object]], *, regex_mode: bool) -> tuple[str, int]:
    """
    Aplica reglas al texto respetando:
    - regex=True: usa el motor de tokens (^p,^w,...) y ESCAPA por default los metacaracteres.
      Si el usuario quiere regex crudo, puede escribir el patrón como 're:...'.
    - mode='literal': reemplazo de substring literal (ej. //, ....)
    - mode='set': reemplaza cualquier carácter del patrón (solo 'Eliminar todos')
    """
    out = text
    total = 0

    def looks_like_charset(s: str) -> bool:
        # Heurística para presets viejos sin 'mode'
        if not s or any(ch.isspace() for ch in s):
            return False
        uniq = len(set(s))
        if len(s) < 12 or uniq < 8:
            return False
        non_alnum = sum(1 for ch in s if not ch.isalnum())
        return (non_alnum / max(1, len(s))) > 0.85 and (uniq / max(1, len(s))) > 0.55

    for item in rules:
        if not isinstance(item, dict):
            # tolera formatos viejos [(pat, rep), ...]
            try:
                pat = str(item[0])
                rep = str(item[1]) if len(item) > 1 else ""
                rx = True
                mode = "literal"
            except Exception:
                continue
        else:
            pat = str(item.get("pattern", ""))
            rep = str(item.get("replace", ""))
            rx = bool(item.get("regex", True))
            mode = str(item.get("mode", item.get("kind", "literal")))

        if not pat:
            continue

        # Tokens o 're:' => fuerza ruta regex
        if pat.startswith("re:") or any(tok in pat for tok in TOKEN_MAP_PATTERN.keys()):
            rx = True

        flags = re.IGNORECASE | re.DOTALL | re.MULTILINE

        if rx:
            rx_pat, rx_rep = compile_rule(pat, rep, regex_mode=True)
            out, n = rx_pat.subn(rx_rep, out)
            total += int(n)
            continue

        # no-regex
        if mode != "set" and looks_like_charset(pat) and (rep == ""):
            # compat: presets viejos de "Eliminar todos"
            mode = "set"

        if mode == "set":
            seen = set()
            chars = []
            for ch in pat:
                if ch in seen:
                    continue
                seen.add(ch)
                chars.append(ch)
            if not chars:
                continue
            charset = "".join(re.escape(ch) for ch in chars)
            pat_re = re.compile(f"[{charset}]", flags=flags)
            out, n = pat_re.subn(rep, out)
            total += int(n)
        else:
            # literal substring exacto
            pat_re = re.compile(re.escape(pat), flags=flags)
            out, n = pat_re.subn(rep, out)
            total += int(n)

    return out, total


def apply_on_file(path: Union[Path, str], rules: List[Dict[str, object]], *, regex_mode: bool) -> int:
    """Aplica reglas a un archivo. Si lo único es '__GEOM_STRIP__', hace un pase
    sobre bytes y escribe sin recodificar. Devuelve número de cambios."""
    path = _coerce_path(path)
    orig_raw = path.read_bytes()

    if b"\r\n" in orig_raw and orig_raw.count(b"\r\n") >= max(1, orig_raw.count(b"\n")//2):
        nl = "\r\n"
    elif b"\r" in orig_raw and b"\n" not in orig_raw:
        nl = "\r"
    else:
        nl = "\n"

    if orig_raw.startswith(b"\xef\xbb\xbf"):
        enc_guess = "utf-8-sig"
    elif orig_raw.startswith(b"\xff\xfe") or orig_raw.startswith(b"\xfe\xff"):
        enc_guess = "utf-16"
    else:
        enc_guess = None
        for enc_try in ("utf-8", "cp1252", "latin-1"):
            try:
                orig_raw.decode(enc_try)
                enc_guess = enc_try
                break
            except Exception:
                continue
        if enc_guess is None:
            enc_guess = "latin-1"

    try:
        data = orig_raw.decode(enc_guess)
    except Exception:
        enc_guess = "latin-1"
        data = orig_raw.decode("latin-1")

    new_text, cnt_text = apply_on_text(data, rules, regex_mode=regex_mode)

    if new_text != data:
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            if not bak.exists():
                bak.write_bytes(orig_raw)
        except Exception:
            pass
        try:
            _canon = new_text.replace("\r\n", "\n").replace("\r", "\n")
            _to_write = _canon.replace("\n", nl)
            with path.open("w", encoding=enc_guess, newline="") as fh:
                fh.write(_to_write)
        except UnicodeEncodeError:
            with path.open("w", encoding=enc_guess, errors="ignore", newline="") as fh:
                fh.write(_to_write)

    return int(cnt_text)

# =============================================================================
# App
class App(tk.Tk):
    def __init__(self, files: Optional[List[Path]] = None):
        super().__init__()
        self.withdraw()

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ww, wh = window_geometry("reemplazos", sw, sh)
        x = int((sw - ww) / 2)
        y = int((sh - wh) / 2)
        self.geometry(f"{ww}x{wh}+{x}+{y}")
        self.minsize(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"])

        apply_theme(self, app="reemplazos", title=CONFIG["window"]["title"], icon=CONFIG["window"]["icon_name"])

        self.configure(bg=CONFIG["colors"]["app_bg"])
        self.title(CONFIG["window"]["title"])

        self._rw, self._rh = _metrics(ww, wh)
        rw, rh = self._rw, self._rh

        fam = CONFIG["fonts"]["family"]
        self.fonts = {
            "title": _font(fam, rh(CONFIG["fonts"]["title"]["rel_px"]), bold=CONFIG["fonts"]["title"]["bold"], italic=CONFIG["fonts"]["title"]["italic"]),
            "sub":   _font(fam, rh(CONFIG["fonts"]["sub"]["rel_px"]),   bold=CONFIG["fonts"]["sub"]["bold"],   italic=CONFIG["fonts"]["sub"]["italic"]),
            "head":  _font(fam, rh(CONFIG["fonts"]["head"]["rel_px"]),  bold=CONFIG["fonts"]["head"]["bold"]),
            "row":   _font(fam, rh(CONFIG["fonts"]["row"]["rel_px"]),   bold=CONFIG["fonts"]["row"]["bold"]),
            "btn":   _font(fam, rh(CONFIG["fonts"]["btn"]["rel_px"]),   bold=CONFIG["fonts"]["btn"]["bold"]),
        }

        pad = int(max(4, CONFIG["layout"].get("row_pad_min_px", 6)))
        c = CONFIG["colors"]

        header = tk.Frame(self, bg=c["app_bg"], bd=0, highlightthickness=0)
        header.grid(row=0, column=0, sticky="new", padx=pad, pady=(pad, 0))
        self.table = RulesTable(self, self.fonts)
        self.table.grid(row=1, column=0, sticky="nsew", padx=pad, pady=pad)

        footer = tk.Frame(self, bg=c["app_bg"], bd=0, highlightthickness=0)
        footer.grid(row=2, column=0, sticky="sew")
        footer.columnconfigure(0, weight=1)
        btns = tk.Frame(footer, bg=c["app_bg"])
        btns.grid(row=0, column=1, sticky="e", padx=pad, pady=pad)

        # solo quedan "Añadir archivos…" y "Aplicar"
        bw0, bh0 = shared_button_box_px(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"], CONFIG)

        btn_fs0 = shared_button_font_px(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"], CONFIG)
        fam_btn = CONFIG["fonts"]["family"] or CONFIG["fonts"]["fallback_family"]
        weight = "bold" if bool(CONFIG["fonts"]["btn"].get("bold", True)) else "normal"
        slant  = "italic" if bool(CONFIG["fonts"]["btn"].get("italic", False)) else "roman"
        self._shared_btn_font = tkfont.Font(family=fam_btn, size=int(btn_fs0), weight=weight, slant=slant)


        self.btn_pick = FlatButton(
            btns, "Añadir archivos…", command=self._add_files_dialog, bg=c["app_bg"],
            width=bw0, height=bh0,
            radius=CONFIG["layout"]["btn_radius_px"], font=self._shared_btn_font
        )
        self.btn_pick.set_shared_font(self._shared_btn_font)
        self.btn_pick.grid(row=0, column=0, padx=(0, 12))

        self.btn_run = FlatButton(
            btns, "Aplicar", command=self._run, bg=c["app_bg"],
            width=bw0, height=bh0,
            radius=CONFIG["layout"]["btn_radius_px"], font=self._shared_btn_font
        )
        self.btn_run.set_shared_font(self._shared_btn_font)
        self.btn_run.grid(row=0, column=1)

        self._files: List[Path] = []
        for _f in (files or []):
            try:
                _p = _coerce_path(_f)
                # Mantener comportamiento: solo archivos reales (evita crashes posteriores)
                if _p and _p.exists() and _p.is_file():
                    self._files.append(_p)
            except Exception:
                pass

        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.bind("<Configure>", self._on_root_configure)

        self.deiconify()

    def _on_root_configure(self, *_):
        ww = max(self.winfo_width(),  CONFIG["window"]["min_width"])
        wh = max(self.winfo_height(), CONFIG["window"]["min_height"])
        self._rw, self._rh = _metrics(ww, wh)
        rw, rh = self._rw, self._rh
        fam = CONFIG["fonts"]["family"]
        self.fonts = {
            "title": _font(fam, rh(CONFIG["fonts"]["title"]["rel_px"]), bold=CONFIG["fonts"]["title"]["bold"], italic=CONFIG["fonts"]["title"]["italic"]),
            "sub":   _font(fam, rh(CONFIG["fonts"]["sub"]["rel_px"]),   bold=CONFIG["fonts"]["sub"]["bold"],   italic=CONFIG["fonts"]["sub"]["italic"]),
            "head":  _font(fam, rh(CONFIG["fonts"]["head"]["rel_px"]),  bold=CONFIG["fonts"]["head"]["bold"]),
            "row":   _font(fam, rh(CONFIG["fonts"]["row"]["rel_px"]),   bold=CONFIG["fonts"]["row"]["bold"]),
            "btn":   _font(fam, rh(CONFIG["fonts"]["btn"]["rel_px"]),   bold=CONFIG["fonts"]["btn"]["bold"]),
        }
        bw, bh = shared_button_box_px(ww, wh, CONFIG)
        for b in (self.btn_pick, self.btn_run):
            try:
                b.config(width=bw, height=bh)
            except Exception:
                pass

        self._schedule_shared_btn_font()

        rowheight = max(int(CONFIG["layout"]["row_min_px"]), int(self.winfo_toplevel().winfo_height() * float(CONFIG["layout"]["row_height_rel"])))
        
        rowheight = min(int(CONFIG["layout"].get("row_max_px", 50)), int(rowheight))
        self.table.refresh_style_metrics(rowheight, self.fonts["head"], self.fonts["row"])

    
    def _schedule_shared_btn_font(self):
        # Unifica tipografía de botones al tamaño más pequeño requerido.
        job = getattr(self, "_btn_font_job", None)
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        self._btn_font_job = self.after(1, self._sync_shared_btn_font)

    def _sync_shared_btn_font(self):
        """Fuente única para botones principales (idéntica entre apps y responsiva)."""
        self._btn_font_job = None
        ww = max(self.winfo_width(),  CONFIG["window"]["min_width"])
        wh = max(self.winfo_height(), CONFIG["window"]["min_height"])
        btn_sz = shared_button_font_px(ww, wh, CONFIG)
        fam_btn = CONFIG["fonts"]["family"] or CONFIG["fonts"]["fallback_family"]
        weight = "bold" if bool(CONFIG["fonts"]["btn"].get("bold", True)) else "normal"
        slant  = "italic" if bool(CONFIG["fonts"]["btn"].get("italic", False)) else "roman"
        if getattr(self, "_shared_btn_font", None) is None:
            self._shared_btn_font = tkfont.Font(family=fam_btn, size=int(btn_sz), weight=weight, slant=slant)
        else:
            try:
                self._shared_btn_font.configure(family=fam_btn, weight=weight, slant=slant, size=int(btn_sz))
            except Exception:
                self._shared_btn_font = tkfont.Font(family=fam_btn, size=int(btn_sz), weight=weight, slant=slant)

        for b in (self.btn_pick, self.btn_run):
            try:
                b.set_shared_font(self._shared_btn_font)
            except Exception:
                pass



    def _add_files_dialog(self):
        paths = filedialog.askopenfilenames(title="Selecciona archivos de texto",
                                            filetypes=[("Texto", "*.txt *.md *.csv *.tsv *.log"), ("Todos", "*.*")])
        if not paths: return
        self._files.extend(Path(p) for p in paths)

    def _collect_rules(self) -> List[Dict[str, object]]:
        return self.table.export_rules()

    def _run(self):
        try:
            self.table.commit_pending_edits()
        except Exception:
            pass

        rules = self._collect_rules()
        if not rules:
            messagebox.showwarning("Reemplazos", "No hay reglas definidas.")
            return
        if not self._files:
            messagebox.showwarning("Reemplazos", "No hay archivos seleccionados.")
            return

        regex_mode = True

        dlg = ProgressDialog(self)
        dlg.set_title("Procesando…")
        dlg.set_subtitle("Preparando archivos")
        dlg.set_global_span(len(self._files))
        dlg.show()

        def _worker():
            try:
                total_changes = 0
                done = 0
                for p in list(self._files):
                    cnt = apply_on_file(p, rules, regex_mode=regex_mode)
                    total_changes += int(cnt)
                    done += 1
                    dlg.set_done(done)
                    dlg.set_subtitle(f"{p.name} ({done}/{len(self._files)})")
                dlg.set_progress(1.0)
                dlg.set_title("Listo")
                dlg.set_subtitle(f"Se realizaron {total_changes} cambios")
                self.after(int(CONFIG["app"].get("close_delay_ms", 1200)), dlg.close)
            except Exception as e:
                try: dlg.close()
                except Exception: pass
                messagebox.showerror("Error", str(e))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
def _merge_multiselect_startup(
    files_in,
    *,
    mutex_name: str,
    spool_name: str,
    settle_ms: int = 7000,
    stable_ms: int = 650,
    poll_ms: int = 60,
):
    """
    Agrupa multi-selección de Explorer cuando éste ejecuta 1 proceso por archivo.
    - Si ya existe otra instancia (mutex), este proceso solo APPENDEA sus rutas a un spool y SALE.
    - La instancia "primaria" colecta rutas desde el spool hasta que el set se estabiliza.
    Mantiene el arranque ágil: si solo hay 1 archivo, normalmente se estabiliza en <200 ms.
    """
    import os
    from pathlib import Path
    from typing import List

    if os.name != "nt":
        return list(files_in), True

    try:
        import time
        import tempfile
        import ctypes
        import ctypes.wintypes as _wt
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        CreateMutexW = kernel32.CreateMutexW
        CreateMutexW.argtypes = [_wt.LPVOID, _wt.BOOL, _wt.LPCWSTR]
        CreateMutexW.restype = _wt.HANDLE
        GetLastError = kernel32.GetLastError
        CloseHandle = kernel32.CloseHandle

        ERROR_ALREADY_EXISTS = 183
        mutex = CreateMutexW(None, False, mutex_name)
        already = (GetLastError() == ERROR_ALREADY_EXISTS)

        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or tempfile.gettempdir()
        spool_dir = Path(base) / "MiausoftSuite" / "IPC"
        spool_dir.mkdir(parents=True, exist_ok=True)
        spool = spool_dir / spool_name

        def _append(paths: List[str]) -> None:
            if not paths:
                return
            with spool.open("a+", encoding="utf-8") as f:
                try:
                    # lock 1 byte region (suficiente para exclusión mutua entre procesos)
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                except Exception:
                    pass
                f.seek(0, os.SEEK_END)
                for p in paths:
                    p = (p or "").replace("\n", "").strip()
                    if p:
                        f.write(p + "\n")
                f.flush()
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass

        def _drain() -> List[str]:
            out: List[str] = []
            with spool.open("a+", encoding="utf-8") as f:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                except Exception:
                    pass
                f.seek(0)
                out = [(line or "").strip() for line in f]
                try:
                    f.seek(0)
                    f.truncate()
                except Exception:
                    pass
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            return [p for p in out if p]

        files_norm = []
        for x in list(files_in):
            try:
                files_norm.append(str(x))
            except Exception:
                pass

        if not files_norm:
            return list(files_in), True

        # Preferir tiempos desde config central (sin configs locales)
        try:
            from config import get_config as _get_cfg  # centralizado
            _cfg = _get_cfg(None) if callable(_get_cfg) else {}
            _ms = ((_cfg or {}).get("app", {}) or {}).get("multiselect", {}) or {}
            settle_ms = int(_ms.get("settle_ms", settle_ms))
            stable_ms = int(_ms.get("stable_ms", stable_ms))
            poll_ms = int(_ms.get("poll_ms", poll_ms))
        except Exception:
            pass

        # Si ya llegó más de 1 ruta en un mismo proceso, no hay que esperar casi nada.
        if len(files_norm) > 1:
            stable_ms = min(stable_ms, 220)
            settle_ms = min(settle_ms, 1200)


        if already:
            _append(files_norm)
            try:
                CloseHandle(mutex)
            except Exception:
                pass
            return [], False

        # Primaria: escribe sus rutas y colecta las demás hasta estabilizar.
        _append(files_norm)

        merged: List[Path] = []
        seen: set[str] = set()
        stable = 0
        deadline = time.monotonic() + (max(200, int(settle_ms)) / 1000.0)

        while time.monotonic() < deadline and stable < int(stable_ms):
            new = _drain()
            added = 0
            for p in new:
                if not p or p in seen:
                    continue
                try:
                    if Path(p).exists():
                        seen.add(p)
                        merged.append(Path(p))
                        added += 1
                except Exception:
                    continue
            if added == 0:
                stable += int(poll_ms)
            else:
                stable = 0
            time.sleep(max(10, int(poll_ms)) / 1000.0)

        # Drenaje final por si llegó algo entre el último sleep y el cierre.
        for p in _drain():
            if p and p not in seen:
                try:
                    if Path(p).exists():
                        seen.add(p)
                        merged.append(Path(p))
                except Exception:
                    pass

        try:
            spool.unlink(missing_ok=True)
        except Exception:
            pass
        # Intencionalmente NO cerramos el mutex aquí.
        # Motivo: Explorer puede lanzar 1 proceso por archivo seleccionado;
        # si liberamos el mutex demasiado pronto, otra instancia puede volverse
        # "primaria" y abrir otra ventana. El sistema cerrará el handle al
        # terminar el proceso, y el objeto se destruye cuando se cierra el último handle.

        # Si por alguna razón quedó vacío, conservamos el input original.
        return (merged if merged else [Path(p) for p in files_norm]), True

    except Exception:
        return list(files_in), True


def _safe_main(argv):
    import os, traceback
    try:
        raw_args = list(argv[1:])
        files = []
        preset_path = None
        for a in raw_args:
            if not a or a.startswith("-"):
                continue
            s = a.strip().strip('"').strip("'")
            s = os.path.expandvars(s)
            try:
                from pathlib import Path as _P
                pp = _P(s).expanduser().resolve()
            except Exception:
                pp = _P(s)
            if pp.exists() and pp.is_file():
                # Si es preset .miau, cargarlo como configuración, no como archivo a procesar
                try:
                    if str(pp).lower().endswith(str(PRESET_EXT).lower()):
                        preset_path = pp
                        continue
                except Exception:
                    pass
                files.append(pp)

        # Agrupar multi-selección cuando Explorer ejecuta 1 instancia POR archivo seleccionado.
        files, _launch = _merge_multiselect_startup(
            files,
            mutex_name=r"Local\MiausoftSuite_Reemplazos_MultiSelect",
            spool_name="reemplazos.paths.txt",
        )
        if not _launch:
            return

        app = App(files=files)
        if preset_path is not None:
            try:
                app.load_preset(preset_path)
            except Exception as e:
                try:
                    from tkinter import messagebox
                    messagebox.showerror("Preset", str(e))
                except Exception:
                    pass
        app.mainloop()
    except Exception:
        try:
            log_path = Path(__file__).with_suffix(".error.log")
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            log_path = None
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb
            root = _tk.Tk(); root.withdraw()
            msg = "Error al iniciar la aplicación."
            if log_path: msg += f" Revisa el log: {log_path}"
            _mb.showerror("Reemplazos", msg)
        except Exception:
            print("Error al iniciar:", traceback.format_exc())

if __name__ == "__main__":
    _safe_main(sys.argv)