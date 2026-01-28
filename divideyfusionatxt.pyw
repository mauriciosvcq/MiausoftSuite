# dividey fusiona.py
#Esta es una aplicación que une o divide exclusivamente archivos TXT
#Este sí tiene una interfaz de inicio que consiste en una tabla de 3 columnas Archivo, un slidebutton que te permite seleccionar si quieres unir o dividir, u una columna que solamente se activa cuando quieres dividir para que pongas en cuántas partes.
#los archivos a procesar se agregan desde el botón "Agregar Archivos" (o toma los previamente seleccionados desde el menú contextual cuando este programa es ejecutado desde el menú contextual)

from __future__ import annotations

import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

from pathlib import Path
from typing import Optional, List
from config import get_config, apply_theme, window_geometry, PHI, shared_button_box_px, shared_button_font_px, ProgressDialog

# ===== Config central (única) =====
CONFIG = get_config("fusionador")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de fuente/medidas (patrón igual a homólogos)
def _font(family, size_px, *, bold=False, italic=False):
    style_bits = []
    if bold: style_bits.append("bold")
    if italic: style_bits.append("italic")
    return (family, max(int(CONFIG["utilities"]["min_px"]), int(size_px)),
            " ".join(style_bits) if style_bits else "")

def _metrics(win_w: int, win_h: int):
    def rw(f): return max(1, int(win_w * float(f)))
    def rh(f): return max(1, int(win_h * float(f)))
    return rw, rh

# ──────────────────────────────────────────────────────────────────────────────
# Botón plano (pie) como en los homólogos
class FlatButton(tk.Canvas):
    def __init__(self, parent, text, command=None, bg=None,
                 width=120, height=28, radius=None, font=None, colors=None, **kw):
        super().__init__(
            parent,
            width=int(width),
            height=int(height),
            bd=0,
            highlightthickness=0,
            bg=bg or parent.cget("bg"),
            **kw,
        )
        self._wpx, self._hpx = int(width), int(height)
        self._r = int(max(0, (radius if radius is not None else int(CONFIG["layout"]["btn_radius_px"]))))

        self._text = str(text)
        self._cmd = command
        self._bg = bg or parent.cget("bg")
        self._hover = False
        self._pressed = False

        # Base desde CONFIG (tuple). La fuente real se ajusta en _fit_font()
        self._font = font or _font(
            CONFIG["utilities"]["family"],
            int(CONFIG["utilities"]["min_px"]),
            bold=bool(CONFIG["utilities"]["btn"]["bold"]),
            italic=bool(CONFIG["utilities"]["btn"]["italic"]),
        )
        self._tkfont_cache: Optional[tkfont.Font] = None  # evitar GC (Canvas necesita ref viva)
        self._shared_font: Optional[tkfont.Font] = None  # fuente bloqueada (compartida)

        c = CONFIG["colors"]
        self._colors = colors or {
            "text": c["text"],
            "base": c["btn_base"],
            "face": c["btn_face"],
            "face_hover": c["btn_face_hover"],
            "face_down": c["btn_face_down"],
        }

        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonPress-1>", lambda e: self._set_pressed(True))
        self.bind("<ButtonRelease-1>", self._on_click)
        self.bind("<Configure>", lambda e: self.after_idle(self._redraw))
        self.after_idle(self._redraw)

    def configure(self, cnf=None, **kw):  # type: ignore[override]
        # tk acepta cnf dict + kwargs
        if cnf:
            kw.update(cnf)

        if "width" in kw:
            self._wpx = int(kw.pop("width"))
            super().configure(width=self._wpx)
        if "height" in kw:
            self._hpx = int(kw.pop("height"))
            super().configure(height=self._hpx)
        if "font" in kw:
            self._font = kw.pop("font")

        out = super().configure(**kw) if kw else super().configure()
        self.after_idle(self._redraw)
        return out

    config = configure  # alias Tk

    def set_shared_font(self, f: Optional[tkfont.Font]) -> None:
        """Bloquea la fuente (misma para todos los botones). Gobernado por CONFIG."""
        self._shared_font = f
        self.after_idle(self._redraw)

    def clear_shared_font(self) -> None:
        self._shared_font = None
        self.after_idle(self._redraw)

    def _set_hover(self, v: bool):
        self._hover = bool(v)
        self.after_idle(self._redraw)

    def _set_pressed(self, v: bool):
        self._pressed = bool(v)
        self.after_idle(self._redraw)

    def _on_click(self, _):
        was_down = self._pressed
        self._pressed = False
        self.after_idle(self._redraw)
        if was_down and callable(self._cmd):
            try:
                self._cmd()
            except Exception:
                pass

    def _fit_font(self, w: int, h: int) -> tkfont.Font:
        """
        Ajusta la fuente para que el texto quepa al ancho del botón.
        - Siempre gobernado por CONFIG
        - Relativo a PHI para padding
        - Mantiene referencia viva (evita texto invisible)
        """
        fam = CONFIG["utilities"]["family"] or CONFIG["utilities"]["fallback_family"]
        min_px = int(CONFIG["utilities"]["min_px"])

        # Base size preferido: el que nos pasen vía tuple (resize), si existe
        base_size = None
        if isinstance(self._font, (tuple, list)) and len(self._font) >= 2:
            try:
                base_size = int(self._font[1])
            except Exception:
                base_size = None

        # padding cómodo (φ) + relativo al alto
        pad = max(int(CONFIG["layout"]["btn_text_pad_px"]), int(h * float(CONFIG["layout"]["btn_text_pad_rel_h"])))
        target_w = max(10, int(w) - 2 * pad)

        style = CONFIG["utilities"]["btn"]
        from typing import cast, Literal
        weight = cast(Literal["normal","bold"], "bold" if bool(style["bold"]) else "normal")
        slant  = cast(Literal["roman","italic"], "italic" if bool(style["italic"]) else "roman")

        # tamaño base por altura, pero nunca menos que min_px
        size = int(base_size if base_size is not None else max(min_px, int(h * 0.55)))
        max_sz = max(min_px, int(h * 0.72))
        size = max(min_px, min(max_sz, size))

        f = tkfont.Font(family=fam, size=int(size), weight=weight, slant=slant)
        tw = int(f.measure(self._text) or 0)

        if tw > 0:
            scaled = int(size * (target_w / float(tw)))
            scaled = max(min_px, min(max_sz, scaled))
            f.configure(size=int(scaled))

            # fine tune (pocos pasos) para clavar el ajuste
            for _ in range(10):
                tw = int(f.measure(self._text) or 0)
                if tw > target_w and f.cget("size") > min_px:
                    f.configure(size=f.cget("size") - 1)
                elif tw < target_w * 0.93 and f.cget("size") < max_sz:
                    f.configure(size=f.cget("size") + 1)
                else:
                    break

        self._tkfont_cache = f
        return f

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        r = max(0, min(int(r), (x2 - x1) // 2, (y2 - y1) // 2))
        pts = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return self.create_polygon(pts, smooth=True, **kw)

    def _redraw(self):
        try:
            self.delete("all")
        except tk.TclError:
            return

        w = max(1, int(self._wpx))
        h = max(1, int(self._hpx))
        r = int(CONFIG["layout"]["btn_radius_px"])

        face = (
            self._colors["face_down"]
            if self._pressed
            else (self._colors["face_hover"] if self._hover else self._colors["face"])
        )
        base = self._colors["base"]

        self._rounded_rect(0, 0, w, h, r, fill=base, outline="")
        self._rounded_rect(0, 0, w, h, r, fill=face, outline="")

        fnt = self._shared_font or self._fit_font(w, h)
        self.create_text(
            w // 2,
            h // 2,
            text=self._text,
            anchor="center",
            fill=self._colors["text"],
            font=fnt,
        )

# ──────────────────────────────────────────────────────────────────────────────
# Aviso flotante con desvanecimiento
class FadePopover(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.withdraw()
        self.overrideredirect(True)
        self.configure(bg=CONFIG["colors"]["app_bg"])
        self.label = tk.Label(self, text="", bg=CONFIG["colors"]["app_bg"], fg=CONFIG["colors"]["text"],
                              font=_font(CONFIG['utilities']['family'], CONFIG['utilities']['min_px']))
        self.label.pack(padx=10, pady=6)
        try: self.attributes("-alpha", 0.0)
        except Exception: pass
        self._alpha = 0.0; self._aft = None
        self._frame_ms = int(CONFIG["app"]["popover_frame_ms"])
        self._alpha_step = float(CONFIG["app"]["popover_alpha_step"])

    def show(self, text: str, x: int, y: int, ms: int = 2400):
        self.label.config(text=text); self.geometry(f"+{int(x)}+{int(y)}")
        self.deiconify(); self.lift(); self._alpha = 1.0
        try: self.attributes("-alpha", self._alpha)
        except Exception: pass
        if self._aft:
            try: self.after_cancel(self._aft)
            except Exception: pass
        visible_ms = int(CONFIG["app"]["close_delay_ms"])
        alpha_step = float(self._alpha_step)
        frame = int(self._frame_ms)
        steps_to_zero = max(1, int((1.0 / max(1e-6, alpha_step)) + 0.9999))
        def _start_fade():
            try: self._aft = self.after(frame, self._fade, steps_to_zero)
            except Exception: pass
        try: self._aft = self.after(int(visible_ms), _start_fade)
        except Exception: _start_fade()

    def _fade(self, steps_left: int):
        if steps_left <= 0:
            try: self.attributes("-alpha", 0.0)
            except Exception: pass
            self.withdraw(); return
        step = float(self._alpha_step)
        self._alpha = max(0.0, self._alpha - step)
        try: self.attributes("-alpha", self._alpha)
        except Exception: pass
        self._aft = self.after(int(self._frame_ms), self._fade, steps_left-1)

    def hold_above_widget(self, widget, text: str, *, margin: int = 6, align: str = "center"):
        self.label.config(text=text)
        if self._aft:
            try: self.after_cancel(self._aft)
            except Exception: pass
        self._aft = None
        self._alpha = 1.0
        try: self.attributes("-alpha", 1.0)
        except Exception: pass
        self.deiconify(); self.lift()
        try:
            self.update_idletasks()
            w = max(1, self.winfo_width()); h = max(1, self.winfo_height())
            wx = widget.winfo_rootx(); wy = widget.winfo_rooty(); ww = max(1, widget.winfo_width())
            gx = int(wx + (ww - w)/2) if align == "center" else int(wx)
            gy = int(wy - h - int(margin))
            self.geometry(f"+{gx}+{gy}")
        except Exception: pass

    def fade_now(self):
        if self._aft:
            try: self.after_cancel(self._aft)
            except Exception: pass
        frame = int(self._frame_ms)
        step = float(self._alpha_step)
        steps_to_zero = max(1, int((self._alpha / max(1e-6, step)) + 0.9999))
        self._aft = self.after(frame, self._fade, steps_to_zero)

# ──────────────────────────────────────────────────────────────────────────────
# SlideButton neumórfico por fila (Canvas) – Tokens 100% de CONFIG["controls"]["slide"]
class SlideButton(tk.Canvas):
    def __init__(self, parent, *, state="Unir", on_change=None, row_height_px=22):
        h = max(18, int(row_height_px) - 4)
        w = max(80, int(h * 2.2))
        super().__init__(parent, width=w, height=h, bd=0, highlightthickness=0,
                         bg=CONFIG["colors"]["box_bg"])

        t = CONFIG["controls"]["slide"]
        self._r = int(t["radius_px"])
        self._pad = int(t["pad_px"])
        self._trk_off = t["track_off"]
        self._trk_on  = t["track_on"]
        self._knob    = t["knob"]
        self._bdr     = t["border"]
        self._txt_on  = t["text_on"]
        self._txt_off = t["text_off"]
        self._sd_l    = t["shadow_light"]
        self._sd_d    = t["shadow_dark"]
        self._sd_off  = int(t["shadow_offset_px"])
        self._sd_in   = int(t["shadow_inner_offset_px"])
        self._anim_ms = int(t["anim_ms"])
        self._frame_ms = int(t["frame_ms"])
        self._notify_ms = int(t["notify_ms"])
        self._font = _font(CONFIG["utilities"]["family"], max(int(CONFIG["utilities"]["min_px"]), int(h * 0.42)))

        self._pos = 0.0 if state == "Unir" else 1.0
        self._dragging = False
        self._x0 = 0
        self._on_change = on_change

        self.bind("<Configure>", lambda e: self.after_idle(self._redraw))
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self._redraw()

    def value(self) -> str:
        return "Dividir" if self._pos >= 0.5 else "Unir"

    def _rounded_rect(self, x0, y0, x1, y1, r, **kw):
        r = int(max(0, min(r, (min(x1 - x0, y1 - y0) // 2))))
        self.create_rectangle(x0 + r, y0, x1 - r, y1, **kw)
        self.create_rectangle(x0, y0 + r, x1, y1 - r, **kw)
        self.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, style="pieslice", **kw)
        self.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, style="pieslice", **kw)
        self.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, style="pieslice", **kw)
        self.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, style="pieslice", **kw)

    def _redraw(self):
        try:
            self.delete("all")
        except tk.TclError:
            return

        w = max(74, int(self.winfo_width()))
        h = max(18, int(self.winfo_height()))
        r = int(self._r)
        pad = int(self._pad)

        # Pista base sin bordes visibles para evitar la 'cuadrícula'
        self._rounded_rect(0, 0, w - 1, h - 1, r, fill=self._trk_off, outline="")

        # Mitad activa (derecha)
        self._rounded_rect(int(w * 0.5), 1, w - 2, h - 2, max(0, r - 2), fill=self._trk_on, outline="")

        # Etiquetas
        self.create_text(int(w * 0.25), h // 2, text="Unir",
                         fill=(self._txt_on if self._pos < 0.5 else self._txt_off), font=self._font)
        self.create_text(int(w * 0.75), h // 2, text="Dividir",
                         fill=(self._txt_on if self._pos >= 0.5 else self._txt_off), font=self._font)

        # Knob con sombras suaves
        knob_w = max(26, int((w - 2 * pad) * 0.46))
        knob_h = h - 2 * pad
        x0 = pad + int((w - knob_w - 2 * pad) * self._pos)
        y0 = pad
        x1 = x0 + knob_w
        y1 = y0 + knob_h

        self._rounded_rect(x0 + 1, y0 + 1, x1 + 1, y1 + 1, max(0, r - 3), fill=self._sd_d, outline="")
        self._rounded_rect(x0 - 1, y0 - 1, x1 - 1, y1 - 1, max(0, r - 3), fill=self._sd_l, outline="")
        self._rounded_rect(x0, y0, x1, y1, max(0, r - 3), fill=self._knob, outline="")

    def _animate_to(self, target: float, *, duration_ms: int | None = None):
        # Animación corta y fluida
        try:
            if hasattr(self, "_anim") and getattr(self, "_anim"):
                self.after_cancel(getattr(self, "_anim"))
        except Exception:
            pass

        target = 1.0 if target >= 0.5 else 0.0
        duration_ms = int(self._anim_ms if duration_ms is None else duration_ms)
        steps = max(5, duration_ms // max(1, int(self._frame_ms)))
        start = float(self._pos)
        delta = (target - start) / steps

        def _step(k=0):
            if k >= steps:
                self._pos = target
                self._redraw()
                return
            self._pos = start + delta * (k + 1)
            self._redraw()
            self._anim = self.after(int(self._frame_ms), _step, k + 1)

        _step()

    def _press(self, e):
        self._dragging = True
        self._x0 = e.x

    def _drag(self, e):
        if not self._dragging:
            return
        w = max(74, int(self.winfo_width()))
        pad = int(self._pad)
        travel = max(1, w - 2 * pad - max(26, int((w - 2 * pad) * 0.46)))
        rel = min(1.0, max(0.0, (e.x - pad) / float(travel)))
        self._pos = rel
        self._redraw()

    def _release(self, e):
        if not self._dragging:
            return
        self._dragging = False

        if abs(e.x - self._x0) < 6:
            target = 0.0 if self._pos >= 0.5 else 1.0
        else:
            target = 1.0 if self._pos >= 0.5 else 0.0

        self._animate_to(target)

        def _notify():
            if callable(self._on_change):
                try:
                    self._on_change(self.value())
                except Exception:
                    pass

        self.after(int(self._notify_ms), _notify)


# ──────────────────────────────────────────────────────────────────────────────
# I/O de texto
def _read_text(p: Path) -> str:
    for enc in ("utf-8-sig","utf-8","utf-16","latin-1","cp1252"):
        try: return Path(p).read_text(encoding=enc)
        except Exception: pass
    return Path(p).read_bytes().decode("utf-8", "ignore")

def _write_text(p: Path, s: str, like: Optional[Path] = None):
    enc = "utf-8"
    try:
        if like is not None:
            rb = Path(like).read_bytes()
            if rb.startswith(b"\\xff\\xfe") or rb.startswith(b"\\xfe\\xff"):
                enc = "utf-16"
    except Exception:
        pass
    Path(p).write_text(s, encoding=enc, newline="\n")

def split_text_smart(txt: str, parts: int) -> list[str]:
    s = (txt or "").replace("\\r\\n","\\n"); n = max(2, int(parts)); L = len(s)
    if L == 0: return ["" for _ in range(n)]
    target = L / n; out = []; start = 0
    for i in range(n-1):
        ideal = int(round((i+1)*target)); span = max(64, int(target*0.12))
        right_space = s.find(" ", ideal, min(L, ideal+span))
        left_space  = s.rfind(" ", max(0, ideal-span), ideal)
        candidates = [c for c in (right_space, left_space) if c != -1]
        if not candidates:
            rn = s.find("\\n", ideal, min(L, ideal+span)); ln = s.rfind("\\n", max(0, ideal-span), ideal)
            if rn != -1: candidates.append(rn)
            if ln != -1: candidates.append(ln)
        cut = min(candidates, key=lambda c: abs(c-ideal)) if candidates else max(start, min(L, ideal))
        if cut <= start: cut = max(start, min(L, ideal))
        out.append(s[start:cut].rstrip()); start = cut+1 if cut < L else cut
    out.append(s[start:]); return out

# ──────────────────────────────────────────────────────────────────────────────
# Tabla con Slide por fila y menú contextual
class FilesTable(ttk.Frame):
    COLS = ("drag", "archivo", "modo", "partes")
    def __init__(self, parent):
        super().__init__(parent)
        c = CONFIG["colors"]; lay = CONFIG["layout"]
        style = ttk.Style(self)
        try: style.theme_use('clam')
        except Exception: pass

        # fuentes y métricas igual a homólogos
        fam = CONFIG["utilities"]["family"]
        min_px = int(CONFIG["utilities"]["min_px"])
        row_px = max(min_px, int(parent.winfo_height() * float(CONFIG["utilities"]["row"]["rel_px"])))
        head_px = max(row_px, int(parent.winfo_height() * float(CONFIG["utilities"]["head"]["rel_px"])))
        font_row  = _font(fam, row_px,  bold=CONFIG["utilities"]["row"]["bold"])
        font_head = _font(fam, head_px, bold=CONFIG["utilities"]["head"]["bold"])

        rowheight = max(int(lay["row_min_px"]),
                        int(self.winfo_toplevel().winfo_height() * float(lay["row_height_rel"])))
        rowheight = min(int(lay.get("row_max_px", 50)), int(rowheight))

        style.configure("Miau.Treeview",
                        background=c["box_bg"], fieldbackground=c["box_bg"],
                        foreground=c["text"], borderwidth=0, relief="flat",
                        rowheight=rowheight, font=font_row)
        style.layout("Miau.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.configure("Miau.Treeview.Heading",
                        background=c["app_bg"], foreground=c["text"],
                        font=font_head, relief="flat", borderwidth=0)
        style.map("Miau.Treeview",
                  background=[("selected", c["box_bg"])],
                  foreground=[("selected", c["text"])])

        style.configure("Flat.Vertical.TScrollbar",
                        gripcount=0, background=c["app_bg"],
                        bordercolor=c["app_bg"], darkcolor=c["app_bg"], lightcolor=c["app_bg"],
                        troughcolor=c["scroll_trough"], arrowcolor=c["scroll_arrow"],
                        relief="flat", borderwidth=0)

        self.tree = ttk.Treeview(self, columns=self.COLS, show="headings",
                                 style="Miau.Treeview", selectmode="extended")
        self.vs = ttk.Scrollbar(self, orient="vertical", command=self._on_scroll,
                                style="Flat.Vertical.TScrollbar")
        def _yview(*args):
            self.vs.set(*args)
            self.after_idle(self._reposition_all)
        self.tree.configure(yscrollcommand=_yview)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vs.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1); self.columnconfigure(0, weight=1)

        self.tree.heading("drag", text="⇅"); self.tree.heading("archivo", text="Archivo")
        self.tree.heading("modo", text="Modo"); self.tree.heading("partes", text="Partes")

        # zebra
        self.tree.tag_configure("odd", background=self._alpha(c["box_bg"], lay["zebra_alpha"]))
        self.tree.tag_configure("even", background=c["box_bg"])

        # estado
        self._slides: dict[str, SlideButton] = {}; self._paths: dict[str, Path] = {}
        self._spin_parts: Optional[tk.Spinbox] = None; self._edit_iid: Optional[str] = None
        self._popover = FadePopover(self); self._dragging: Optional[str] = None

        # menú contextual
        self._menu = tk.Menu(self, tearoff=False, bg=c["app_bg"], fg=c["text"],
                             activebackground=c["btn_face_hover"])
        self._menu.add_command(label="Quitar", command=self._menu_remove_selected)

        # binds
        self.tree.bind("<Button-1>", self._on_click); self.tree.bind("<B1-Motion>", self._on_drag)
        self.tree.bind("<ButtonRelease-1>", self._on_drag_end)
        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<Button-3>", self._on_right_click)
        self.bind("<Configure>", lambda _e: self.after_idle(self._reposition_all))

        self._apply_columns()

    def _alpha(self, hex_color: str, alpha: float) -> str:
        try:
            hc = hex_color.lstrip("#")
            r = int(hc[0:2], 16)/255.0; g = int(hc[2:4], 16)/255.0; b = int(hc[4:6], 16)/255.0
            r = r + (1.0 - r) * alpha; g = g + (1.0 - g) * alpha; b = b + (1.0 - b) * alpha
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        except Exception:
            return hex_color

    def refresh_style_metrics(self, rowheight_px: int, head_font, row_font):
        style = ttk.Style(self)
        # cap duro de altura de fila (evita filas gigantes al redimensionar)
        _min = int(CONFIG["layout"].get("row_min_px", 10))
        _max = int(CONFIG["layout"].get("row_max_px", 50))
        rowheight_px = max(_min, min(_max, int(rowheight_px)))
        style.configure("Miau.Treeview", rowheight=int(rowheight_px), font=row_font)
        style.configure("Miau.Treeview.Heading", font=head_font)
        self._reposition_all()

    def _apply_columns(self):
        """
        Ajusta columnas sin que la suma supere el ancho disponible (evita scroll horizontal).
        100% gobernado por CONFIG y proporciones relativas a PHI.
        """
        w = max(1, int(self.winfo_width() or self.winfo_toplevel().winfo_width()))
        sb = int(CONFIG["layout"]["scrollbar_fallback_w_px"])
        pad = int(CONFIG["layout"]["table_pad_px"])
        avail = max(1, w - sb - (pad * 2))
        mins = CONFIG["columns"]["min_px"]

        c_drag = int(CONFIG["layout"]["drag_min_px"])
        min_file  = int(mins["archivo"])
        min_mode  = int(mins["modo"])
        min_parts = int(mins["partes"])

        # Proporciones phi-friendly (modo y partes chicos, archivo manda)
        c_mode  = max(min_mode,  int(avail * (1/(PHI*3.2))))
        c_parts = max(min_parts, int(avail * (1/(PHI*4.2))))

        c_file = avail - (c_drag + c_mode + c_parts)

        # Si archivo no alcanza su mínimo, recorta primero modo/partes (hasta sus mínimos)
        if c_file < min_file:
            deficit = min_file - c_file
            flex_mode  = max(0, c_mode  - min_mode)
            flex_parts = max(0, c_parts - min_parts)
            flex_total = flex_mode + flex_parts
            if flex_total > 0:
                cut_mode = int(deficit * (flex_mode / flex_total))
                cut_parts = deficit - cut_mode
                c_mode  = max(min_mode,  c_mode  - cut_mode)
                c_parts = max(min_parts, c_parts - cut_parts)
            c_file = avail - (c_drag + c_mode + c_parts)

        # Último seguro: nunca exceder avail
        total = c_drag + c_file + c_mode + c_parts
        if total > avail:
            c_file = max(1, c_file - (total - avail))

        self.tree.column("drag",    width=int(c_drag),  anchor="center", stretch=False)
        self.tree.column("archivo", width=int(max(1, c_file)), anchor="w", stretch=True)
        self.tree.column("modo",    width=int(max(1, c_mode)), anchor="center", stretch=False)
        self.tree.column("partes",  width=int(max(1, c_parts)), anchor="center", stretch=False)
    def add_files(self, paths: List[Path]):
        for p in paths:
            iid = self.tree.insert("", "end", values=("⇅", Path(p).name, "Unir", "—"))
            self._paths[iid] = Path(p)
            idx = self.tree.index(iid)
            self.tree.item(iid, tags=("odd" if idx%2 else "even",))
            self._ensure_slide(iid)
        self.after_idle(self._reposition_all)

    def get_rows(self):
        rows = []
        for iid in self.tree.get_children(""):
            _, name, modo, parts = self.tree.item(iid, "values")
            p = self._paths.get(iid, Path(name))
            n = None if parts in ("—","") else int(parts)
            rows.append((p, modo, n))
        return rows

    def remove_selected(self):
        sel = self.tree.selection()
        for iid in sel:
            if iid in self._slides:
                try: self._slides[iid].destroy()
                except Exception: pass
                self._slides.pop(iid, None)
            self._paths.pop(iid, None)
            self.tree.delete(iid)
        self.after_idle(self._reposition_all)

    # contexto + DnD
    def _on_right_click(self, e):
        try: self.tree.selection_set(self.tree.identify_row(e.y))
        except Exception: pass
        self._menu.tk_popup(e.x_root, e.y_root)

    def _menu_remove_selected(self): self.remove_selected()

    def _on_click(self, e):
        self._dragging = self.tree.identify_row(e.y)

    def _on_drag(self, e):
        if not self._dragging: return
        target = self.tree.identify_row(e.y)
        if target and target != self._dragging:
            self.tree.move(self._dragging, "", self.tree.index(target))

    def _on_drag_end(self, _e):
        self._dragging = None
        for i, iid in enumerate(self.tree.get_children("")):
            self.tree.item(iid, tags=("odd" if i%2 else "even",))
        self.after_idle(self._reposition_all)

    # ediciones
    def _on_double_click(self, e):
        col = self.tree.identify_column(e.x); iid = self.tree.identify_row(e.y)
        if not iid: return
        if col == "#4":  # partes
            if self.tree.set(iid, "modo") != "Dividir": return
            x, y, w, h = self.tree.bbox(iid, col)
            self._show_parts_spin(iid, x, y, w, h)

    def _show_parts_spin(self, iid: str, x: int, y: int, w: int, h: int):
        if self._spin_parts and self._spin_parts.winfo_exists():
            try: self._spin_parts.destroy()
            except Exception: pass
        curr = self.tree.set(iid, "partes")
        if curr in ("—",""): curr = "2"
        sb = tk.Spinbox(self, from_=2, to=99, increment=1,
                        font=(CONFIG["utilities"]["family"], int(CONFIG["utilities"]["min_px"])),
                        width=6, justify="center")
        tx, ty = self.tree.winfo_x(), self.tree.winfo_y()
        pad = 2
        bw = max(24, int(w) - 2 * pad)
        bh = max(18, int(h) - 2 * pad)
        sb.place(x=int(tx) + int(x) + pad, y=int(ty) + int(y) + pad, width=bw, height=bh)
        sb.delete(0, "end"); sb.insert(0, curr); sb.focus_set()
        self._spin_parts = sb
        sb.bind("<Return>", lambda _e: self._commit_parts(iid))
        sb.bind("<Escape>", lambda _e: self._cancel_parts_spin())
        sb.bind("<FocusOut>", lambda _e: self._cancel_parts_spin())

    def _commit_parts(self, iid: str):
        if not self._spin_parts: return
        try: n = max(2, int(self._spin_parts.get()))
        except Exception: n = 2
        self.tree.set(iid, "partes", str(n))
        self._cancel_parts_spin()

    def _cancel_parts_spin(self):
        if self._spin_parts and self._spin_parts.winfo_exists():
            try: self._spin_parts.destroy()
            except Exception: pass
        self._spin_parts = None

    # Slide por fila
    def _ensure_slide(self, iid: str):
        if iid in self._slides and self._slides[iid].winfo_exists(): return
        cur = self.tree.set(iid, "modo") or "Unir"
        # altura basada en rowheight de estilo
        rowh = int(ttk.Style(self).lookup("Miau.Treeview", "rowheight") or 22)
        sb = SlideButton(self, state=cur, row_height_px=rowh,
                         on_change=lambda v, _iid=iid: self._apply_mode(_iid, v))
        self._slides[iid] = sb
        try:
            sb.bind("<Enter>", lambda e, _iid=iid, _sb=sb: self._on_slide_hover(_iid, _sb))
            sb.bind("<Leave>", lambda e, _iid=iid: self._on_slide_leave(_iid))
        except Exception:
            pass
    def _on_slide_hover(self, iid: str, sb):
        try:
            mode = self.tree.set(iid, "modo") or "Unir"
        except Exception:
            mode = "Unir"
        if mode == "Dividir":
            msg = "Modo: Dividir • contador activo"
        else:
            msg = "Modo: Unir • se empareja con el siguiente Unir"
        try:
            self._popover.hold_above_widget(sb, msg, margin=6, align="center")
        except Exception:
            pass

    def _on_slide_leave(self, iid: str):
        try:
            self._popover.fade_now()
        except Exception:
            pass



    def _apply_mode(self, iid: str, mode: str):
        self.tree.set(iid, "modo", mode)
        if mode == "Dividir":
            if self.tree.set(iid, "partes") in ("—",""): self.tree.set(iid, "partes", "2")
            msg = "Modo: Dividir • contador activo"
        else:
            self.tree.set(iid, "partes", "—")
            msg = "Modo: Unir • se empareja con el siguiente Unir"
        sb = self._slides.get(iid)
        if sb and sb.winfo_exists():
            try:
                self._popover.hold_above_widget(sb, msg, margin=6, align="center")
            except Exception:
                pass
    def _reposition_all(self):
        for iid, sb in list(self._slides.items()):
            if not sb.winfo_exists():
                self._slides.pop(iid, None); continue
            bbox = self.tree.bbox(iid, "#3")
            if not bbox:
                sb.place_forget(); continue
            x, y, w, h = bbox
            tx, ty = self.tree.winfo_x(), self.tree.winfo_y()
            pad = 2
            bw = max(1, int(w) - 2 * pad)
            bh = max(1, int(h) - 2 * pad)
            sb.place(x=int(tx) + int(x) + pad, y=int(ty) + int(y) + pad, width=bw, height=bh)
            try:
                sb.tk.call("raise", sb._w)
            except Exception:
                pass
    def _on_scroll(self, *args):
        self.tree.yview(*args)
        self.after_idle(self._reposition_all)

# ──────────────────────────────────────────────────────────────────────────────
# App
class App(tk.Tk):
    def __init__(self, files: Optional[List[Path]] = None):
        super().__init__(); self.withdraw()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ww, wh = window_geometry("fusionador", sw, sh)
        self.geometry(f"{ww}x{wh}+80+60")
        self.minsize(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"])
        try:
            apply_theme(self, app="fusionador", title=CONFIG["window"]["title"], icon=CONFIG["window"].get("icon_name"))
        except Exception:
            pass
        self.configure(bg=CONFIG["colors"]["app_bg"])
        self.title(CONFIG["window"]["title"])

        self._rw, self._rh = _metrics(ww, wh); rw, rh = self._rw, self._rh
        fam = CONFIG["utilities"]["family"]
        self.fonts = {
            "title": _font(fam, rh(CONFIG["utilities"]["title"]["rel_px"]), bold=CONFIG["utilities"]["title"]["bold"], italic=CONFIG["utilities"]["title"]["italic"]),
            "sub":   _font(fam, rh(CONFIG["utilities"]["sub"]["rel_px"]),   bold=CONFIG["utilities"]["sub"]["bold"], italic=CONFIG["utilities"]["sub"]["italic"]),
            "head":  _font(fam, rh(CONFIG["utilities"]["head"]["rel_px"]),  bold=CONFIG["utilities"]["head"]["bold"]),
            "row":   _font(fam, rh(CONFIG["utilities"]["row"]["rel_px"]),   bold=CONFIG["utilities"]["row"]["bold"]),
            "btn":   _font(fam, rh(CONFIG["utilities"]["btn"]["rel_px"]),   bold=CONFIG["utilities"]["btn"]["bold"]),
        }

        self._shared_btn_font = None

        pad = int(max(4, CONFIG["layout"]["row_pad_min_px"]))
        c = CONFIG["colors"]

        # centro
        self.table = FilesTable(self)
        self.table.grid(row=1, column=0, sticky="nsew", padx=pad, pady=pad)

        # pie (botones a la derecha, como homólogos)
        footer = tk.Frame(self, bg=c["app_bg"], bd=0, highlightthickness=0)
        footer.grid(row=2, column=0, sticky="sew"); footer.columnconfigure(0, weight=1)
        btns = tk.Frame(footer, bg=c["app_bg"]); btns.grid(row=0, column=1, sticky="e", padx=pad, pady=pad)

        bw0, bh0 = shared_button_box_px(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"], CONFIG)
        btn_fs0 = shared_button_font_px(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"], CONFIG)
        fam_btn = CONFIG["utilities"]["family"] or CONFIG["utilities"]["fallback_family"]
        weight = "bold" if bool(CONFIG["utilities"]["btn"].get("bold", True)) else "normal"
        slant  = "italic" if bool(CONFIG["utilities"]["btn"].get("italic", False)) else "roman"
        self._shared_btn_font = tkfont.Font(family=fam_btn, size=int(btn_fs0), weight=weight, slant=slant)

        self.btn_pick = FlatButton(btns, "Añadir archivos…", command=self._add_files_dialog, bg=c["app_bg"],
                                   width=bw0, height=bh0,
                                   radius=CONFIG["layout"]["btn_radius_px"], font=self._shared_btn_font)
        self.btn_pick.set_shared_font(self._shared_btn_font)
        self.btn_pick.grid(row=0, column=0, padx=(0, 12))

        self.btn_run  = FlatButton(btns, "Ejecutar", command=self._run, bg=c["app_bg"],
                                   width=bw0, height=bh0,
                                   radius=CONFIG["layout"]["btn_radius_px"], font=self._shared_btn_font)
        self.btn_run.set_shared_font(self._shared_btn_font)
        self.btn_run.grid(row=0, column=1)

        self.rowconfigure(1, weight=1); self.columnconfigure(0, weight=1)
        self.bind("<Configure>", self._on_root_configure)
        self.deiconify()

        if files: self.table.add_files([Path(p) for p in files])
    def _sync_shared_btn_font(self, bw: int, bh: int) -> None:
        """Unifica el tamaño de letra de los botones: todos usan el tamaño del que quedó más pequeño."""
        try:
            # forzamos cálculo individual con su ancho/alto actual
            btns = [self.btn_pick, self.btn_run]
            sizes = []
            for b in btns:
                try:
                    b.clear_shared_font()
                except Exception:
                    pass
                try:
                    f = b._fit_font(int(bw), int(bh))
                    sizes.append(int(f.cget("size")))
                except Exception:
                    pass

            if not sizes:
                return

            shared_sz = int(min(sizes))
            fam_btn = (CONFIG["utilities"].get("family") or CONFIG["utilities"].get("fallback_family") or "Segoe UI")
            style = CONFIG["utilities"]["btn"]
            # Literales para evitar warnings de type-checkers
            from typing import cast, Literal
            weight = cast(Literal["normal","bold"], "bold" if bool(style.get("bold")) else "normal")
            slant  = cast(Literal["roman","italic"], "italic" if bool(style.get("italic")) else "roman")

            if getattr(self, "_shared_btn_font", None) is None:
                self._shared_btn_font = tkfont.Font(family=fam_btn, size=shared_sz, weight=weight, slant=slant)
            else:
                try:
                    if int(self._shared_btn_font.cget("size")) != shared_sz:
                        self._shared_btn_font.configure(size=shared_sz)
                    # por si cambió familia/estilo
                    self._shared_btn_font.configure(family=fam_btn, weight=weight, slant=slant)
                except Exception:
                    self._shared_btn_font = tkfont.Font(family=fam_btn, size=shared_sz, weight=weight, slant=slant)

            for b in btns:
                try:
                    b.set_shared_font(self._shared_btn_font)
                except Exception:
                    pass
        except Exception:
            return


    def _on_root_configure(self, *_):
        ww = max(self.winfo_width(),  CONFIG["window"]["min_width"])
        wh = max(self.winfo_height(), CONFIG["window"]["min_height"])
        self._rw, self._rh = _metrics(ww, wh); rw, rh = self._rw, self._rh
        fam = CONFIG["utilities"]["family"]
        self.fonts = {
            "title": _font(fam, rh(CONFIG["utilities"]["title"]["rel_px"]), bold=CONFIG["utilities"]["title"]["bold"], italic=CONFIG["utilities"]["title"]["italic"]),
            "sub":   _font(fam, rh(CONFIG["utilities"]["sub"]["rel_px"]),   bold=CONFIG["utilities"]["sub"]["bold"], italic=CONFIG["utilities"]["sub"]["italic"]),
            "head":  _font(fam, rh(CONFIG["utilities"]["head"]["rel_px"]),  bold=CONFIG["utilities"]["head"]["bold"]),
            "row":   _font(fam, rh(CONFIG["utilities"]["row"]["rel_px"]),   bold=CONFIG["utilities"]["row"]["bold"]),
            "btn":   _font(fam, rh(CONFIG["utilities"]["btn"]["rel_px"]),   bold=CONFIG["utilities"]["btn"]["bold"]),
        }
        bw, bh = shared_button_box_px(ww, wh, CONFIG)
        btn_sz = shared_button_font_px(ww, wh, CONFIG)
        fam_btn = CONFIG["utilities"]["family"] or CONFIG["utilities"]["fallback_family"]
        weight = "bold" if bool(CONFIG["utilities"]["btn"].get("bold", True)) else "normal"
        slant  = "italic" if bool(CONFIG["utilities"]["btn"].get("italic", False)) else "roman"
        if getattr(self, "_shared_btn_font", None) is None:
            self._shared_btn_font = tkfont.Font(family=fam_btn, size=int(btn_sz), weight=weight, slant=slant)
        else:
            try:
                self._shared_btn_font.configure(family=fam_btn, weight=weight, slant=slant, size=int(btn_sz))
            except Exception:
                self._shared_btn_font = tkfont.Font(family=fam_btn, size=int(btn_sz), weight=weight, slant=slant)
        for b in (self.btn_pick, self.btn_run):
            try:
                b.config(width=bw, height=bh)
                b.set_shared_font(self._shared_btn_font)
            except Exception:
                pass
        try:
            rowheight = max(int(CONFIG["layout"]["row_min_px"]),
                            int(self.winfo_height() * float(CONFIG["layout"]["row_height_rel"])))
            rowheight = min(int(CONFIG["layout"].get("row_max_px", 50)), int(rowheight))
            self.table.refresh_style_metrics(rowheight, self.fonts["head"], self.fonts["row"])
        except Exception: pass

    def _add_files_dialog(self):
        paths = filedialog.askopenfilenames(title="Selecciona archivos de texto", filetypes=[("Texto","*.txt"),("Todos","*.*")])
        if not paths: return
        self.table.add_files([Path(p) for p in paths])

    def _run(self):

        rows = self.table.get_rows()
        if not rows:
            messagebox.showinfo("Miausoft", "Agrega archivos primero."); return
        # Salida: misma carpeta que input
        # unir por parejas consecutivas de "Unir"
        indices_unir = [i for i, (_, m, _) in enumerate(rows) if m == "Unir"]
        paired = []; used = set()
        for i in indices_unir:
            if i in used: continue
            j = next((k for k in indices_unir if k > i and k not in used), None)
            if j is None: continue
            paired.append((i, j)); used.add(i); used.add(j)

        # total de escrituras (pares unidos + partes divididas)
        total_units = len(paired)
        for p, m, n in rows:
            if m == "Dividir":
                try: nn = max(2, int(n or 2))
                except Exception: nn = 2
                total_units += nn

        dlg = None
        try:
            dlg = ProgressDialog(self)
            dlg.set_title("Procesando…")
            dlg.set_subtitle("Preparando")
            dlg.set_global_span(max(1, int(total_units)))
            dlg.show()

            done = 0

            # unir
            for (i, j) in paired:
                p1, _, _ = rows[i]; p2, _, _ = rows[j]
                merged = f"{_read_text(p1).rstrip()}\n\n{_read_text(p2).rstrip()}\n"
                _write_text((p1.parent) / f"{p1.stem}__{p2.stem}_unidos.txt", merged, like=p1)
                done += 1; dlg.set_done(done)
                dlg.set_subtitle(f"Unidos: {p1.name} + {p2.name}")

            # dividir
            for p, m, n in rows:
                if m != "Dividir": continue
                try: n = max(2, int(n or 2))
                except Exception: n = 2
                txt = _read_text(p)
                parts = split_text_smart(txt, n)
                for idx, part in enumerate(parts, 1):
                    _write_text((p.parent) / f"{p.stem}_part{idx:02d}.txt", part, like=p)
                    done += 1; dlg.set_done(done)
                    dlg.set_subtitle(f"Dividido: {p.name} ({idx}/{n})")

            lonely = [rows[i][0].name for i in indices_unir if i not in used]
            msg = "Listo."
            if lonely: msg += "\\nSin pareja (no se unieron): " + ", ".join(lonely)

            dlg.set_title("Listo")
            dlg.set_subtitle("Cambios aplicados")
            delay = int(CONFIG["app"]["close_delay_ms"])
            self.after(delay, dlg.close)
            None  # aviso final suprimido
        except Exception as ex:
            try:
                if dlg: dlg.close()
            except Exception:
                pass
            messagebox.showerror("Miausoft", f"Error al procesar: {ex}")
    

# ──────────────────────────────────────────────────────────────────────────────
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

        merged: List[str] = []
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
                        merged.append(p)
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
                        merged.append(p)
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
        return (merged if merged else files_norm), True

    except Exception:
        return list(files_in), True


def main():
    files = [Path(a) for a in sys.argv[1:] if os.path.isfile(a)]

    # Agrupar multi-selección cuando Explorer ejecuta 1 instancia POR archivo seleccionado.
    files, _launch = _merge_multiselect_startup(
        files,
        mutex_name=r"Local\MiausoftSuite_DivideFusiona_MultiSelect",
        spool_name="dividefusiona.paths.txt",
    )
    if not _launch:
        return

    App(files=files).mainloop()

if __name__ == "__main__":
    main()