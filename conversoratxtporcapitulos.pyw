# conversoratxtporcapitulos.py
#Este es un conversor de archivos PDF, DOC, DOCX, PPT, PPTX a TXT
#Este sí tiene una interfaz de inicio que consiste en una tabla de 3 columnas Archivo, inicio, fin, prefijo, y un selector para elegir el capítulo a procesar
#los archivos a procesar se agregan desde el botón "Agregar Archivos" (o toma los previamente seleccionados desde el menú contextual cuando este programa es ejecutado desde el menú contextual)
#y transforma eficientemente a txt.
#Solo aplica OCR de 75 dpi a archivos que tengan más de 10% de páginas con menos de 20 caracteres cada una.

import sys
import threading
import queue
from pathlib import Path
import ctypes
from ctypes import wintypes, byref
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
try:
    from PIL import Image, ImageTk  # type: ignore
    _HAS_PIL = True
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore
    _HAS_PIL = False
try:
    import ocrmypdf  # type: ignore
    _HAS_OCR = True
except Exception:
    ocrmypdf = None  # type: ignore
    _HAS_OCR = False
try:
    import winreg  # type: ignore
except Exception:
    winreg = None  # macOS/Linux: no Registry
from pypdf import PdfReader
from pypdf.generic import Destination, IndirectObject, DictionaryObject, ArrayObject
try:
    from ebooklib import epub  # type: ignore
    from ebooklib.epub import EpubHtml, Link as EpubLink  # type: ignore
    _HAS_EPUB = True
except Exception:
    epub = None  # type: ignore
    EpubHtml = object  # type: ignore
    EpubLink = object  # type: ignore
    _HAS_EPUB = False
try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:
    BeautifulSoup = None  # type: ignore
    _HAS_BS4 = False
import re
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import string
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Tuple, List, Optional
import json, time, zipfile, hashlib, tempfile, shutil, subprocess, xml.etree.ElementTree as ET

# ──────────────────────────────────────────────────────────────────────────────
# Extracción PDF con timeout sin crear un hilo por página (y apto para paralelismo)
class _PDFTextExtractor:
    """Hilo *daemon* reutilizable para extraer texto de una página PDF con timeout.
    Si una extracción se cuelga, se reemplaza el extractor y el hilo anterior queda como daemon."""
    def __init__(self):
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            page_obj, out, ev = self._q.get()
            try:
                out["text"] = page_obj.extract_text() or ""
            except Exception:
                out["text"] = ""
            finally:
                try:
                    ev.set()
                except Exception:
                    pass

    def extract(self, page_obj, timeout_s: float) -> tuple[bool, str]:
        out: dict = {}
        ev = threading.Event()
        try:
            self._q.put((page_obj, out, ev))
        except Exception:
            return True, ""
        if ev.wait(float(timeout_s)):
            return True, str(out.get("text", "") or "")
        return False, ""


class _PDFTextExtractorPool:
    def __init__(self, workers: int):
        self.workers = max(1, int(workers) if workers else 1)
        self._lock = threading.Lock()
        self._rr = 0
        self._extractors = [_PDFTextExtractor() for _ in range(self.workers)]

    def extract(self, page_obj, timeout_s: float) -> tuple[bool, str]:
        with self._lock:
            idx = self._rr
            self._rr = (self._rr + 1) % self.workers
            ex = self._extractors[idx]

        ok, txt = ex.extract(page_obj, timeout_s)
        if ok:
            return True, txt

        # Timeout: reemplazar extractor para no bloquear futuros.
        try:
            with self._lock:
                self._extractors[idx] = _PDFTextExtractor()
        except Exception:
            pass
        return False, ""


from config import CONFIG, PHI, shared_button_box_px, shared_button_font_px, attach_activation, ProgressDialog
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMSBT_NONE        = 0
DWMSBT_MAINWINDOW  = 2
DWMSBT_TRANSIENT   = 3
DWMSBT_TABBED      = 4

def _get_dwmapi():
    """Carga dwmapi con firmas seguras (evita crashes por ctypes)."""
    try:
        if os.name != "nt":
            return None
    except Exception:
        return None
    try:
        dll = ctypes.WinDLL("dwmapi", use_last_error=True)
    except Exception:
        return None

    # Prototipos (solo si existen)
    try:
        fn = getattr(dll, "DwmGetColorizationColor", None)
        if fn is not None:
            fn.argtypes = [ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.BOOL)]
            fn.restype = wintypes.HRESULT
    except Exception:
        pass
    try:
        fn = getattr(dll, "DwmSetWindowAttribute", None)
        if fn is not None:
            fn.argtypes = [wintypes.HWND, wintypes.DWORD, wintypes.LPCVOID, wintypes.DWORD]
            fn.restype = wintypes.HRESULT
    except Exception:
        pass
    return dll


_DWMAPI = None

def _accent_from_dwm():
    if not CONFIG["win11"]["use_accent_from_dwm"]:
        return "#0078d4"

    # 1) DwmGetColorizationColor (más fiable si está)
    try:
        global _DWMAPI
        if _DWMAPI is None:
            _DWMAPI = _get_dwmapi()
        if _DWMAPI is not None:
            fn = getattr(_DWMAPI, "DwmGetColorizationColor", None)
            if fn is not None:
                color = wintypes.DWORD()
                opaque = wintypes.BOOL()
                hr = fn(byref(color), byref(opaque))
                if int(hr) >= 0:
                    c = color.value & 0x00FFFFFF
                    r = (c >> 16) & 0xFF
                    g = (c >> 8) & 0xFF
                    b = c & 0xFF
                    return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        pass

    # 2) Registry fallback
    try:
        if winreg is None:
            raise Exception("winreg no disponible fuera de Windows")
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\DWM")
        val, _ = winreg.QueryValueEx(k, "ColorizationColor")
        r = (val >> 16) & 0xFF
        g = (val >> 8) & 0xFF
        b = val & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#0078d4"


def _set_win11_backdrop(hwnd: int, mode: str, prefer_dark=True):
    # Asegura que NUNCA crashee por ctypes (signature/handles)
    try:
        global _DWMAPI
        if _DWMAPI is None:
            _DWMAPI = _get_dwmapi()
        if _DWMAPI is None:
            return
        fn = getattr(_DWMAPI, "DwmSetWindowAttribute", None)
        if fn is None:
            return

        # Dark mode flag
        try:
            pv = ctypes.c_int(1 if prefer_dark else 0)
            fn(wintypes.HWND(int(hwnd)),
               wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE),
               byref(pv),
               ctypes.sizeof(pv))
        except Exception:
            pass

        # Backdrop type
        try:
            if mode == "mica":
                typ = ctypes.c_int(DWMSBT_MAINWINDOW)
            elif mode == "acrylic":
                typ = ctypes.c_int(DWMSBT_TRANSIENT)
            elif mode == "tabbed":
                typ = ctypes.c_int(DWMSBT_TABBED)
            else:
                typ = ctypes.c_int(DWMSBT_NONE)

            fn(wintypes.HWND(int(hwnd)),
               wintypes.DWORD(DWMWA_SYSTEMBACKDROP_TYPE),
               byref(typ),
               ctypes.sizeof(typ))
        except Exception:
            pass
    except Exception:
        return


# ──────────────────────────────────────────────────────────────────────────────
# Helpers tamaño/posición / fuentes
def _as_px(v, ref):
    try:
        v = float(v)
    except Exception:
        try:
            return max(1, int(v))
        except Exception:
            return 1
    if 0.0 < v <= 1.0:
        return max(1, int(round(v * ref)))
    return max(1, int(round(v)))

def _px_w(v, widget):
    try:
        ref = widget.winfo_width()
        if ref <= 1:
            ref = widget.winfo_reqwidth()
        if ref <= 1 and widget.winfo_toplevel():
            ref = widget.winfo_toplevel().winfo_width()
    except Exception:
        ref = 800
    return _as_px(v, ref)

def _px_h(v, widget):
    try:
        ref = widget.winfo_height()
        if ref <= 1:
            ref = widget.winfo_reqheight()
        if ref <= 1 and widget.winfo_toplevel():
            ref = widget.winfo_toplevel().winfo_height()
    except Exception:
        ref = 600
    return _as_px(v, ref)

def _font(family, size_px, bold=False, italic=False):
    style_bits = []
    if bold: style_bits.append("bold")
    if italic: style_bits.append("italic")
    return (family, max(CONFIG["utilities"]["min_px"], int(size_px)), " ".join(style_bits) if style_bits else "")

def _metrics(win_w, win_h):
    def rw(f): return max(1, int(win_w * f))
    def rh(f): return max(1, int(win_h * f))
    return rw, rh

# ──────────────────────────────────────────────────────────────────────────────
# ======= UTILIDADES PDF/EPUB =======
def _normalize_title_for_similarity(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[\s_]+", " ", t)
    t = re.sub(r"[^\w\sáéíóúüñ]", "", t, flags=re.UNICODE)
    return t.strip()

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None,
        _normalize_title_for_similarity(a),
        _normalize_title_for_similarity(b)
    ).ratio()

_ROMAN_MAP = {'m':1000,'d':500,'c':100,'l':50,'x':10,'v':5,'i':1}
def _roman_to_int(s: str):
    s = re.sub(r'[^ivxlcdm]', '', s.lower()); total = 0; prev = 0
    for ch in reversed(s):
        val = _ROMAN_MAP.get(ch, 0)
        if val < prev: total -= val
        else: total += val; prev = val
    return total if total > 0 else None

_NUM_RX = re.compile(r'(?:^|[\s\-:])([0-9]+|[ivxlcdm]{1,8})(?:\b|$)', re.I)
def _extract_numeric_tag(title: str):
    if not title: return None
    m = _NUM_RX.search(title or "")
    if not m: return None
    tok = m.group(1)
    return int(tok) if tok.isdigit() else _roman_to_int(tok)

def _merge_similar_contiguous(starts, threshold=0.50, max_gap=2):
    if not starts: return []
    merged = [starts[0]]
    for title, idx in starts[1:]:
        last_title, last_idx = merged[-1]
        n1 = _extract_numeric_tag(last_title); n2 = _extract_numeric_tag(title)
        if n1 is not None and n2 is not None and n1 != n2:
            merged.append((title, idx)); continue
        if (idx - last_idx) <= max_gap and _similar(title, last_title) >= threshold:
            continue
        merged.append((title, idx))
    return merged

def _filter_starts_by_min_gap(starts: List[tuple], min_gap: int) -> List[tuple]:
    if not starts or min_gap <= 0:
        return starts
    starts = sorted(starts, key=lambda t: t[1])
    kept = []
    last_idx = -10**9
    for t, idx in starts:
        if (idx - last_idx) >= min_gap:
            kept.append((t, idx))
            last_idx = idx
    return kept

def _chapter_ranges_from_starts(starts, total_pages):
    """
    Convierte [(title, idx0)] en [(title, ini1, fin1)], 1-based inclusive.
    Cierra en el PRÓXIMO inicio cuya página sea estrictamente mayor.
    """
    if not starts: return []
    ranges = []
    starts = sorted(starts, key=lambda t: t[1])
    n = len(starts)
    for i, (title, idx0) in enumerate(starts):
        s = idx0 + 1
        e = total_pages
        for j in range(i + 1, n):
            if starts[j][1] > idx0:
                e = starts[j][1]
                break
        if e >= s: ranges.append((title, s, e))
    return ranges

# ───────────────── PDF: Marcadores (vía PyMuPDF) ─────────────────
# ───────────────── PDF: Marcadores/miniaturas (PyMuPDF) ─────────────────
# PyMuPDF (fitz) es rápido, pero puede provocar crashes nativos con ciertos PDFs.
# Por estabilidad, se habilita SOLO si:
#   - MIAUSOFT_ENABLE_FITZ=1 en variables de entorno.
try:
    import os as _os
    if str(_os.environ.get("MIAUSOFT_ENABLE_FITZ", "")).strip() == "1":
        import fitz  # type: ignore  # PyMuPDF
        _HAS_FITZ = True
    else:
        _HAS_FITZ = False
except Exception:
    _HAS_FITZ = False


def _pdf_outline_starts_pymupdf(path_str: str) -> List[tuple]:
    """Devuelve marcadores filtrados por nivel como [(titulo, page_idx0)]."""
    if not _HAS_FITZ:
        return []
    try:
        with fitz.open(path_str) as doc:
            toc = doc.get_toc(simple=True) or []
            starts = []
            max_lvl = int(CONFIG["chapters"].get("max_bookmark_level", 9999))
            for item in toc:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                level, title, page1 = int(item[0] or 1), str(item[1] or "").strip(), int(item[2] or 0)
                if page1 >= 1 and level <= max_lvl:
                    starts.append(((title or "Capítulo"), page1 - 1))
            starts.sort(key=lambda t: t[1])
            # aplica separación mínima
            min_gap = int(CONFIG["chapters"].get("min_gap_pages", 0))
            starts = _filter_starts_by_min_gap(starts, min_gap)
            return starts
    except Exception:
        return []

# ───────────────── PDF: contar páginas ─────────────────
def _pdf_page_count(path_str: str) -> int:
    """Cuenta páginas de forma segura.
    - Prioriza pypdf (puro Python) para evitar crashes de backends C con PDFs problemáticos.
    - Usa PyMuPDF (fitz) SOLO si pypdf falla y está disponible.
    """
    try:
        reader = PdfReader(path_str)
        return len(reader.pages)
    except Exception:
        pass
    if _HAS_FITZ:
        try:
            import fitz  # type: ignore
            with fitz.open(path_str) as doc:
                return len(doc)
        except Exception:
            pass
    return 0


# ───────────────── PDF: Marcadores (vía pypdf/PyPDF2) ─────────────────
def _page_index_from_dest(reader: PdfReader, dest) -> Optional[int]:
    try:
        return reader.get_destination_page_number(dest)
    except Exception:
        pass
    try:
        if isinstance(dest, Destination):
            try: return dest.page_number
            except Exception: return dest.page.index
        if isinstance(dest, ArrayObject) and dest and isinstance(dest[0], (IndirectObject, DictionaryObject)):
            page_ref = dest[0]
            for i, pg in enumerate(reader.pages):
                if getattr(pg, "indirect_reference", None) == page_ref or pg == page_ref:
                    return i
            return reader.pages.index(page_ref)
        if isinstance(dest, DictionaryObject):
            if "/Dest" in dest: return _page_index_from_dest(reader, dest["/Dest"])
            if "/A" in dest and isinstance(dest["/A"], DictionaryObject) and "/D" in dest["/A"]:
                return _page_index_from_dest(reader, dest["/A"]["/D"])
    except Exception:
        pass
    return None

def _flatten_pdf_outlines_any(reader: PdfReader) -> List[tuple]:
    flat = []
    try:
        outlines = getattr(reader, "outlines", None)
        if outlines is None:
            outlines = reader.get_outlines()
    except Exception:
        outlines = []
    def rec(items):
        for it in items:
            if isinstance(it, list):
                rec(it); continue
            title = None; idx = None
            try:
                if isinstance(it, Destination):
                    title = (it.title or "Capítulo").strip()
                    idx = _page_index_from_dest(reader, it)
                elif isinstance(it, DictionaryObject):
                    title = (str(it.get("/Title","Capítulo"))).strip()
                    idx = _page_index_from_dest(reader, it)
                elif isinstance(it, ArrayObject):
                    title = "Capítulo"; idx = _page_index_from_dest(reader, it)
            except Exception:
                pass
            if idx is not None: flat.append((title or "Capítulo", idx))
    try:
        rec(outlines if isinstance(outlines, list) else [outlines])
    except Exception:
        pass
    flat.sort(key=lambda t: t[1])
    return flat

# ───────────────── Caché persistente y cancelación ─────────────────
class CancelToken:
    def __init__(self): self._flag = False
    def cancel(self):    self._flag = True
    def canceled(self):  return self._flag

def _cache_dir_init():
    cf = Path(CONFIG["persistence"]["cache_file"])
    try:
        cf.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return cf

def _file_sig(p: Path) -> str:
    try:
        st = p.stat()
        raw = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return hashlib.sha1(str(p).encode("utf-8")).hexdigest()

def _cache_load() -> dict:
    if not CONFIG["persistence"]["enable_cache"]:
        return {}
    cf = _cache_dir_init()
    try:
        return json.loads(cf.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _cache_save(data: dict):
    if not CONFIG["persistence"]["enable_cache"]:
        return
    cf = _cache_dir_init()
    try:
        cf.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _cache_get_ranges(path: Path):
    if not CONFIG["persistence"]["enable_cache"]:
        return None
    db = _cache_load()
    key = str(path.resolve())
    sig = _file_sig(path)
    ent = db.get(key)
    if ent and ent.get("sig") == sig:
        return ent.get("ranges")
    return None

def _cache_put_ranges(path: Path, ranges: list):
    if not CONFIG["persistence"]["enable_cache"]:
        return
    db = _cache_load()
    key = str(path.resolve())
    db[key] = {"sig": _file_sig(path), "ranges": ranges, "ts": int(time.time())}
    _cache_save(db)

# ───────────────── EPUB rápido vía nav.xhtml/OPF ─────────────────
def _epub_chapter_starts_fast(path_str: str):
    thr = float(CONFIG["chapters"].get("merge_similarity_threshold", 0.50))
    mgap = int(CONFIG["chapters"].get("merge_max_gap", 1))
    min_gap = int(CONFIG["chapters"].get("min_gap_pages", 0))
    try:
        with zipfile.ZipFile(path_str, "r") as z:
            if "META-INF/container.xml" not in z.namelist():
                return [], 0
            with z.open("META-INF/container.xml") as fh:
                cont = fh.read().decode("utf-8", "ignore")
            soup = BeautifulSoup(cont, "xml")
            rootfile = soup.find("rootfile")
            opf_path = rootfile["full-path"] if rootfile and rootfile.has_attr("full-path") else None
            if not opf_path or opf_path not in z.namelist():
                return [], 0
            with z.open(opf_path) as fh:
                opf = BeautifulSoup(fh.read().decode("utf-8", "ignore"), "xml")
            manifest = {it["id"]: it["href"] for it in opf.find_all("item") if it.has_attr("id") and it.has_attr("href")}
            spine_ids = [it["idref"] for it in opf.find_all("itemref") if it.has_attr("idref")]
            spine_hrefs = [manifest.get(i) for i in spine_ids if i in manifest]
            base = "/".join(opf_path.split("/")[:-1])
            def norm(p): return (base + "/" + p).replace("//", "/")
            spine = [norm(h) for h in spine_hrefs if h]
            by_name = {h.replace("\\","/"): i for i, h in enumerate(spine)}
            nav_id = None
            for it in opf.find_all("item"):
                if it.get("properties","").find("nav") >= 0:
                    nav_id = it.get("id"); break
            nav_href = manifest.get(nav_id) if nav_id else None
            starts = []
            if nav_href:
                nav_full = norm(nav_href)
                if nav_full in z.namelist():
                    with z.open(nav_full) as fh:
                        nav_html = BeautifulSoup(fh.read().decode("utf-8", "ignore"), "html.parser")
                    items = nav_html.select("nav ol li a")
                    seen=set()
                    for a in items:
                        href = (a.get("href","") or "").split("#",1)[0]
                        full = norm(href)
                        if full in by_name and full not in seen:
                            starts.append(((a.get_text(strip=True) or full), by_name[full]))
                            seen.add(full)
            else:
                if spine:
                    cand = spine[0]
                    if cand in z.namelist():
                        with z.open(cand) as fh:
                            html = BeautifulSoup(fh.read().decode("utf-8", "ignore"), "html.parser")
                        nav = html.find("nav")
                        if nav:
                            items = nav.find_all("a")
                            seen=set()
                            for a in items:
                                href = (a.get("href","") or "").split("#",1)[0]
                                full = norm(href)
                                if full in by_name and full not in seen:
                                    starts.append(((a.get_text(strip=True) or "Capítulo"), by_name[full]))
                                    seen.add(full)
            starts.sort(key=lambda t: t[1])
            starts = _merge_similar_contiguous(starts, threshold=thr, max_gap=mgap)
            starts = _filter_starts_by_min_gap(starts, min_gap)
            return starts, len(spine)
    except Exception:
        return [], 0

def _epub_docs_and_pages(path_str: str):
    """Devuelve (docs, total_docs) para EPUB (cada doc ~ "página" lógica)."""
    if (not _HAS_EPUB) or (epub is None):
        return [], 0
    try:
        book = epub.read_epub(path_str)
        docs = [item for item in book.get_items() if isinstance(item, EpubHtml)]
        return docs, len(docs)
    except Exception:
        return [], 0


# ─────────────────── DOCX: secciones por H1–H3 ───────────────────
_WNS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def _is_hlevel(style_val: str, outline_lvl: Optional[int]) -> Optional[int]:
    """Devuelve nivel 1–3 si el estilo/nivel corresponde a H1..H3."""
    if outline_lvl is not None:
        try:
            lvl = int(outline_lvl)
            if 0 <= lvl <= 2:
                return lvl + 1
        except Exception:
            pass
    s = (style_val or "").lower().strip()
    # variantes comunes en español/inglés
    patterns = ["heading 1","heading1","título 1","titulo 1","title 1","h1",
                "heading 2","heading2","título 2","titulo 2","title 2","h2",
                "heading 3","heading3","título 3","titulo 3","title 3","h3"]
    for i, pat in enumerate(patterns):
        if s == pat or s.replace(" ","") == pat.replace(" ",""):
            return 1 + (i // 6)
    # comienza con "heading"/"título" y termina 1..3
    m = re.search(r"(heading|t[íi]tulo|title)\s*([123])", s)
    if m:
        return int(m.group(2))
    return None

def _docx_sections_from_zip(path: Path) -> List[tuple]:
    """
    Construye [(titulo, texto)] por secciones basadas en H1–H3.
    Si no hay encabezados, devuelve un único bloque con todo el texto.
    """
    sections: List[tuple] = []
    cur_title = None
    cur_lines: List[str] = []
    try:
        with zipfile.ZipFile(str(path), "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return [(path.stem, "")]
            root = ET.fromstring(zf.read("word/document.xml"))
            for p in root.iter("{%s}p" % _WNS["w"]):
                pPr = p.find("{%s}pPr" % _WNS["w"])
                style_val = None
                outline_lvl = None
                if pPr is not None:
                    pStyle = pPr.find("{%s}pStyle" % _WNS["w"])
                    if pStyle is not None:
                        style_val = pStyle.get("{%s}val" % _WNS["w"])
                    outLvl = pPr.find("{%s}outlineLvl" % _WNS["w"])
                    if outLvl is not None:
                        try:
                            outline_lvl = int(outLvl.get("{%s}val" % _WNS["w"]))
                        except Exception:
                            outline_lvl = None
                # recoger texto del párrafo
                runs = []
                for t in p.iter("{%s}t" % _WNS["w"]):
                    if t.text:
                        runs.append(t.text)
                para_text = "".join(runs).strip()

                h = _is_hlevel(style_val or "", outline_lvl)
                if h is not None and h <= 3:
                    # cerrar sección previa
                    if cur_title is not None:
                        sections.append((cur_title, "\n".join(cur_lines).strip()))
                        cur_lines = []
                    cur_title = para_text if para_text else f"Sección {len(sections)+1}"
                else:
                    if para_text:
                        cur_lines.append(para_text)
            # cerrar final
            if cur_title is None:
                # sin encabezados: todo es una sección
                all_text = []
                for t in root.iter("{%s}t" % _WNS["w"]):
                    if t.text:
                        all_text.append(t.text)
                return [(path.stem, "\n".join(all_text).strip())]
            sections.append((cur_title, "\n".join(cur_lines).strip()))
    except Exception:
        return [(path.stem, "")]
    return sections

def _docx_chapter_starts_fast(path: Path):
    secs = _docx_sections_from_zip(path)
    starts = [ (t or f"Sección {i+1}", i) for i,(t,_) in enumerate(secs) ]
    return starts, len(secs)

# ─────────────────── PPTX: títulos de diapositiva ───────────────────
_PNS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

def _pptx_slides_from_zip(path: Path) -> List[tuple]:
    """Devuelve [(titulo, texto)] por diapositiva."""
    slides: List[tuple] = []
    try:
        with zipfile.ZipFile(str(path), "r") as zf:
            names = [n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            def index_of(n):
                m = re.search(r"slide(\d+)\.xml", n)
                return int(m.group(1)) if m else 0
            names.sort(key=index_of)
            for i, name in enumerate(names, 1):
                try:
                    root = ET.fromstring(zf.read(name))
                    # título preferente: shape con p:ph type="title" o "ctrTitle"
                    title = None
                    texts_all = []
                    for sp in root.iter("{%s}sld" % _PNS["p"]):
                        for shp in sp.iter("{%s}sp" % _PNS["p"]):
                            nv = shp.find(".//{%s}ph" % _PNS["p"])
                            is_title = bool(nv is not None and nv.get("type") in ("title","ctrTitle"))
                            # text body
                            body = shp.find("{%s}txBody" % _PNS["p"])
                            if body is not None:
                                parts=[]
                                for t in body.iter("{%s}t" % _PNS["a"]):
                                    if t.text: parts.append(t.text)
                                txt = "\n".join(parts).strip()
                                if txt:
                                    texts_all.append(txt)
                                    if is_title and not title:
                                        title = txt
                    if not title:
                        title = texts_all[0] if texts_all else f"Diapositiva {i}"
                    slides.append((title, "\n".join(texts_all).strip()))
                except Exception:
                    slides.append((f"Diapositiva {i}", ""))
    except Exception:
        return []
    return slides

def _pptx_chapter_starts_fast(path: Path):
    slides = _pptx_slides_from_zip(path)
    starts = [ (t or f"Diapositiva {i+1}", i) for i,(t,_) in enumerate(slides) ]
    return starts, len(slides)

# ─────────────────── Fallbacks Office/libreoffice ───────────────────
def _which(bin_names: List[str]) -> Optional[str]:
    for n in bin_names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _find_pdftotext() -> Optional[str]:
    """Encuentra pdftotext (Poppler/Xpdf). Best-effort, sin escaneo agresivo."""
    p = _which(["pdftotext", "pdftotext.exe"])
    if p:
        return p
    if os.name != "nt":
        return None
    try:
        candidates: List[Path] = []
        for env in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA", "USERPROFILE"):
            base = os.environ.get(env)
            if base:
                candidates.append(Path(base))
        rels = [
            r"scoop\apps\poppler\current\Library\bin\pdftotext.exe",
            r"chocolatey\bin\pdftotext.exe",
            r"poppler\Library\bin\pdftotext.exe",
            r"Poppler\Library\bin\pdftotext.exe",
        ]
        for base in candidates:
            for rel in rels:
                c = base / rel
                if c.exists():
                    return str(c)
    except Exception:
        pass
    return None


def _iter_pdftotext_pages(
    pdftotext_bin: str,
    pdf_path: Path,
    start1: int,
    end1: int,
    *,
    layout: bool = False,
    raw: bool = False,
    table: bool = False,
    extra_args: Optional[List[str]] = None,
) -> 'Iterable[str]':
    """Itera páginas en streaming (separadas por \f) desde pdftotext hacia Python."""
    cmd: List[str] = [
        str(pdftotext_bin),
        "-q",
        "-f", str(int(start1)),
        "-l", str(int(end1)),
        "-enc", "UTF-8",
        "-eol", "unix",
    ]
    if layout:
        cmd.append("-layout")
    if raw:
        cmd.append("-raw")
    if table:
        cmd.append("-table")
    if extra_args:
        cmd.extend([str(a) for a in extra_args if str(a).strip()])
    cmd.extend([str(pdf_path), "-"])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        buf = bytearray()
        CHUNK = 1 << 16
        while True:
            chunk = proc.stdout.read(CHUNK)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                i = buf.find(b"\x0c")
                if i < 0:
                    break
                seg = bytes(buf[:i])
                del buf[:i+1]
                yield seg.decode("utf-8", errors="ignore")
        if buf:
            yield bytes(buf).decode("utf-8", errors="ignore")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"pdftotext falló (exit={rc})")
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


def _convert_with_libreoffice(src: Path, target_ext: str) -> Optional[Path]:
    soffice = _which(["soffice", "lowriter", "soffice.bin"])
    if not soffice:
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            proc = subprocess.run(
                [soffice, "--headless", "--convert-to", target_ext, "--outdir", str(outdir), str(src)],
                capture_output=True, text=True
            )
            if proc.returncode == 0:
                out = outdir / (src.stem + "." + target_ext.split(":")[0])
                if out.exists():
                    # copiar a junto al original
                    final = src.with_suffix("." + target_ext.split(":")[0])
                    try:
                        shutil.copy2(out, final)
                    except Exception:
                        final.write_bytes(out.read_bytes())
                    return final
    except Exception:
        return None
    return None

def _doc_to_docx_path(src: Path) -> Optional[Path]:
    # 1) LibreOffice
    p = _convert_with_libreoffice(src, "docx")
    if p and p.exists():
        return p
    # 2) Word COM
    try:
        import pythoncom, win32com.client  # type: ignore
        pythoncom.CoInitialize()
        try:
            wd = win32com.client.gencache.EnsureDispatch("Word.Application")
        except Exception:
            wd = win32com.client.Dispatch("Word.Application")
        wd.Visible = False; wd.DisplayAlerts = 0
        doc = wd.Documents.Open(str(src))
        out = src.with_suffix(".docx")
        doc.SaveAs(str(out), FileFormat=16)  # wdFormatXMLDocument
        doc.Close(False); wd.Quit()
        pythoncom.CoUninitialize()
        return out if out.exists() else None
    except Exception:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass
        return None

def _ppt_to_pptx_path(src: Path) -> Optional[Path]:
    # 1) LibreOffice
    p = _convert_with_libreoffice(src, "pptx")
    if p and p.exists():
        return p
    # 2) PowerPoint COM
    try:
        import pythoncom, win32com.client  # type: ignore
        pythoncom.CoInitialize()
        try:
            pp = win32com.client.gencache.EnsureDispatch("PowerPoint.Application")
        except Exception:
            pp = win32com.client.Dispatch("PowerPoint.Application")
        pp.Visible = False
        pres = pp.Presentations.Open(str(src), WithWindow=False)
        out = src.with_suffix(".pptx")
        pres.SaveAs(str(out))  # PPTX por default
        pres.Close(); pp.Quit()
        pythoncom.CoUninitialize()
        return out if out.exists() else None
    except Exception:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass
        return None

# ───────────────── Wrappers de detección con caché ─────────────────
def _detect_pdf_chapters_prior_bookmarks_fast(path: Path,
                                              time_budget_ms: int,
                                              quick_pages: Optional[int],
                                              token: Optional[CancelToken]=None) -> List[tuple]:
    cached = _cache_get_ranges(path)
    if isinstance(cached, list) and cached:
        return cached

    path_str = str(path)
    min_gap = int(CONFIG["chapters"].get("min_gap_pages", 0))

    starts = _pdf_outline_starts_pymupdf(path_str)
    total = _pdf_page_count(path_str)
    if starts:
        starts = _filter_starts_by_min_gap(starts, min_gap)
        ranges = _chapter_ranges_from_starts(starts, total)
        _cache_put_ranges(path, ranges)
        return ranges

    # Fallback de marcadores vía pypdf (puro Python) por estabilidad.
    if True:
        try:
            reader = PdfReader(path_str)
            starts2 = _flatten_pdf_outlines_any(reader)
            if starts2:
                starts2 = _filter_starts_by_min_gap(starts2, min_gap)
                ranges = _chapter_ranges_from_starts(starts2, len(reader.pages))
                _cache_put_ranges(path, ranges)
                return ranges
        except Exception:
            pass

    _cache_put_ranges(path, [])
    return []

def _detect_epub_chapters_prior_fast(path: Path) -> List[tuple]:
    cached = _cache_get_ranges(path)
    if isinstance(cached, list) and cached:
        return cached
    starts, pages = _epub_chapter_starts_fast(str(path))
    ranges = _chapter_ranges_from_starts(starts, pages) if starts else []
    if not ranges:
        s2, p2 = _epub_chapter_starts(str(path))
        ranges = _chapter_ranges_from_starts(s2, p2) if s2 else []
    _cache_put_ranges(path, ranges)
    return ranges

def _detect_docx_chapters_fast(path: Path) -> List[tuple]:
    cached = _cache_get_ranges(path)
    if isinstance(cached, list) and cached:
        return cached
    starts, total = _docx_chapter_starts_fast(path)
    ranges = _chapter_ranges_from_starts(starts, total) if starts else []
    _cache_put_ranges(path, ranges)
    return ranges

def _detect_pptx_chapters_fast(path: Path) -> List[tuple]:
    cached = _cache_get_ranges(path)
    if isinstance(cached, list) and cached:
        return cached
    starts, total = _pptx_chapter_starts_fast(path)
    ranges = _chapter_ranges_from_starts(starts, total) if starts else []
    _cache_put_ranges(path, ranges)
    return ranges

# ───────────────── Preview PDF (miniatura) ─────────────────
@lru_cache(maxsize=256)
def _render_pdf_page_thumbnail(path_str: str, page1: int):
    if (not _HAS_FITZ) or (not _HAS_PIL): return None
    try:
        with fitz.open(path_str) as doc:
            idx0 = max(0, min(page1-1, len(doc)-1))
            scale = CONFIG["performance"]["fitz_raster_scale"] or 1.0
            pix = doc[idx0].get_pixmap(matrix=fitz.Matrix(scale, scale))
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None

# ───────────────── Archivos/columnas helpers ─────────────────
_INVALID_WIN = set('<>:"/\\|?*') | {chr(i) for i in range(0,32)}
def _sanitize_suffix(s: str, maxlen: int = 60) -> str:
    if not s: return ""
    s = s.strip(); s = re.sub(r"\s+", "_", s)
    out=[]
    for ch in s:
        if ch in _INVALID_WIN: out.append("_")
        elif ch in string.printable or ch.isalnum() or ch in "_-.": out.append(ch)
        else: out.append(ch if (ch.isalnum() or ch in "_-.") else "_")
    slug = re.sub(r"_+", "_", "".join(out)).strip("._-")
    return slug[:maxlen] if slug else "cap"



def _norm_txt(s: str) -> str:
    """Normaliza texto extraído/OCR sin destruir saltos de línea."""
    if s is None:
        return ""
    try:
        s = str(s)
    except Exception:
        return ""
    # limpia NULLs y normaliza EOL
    s = s.replace("\x00", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # tabs múltiples -> un espacio
    s = re.sub(r"[ \t\f\v]+", " ", s)
    # limpia espacios antes de salto
    s = re.sub(r"[ \t]+\n", "\n", s)
    # colapsa saltos excesivos
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip()


def _pages_to_ranges(pages_1based: List[int]) -> str:
    """Convierte [1,2,3,7,8] -> '1-3,7-8' (formato OCRmyPDF --pages)."""
    try:
        pages = sorted({int(p) for p in pages_1based if int(p) >= 1})
    except Exception:
        return ""
    if not pages:
        return ""
    out: List[str] = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = p
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def _column_weights_and_minima() -> Tuple[dict, dict]:
    raw = CONFIG["columns"]["weights_raw"].copy()
    s = sum(raw.values()) or 1.0
    rel = {k: v/s for k, v in raw.items()}
    base = CONFIG["columns"]["base_min_px"]
    mins_cfg = CONFIG["columns"]["min_px"]
    phi2 = PHI**2
    defaults = {
        "archivo": int(base * phi2),
        "sufijo":  int(base * phi2),
        "inicio":  int(base * PHI),
        "fin":     int(base * PHI),
        "caps":    int(base * PHI)
    }
    mins = {k: (mins_cfg.get(k) if mins_cfg.get(k) is not None else defaults[k]) for k in defaults}
    return rel, mins

def _open_editor_at(self, iid, colid):
    x, y, w, h = self.tree.bbox(iid, colid)
    val = self.tree.set(iid, self.COLS[int(colid[1:]) - 1])
    self._editing = (iid, colid)
    self._editor.place(x=x + 1, y=y + 1, width=max(30, w - 2), height=max(20, h - 2))
    self._editor.delete(0, "end")
    self._editor.insert(0, val or "")
    self._editor.focus_set()
    self._editor.unbind("<KeyRelease>")
    if colid in ("#2", "#3"):
        def _preview(_=None):
            gx, gy = self.tree.winfo_rootx() + x, self.tree.winfo_rooty() + y + h + 6
            txt = self._editor.get().strip()
            page1 = int(txt) if txt.isdigit() else 1
            row = self.rows[iid]
            if row["suffix"] == ".pdf" and row["total"]:
                page1 = max(1, min(page1, row["total"]))
                img = _render_pdf_page_thumbnail(str(row["path"]), page1)
                if img is not None:
                    self._popover.show_image_at(gx, gy, img)
                else:
                    self._popover.show_text_at(gx, gy, f"Pág. {page1}/{row['total']}")
            elif row["suffix"] == ".epub" and row["total"]:
                page1 = max(1, min(page1, row["total"]))
                try:
                    book = epub.read_epub(str(row["path"]))
                    docs = [it for it in book.get_items() if isinstance(it, EpubHtml)]
                    raw = docs[page1 - 1].get_content()
                    html = raw.decode('utf-8', errors='ignore')
                    txtp = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
                    txtp = (txtp[:300] + "…") if len(txtp) > 300 else (txtp or "(vacío)")
                    self._popover.show_text_at(gx, gy, txtp)
                except Exception:
                    self._popover.show_text_at(gx, gy, "(No fue posible previsualizar)")
            else:
                self._popover.show_text_at(gx, gy, "(Sin previsualización)")
        self._editor.bind("<KeyRelease>", _preview)
    else:
        self._popover.hide()


def _begin_edit(self, event):
    iid, colid = self._cell_info_at(event)
    if not iid or colid not in ("#2", "#3", "#4"):
        return
    self._open_editor_at(iid, colid)


# ────────────────── Botón plano ──────────────────
class FlatButton(tk.Canvas):
    def __init__(self, parent, text, command=None, bg=None, accent=None,
                 width=106, height=28, radius=None, font=None, colors=None, **kw):
        super().__init__(parent, width=width, height=height, bd=0,
                         highlightthickness=0, bg=bg or parent.cget("bg"), **kw)
        self._wpx, self._hpx = int(width), int(height)
        self._r = int(max(0, radius if radius is not None else int(CONFIG["layout"]["btn_radius_px"])))
        self._text = text; self._cmd = command
        self._bg = bg or parent.cget("bg")
        self._accent = accent or _accent_from_dwm()
        self._hover = False; self._pressed = False; self._font = font or _font(CONFIG["utilities"]["family"],  max(CONFIG["utilities"]["min_px"],  int(12)))
        c = CONFIG["colors"]; self._colors = colors or {
            "text": c["text"],
            "base": c["btn_base"],
            "face": c["btn_face"],
            "face_hover": c["btn_face_hover"],
            "face_down": c["btn_face_down"]
        }
        self.bind("<Enter>",  lambda e: self._set_hover(True))
        self.bind("<Leave>",  lambda e: self._set_hover(False))
        self.bind("<ButtonPress-1>",  lambda e: self._set_pressed(True))
        self.bind("<ButtonRelease-1>", self._on_click)
        self.bind("<Configure>", lambda e: self.after_idle(self._redraw))
        self.after_idle(self._redraw)

    def set_size(self, width, height, radius=None):
        self._wpx, self._hpx = int(width), int(height)
        if radius is not None: self._r = int(max(0, radius if radius is not None else int(CONFIG["layout"]["btn_radius_px"])))
        self.config(width=self._wpx, height=self._hpx)
        self.after_idle(self._redraw)

    def set_font(self, font_tuple):
        self._font = font_tuple
        self.after_idle(self._redraw)

    def set_colors(self, colors: dict):
        self._colors.update(colors or {})
        self.after_idle(self._redraw)

    def _set_hover(self, v): self._hover = v; self._redraw()
    def _set_pressed(self, v): self._pressed = v; self._redraw()
    def _on_click(self, _):
        if self._pressed:
            self._pressed = False; self._redraw()
            if callable(self._cmd): self.after(0, self._cmd)
    def _rounded_rect(self, x1, y1, x2, y2, r, **opts):
        self.create_rectangle(x1+r, y1, x2-r, y2, **opts)
        if r > 0:
            self.create_oval(x1, y1, x1+2*r, y2, **opts)
            self.create_oval(x2-2*r, y1, x2, y2, **opts)
    def _fit_font(self, w: int, h: int):
        """Auto-ajusta la letra al ancho del botón."""
        try:
            family, base_size, style = self._font
        except Exception:
            return self._font
        min_size = int(CONFIG.get("utilities", {}).get("min_px", 10))
        base_size = int(max(min_size, int(base_size)))

        pad = max(int(6*PHI), int(h*0.30))
        target_w = max(10, int(w - 2*pad))

        weight = "bold" if ("bold" in str(style).lower()) else "normal"
        slant = "italic" if ("italic" in str(style).lower()) else "roman"

        try:
            f0 = tkfont.Font(family=family, size=base_size, weight=weight, slant=slant)
            tw = max(1, int(f0.measure(self._text)))
        except Exception:
            return self._font

        max_size = max(min_size, int(h*0.60))
        size = int(base_size * (target_w / float(tw)))
        size = max(min_size, min(max_size, size))

        try:
            f = tkfont.Font(family=family, size=size, weight=weight, slant=slant)
            while size > min_size and int(f.measure(self._text)) > target_w:
                size -= 1
                f.configure(size=size)
            while size < max_size:
                f2 = tkfont.Font(family=family, size=size+1, weight=weight, slant=slant)
                if int(f2.measure(self._text)) <= target_w:
                    size += 1
                    f = f2
                else:
                    break
            return f
        except Exception:
            return self._font


    def _redraw(self):
        try:
            if not self.winfo_exists(): return
            self.delete("all")
        except tk.TclError:
            return
        w = self.winfo_width() or self._wpx
        h = self.winfo_height() or self._hpx
        r = min(self._r, h//2, w//2)
        base  = self._colors["base"]
        face = self._colors["face"]
        if self._hover:
            face = self._colors["face_hover"]
        if self._pressed:
            face = self._colors["face_down"]
        self._rounded_rect(0, 0, w, h, r, fill=base, outline="")
        self._rounded_rect(0, 0, w, h, r, fill=face, outline="")
        self.create_text(w // 2, h // 2, text=self._text, anchor="center",
                         fill=self._colors["text"], font=self._font)


# ────────────────── Popover anclado ──────────────────
class AnchoredPopover(tk.Toplevel):
    """Popover anclado; usado para preview Inicio/Fin."""

    def __init__(self, root, w, h, font):
        super().__init__(root)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.withdraw()
        self.width, self.height = int(w), int(h)
        self.canvas = tk.Canvas(self, width=self.width, height=self.height,
                                highlightthickness=0, bd=0, bg=CONFIG["colors"]["app_bg"])
        self.canvas.pack(fill="both", expand=True)
        self._img_ref = None
        self._font = font

    def set_size(self, w, h):
        self.width, self.height = int(w), int(h)
        self.canvas.config(width=self.width, height=self.height)

    def place_at(self, x_abs: int, y_abs: int):
        self.geometry(f"{int(self.width)}x{int(self.height)}+{int(x_abs)}+{int(y_abs)}")

    def _clear(self):
        self.canvas.delete("all")
        self._img_ref = None

    def show_image_at(self, x_abs: int, y_abs: int, pil_img):
        self.place_at(x_abs, y_abs)
        self._clear()
        if pil_img and _HAS_PIL and Image is not None and ImageTk is not None:
            iw, ih = pil_img.size
            scale = min(self.width / iw, self.height / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            img = pil_img.resize((nw, nh), getattr(getattr(Image, "Resampling", Image), "LANCZOS"))
            self._img_ref = ImageTk.PhotoImage(img)
            self.canvas.create_image(self.width // 2, self.height // 2, image=self._img_ref)
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)

    def show_text_at(self, x_abs: int, y_abs: int, text):
        self.place_at(x_abs, y_abs)
        self._clear()
        self.canvas.create_text(8, 8, anchor="nw", width=self.width - 16, text=text,
                                fill=CONFIG["colors"]["text"], font=self._font)
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)

    def hide(self): self.withdraw()


# ────────────────── Dropdown de capítulos (no bloqueante) ──────────────────
class ChaptersDropdown(tk.Toplevel):
    def __init__(self, parent, width=420, height=420, font=None,
                 on_select=None, instant_apply=True, destroy_on_select=False):
        super().__init__(parent)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.transient(parent.winfo_toplevel())
        self.on_select = on_select or (lambda *args: None)
        self.instant_apply = instant_apply
        self.destroy_on_select = destroy_on_select
        c = CONFIG["colors"]
        self.configure(bg=c["app_bg"])

        self._search_var = tk.StringVar()
        wrap = tk.Frame(self, bg=c["app_bg"])
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        if font is None:
            fam = CONFIG["utilities"]["family"]
            font = _font(fam, max(CONFIG["utilities"]["min_px"], 12))
        self.entry = tk.Entry(wrap, textvariable=self._search_var,
                              relief="flat", bd=0, highlightthickness=1,
                              bg=c["box_bg"], fg=c["text"], insertbackground=c["text"],
                              font=font)
        self.entry.pack(fill="x", padx=(0, 0), pady=(0, 6))
        self.entry.bind("<KeyRelease>", self._apply_filter)

        self.listbox = tk.Listbox(
            wrap,
            activestyle="dotbox",
            bd=0,
            highlightthickness=0,
            bg=c["box_bg"],
            fg=c["text"],
            font=font
        )

        self.listbox.pack(fill="both", expand=True)
        self.scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.listbox.yview,
                                    style="Flat.Vertical.TScrollbar")

        self.listbox.configure(yscrollcommand=self.scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")
        self.listbox.bind("<Return>", self._choose)
        if self.instant_apply:
            self.listbox.bind("<Double-Button-1>", self._choose)
        self.bind("<Escape>", lambda e: self.destroy())

        self._items: List[tuple] = []
        self._view_items: List[tuple] = []
        self.geometry(f"{int(width)}x{int(height)}+0+0")
        self._outside_bind_id = None

    def show_at(self, x, y):
        self.geometry(f"+{int(x)}+{int(y)}")
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)
        self.entry.focus_set()
        if self._outside_bind_id is None:
            self._outside_bind_id = self.bind_all("<Button-1>", self._on_global_click, add="+")

    def _format_item(self, item: tuple) -> str:
        t, s, e = item
        if isinstance(s, int) and isinstance(e, int) and s >= 1 and e >= 1:
            return f"{t}  ({s}–{e})"
        return str(t)

    def set_items(self, items: List[tuple]):
        self._items = items or []
        self._rebuild_listbox(self._items)

    def show_loading(self, text="Cargando…"):
        self._items = []
        self._view_items = []
        self.listbox.delete(0, "end")
        self.listbox.insert("end", text)

    def replace_items(self, items: List[tuple]):
        self.set_items(items)

    def _rebuild_listbox(self, src):
        self._view_items = list(src)
        self.listbox.delete(0, "end")
        for it in src:
            self.listbox.insert("end", self._format_item(it))
        if src:
            self.listbox.selection_set(0)

    def _apply_filter(self, *_):
        q = (self._search_var.get() or "").strip().lower()
        if not q:
            self._rebuild_listbox(self._items)
            return
        filt = []
        for it in self._items:
            if q in str(it[0]).lower():
                filt.append(it)
        self._rebuild_listbox(filt)

    def _contains_point(self, x_root, y_root) -> bool:
        try:
            x0 = self.winfo_rootx()
            y0 = self.winfo_rooty()
            x1 = x0 + self.winfo_width()
            y1 = y0 + self.winfo_height()
            return (x0 <= x_root <= x1) and (y0 <= y_root <= y1)
        except Exception:
            return False

    def _on_global_click(self, event):
        if self._contains_point(event.x_root, event.y_root):
            return
        try:
            self.destroy()
        except Exception:
            pass

    def _choose(self, *_):
        i = self.listbox.curselection()
        if not i: return
        item = self._view_items[i[0]]
        t, s, e = item
        self.on_select(t, s, e)
        if isinstance(s, int) and isinstance(e, int) and s >= 1 and e >= 1 and self.destroy_on_select:
            self.destroy()

    def destroy(self):
        try:
            if self._outside_bind_id is not None:
                self.unbind_all("<Button-1>")
                self._outside_bind_id = None
        except Exception:
            pass
        super().destroy()


# ────────────────── Tabla / detección de capítulos ──────────────────
class RangeTable(ttk.Frame):
    COLS = ("archivo", "inicio", "fin", "sufijo", "caps")

    def __init__(self, parent, files, fonts):
        super().__init__(parent)
        self.configure(style="Flat.TFrame")
        self.fonts = fonts
        self.rows = {}
        self._accent = _accent_from_dwm()
        self._cols_rel, self._col_min = _column_weights_and_minima()
        self._chapters_cache: dict[str, list | None] = {}
        self._chapters_lock = threading.Lock()
        self._detect_workers: dict[str, threading.Thread] = {}
        self._detect_tokens: dict[str, CancelToken] = {}

        c = CONFIG["colors"]
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure("Flat.TFrame", background=c["app_bg"], borderwidth=0, relief="flat")
        style.configure("Miau.Treeview",
                        background=c["box_bg"],
                        fieldbackground=c["box_bg"],
                        foreground=c["text"],
                        borderwidth=0,
                        relief="flat",
                        rowheight=max(24,
                                      int(self.winfo_toplevel().winfo_height() * CONFIG["layout"]["row_height_rel"])) )
        style.layout("Miau.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.configure("Miau.Treeview.Heading",
                        background=c["app_bg"], foreground=c["text"],
                        font=self.fonts["head"], relief="flat", borderwidth=0)
        style.map("Miau.Treeview",
                  background=[("selected", c["box_bg"])],
                  foreground=[("selected", c["text"])])

        style.configure("Flat.Vertical.TScrollbar",
            gripcount=0, background=c["app_bg"],
            bordercolor=c["app_bg"], darkcolor=c["app_bg"], lightcolor=c["app_bg"],
            troughcolor=c["scroll_trough"], arrowcolor=c["scroll_arrow"],
            relief="flat", borderwidth=0)

        self.tree = ttk.Treeview(self, columns=self.COLS, show="headings",
                                 style="Miau.Treeview", selectmode="none")
        self.vs = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview,
                                style="Flat.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=self.vs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vs.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.heading("archivo", text="Archivo")
        self.tree.heading("inicio", text="Inicio")
        self.tree.heading("fin", text="Fin")
        self.tree.heading("sufijo", text="Sufijo")
        self.tree.heading("caps", text="Capítulos")

        for k in self.COLS:
            anchor = "w" if k in ("archivo", "sufijo", "caps") else "e"
            self.tree.column(k, anchor=anchor, stretch=True, minwidth=self._col_min[k])

        self._editor = tk.Entry(self.tree, bd=0, highlightthickness=1,
                                bg=c["box_bg"], fg=c["text"], insertbackground=c["text"],
                                font=self.fonts["row"], relief="flat")
        self._editor.place_forget()
        self._editor.bind("<Return>", self._commit_edit)
        self._editor.bind("<Escape>", lambda e: self._cancel_edit())
        self._editor.bind("<FocusOut>", lambda e: self._commit_edit())

        pop_w = _px_w(CONFIG["layout"]["preview_rel_w"], self.tree)
        pop_h = _px_h(CONFIG["layout"]["preview_rel_h"], self.tree)
        self._popover = AnchoredPopover(self.winfo_toplevel(), pop_w, pop_h, font=self.fonts["preview"])

        self.tree.bind("<Configure>", self._on_tree_configure)
        self.tree.bind("<Double-1>", self._begin_edit)
        self.tree.bind("<Button-1>", self._maybe_dropdown)
        self.tree.bind("<MouseWheel>", lambda e: self.tree.yview_scroll(int(-1 * (e.delta / 106)), "units"))

        for path in [Path(f) for f in files]:
            self._add_row(path)

    def _on_tree_configure(self, *_):
        self._resize_columns()
        pop_w = _px_w(CONFIG["layout"]["preview_rel_w"], self.tree)
        pop_h = _px_h(CONFIG["layout"]["preview_rel_h"], self.tree)
        self._popover.set_size(pop_w, pop_h)

    def refresh_style_metrics(self, rowheight_px: int, head_font, row_font):
        ttk.Style(self).configure("Miau.Treeview", rowheight=rowheight_px)
        ttk.Style(self).configure("Miau.Treeview.Heading", font=head_font)
        self._editor.config(font=row_font)
        self.after_idle(self._resize_columns)

    def _normalized_widths(self, total_w):
        widths = {}
        acc = 0
        order = self.COLS
        for i, k in enumerate(order):
            if i < len(order) - 1:
                w = int(total_w * self._cols_rel[k])
                widths[k] = w
                acc += w
            else:
                widths[k] = max(1, total_w - acc)

        def enforce_min(wdict, total):
            for k in wdict: wdict[k] = max(wdict[k], self._col_min[k])
            cur = sum(wdict.values())
            if cur > total:
                exceso = cur - total
                marg = {k: max(0, wdict[k] - self._col_min[k]) for k in wdict}
                msum = sum(marg.values())
                if msum > 0:
                    for k in wdict:
                        take = min(marg[k], int(exceso * (marg[k] / msum))) if msum else 0
                        wdict[k] -= take
            cur = sum(wdict.values())
            if cur > total and cur > 0:
                f = total / cur
                for k in wdict: wdict[k] = max(1, int(wdict[k] * f))
            cur = sum(wdict[k] for k in order[:-1])
            wdict[order[-1]] = max(1, total - cur)

        enforce_min(widths, total_w)
        return widths

    def _resize_columns(self, *_):
        total = max(100, self.tree.winfo_width() - (self.vs.winfo_width() or 16))
        w = self._normalized_widths(total)
        for k in self.COLS:
            self.tree.column(k, width=w[k], stretch=False)

    def _quick_total_units(self, path: Path, suffix: str) -> Optional[int]:
        try:
            if suffix == ".pdf":
                return _pdf_page_count(str(path))
            if suffix == ".epub":
                _, total = _epub_docs_and_pages(str(path)); return total
            if suffix == ".docx":
                return _docx_chapter_starts_fast(path)[1]
            if suffix == ".pptx":
                return _pptx_chapter_starts_fast(path)[1]
        except Exception:
            return None
        return None

    def _add_row(self, path: Path):
        suffix = path.suffix.lower()
        total_pages = self._quick_total_units(path, suffix)
        iid = self.tree.insert("", "end", values=(path.name, "", "", "", "▾ Seleccionar…"))
        self.rows[iid] = {"path": path, "suffix": suffix, "total": total_pages,
                          "inicio": "", "fin": "", "sufijo": ""}
        with self._chapters_lock:
            self._chapters_cache[iid] = None

    def _cell_info_at(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region not in ("cell", "tree"): return None, None
        iid = self.tree.identify_row(event.y)
        colid = self.tree.identify_column(event.x)
        return iid, colid

    def _place_below_cell(self, iid, colid):
        bx, by, bw, bh = self.tree.bbox(iid, colid)
        tx = self.tree.winfo_rootx() + bx
        ty = self.tree.winfo_rooty() + by + bh + 2

        cfg = CONFIG["chapters_popup"]
        if cfg.get("use_relative", True):
            w = max(int(cfg.get("min_w_px", 420)), int(self.tree.winfo_width() * float(cfg.get("rel_w_tree", 0.5))))
            h = min(int(cfg.get("max_h_px", 720)), int(self.tree.winfo_height() * float(cfg.get("rel_h_tree", 1.0))))
        else:
            w = int(cfg.get("fixed_w_px", 520))
            h = int(cfg.get("fixed_h_px", 520))

        return tx, ty, w, h

    def _detect_pdf_chapters_prior_bookmarks(self, pdf_path: Path) -> List[tuple]:
        return _detect_pdf_chapters_prior_bookmarks_fast(
            pdf_path,
            time_budget_ms=10_000,
            quick_pages=CONFIG["chapters_runtime"]["full_scan_pages"] or None,
            token=None
        )

    def _detect_epub_chapters_prior_toc(self, epub_path: Path) -> List[tuple]:
        starts, pages = _epub_chapter_starts(str(epub_path))
        return _chapter_ranges_from_starts(starts, pages) if starts else []

    # ==================== edición y dropdown ====================
    def _open_editor_at(self, iid, colid):
        try:
            x, y, w, h = self.tree.bbox(iid, colid)
        except Exception:
            return

        val = self.tree.set(iid, self.COLS[int(colid[1:]) - 1])
        self._editing = (iid, colid)

        self._editor.place(x=x + 1, y=y + 1, width=max(30, w - 2), height=max(20, h - 2))
        self._editor.delete(0, "end")
        self._editor.insert(0, val or "")
        self._editor.selection_range(0, "end")
        self._editor.icursor("end")
        self.after(0, self._editor.focus_force)

        self._editor.unbind("<KeyRelease>")
        if colid in ("#2", "#3"):
            def _preview(_=None):
                gx, gy = self.tree.winfo_rootx() + x, self.tree.winfo_rooty() + y + h + 6
                txt = self._editor.get().strip()
                page1 = int(txt) if txt.isdigit() else 1
                row = self.rows[iid]

                if row["suffix"] == ".pdf" and row["total"]:
                    page1 = max(1, min(page1, row["total"]))
                    img = _render_pdf_page_thumbnail(str(row["path"]), page1)
                    if img is not None:
                        self._popover.show_image_at(gx, gy, img)
                    else:
                        self._popover.show_text_at(gx, gy, f"Pág. {page1}/{row['total']}")
                elif row["suffix"] == ".epub" and row["total"]:
                    page1 = max(1, min(page1, row["total"]))
                    try:
                        book = epub.read_epub(str(row["path"]))
                        docs = [it for it in book.get_items() if isinstance(it, EpubHtml)]
                        raw = docs[page1 - 1].get_content()
                        html = raw.decode('utf-8', errors='ignore')
                        txtp = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
                        txtp = (txtp[:300] + "…") if len(txtp) > 300 else (txtp or "(vacío)")
                        self._popover.show_text_at(gx, gy, txtp)
                    except Exception:
                        self._popover.show_text_at(gx, gy, "(No fue posible previsualizar)")
                else:
                    self._popover.show_text_at(gx, gy, "(Sin previsualización)")

            self._editor.bind("<KeyRelease>", _preview)
        else:
            self._popover.hide()

    def _begin_edit(self, event):
        iid, colid = self._cell_info_at(event)
        if not iid or colid not in ("#2", "#3", "#4"):
            return
        self._open_editor_at(iid, colid)

    def _commit_edit(self, event=None):
        if not hasattr(self, "_editing") or not self._editing: return
        iid, colid = self._editing
        val = self._editor.get().strip()
        key = self.COLS[int(colid[1:]) - 1]
        if key in ("inicio", "fin") and val and not val.isdigit():
            messagebox.showerror("Rango inválido", "Usa números enteros en Inicio/Fin")
            self._editor.focus_set()
            return
        self.tree.set(iid, key, val)
        self.rows[iid][key] = val
        self._editor.place_forget()
        self._popover.hide()
        self._editing = None

    def _cancel_edit(self):
        if hasattr(self, "_editing"):
            self._editor.place_forget()
            self._popover.hide()
            self._editing = None

    def _maybe_dropdown(self, event):
        iid, colid = self._cell_info_at(event)
        if not iid or colid != "#5":
            return
        row = self.rows[iid]
        tx, ty, w, h = self._place_below_cell(iid, colid)
        dd = ChaptersDropdown(
            self, width=w, height=h, font=self.fonts["row"],
            on_select=lambda t, s, e: _handle_select(t, s, e),
            instant_apply=True, destroy_on_select=True
        )

        def _menu_items():
            items = []
            total = row["total"] or 1
            if CONFIG["chapters_fastmenu"]["show_full_doc"]:
                items.append(("Todo el documento", 1, int(total)))
            if row["suffix"] == ".pdf":
                if CONFIG["chapters_fastmenu"]["show_pdf_bookmarks"]:
                    items.append(("Marcadores del PDF (rápido)", -101, -101))
            elif row["suffix"] == ".epub":
                if CONFIG["chapters_fastmenu"]["show_epub_toc"]:
                    items.append(("TOC del EPUB (rápido)", -201, -201))
            elif row["suffix"] in (".docx", ".doc"):
                items.append(("Títulos Word H1–H3 (rápido)", -301, -301))
            elif row["suffix"] in (".pptx", ".ppt"):
                items.append(("Títulos de diapositiva (rápido)", -401, -401))
            if iid in self._detect_workers and self._detect_workers[iid].is_alive():
                items.append(("Cancelar tarea en curso", -999, -999))
            return items

        def _preload_ranges_items():
            lst = []
            with self._chapters_lock:
                cached_runtime = self._chapters_cache.get(iid, None)
            if isinstance(cached_runtime, list) and cached_runtime:
                lst.extend(cached_runtime)
            else:
                persisted = _cache_get_ranges(row["path"])
                if isinstance(persisted, list) and persisted:
                    lst.extend(persisted)
            return [(str(t).strip(), s, e) for (t, s, e) in lst] if lst else []

        def _compose_items_with(ranges):
            base = _menu_items()
            return base + ranges if ranges else base

        dd.set_items(_compose_items_with(_preload_ranges_items()))
        dd.show_at(tx, ty)

        def _start_work(target):
            tok_prev = self._detect_tokens.get(iid)
            if tok_prev: tok_prev.cancel()
            tok = CancelToken()
            self._detect_tokens[iid] = tok
            th = threading.Thread(target=target, daemon=True)
            self._detect_workers[iid] = th
            th.start()

        def _update_dd_with_ranges(ranges):
            with self._chapters_lock:
                self._chapters_cache[iid] = ranges
            items = _compose_items_with([(str(tt).strip(), ss, ee) for (tt, ss, ee) in (ranges or [])])
            if dd.winfo_exists():
                dd.after(0, lambda: dd.replace_items(items if items else
                                                     _compose_items_with(
                                                         [("Sin capítulos detectados", 1, max(1, row["total"] or 1))])))

        def _run_pdf_bookmarks():
            dd.show_loading("Leyendo marcadores…")
            def job():
                try:
                    ranges = _detect_pdf_chapters_prior_bookmarks_fast(
                        row["path"],
                        CONFIG["chapters_runtime"]["time_budget_ms"],
                        CONFIG["chapters_runtime"]["fast_scan_pages"],
                        token=self._detect_tokens[iid]
                    )
                    _update_dd_with_ranges(ranges)
                except Exception:
                    _update_dd_with_ranges([])
            _start_work(job)

        def _run_epub_toc():
            dd.show_loading("Leyendo TOC EPUB…")
            def job():
                try:
                    ranges = _detect_epub_chapters_prior_fast(row["path"])
                    _update_dd_with_ranges(ranges)
                except Exception:
                    _update_dd_with_ranges([])
            _start_work(job)

        def _run_docx_titles():
            dd.show_loading("Leyendo títulos H1–H3…")
            def job():
                try:
                    p = row["path"]
                    if row["suffix"] == ".doc":
                        conv = _doc_to_docx_path(p)
                        src = conv if conv and conv.exists() else None
                        if src is None:
                            _update_dd_with_ranges([])
                            return
                        ranges = _detect_docx_chapters_fast(src)
                    else:
                        ranges = _detect_docx_chapters_fast(p)
                    _update_dd_with_ranges(ranges)
                except Exception:
                    _update_dd_with_ranges([])
            _start_work(job)

        def _run_pptx_titles():
            dd.show_loading("Leyendo títulos de diapositiva…")
            def job():
                try:
                    p = row["path"]
                    if row["suffix"] == ".ppt":
                        conv = _ppt_to_pptx_path(p)
                        src = conv if conv and conv.exists() else None
                        if src is None:
                            _update_dd_with_ranges([])
                            return
                        ranges = _detect_pptx_chapters_fast(src)
                    else:
                        ranges = _detect_pptx_chapters_fast(p)
                    _update_dd_with_ranges(ranges)
                except Exception:
                    _update_dd_with_ranges([])
            _start_work(job)

        def _handle_select(title, s, e):
            if s is None or e is None:
                return
            if s >= 1 and e >= 1:
                self._apply_range(iid, title, s, e)
                dd.destroy()
                self.after(30, lambda: (self.tree.see(iid), self._open_editor_at(iid, "#2")))
                return

            if s == -101: _run_pdf_bookmarks(); return
            if s == -201: _run_epub_toc(); return
            if s == -301: _run_docx_titles(); return
            if s == -401: _run_pptx_titles(); return
            if s == -999:
                tok = self._detect_tokens.get(iid)
                if tok: tok.cancel()
                if dd.winfo_exists():
                    dd.replace_items(_compose_items_with(_preload_ranges_items()))
                return

    def _apply_range(self, iid, title, s, e):
        self.tree.set(iid, "inicio", str(s))
        self.tree.set(iid, "fin", str(e))
        self.tree.set(iid, "sufijo", _sanitize_suffix((title or "cap").strip()))
        self.rows[iid]["inicio"] = str(s)
        self.rows[iid]["fin"] = str(e)
        self.rows[iid]["sufijo"] = _sanitize_suffix((title or "cap").strip())

    def collect_ranges(self):
        out = []
        for iid, r in self.rows.items():
            s = r["inicio"].strip()
            e = r["fin"].strip()
            path = r["path"]; suffix = r["suffix"]; total = r["total"]

            if s == "" and e == "":
                # determinar total si no estaba
                try:
                    if (not total) and path.is_file():
                        if suffix == ".pdf":
                            total = _pdf_page_count(str(path))
                        elif suffix == ".epub":
                            _, total = _epub_docs_and_pages(str(path))
                        elif suffix == ".docx":
                            total = _docx_chapter_starts_fast(path)[1]
                        elif suffix == ".pptx":
                            total = _pptx_chapter_starts_fast(path)[1]
                except Exception:
                    total = None
                if not total or total < 1:
                    messagebox.showerror("Sin páginas", f"No se pudo determinar el total en:\n{path.name}")
                    return None
                out.append((path, 1, int(total), _sanitize_suffix(r["sufijo"].strip())))
                continue

            if (s == "") ^ (e == ""):
                messagebox.showerror("Rango incompleto",
                                     f"Rellena Inicio y Fin o deja ambos vacíos en:\n{r['path'].name}")
                return None
            if not (s.isdigit() and e.isdigit()):
                messagebox.showerror("Rango inválido", f"Usa números enteros en:\n{r['path'].name}")
                return None
            si = int(s); ei = int(e)
            if total and (si < 1 or ei < 1 or si > total or ei > total or si > ei):
                lim = f" (1–{total})" if total else ""
                messagebox.showerror("Fuera de límites", f"Rango fuera de límites en:\n{r['path'].name}{lim}")
                return None
            out.append((path, si, ei, _sanitize_suffix(r["sufijo"].strip())))
        return out


# ──────────────────────────────────────────────────────────────────────────────
class MiausoftApp(tk.Tk):
    def __init__(self, files):
        super().__init__()
        self.title(CONFIG["window"]["title"])
        c = CONFIG["colors"]
        self.configure(bg=c["app_bg"])

        pol = CONFIG["window"]["dpi_scale_policy"]
        try:
            if pol != "auto":
                self.tk.call('tk', 'scaling', float(pol))
        except Exception:
            pass

        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure("Flat.TFrame", background=c["app_bg"], borderwidth=0, relief="flat")
        style.configure("Flat.TLabel", background=c["app_bg"], borderwidth=0, relief="flat")
        style.configure("Flat.Vertical.TScrollbar", gripcount=0, background=c["app_bg"],
                        bordercolor=c["app_bg"], darkcolor=c["app_bg"], lightcolor=c["app_bg"],
                        troughcolor=c["scroll_trough"], arrowcolor=c["scroll_arrow"],
                        relief="flat", borderwidth=0)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        if CONFIG["window"]["use_relative"]:
            w = int(sw * CONFIG["window"]["rel_width"])
            h = int(sh * CONFIG["window"]["rel_height"])
        else:
            w, h = 980, 520
        w = min(w, int(sw * CONFIG["window"]["max_width_rel_screen"]))
        h = min(h, int(sh * CONFIG["window"]["max_height_rel_screen"]))

        self.geometry(f"{int(w)}x{int(h)}")
        self.minsize(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"])
        self.resizable(True, True)
        pass  # update_idletasks deshabilitado por estabilidad (Tk 9 / Python 3.14)
        self.geometry(f"{int(w)}x{int(h)}+{int((sw - w) // 2)}+{int((sh - h) // 2)}")

        try:
            attach_activation(self, app="conversor_capitulos")
        except Exception:
            pass

        ico = Path(__file__).parent / CONFIG["window"]["icon_name"]
        try:
            if ico.exists():
                self.iconbitmap(ico)
        except Exception:
            pass
        _set_win11_backdrop(self.winfo_id(), CONFIG["win11"]["backdrop"], CONFIG["win11"]["prefer_dark_mode"])

        rw, rh = _metrics(w, h)
        self._rw, self._rh = rw, rh
        fam = CONFIG["utilities"]["family"]
        self.fonts = {
            "title": _font(fam, rh(CONFIG["utilities"]["title"]["rel_px"]), bold=CONFIG["utilities"]["title"]["bold"],
                           italic=CONFIG["utilities"]["title"]["italic"]),
            "sub": _font(fam, rh(CONFIG["utilities"]["sub"]["rel_px"]), bold=CONFIG["utilities"]["sub"]["bold"]),
            "head": _font(fam, rh(CONFIG["utilities"]["head"]["rel_px"]), bold=CONFIG["utilities"]["head"]["bold"]),
            "row": _font(fam, rh(CONFIG["utilities"]["row"]["rel_px"]), bold=CONFIG["utilities"]["row"]["bold"]),
            "btn": _font(fam, rh(CONFIG["utilities"]["btn"]["rel_px"]), bold=CONFIG["utilities"]["btn"]["bold"]),
            "err": _font(fam, rh(CONFIG["utilities"]["err"]["rel_px"]), bold=CONFIG["utilities"]["err"]["bold"]),
            "preview": _font(fam, rh(CONFIG["utilities"]["preview"]["rel_px"]), bold=CONFIG["utilities"]["preview"]["bold"]),
        }

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.table = RangeTable(self, files, self.fonts)
        self.table.grid(row=0, column=0, sticky="nsew")

        footer = tk.Frame(self, bg=c["app_bg"], bd=0, highlightthickness=0)
        footer.grid(row=1, column=0, sticky="sew")
        footer.columnconfigure(0, weight=1)
        btns = tk.Frame(footer, bg=c["app_bg"])
        btns.grid(row=0, column=1, sticky="e", padx=8, pady=8)

        bw0, bh0 = shared_button_box_px(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"], CONFIG)
        btn_fs0 = shared_button_font_px(CONFIG["window"]["min_width"], CONFIG["window"]["min_height"], CONFIG)
        fam_btn = CONFIG["utilities"]["family"] or CONFIG["utilities"]["fallback_family"]
        weight = "bold" if bool(CONFIG["utilities"]["btn"].get("bold", True)) else "normal"
        slant  = "italic" if bool(CONFIG["utilities"]["btn"].get("italic", False)) else "roman"
        self._shared_btn_font = tkfont.Font(family=fam_btn, size=int(btn_fs0), weight=weight, slant=slant)


        self.btn_add = FlatButton(
            btns, "Añadir archivos…", command=self._add_files_dialog, bg=c["app_bg"],
            width=bw0, height=bh0,
            radius=CONFIG["layout"]["btn_radius_px"], font=self._shared_btn_font
        )
        self.btn_add.grid(row=0, column=0, padx=(0, 12))
        self.btn_convert = FlatButton(
            btns, "Convertir", command=self._convert, bg=c["app_bg"],
            width=bw0, height=bh0,
            radius=CONFIG["layout"]["btn_radius_px"], font=self._shared_btn_font
        )
        self.btn_convert.grid(row=0, column=1)

        self.queue = queue.Queue()
        self._ranges = None

        self.pdlg: ProgressDialog | None = None

        self._cfg_job = None

    def report_callback_exception(self, exc, val, tb):
        try:
            import traceback
            log_path = Path(__file__).with_suffix(".tk.log")
            with open(log_path, "a", encoding="utf-8", errors="ignore") as fh:
                fh.write("\n\n" + ("=" * 80) + "\n")
                fh.writelines(traceback.format_exception(exc, val, tb))
        except Exception:
            pass
        try:
            super().report_callback_exception(exc, val, tb)
        except Exception:
            pass


    def _on_root_configure(self, *_):
        if self._cfg_job:
            try:
                self.after_cancel(self._cfg_job)
            except Exception:
                pass
        self._cfg_job = self.after(80, self._apply_metrics)

    def _apply_metrics(self):
        w = max(self.winfo_width(), CONFIG["window"]["min_width"])
        h = max(self.winfo_height(), CONFIG["window"]["min_height"])
        self._rw, self._rh = _metrics(w, h)
        rw, rh = self._rw, self._rh
        fam = CONFIG["utilities"]["family"]

        self.fonts.update({
            "title": _font(fam, rh(CONFIG["utilities"]["title"]["rel_px"]), bold=CONFIG["utilities"]["title"]["bold"],
                           italic=CONFIG["utilities"]["title"]["italic"]),
            "sub": _font(fam, rh(CONFIG["utilities"]["sub"]["rel_px"]), bold=CONFIG["utilities"]["sub"]["bold"]),
            "head": _font(fam, rh(CONFIG["utilities"]["head"]["rel_px"]), bold=CONFIG["utilities"]["head"]["bold"]),
            "row": _font(fam, rh(CONFIG["utilities"]["row"]["rel_px"]), bold=CONFIG["utilities"]["row"]["bold"]),
            "btn": _font(fam, rh(CONFIG["utilities"]["btn"]["rel_px"]), bold=CONFIG["utilities"]["btn"]["bold"]),
            "err": _font(fam, rh(CONFIG["utilities"]["err"]["rel_px"]), bold=CONFIG["utilities"]["err"]["bold"]),
            "preview": _font(fam, rh(CONFIG["utilities"]["preview"]["rel_px"]), bold=CONFIG["utilities"]["preview"]["bold"]),
        })

        rowheight = max(24, int(h * CONFIG["layout"]["row_height_rel"]))
        self.table.refresh_style_metrics(rowheight, self.fonts["head"], self.fonts["row"])
        bw, bh = shared_button_box_px(w, h, CONFIG)
        btn_sz = shared_button_font_px(w, h, CONFIG)
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
        radius = CONFIG["layout"]["btn_radius_px"]
        for b in (self.btn_add, self.btn_convert):
            b.set_size(bw, bh, radius)
            b.set_font(self._shared_btn_font)

    def _add_files_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Selecciona archivos",
            filetypes=[
                ("PDF/EPUB/Word/PPT", "*.pdf *.epub *.doc *.docx *.ppt *.pptx"),
                ("PDF", "*.pdf"), ("EPUB", "*.epub"),
                ("Word", "*.doc *.docx"), ("PowerPoint", "*.ppt *.pptx"),
                ("Todos", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            self.table._add_row(Path(p))

    def _convert(self):
        ranges = self.table.collect_ranges()
        if ranges is None:
            return
        self._ranges = ranges

        if self.pdlg is None or not self.pdlg.winfo_exists():
            self.pdlg = ProgressDialog(parent=self, icon_search_dir=Path(__file__).parent)
        self.pdlg.set_title("Preparando…")
        self.pdlg.set_subtitle("Extrayendo texto...")

        total_pages_all = sum((e1 - s1 + 1) for (_, s1, e1, _) in self._ranges)
        self.pdlg.set_global_span(max(1, int(total_pages_all)))
        self.pdlg.show_counts_in_subtitle(True)
        self.pdlg.show()
        try:
            pass  # update_idletasks deshabilitado por estabilidad (Tk 9 / Python 3.14)
        except Exception:
            pass

        self.after(100, self.start)

    def start(self):
        threading.Thread(target=self.process_files, daemon=True).start()
        self.after(106, self.update_progress)

    def _safe_extract(self, page, p, pages):

        """Extrae texto de una página PDF con timeout, sin crear un hilo por página."""

        try:

            timeout = float((CONFIG.get("app", {}) or {}).get("safe_extract_timeout", 6.0))

        except Exception:

            timeout = 6.0


        if timeout <= 0:

            try:

                return page.extract_text() or ""

            except Exception:

                return ""


        try:

            workers = int((CONFIG.get("app", {}) or {}).get("pdf_text_workers", 0) or 0)

        except Exception:

            workers = 0

        if not workers:

            workers = max(1, min(4, (os.cpu_count() or 4)))


        if (not hasattr(self, "_pdf_extractor_pool")) or (self._pdf_extractor_pool is None) or (getattr(self._pdf_extractor_pool, "workers", None) != workers):

            self._pdf_extractor_pool = _PDFTextExtractorPool(workers)


        ok, txt = self._pdf_extractor_pool.extract(page, timeout)

        return (txt or "") if ok else ""


    def process_files(self):
        total_files = len({p for (p, *_) in self._ranges})
        for idx, (path, s1, e1, suf) in enumerate(self._ranges, 1):
            self.queue.put(("file", idx, total_files, path.name))
            if not path.is_file():
                self.queue.put(("error", f"No existe: {path.name}"))
                continue

            out_dir = path.parent
            suffix = path.suffix.lower()
            try:

                if suffix == '.pdf':
                    # Conversión PDF optimizada: RAM mínima + escritura inmediata por página + OCR inteligente.
                    import os
                    import io
                    import gc
                    import tempfile
                    import contextlib

                    start0, end0 = int(s1) - 1, int(e1) - 1
                    total_sel = int(end0 - start0 + 1)
                    if total_sel <= 0:
                        self.queue.put(("error", f"Rango inválido: {s1}-{e1}"))
                        continue

                    # ---- OCR config
                    ocr_cfg = (CONFIG.get("ocr", {}) or {})
                    pl = (CONFIG.get("progress_logic", {}) or {})

                    enable_ocr = bool(ocr_cfg.get("enable", pl.get("enable_ocr", False)))
                    ocr_force = bool(ocr_cfg.get("force", pl.get("ocr_force", False)))
                    ocr_min_chars = int(ocr_cfg.get("min_chars", pl.get("ocr_min_chars", 20)))
                    ocr_trigger_ratio = float(ocr_cfg.get("trigger_ratio", 0.10))
                    ocr_timeout = float(ocr_cfg.get("timeout_sec", ocr_cfg.get("timeout", pl.get("ocr_timeout", 60))))
                    ocr_dpi = int(ocr_cfg.get("dpi", pl.get("ocr_dpi", 300)))
                    ocr_lang = str(ocr_cfg.get("lang", "spa+eng") or "spa+eng")
                    ocr_skip_text = bool(ocr_cfg.get("skip_text", pl.get("ocr_skip_text", True)))
                    ocr_jobs_cfg = int(ocr_cfg.get("jobs", pl.get("ocr_jobs", 0)) or 0)
                    ocr_whitelist = str(ocr_cfg.get("tessedit_char_whitelist", ocr_cfg.get("whitelist", "")) or "")
                    ocr_whitelist = re.sub(r"[^A-Za-z0-9]", "", ocr_whitelist)
                    ocr_sanitize = bool(ocr_cfg.get("sanitize_ocr_output", True))

                    # disponibilidad OCR
                    ocrmypdf_mod = None
                    has_ocr = False
                    if enable_ocr or ocr_force:
                        try:
                            import ocrmypdf as _ocrmypdf  # type: ignore
                            ocrmypdf_mod = _ocrmypdf
                            has_ocr = True
                        except Exception:
                            has_ocr = False
                    enable_ocr = bool(enable_ocr and has_ocr)
                    if ocr_force and not has_ocr:
                        ocr_force = False

                    cpu = os.cpu_count() or 4
                    def _pick_ocr_jobs() -> int:
                        if ocr_jobs_cfg and ocr_jobs_cfg > 0:
                            return max(1, min(int(ocr_jobs_cfg), cpu))
                        return max(1, min(4, max(1, cpu - 1)))

                    def _filter_to_whitelist(s: str) -> str:
                        if not ocr_sanitize:
                            return s or ""
                        if not s:
                            return ""
                        if not ocr_whitelist:
                            return s
                        allowed = set(ocr_whitelist)
                        allowed.update({" ", "\n", "\r", "\t", "\f"})
                        return "".join(ch for ch in s if ch in allowed)

                    # ---- Abrir PDF preferentemente con fitz (menos RAM)
                    use_fitz = bool(globals().get('_HAS_FITZ', False) and globals().get('fitz', None) is not None)
                    doc = None
                    reader = None
                    try:
                        if use_fitz:
                            doc = fitz.open(str(path))
                            total_pages_doc = int(getattr(doc, 'page_count', 0) or 0)
                        else:
                            reader = PdfReader(str(path))
                            try:
                                total_pages_doc = int(reader.get_num_pages())
                            except Exception:
                                total_pages_doc = len(reader.pages)
                    except Exception as e:
                        self.queue.put(("error", f"Falló lectura PDF: {e}"))
                        try:
                            if doc is not None:
                                doc.close()
                        except Exception:
                            pass
                        continue

                    if total_pages_doc <= 0:
                        try:
                            if doc is not None:
                                doc.close()
                        except Exception:
                            pass
                        self.queue.put(("error", "PDF vacío."))
                        continue

                    # Clamp rango a páginas reales
                    start0 = max(0, min(start0, total_pages_doc - 1))
                    end0 = max(0, min(end0, total_pages_doc - 1))
                    if end0 < start0:
                        self.queue.put(("error", f"Rango inválido tras clamp: {start0+1}-{end0+1}"))
                        try:
                            if doc is not None:
                                doc.close()
                        except Exception:
                            pass
                        continue
                    total_sel = int(end0 - start0 + 1)

                    stem = path.stem + (f"_{_sanitize_suffix(suf)}" if suf else "")
                    txt_file = out_dir / f"{stem}.txt"

                    # ---- Cache base: len + bytes por página seleccionada
                    def _cache_write(fh, s: str):
                        b = (s or "").encode("utf-8", errors="ignore")
                        fh.write(len(b).to_bytes(4, "little", signed=False))
                        fh.write(b)

                    def _cache_iter(cache_path: Path):
                        with open(cache_path, "rb") as fh:
                            while True:
                                lb = fh.read(4)
                                if not lb:
                                    break
                                n = int.from_bytes(lb, "little", signed=False)
                                if n <= 0:
                                    yield ""
                                    continue
                                yield fh.read(n).decode("utf-8", errors="ignore")

                    def _ocrrec_write(fh, rel0: int, s: str):
                        b = (s or "").encode("utf-8", errors="ignore")
                        fh.write(int(rel0).to_bytes(4, "little", signed=False))
                        fh.write(len(b).to_bytes(4, "little", signed=False))
                        fh.write(b)

                    def _ocrrec_iter(cache_path: Path):
                        with open(cache_path, "rb") as fh:
                            while True:
                                rb = fh.read(4)
                                if not rb:
                                    break
                                rel0 = int.from_bytes(rb, "little", signed=False)
                                lb = fh.read(4)
                                if not lb:
                                    break
                                ln = int.from_bytes(lb, "little", signed=False)
                                if ln <= 0:
                                    yield rel0, ""
                                    continue
                                yield rel0, fh.read(ln).decode("utf-8", errors="ignore")

                    def _pages_arg_from_gp0(gp0_list):
                        if not gp0_list:
                            return ""
                        nums = sorted({int(gp0) + 1 for gp0 in gp0_list})
                        ranges = []
                        a = b = nums[0]
                        for n in nums[1:]:
                            if n == b + 1:
                                b = n
                            else:
                                ranges.append(f"{a}-{b}" if a != b else f"{a}")
                                a = b = n
                        ranges.append(f"{a}-{b}" if a != b else f"{a}")
                        return ",".join(ranges)

                    def _iter_sidecar_segments(sidecar_path: Path):
                        CHUNK = 65536
                        buf = bytearray()
                        with open(sidecar_path, "rb") as fh:
                            while True:
                                chunk = fh.read(CHUNK)
                                if not chunk:
                                    break
                                buf.extend(chunk)
                                while True:
                                    i = buf.find(b"\x0c")
                                    if i < 0:
                                        break
                                    seg = bytes(buf[:i])
                                    del buf[:i+1]
                                    yield seg.decode("utf-8", errors="ignore")
                        if buf:
                            yield bytes(buf).decode("utf-8", errors="ignore")

                    cache_path = None
                    short_gp0 = []
                    textful_count = 0

                    try:
                        with tempfile.NamedTemporaryFile("wb", suffix=".miaucache", delete=False) as tf, \
                             open(txt_file, "w", encoding="utf-8", errors="ignore") as out:
                            cache_path = Path(tf.name)

                            perf_cfg = (CONFIG.get("performance", {}) or {})
                            pdf_backend = str(perf_cfg.get("pdf_text_backend", perf_cfg.get("pdf_backend", "auto")) or "auto").lower()
                            txt_flush_pages = int(perf_cfg.get("txt_flush_pages", 25) or 25)
                            pdftotext_layout = bool(perf_cfg.get("pdftotext_layout", False))
                            pdftotext_raw = bool(perf_cfg.get("pdftotext_raw", False))
                            pdftotext_table = bool(perf_cfg.get("pdftotext_table", False))
                            pdftotext_extra_args = perf_cfg.get("pdftotext_extra_args", []) or []
                            try:
                                pdftotext_extra_args = [str(a) for a in pdftotext_extra_args if str(a).strip()]
                            except Exception:
                                pdftotext_extra_args = []

                            pdftotext_bin = None
                            use_pdftotext = False
                            if pdf_backend in ("auto", "pdftotext"):
                                pdftotext_bin = _find_pdftotext()
                                use_pdftotext = bool(pdftotext_bin)
                            if pdf_backend in ("pymupdf", "fitz", "pypdf"):
                                use_pdftotext = False

                            flush_every = max(0, int(txt_flush_pages))

                            if use_pdftotext and pdftotext_bin:
                                page_iter = _iter_pdftotext_pages(
                                    pdftotext_bin,
                                    path,
                                    int(start0) + 1,
                                    int(end0) + 1,
                                    layout=pdftotext_layout,
                                    raw=pdftotext_raw,
                                    table=pdftotext_table,
                                    extra_args=pdftotext_extra_args,
                                )
                                for rel0, gp0 in enumerate(range(start0, end0 + 1)):
                                    rel = rel0 + 1
                                    self.queue.put(("extract_page", rel, total_sel))
                                    try:
                                        txt = next(page_iter)
                                    except StopIteration:
                                        txt = ""
                                    except Exception:
                                        txt = ""
                                    txt = _norm_txt(txt)
                                    _cache_write(tf, txt)

                                    out.write(txt or "")
                                    if rel < total_sel:
                                        out.write("\n")
                                    if flush_every and (rel % flush_every == 0):
                                        out.flush()

                                    st = (txt or "").strip()
                                    if len(st) < ocr_min_chars:
                                        short_gp0.append(int(gp0))
                            else:
                                for rel0, gp0 in enumerate(range(start0, end0 + 1)):
                                    rel = rel0 + 1
                                    self.queue.put(("extract_page", rel, total_sel))
                                    try:
                                        if use_fitz and doc is not None:
                                            page = doc.load_page(gp0)
                                            txt = page.get_text("text") or ""
                                        else:
                                            txt = self._safe_extract_pdf_page(reader.pages[gp0]) if reader is not None else ""
                                    except Exception:
                                        txt = ""
                                    txt = _norm_txt(txt)
                                    _cache_write(tf, txt)

                                    out.write(txt or "")
                                    if rel < total_sel:
                                        out.write("\n")
                                    if flush_every and (rel % flush_every == 0):
                                        out.flush()

                                    st = (txt or "").strip()
                                    if len(st) < ocr_min_chars:
                                        short_gp0.append(int(gp0))
                    finally:
                        try:
                            if doc is not None:
                                doc.close()
                        except Exception:
                            pass
                        try:
                            if use_fitz and globals().get('fitz', None) is not None:
                                try:
                                    fitz.TOOLS.store_shrink(100)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    # ---- Regla: NO OCR si >=80% de páginas del rango tienen texto >= min_chars
                    textful_ratio = (textful_count / float(total_sel)) if total_sel else 0.0
                    short_ratio = (len(short_gp0) / float(total_sel)) if total_sel else 0.0

                    do_ocr = False
                    if ocr_force and enable_ocr:
                        do_ocr = True
                    elif enable_ocr:
                        if textful_ratio >= 0.80:
                            do_ocr = False
                        else:
                            do_ocr = bool(short_ratio > float(ocr_trigger_ratio))

                    if (not do_ocr) or (not short_gp0) or (ocrmypdf_mod is None):
                        # limpiar cache y listo
                        try:
                            if cache_path is not None:
                                cache_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        try:
                            del reader
                        except Exception:
                            pass
                        gc.collect()
                        self.queue.put(("saved", txt_file.name))
                        continue

                    # ---- OCR chunked (solo páginas cortas) y cache de OCR en disco
                    self.queue.put(("ocr_start", len(short_gp0)))
                    ocr_cache_path = None
                    try:
                        with tempfile.NamedTemporaryFile("wb", suffix=".miaucacheocr", delete=False) as ocf:
                            ocr_cache_path = Path(ocf.name)
                            chunk_size = int((CONFIG.get("performance", {}) or {}).get("ocr_chunk_size", 50) or 50)
                            chunk_size = max(5, min(chunk_size, 200))

                            pages_sorted = sorted(set(short_gp0))
                            done = 0
                            for i0 in range(0, len(pages_sorted), chunk_size):
                                chunk = pages_sorted[i0:i0 + chunk_size]
                                pages_arg = _pages_arg_from_gp0(chunk)
                                if not pages_arg:
                                    done = min(len(pages_sorted), i0 + len(chunk))
                                    self.queue.put(("ocr_page", done, len(pages_sorted)))
                                    continue

                                sidecar = None
                                out_pdf = None
                                try:
                                    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tfp:
                                        out_pdf = Path(tfp.name)
                                    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as tfs:
                                        sidecar = Path(tfs.name)

                                    tcfg = {}
                                    if ocr_whitelist:
                                        tcfg["tessedit_char_whitelist"] = ocr_whitelist

                                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                                        kwargs = dict(
                                            pages=pages_arg,
                                            skip_text=ocr_skip_text,
                                            force_ocr=False,
                                            jobs=int(_pick_ocr_jobs()),
                                            tesseract_timeout=float(ocr_timeout),
                                            oversample=int(ocr_dpi),
                                            language=ocr_lang,
                                            output_type="none",
                                            sidecar=str(sidecar),
                                            tesseract_config=tcfg if tcfg else None,
                                        )
                                        try:
                                            ocrmypdf_mod.ocr(str(path), str(out_pdf), **kwargs)
                                        except TypeError:
                                            kwargs.pop("output_type", None)
                                            try:
                                                ocrmypdf_mod.ocr(str(path), str(out_pdf), **kwargs)
                                            except TypeError:
                                                kwargs.pop("tesseract_config", None)
                                                ocrmypdf_mod.ocr(str(path), str(out_pdf), **kwargs)

                                    seg_iter = _iter_sidecar_segments(sidecar)
                                    for gp0 in chunk:
                                        rel0 = int(gp0 - start0)
                                        try:
                                            seg = next(seg_iter)
                                        except StopIteration:
                                            seg = ''
                                        seg = _filter_to_whitelist(_norm_txt(seg)).strip('\n')
                                        _ocrrec_write(ocf, rel0, seg)

                                except Exception:
                                    for gp0 in chunk:
                                        rel0 = int(gp0 - start0)
                                        _ocrrec_write(ocf, rel0, '')
                                finally:
                                    for tmp in (sidecar, out_pdf):
                                        if tmp is not None:
                                            try:
                                                tmp.unlink()
                                            except Exception:
                                                pass

                                done = min(len(pages_sorted), i0 + len(chunk))
                                self.queue.put(("ocr_page", done, len(pages_sorted)))

                        # ---- Reensamble final sin RAM
                        tmp_out = None
                        try:
                            with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8', errors='ignore') as tfout:
                                tmp_out = Path(tfout.name)

                            ocr_it = _ocrrec_iter(ocr_cache_path)
                            try:
                                cur = next(ocr_it)
                            except StopIteration:
                                cur = None

                            with open(tmp_out, 'w', encoding='utf-8', errors='ignore') as out:
                                for rel0, base_txt in enumerate(_cache_iter(cache_path)):
                                    base_txt = base_txt or ''
                                    ocr_txt = ''
                                    if cur is not None and int(cur[0]) == int(rel0):
                                        ocr_txt = cur[1] or ''
                                        try:
                                            cur = next(ocr_it)
                                        except StopIteration:
                                            cur = None
                                    use_txt = ocr_txt if len((ocr_txt or '').strip()) > len((base_txt or '').strip()) else base_txt
                                    out.write(use_txt or '')
                                    if rel0 + 1 < total_sel:
                                        out.write('\n')

                            os.replace(str(tmp_out), str(txt_file))
                        finally:
                            if tmp_out is not None:
                                try:
                                    tmp_out.unlink(missing_ok=True)
                                except Exception:
                                    pass

                        self.queue.put(("saved", txt_file.name))
                    except Exception as e:
                        self.queue.put(("error", f"Falló OCR/ensamble: {e}"))
                    finally:
                        for tmp in (cache_path, ocr_cache_path):
                            if tmp is not None:
                                try:
                                    tmp.unlink(missing_ok=True)
                                except Exception:
                                    pass
                        try:
                            del reader
                        except Exception:
                            pass
                        gc.collect()
                elif suffix == '.epub':
                    if (not _HAS_EPUB) or (epub is None) or (not _HAS_BS4) or (BeautifulSoup is None):
                        self.queue.put(('error', 'EPUB no disponible: falta ebooklib/lxml o beautifulsoup4 en este intérprete.'))
                        continue
                    book = epub.read_epub(str(path))
                    docs = [item for item in book.get_items() if isinstance(item, EpubHtml)]
                    start0, end0 = s1 - 1, e1 - 1
                    text_pages = []
                    total = end0 - start0 + 1
                    for rel, i in enumerate(range(start0, end0 + 1), 1):
                        self.queue.put(("extract_page", rel, total))
                        raw = docs[i].get_content()
                        html = raw.decode('utf-8', errors='ignore')
                        txt = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
                        text_pages.append(txt)
                        self.queue.put(("tick", 1))

                    stem = path.stem + (f"_{_sanitize_suffix(suf)}" if suf else "")
                    txt_file = out_dir / f"{stem}.txt"
                    txt_file.write_text("\n\n".join(text_pages), encoding='utf-8')
                    self.queue.put(("saved", txt_file.name))

                elif suffix in ('.docx', '.doc'):
                    # construir secciones H1–H3 y tomar rango
                    src = path
                    if suffix == '.doc':
                        conv = _doc_to_docx_path(path)
                        if conv and conv.exists():
                            src = conv
                        else:
                            raise RuntimeError("No se pudo convertir .doc a .docx para extraer capítulos.")
                    sections = _docx_sections_from_zip(src)
                    start0, end0 = s1 - 1, min(e1 - 1, len(sections)-1)
                    parts = []
                    total = end0 - start0 + 1
                    for rel, i in enumerate(range(start0, end0 + 1), 1):
                        self.queue.put(("extract_page", rel, total))
                        title, body = sections[i]
                        chunk = (title.strip() + "\n" if title.strip() else "") + (body or "")
                        parts.append(chunk.strip())
                        self.queue.put(("tick", 1))
                    stem = path.stem + (f"_{_sanitize_suffix(suf)}" if suf else "")
                    (out_dir / f"{stem}.txt").write_text("\n\n".join(parts), encoding="utf-8")
                    self.queue.put(("saved", f"{stem}.txt"))

                elif suffix in ('.pptx', '.ppt'):
                    src = path
                    if suffix == '.ppt':
                        conv = _ppt_to_pptx_path(path)
                        if conv and conv.exists():
                            src = conv
                        else:
                            raise RuntimeError("No se pudo convertir .ppt a .pptx para extraer capítulos.")
                    slides = _pptx_slides_from_zip(src)
                    start0, end0 = s1 - 1, min(e1 - 1, len(slides)-1)
                    total = end0 - start0 + 1
                    chunks = []
                    for rel, i in enumerate(range(start0, end0 + 1), 1):
                        self.queue.put(("extract_page", rel, total))
                        title, txt = slides[i]
                        chunks.append(f"[{title or 'Diapositiva %d' % (i+1)}]\n{txt}".strip())
                        self.queue.put(("tick", 1))
                    stem = path.stem + (f"_{_sanitize_suffix(suf)}" if suf else "")
                    (out_dir / f"{stem}.txt").write_text("\n\n".join(chunks), encoding="utf-8")
                    self.queue.put(("saved", f"{stem}.txt"))

                else:
                    self.queue.put(("error", f"No soportado: {suffix}"))
            except Exception as e:
                self.queue.put(("error", f"Falló {path.name}: {e}"))
        self.queue.put(("done",))

    def update_progress(self):
        """Actualiza UI de progreso sin bloquear y con avance por página (PDF incluido)."""
        import time as _time

        max_events = 250
        max_ms = 12.0
        start = _time.perf_counter()
        processed = 0

        try:
            while processed < max_events and ((_time.perf_counter() - start) * 1000.0) < max_ms:
                try:
                    msg = self.queue.get_nowait()
                except queue.Empty:
                    break

                processed += 1
                tag = msg[0]

                if tag == "file":
                    _, idx, total, name = msg
                    self.pdlg.set_title(f"Procesando ({idx}/{total}): {name}")
                    self.pdlg.set_subtitle("Preparando…")
                    self._last_page_unit = 0
                    self._last_ocr_unit = 0
                    continue

                if tag == "extract_page":
                    _, p, pages = msg
                    p = int(p)
                    pages = int(max(1, pages))
                    last = int(getattr(self, "_last_page_unit", 0) or 0)
                    delta = p - last
                    if delta < 0:
                        delta = p
                    self._last_page_unit = p
                    if delta > 0:
                        self.pdlg.inc(delta)
                    self.pdlg.set_subtitle(f"Página {p}/{pages}")
                    continue

                if tag == "tick":
                    # Avance genérico (EPUB/DOCX/PPTX). No cambia el subtítulo salvo que esté vacío.
                    _, delta = msg
                    try:
                        d = int(delta)
                    except Exception:
                        d = 1
                    if d > 0:
                        try:
                            self.pdlg.inc(d)
                        except Exception:
                            pass
                    continue

                if tag == "ocr_start":
                    # solo feedback visual (no altera el conteo global de páginas)
                    _, total_ocr = msg
                    self._last_ocr_unit = 0
                    self.pdlg.set_subtitle(f"OCR... (0/{int(total_ocr)})")
                    continue

                if tag == "ocr_page":
                    _, i, total = msg
                    i = int(i)
                    total = int(max(1, total))
                    self.pdlg.set_subtitle(f"OCR... ({i}/{total})")
                    continue

                if tag == "saved":
                    _, name = msg
                    self.pdlg.set_subtitle(f"Guardado: {name}")
                    continue

                if tag == "error":
                    _, err = msg
                    self.pdlg.set_subtitle(f"Error: {err}")
                    continue

                if tag == "done":
                    self.pdlg.set_progress(1.0)
                    self.pdlg.set_title("Listo")
                    self.pdlg.set_subtitle("Listo")
                    self.after(2000, self._final_close)
                    return

        except Exception:
            pass

        self.after(60, self.update_progress)



    def _final_close(self):
        """Cierre seguro del diálogo de progreso (evita excepción en callback)."""
        try:
            if self.pdlg is not None:
                try:
                    self.pdlg.close()
                except Exception:
                    try:
                        self.pdlg.destroy()
                    except Exception:
                        pass
                self.pdlg = None
        except Exception:
            pass



def _href_without_fragment(href: str) -> str:
    return href.split("#", 1)[0] if href else ""

def _epub_chapter_starts(path_str: str):
    thr = float(CONFIG["chapters"].get("merge_similarity_threshold", 0.50))
    mgap = int(CONFIG["chapters"].get("merge_max_gap", 1))
    try:
        book = epub.read_epub(path_str)
        docs = [item for item in book.get_items() if isinstance(item, EpubHtml)]
        by_name = {(d.file_name or "").replace("\\", "/"): i for i, d in enumerate(docs)}
        starts = []

        def _flatten(items, out):
            for it in (items if isinstance(items, (list, tuple)) else [items]):
                if isinstance(it, (list, tuple)):
                    if isinstance(it, tuple) and it:
                        out.append(it[0])
                        if len(it) > 1:
                            _flatten(it[1], out)
                    else:
                        _flatten(it, out)
                else:
                    out.append(it)

        toc_list = []
        _flatten(book.toc, toc_list)
        for item in toc_list:
            if isinstance(item, EpubHtml):
                idx = next((i for i, d in enumerate(docs) if d is item), None)
                if idx is not None:
                    title = item.get_name() or (item.file_name or "Capítulo")
                    starts.append((title, idx))
            elif isinstance(item, EpubLink):
                href = _href_without_fragment(getattr(item, "href", None) or "").replace("\\", "/")
                if href in by_name:
                    idx = by_name[href]
                    title = getattr(item, "title", None) or href
                    starts.append((title, idx))

        starts.sort(key=lambda t: t[1])
        filt = []
        seen = set()
        for t in starts:
            if t[1] not in seen:
                filt.append(t)
                seen.add(t[1])
        starts = _merge_similar_contiguous(filt, threshold=thr, max_gap=mgap)
        return starts, len(docs)
    except Exception:
        return [], 0


# ──────────────────────────────────────────────────────────────────────────────
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
        # Motivo: Explorer puede lanzar 1 proceso por archivo seleccionado
        # si liberamos el mutex demasiado pronto, otra instancia puede volverse
        # "primaria" y abrir otra ventana. El sistema cerrará el handle al
        # terminar el proceso, y el objeto se destruye cuando se cierra el último handle.

        # Si por alguna razón quedó vacío, conservamos el input original.
        return (merged if merged else files_norm), True

    except Exception:
        return list(files_in), True


def _safe_main(argv):
    import os, traceback, faulthandler
    try:
        _crash_log = os.path.splitext(__file__)[0] + ".crash.log"
        faulthandler.enable(open(_crash_log, 'a', encoding='utf-8'), all_threads=True)
    except Exception:
        pass
    try:
        raw_args = list(argv[1:])
        files: List[str] = []
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
                files.append(str(pp))

        # Agrupar multi-selección cuando Explorer ejecuta 1 instancia POR archivo seleccionado.
        files, _launch = _merge_multiselect_startup(
            files,
            mutex_name=r"Local\MiausoftSuite_TCapitulos_MultiSelect",
            spool_name="tcapitulos.paths.txt",
        )
        if not _launch:
            return

        MiausoftApp(files).mainloop()
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
            msg = "Error al iniciar el conversor por capítulos."
            if log_path:
                msg += f" Revisa el log: {log_path}"
            _mb.showerror("MiausoftSuite", msg)
        except Exception:
            print("Error al iniciar:", traceback.format_exc())


if __name__ == "__main__":
    _safe_main(sys.argv)