# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\mauri\\Downloads\\MiausoftSuite\\divideyfusionatxt.pyw'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\mauri\\Downloads\\MiausoftSuite\\Miausoft.ico', '.'), ('C:\\Users\\mauri\\Downloads\\MiausoftSuite\\config.py', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MiausoftSuite_DivisorFusionador',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\mauri\\Downloads\\MiausoftSuite\\Miausoft.ico'],
)
