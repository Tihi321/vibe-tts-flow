# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the portable VibeTTS build (one-folder).
# Build via scripts\build.ps1 (generates build\vibetts.ico first).
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# SPECPATH is the folder containing this spec (packaging/); the repo root is its parent.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

datas = [(os.path.join(ROOT, "config.example.toml"), ".")]
datas += collect_data_files("sv_ttk")
datas += collect_data_files("kokoro_onnx")       # config.json (vocabulary)
datas += collect_data_files("espeakng_loader")   # espeak-ng-data (phoneme rules)
binaries = collect_dynamic_libs("espeakng_loader")  # espeak-ng.dll

hiddenimports = [
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "pystray._win32",
    # imported lazily from vibetts.app
    "vibetts.settings_ui",
    "vibetts.model_setup",
    "vibetts.single_instance",
    "vibetts.selection",
    "vibetts.textprep",
    # VTF-04: export to audio files + local MCP server (all imported lazily)
    "vibetts.docload",
    "vibetts.export",
    "vibetts.mcp_server",
    "lameenc",
    "pypdf",
]

a = Analysis(
    [os.path.join(ROOT, "run.pyw")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "_pytest", "tests", "pip", "setuptools", "pyinstaller", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VibeTTS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(ROOT, "build", "vibetts.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VibeTTS",
)
