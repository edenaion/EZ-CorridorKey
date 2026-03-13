# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CorridorKey — AI Green Screen Keyer.

Usage:
    pyinstaller corridorkey.spec

Notes:
    - Checkpoints (~1.5GB) are NOT bundled — they must be placed next to
      the .exe in CorridorKeyModule/checkpoints/
    - CUDA/PyTorch dlls are collected automatically
    - PySide6 is collected automatically
    - QSS theme and fonts are bundled as data files
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Project root
ROOT = os.path.dirname(os.path.abspath(SPEC))

# Data files to bundle
datas = [
    # Theme QSS, fonts, and icon
    (os.path.join(ROOT, 'ui', 'theme', 'corridor_theme.qss'), os.path.join('ui', 'theme')),
    (os.path.join(ROOT, 'ui', 'theme', 'corridorkey.png'), os.path.join('ui', 'theme')),
    (os.path.join(ROOT, 'ui', 'theme', 'icons'), os.path.join('ui', 'theme', 'icons')),
]

# Add fonts directory if it exists
fonts_dir = os.path.join(ROOT, 'ui', 'theme', 'fonts')
if os.path.isdir(fonts_dir):
    datas.append((fonts_dir, os.path.join('ui', 'theme', 'fonts')))

# Hidden imports needed for dynamic loading
hiddenimports = [
    'PySide6.QtWidgets',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'cv2',
    'numpy',
    'backend',
    'backend.service',
    'backend.clip_state',
    'backend.job_queue',
    'backend.validators',
    'backend.errors',
    'ui',
    'ui.app',
    'ui.main_window',
    'ui.preview.natural_sort',
    'ui.preview.frame_index',
    'ui.preview.display_transform',
    'ui.preview.async_decoder',
]

for dynamic_pkg in ('modules.MatAnyone2Module', 'MatAnyone2Module'):
    try:
        hiddenimports += collect_submodules(dynamic_pkg)
        datas += collect_data_files(dynamic_pkg)
    except Exception:
        pass

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'jupyter',
        'IPython',
        'notebook',
        'scipy.spatial',
        'scipy.sparse',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CorridorKey',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, 'ui', 'theme', 'corridorkey.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CorridorKey',
)
