# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CorridorKey — Windows .exe bundle.

Usage:
    pyinstaller corridorkey-windows.spec --noconfirm

Notes:
    - Builds a one-folder .exe for Windows (x64)
    - Checkpoints are NOT bundled — downloaded on first launch via setup wizard
    - CUDA/PyTorch DLLs are collected automatically
    - triton-windows is included for torch.compile support
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Project root
ROOT = os.path.dirname(os.path.abspath(SPEC))

# Data files to bundle
datas = [
    # Theme QSS, fonts, and icons
    (os.path.join(ROOT, 'ui', 'theme', 'corridor_theme.qss'), os.path.join('ui', 'theme')),
    (os.path.join(ROOT, 'ui', 'theme', 'corridorkey.png'), os.path.join('ui', 'theme')),
    (os.path.join(ROOT, 'ui', 'theme', 'corridorkey.svg'), os.path.join('ui', 'theme')),
    (os.path.join(ROOT, 'ui', 'theme', 'corridorkey.ico'), os.path.join('ui', 'theme')),
    (os.path.join(ROOT, 'ui', 'theme', 'icons'), os.path.join('ui', 'theme', 'icons')),
    # UI sounds (.wav files)
    (os.path.join(ROOT, 'ui', 'sounds'), os.path.join('ui', 'sounds')),
    # setup_models.py needed by the first-launch wizard
    (os.path.join(ROOT, 'scripts', 'setup_models.py'), 'scripts'),
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
    'PySide6.QtMultimedia',
    'cv2',
    'numpy',
    'backend',
    'backend.service',
    'backend.clip_state',
    'backend.job_queue',
    'backend.validators',
    'backend.errors',
    'backend.project',
    'backend.ffmpeg_tools',
    'backend.ffmpeg_tools.discovery',
    'backend.ffmpeg_tools.extraction',
    'backend.ffmpeg_tools.probe',
    'backend.ffmpeg_tools.stitching',
    'ui',
    'ui.app',
    'ui.main_window',
    'backend.natural_sort',
    'ui.preview.frame_index',
    'ui.preview.display_transform',
    'ui.preview.async_decoder',
    'pynvml',              # GPU monitoring (nvidia-ml-py)
    'psutil',              # System memory reporting
    'huggingface_hub',     # Model downloads in setup wizard
]

# Try to collect MatAnyone2Module
for dynamic_pkg in ('modules.MatAnyone2Module', 'MatAnyone2Module'):
    try:
        hiddenimports += collect_submodules(dynamic_pkg)
        datas += collect_data_files(dynamic_pkg)
    except Exception:
        pass

# Collect triton-windows for torch.compile support
try:
    hiddenimports += collect_submodules('triton')
    datas += collect_data_files('triton')
except Exception:
    pass

# Windows icon
ico_path = os.path.join(ROOT, 'ui', 'theme', 'corridorkey.ico')
icon_path = ico_path if os.path.exists(ico_path) else None

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        os.path.join(ROOT, 'scripts', 'windows', 'pyi_rth_cv2.py'),
    ],
    excludes=[
        # Not needed in GUI app
        'matplotlib',
        'tkinter',
        'jupyter',
        'IPython',
        'notebook',
        'scipy.spatial',
        'scipy.sparse',
        # macOS-only
        'corridorkey_mlx',
    ],
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
    upx=False,  # UPX can corrupt CUDA DLLs
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CorridorKey',
)
