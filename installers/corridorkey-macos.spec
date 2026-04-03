# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CorridorKey — macOS .app bundle.

Usage:
    pyinstaller installers/corridorkey-macos.spec --noconfirm

Notes:
    - Builds a .app bundle for macOS (Apple Silicon / arm64)
    - Checkpoints are NOT bundled — placed next to .app or downloaded on first launch
    - Uses corridorkey-mlx backend if installed, falls back to torch MPS
    - CUDA/NVIDIA deps are excluded (not available on macOS)
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
    (os.path.join(ROOT, 'ui', 'theme', 'corridorkey.svg'), os.path.join('ui', 'theme')),
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
    'psutil',  # Apple Silicon memory reporting
    'huggingface_hub',  # Model downloads in setup wizard
]

# Try to collect corridorkey-mlx if installed
try:
    hiddenimports += collect_submodules('corridorkey_mlx')
    datas += collect_data_files('corridorkey_mlx')
except Exception:
    pass

# Try to collect MatAnyone2Module
for dynamic_pkg in ('modules.MatAnyone2Module', 'MatAnyone2Module'):
    try:
        hiddenimports += collect_submodules(dynamic_pkg)
        datas += collect_data_files(dynamic_pkg)
    except Exception:
        pass

# macOS icon
icns_path = os.path.join(ROOT, 'ui', 'theme', 'corridorkey.icns')
icon_path = icns_path if os.path.exists(icns_path) else None

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        os.path.join(ROOT, 'scripts', 'macos', 'pyi_rth_cv2.py'),
    ],
    excludes=[
        # Not needed on macOS
        'matplotlib',
        'tkinter',
        'jupyter',
        'IPython',
        'notebook',
        'scipy.spatial',
        'scipy.sparse',
        # NVIDIA/CUDA — not available on macOS
        'pynvml',
        'nvidia',
        'nvidia.cuda_runtime',
        'triton',
        'triton_windows',
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
    upx=False,  # UPX not useful on macOS arm64
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,  # True swallows first click on macOS
    target_arch='arm64',  # Apple Silicon only
    codesign_identity=None,  # Signing done post-build
    entitlements_file=os.path.join(ROOT, 'scripts', 'macos', 'CorridorKey.entitlements'),
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

app = BUNDLE(
    coll,
    name='CorridorKey.app',
    icon=icon_path,
    bundle_identifier='com.corridordigital.corridorkey',
    info_plist={
        'CFBundleName': 'CorridorKey',
        'CFBundleDisplayName': 'EZ-CorridorKey',
        'CFBundleIdentifier': 'com.corridordigital.corridorkey',
        'CFBundleVersion': '1.9.0',
        'CFBundleShortVersionString': '1.9.0',
        'CFBundleExecutable': 'CorridorKey',
        'CFBundlePackageType': 'APPL',
        'LSMinimumSystemVersion': '12.0',
        'NSHighResolutionCapable': True,
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeExtensions': ['mp4', 'mov', 'avi', 'mkv', 'webm'],
                'CFBundleTypeName': 'Video File',
                'CFBundleTypeRole': 'Editor',
            }
        ],
    },
)
