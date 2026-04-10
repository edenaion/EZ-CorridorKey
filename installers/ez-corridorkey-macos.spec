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
import tomllib
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# Single source of truth for version
with open(os.path.join(SPECPATH, '..', 'pyproject.toml'), 'rb') as _f:
    APP_VERSION = tomllib.load(_f)['project']['version']

block_cipher = None

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))

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
    # pyproject.toml for runtime version detection
    (os.path.join(ROOT, 'pyproject.toml'), '.'),
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

# Collect corridorkey-mlx and MLX framework (Apple Silicon acceleration)
# MLX requires: Python submodules, native dylibs (libmlx.dylib),
# Metal GPU kernels (.metallib), and package metadata.
mlx_binaries = []
mlx_metadata_datas = []

for mlx_pkg in ('corridorkey_mlx', 'mlx', 'mlx.core', 'mlx.nn'):
    try:
        hiddenimports += collect_submodules(mlx_pkg)
        datas += collect_data_files(mlx_pkg)
        mlx_binaries += collect_dynamic_libs(mlx_pkg)
    except Exception:
        pass

# mlx-metal GPU kernels (separate package providing .metallib shaders)
for metal_pkg in ('mlx_metal', 'mlx.metallib'):
    try:
        hiddenimports += collect_submodules(metal_pkg)
        datas += collect_data_files(metal_pkg)
        mlx_binaries += collect_dynamic_libs(metal_pkg)
    except Exception:
        pass

# Package metadata (needed by importlib.metadata at runtime)
for meta_pkg in ('mlx', 'mlx-metal', 'corridorkey-mlx'):
    try:
        mlx_metadata_datas += copy_metadata(meta_pkg)
    except Exception:
        pass

datas += mlx_metadata_datas

# Explicitly find and bundle .metallib files from MLX's package directory
# These are the compiled Metal GPU shaders required for any MLX computation.
# PyInstaller's collect_data_files may miss them if they're in lib/ subdirs.
import importlib.util as _ilu
for _probe_pkg in ('mlx', 'mlx.core', 'mlx_metal'):
    try:
        _spec = _ilu.find_spec(_probe_pkg.split('.')[0])
        if _spec and _spec.submodule_search_locations:
            import pathlib as _pl
            for _loc in _spec.submodule_search_locations:
                _pkg_root = _pl.Path(_loc)
                # Search package root and lib/ subdirectory for .metallib files
                for _pattern in ('*.metallib', 'lib/*.metallib', '**/*.metallib'):
                    for _mlib in _pkg_root.glob(_pattern):
                        _dest = str(_mlib.parent.relative_to(_pkg_root.parent))
                        datas.append((str(_mlib), _dest))
                # Also grab any .dylib in lib/ that collect_dynamic_libs might miss
                for _dylib in _pkg_root.glob('lib/*.dylib'):
                    _dest = str(_dylib.parent.relative_to(_pkg_root.parent))
                    mlx_binaries.append((str(_dylib), _dest))
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
    binaries=mlx_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        os.path.join(ROOT, 'scripts', 'macos', 'pyi_rth_mlx.py'),
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
    name='EZ-CorridorKey',
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
    name='EZ-CorridorKey',
)

app = BUNDLE(
    coll,
    name='EZ-CorridorKey.app',
    icon=icon_path,
    bundle_identifier='com.ezscape.ez-corridorkey',
    info_plist={
        'CFBundleName': 'EZ-CorridorKey',
        'CFBundleDisplayName': 'EZ-CorridorKey',
        'CFBundleIdentifier': 'com.ezscape.ez-corridorkey',
        'CFBundleVersion': APP_VERSION,
        'CFBundleShortVersionString': APP_VERSION,
        'CFBundleExecutable': 'EZ-CorridorKey',
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
