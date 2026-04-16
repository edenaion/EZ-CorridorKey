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

# Ensure dist-info metadata for packages that use importlib.metadata at runtime.
# Packages listed here have their .dist-info/ bundled so that code paths
# doing `importlib.metadata.version(pkg)` or `entry_points(group=...)` do
# not crash with PackageNotFoundError on the frozen app. Absence here
# equals runtime crash on the first import of any such package.
for _meta_pkg in ['requests', 'regex', 'filelock', 'numpy',
                   'huggingface-hub', 'safetensors', 'importlib-metadata',
                   # ship blocker - MatAnyone2 imports imageio on Auto GVM
                   'imageio', 'imageio_ffmpeg',
                   # belt-and-braces for packages that currently rely on
                   # PyInstaller hook behaviour to ship their dist-info
                   'av', 'peft', 'pillow', 'tokenizers',
                   'accelerate', 'tqdm', 'packaging', 'diffusers',
                   'transformers', 'timm', 'click']:
    try:
        datas += copy_metadata(_meta_pkg)
    except Exception:
        pass

# Hidden imports -- everything the app needs at runtime
hiddenimports = [
    # -- Qt / GUI --
    'PySide6.QtWidgets',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtMultimedia',

    # -- Core libs --
    'cv2',
    'numpy',
    'PIL',
    'psutil',
    'huggingface_hub',
    'safetensors',
    'filelock',

    # -- ML frameworks --
    'torch',
    'torchvision',
    'torchvision.transforms',
    'timm',
    'timm.layers',
    'einops',
    'kornia',
    'kornia.filters',

    # -- transformers (BiRefNet + VideoMaMa) --
    'transformers',
    'transformers.modeling_utils',
    'transformers.configuration_utils',
    'transformers.models.auto',
    'transformers.image_processing_utils',

    # -- diffusers (GVM + VideoMaMa pipelines) --
    'diffusers',
    'diffusers.models',
    'diffusers.schedulers',
    'diffusers.pipelines',
    'diffusers.image_processor',
    'diffusers.loaders',

    # -- peft (GVM LoRA) --
    'peft',

    # -- Video I/O (GVM) --
    'av',
    'pims',
    'imageio',
    'imageio_ffmpeg',

    # -- MatAnyone2 --
    'omegaconf',

    # -- Backend modules --
    'backend',
    'backend.service',
    'backend.service.core',
    'backend.service.helpers',
    'backend.clip_state',
    'backend.job_queue',
    'backend.validators',
    'backend.errors',
    'backend.project',
    'backend.frame_io',
    'backend.natural_sort',
    'backend.ffmpeg_tools',
    'backend.ffmpeg_tools.discovery',
    'backend.ffmpeg_tools.extraction',
    'backend.ffmpeg_tools.probe',
    'backend.ffmpeg_tools.stitching',
    'backend.ffmpeg_tools.color',

    # -- UI modules --
    'ui',
    'ui.app',
    'ui.main_window',
    'ui.preview.frame_index',
    'ui.preview.display_transform',
    'ui.preview.async_decoder',

    # -- Inference modules --
    'CorridorKeyModule',
    'CorridorKeyModule.inference_engine',
    'CorridorKeyModule.backend',
    'CorridorKeyModule.core',
    'CorridorKeyModule.core.model_transformer',
    'CorridorKeyModule.core.color_utils',
    'modules.BiRefNetModule',
    'modules.BiRefNetModule.wrapper',
    'gvm_core',
    'gvm_core.wrapper',
    'VideoMaMaInferenceModule',
    'VideoMaMaInferenceModule.pipeline',
    'VideoMaMaInferenceModule.inference',

    # -- SAM2 tracker (guided mode) --
    'sam2_tracker',
    'sam2_tracker.wrapper',
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

# ---------------------------------------------------------------------------
# Collect submodules for large packages with lazy/dynamic imports
# ---------------------------------------------------------------------------
_collect_subs = [
    'transformers',
    'diffusers',
    'timm',
    'kornia',
    'safetensors',
    'peft',
    'modules.BiRefNetModule',
    'modules.MatAnyone2Module',
    'MatAnyone2Module',
]

for pkg in _collect_subs:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# Collect data files for packages that need them
_collect_data = [
    'transformers',
    'diffusers',
    'timm',
    'modules.BiRefNetModule',
    'modules.MatAnyone2Module',
    'MatAnyone2Module',
]

for pkg in _collect_data:
    try:
        datas += collect_data_files(
            pkg,
            excludes=['**/_BACKUPS/**', '**/*.bak'],
        )
    except Exception:
        pass

# BiRefNet checkpoint configs (needed for model loading)
for variant in ('BiRefNet-matting', 'BiRefNet_HR'):
    ckpt_dir = os.path.join(ROOT, 'modules', 'BiRefNetModule', 'checkpoints', variant)
    if os.path.isdir(ckpt_dir):
        datas.append((ckpt_dir, os.path.join('modules', 'BiRefNetModule', 'checkpoints', variant)))

# CorridorKeyModule core (model code)
ck_core = os.path.join(ROOT, 'CorridorKeyModule', 'core')
if os.path.isdir(ck_core):
    datas.append((ck_core, os.path.join('CorridorKeyModule', 'core')))

# Custom tree walker that includes .py files and skips backups/weights
def _collect_tree_skipping_backups(src_root, dest_root):
    collected = []
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [
            d for d in dirnames
            if d not in ('_BACKUPS', 'checkpoints', 'weights')
        ]
        for fname in filenames:
            if fname.endswith('.bak'):
                continue
            src = os.path.join(dirpath, fname)
            rel = os.path.relpath(src, src_root)
            dest = os.path.join(dest_root, os.path.dirname(rel))
            collected.append((src, dest))
    return collected

# GVM pipeline code
gvm_dir = os.path.join(ROOT, 'gvm_core', 'gvm')
if os.path.isdir(gvm_dir):
    datas += _collect_tree_skipping_backups(
        gvm_dir, os.path.join('gvm_core', 'gvm')
    )

# VideoMaMa pipeline code
vmm_dir = os.path.join(ROOT, 'VideoMaMaInferenceModule')
if os.path.isdir(vmm_dir):
    datas += _collect_tree_skipping_backups(vmm_dir, 'VideoMaMaInferenceModule')

# MatAnyone2 vendored package (wrapper.py uses runtime sys.path hack)
m2_dir = os.path.join(ROOT, 'modules', 'MatAnyone2Module', 'matanyone2')
if os.path.isdir(m2_dir):
    datas += _collect_tree_skipping_backups(
        m2_dir, os.path.join('modules', 'MatAnyone2Module', 'matanyone2')
    )

# CorridorKey base checkpoint AND the MLX acceleration weight — both
# bundled into the .app so Apple Silicon users never hit a download wall
# on first launch. .pth powers the default PyTorch/MPS path; the MLX
# framework code is already collected above (mlx, mlx.core, metallib),
# but its weights (corridorkey_mlx.safetensors) live next to the .pth
# in CorridorKeyModule/checkpoints and must be bundled explicitly.
#
# Fail loudly if the .pth is missing: shipping without it crashes every
# end user on first inference. The MLX safetensors is a soft miss — we
# warn but still produce an installer, because non-Apple-Silicon Macs
# (x86_64) don't need MLX and the wizard can still fetch it post-install
# if the user really wants it.
ck_checkpoints = os.path.join(ROOT, 'CorridorKeyModule', 'checkpoints')
_ck_pth_files = []
_ck_mlx_files = []
if os.path.isdir(ck_checkpoints):
    for _name in os.listdir(ck_checkpoints):
        if _name.endswith('.pth'):
            _ck_pth_files.append(_name)
        elif _name.endswith('.safetensors'):
            _ck_mlx_files.append(_name)
if not _ck_pth_files:
    raise RuntimeError(
        "CorridorKey checkpoint (.pth) not found in "
        f"{ck_checkpoints}. Run `python scripts/setup_models.py "
        "--corridorkey` on the build machine before running "
        "PyInstaller, or the .app will ship without the core "
        "model and crash for every end user."
    )
if not _ck_mlx_files:
    print(
        "\n[macos-spec] WARNING: no corridorkey_mlx.safetensors in "
        f"{ck_checkpoints}. Shipping without the MLX fast-path weight. "
        "Run `python scripts/setup_models.py --corridorkey-mlx` before "
        "building to include it.\n"
    )
for _weight in _ck_pth_files + _ck_mlx_files:
    datas.append((
        os.path.join(ck_checkpoints, _weight),
        os.path.join('CorridorKeyModule', 'checkpoints'),
    ))

# macOS icon
icns_path = os.path.join(ROOT, 'ui', 'theme', 'corridorkey.icns')
icon_path = icns_path if os.path.exists(icns_path) else None

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT, os.path.join(ROOT, 'modules')],
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
