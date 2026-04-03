# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CorridorKey — Windows .exe bundle.

Usage:
    pyinstaller installers/corridorkey-windows.spec --noconfirm

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

# ---------------------------------------------------------------------------
# Hidden imports — everything the app needs at runtime
# ---------------------------------------------------------------------------
hiddenimports = [
    # ── Qt / GUI ──
    'PySide6.QtWidgets',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtMultimedia',

    # ── Core libs ──
    'cv2',
    'numpy',
    'PIL',
    'psutil',
    'pynvml',
    'huggingface_hub',
    'safetensors',
    'filelock',

    # ── ML frameworks ──
    'torch',
    'torchvision',
    'torchvision.transforms',
    'timm',                          # CorridorKey GreenFormer backbone
    'timm.layers',
    'einops',                        # BiRefNet + VideoMaMa tensor ops
    'kornia',                        # BiRefNet laplacian filter
    'kornia.filters',

    # ── transformers (BiRefNet + VideoMaMa) ──
    'transformers',
    'transformers.modeling_utils',   # PreTrainedModel
    'transformers.configuration_utils',  # PretrainedConfig
    'transformers.models.auto',      # AutoModelForImageSegmentation
    'transformers.image_processing_utils',

    # ── diffusers (GVM + VideoMaMa pipelines) ──
    'diffusers',
    'diffusers.models',
    'diffusers.schedulers',
    'diffusers.pipelines',
    'diffusers.image_processor',
    'diffusers.loaders',

    # ── peft (GVM LoRA) ──
    'peft',

    # ── Video I/O (GVM) ──
    'av',
    'pims',

    # ── MatAnyone2 ──
    'omegaconf',

    # ── Backend modules ──
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

    # ── UI modules ──
    'ui',
    'ui.app',
    'ui.main_window',
    'ui.preview.frame_index',
    'ui.preview.display_transform',
    'ui.preview.async_decoder',

    # ── Inference modules ──
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
]

# ---------------------------------------------------------------------------
# Collect submodules for large packages with lazy/dynamic imports
# ---------------------------------------------------------------------------
_collect_subs = [
    'transformers',
    'diffusers',
    'timm',
    'triton',
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
    'triton',
    'timm',
    'modules.BiRefNetModule',
    'modules.MatAnyone2Module',
    'MatAnyone2Module',
]

for pkg in _collect_data:
    try:
        datas += collect_data_files(pkg)
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

# GVM pipeline code
gvm_dir = os.path.join(ROOT, 'gvm_core', 'gvm')
if os.path.isdir(gvm_dir):
    datas.append((gvm_dir, os.path.join('gvm_core', 'gvm')))

# VideoMaMa pipeline code
vmm_dir = os.path.join(ROOT, 'VideoMaMaInferenceModule')
if os.path.isdir(vmm_dir):
    datas.append((vmm_dir, 'VideoMaMaInferenceModule'))

# Windows icon
ico_path = os.path.join(ROOT, 'ui', 'theme', 'corridorkey.ico')
icon_path = ico_path if os.path.exists(ico_path) else None

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT, os.path.join(ROOT, 'modules')],
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
