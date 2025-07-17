# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('window.py', '.'),    
        ('yolo.py', '.'), 
        ('Logo.png', '.'),
        ('AegisLens-stroked.png', '.'),
        ('latest.pt', '.')
    ],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'cv2',
        'numpy',
        'ultralytics.models.yolo',  # More specific than 'ultralytics'
        'easyocr'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow', 'keras',               # Deep learning
        'matplotlib', 'scipy', 'pandas',     # Data science
        'notebook', 'jupyter',               # Jupyter
        'PIL', 'Pillow',
        "torch.backends.cuda",
        "torch.backends.mps",
        "torch.backends.openmp",
        "torch.backends.mkldnn",
        "torch.utils.tensorboard",
        "tensorboard",
    ],
    noarchive=False,
    optimize=1,  # Mild optimization
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AegisLens',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
