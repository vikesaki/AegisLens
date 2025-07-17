# -*- mode: python ; coding: utf-8 -*-

import os
import ultralytics
import easyocr

easyocr_model_path = '/Users/liswahyuni/.EasyOCR//model' 

ultralytics_path = os.path.dirname(ultralytics.__file__)

app_data = [
    ('latest.pt', '.'),
    ('Logo.png', '.'),
    (easyocr_model_path, 'easyocr/model'),
    (os.path.join(ultralytics_path, 'cfg'), 'ultralytics/cfg')
]

a = Analysis(
    ['main_app.py'],
    pathex=[],
    binaries=[],
    datas=app_data,
    hiddenimports=[
        'easyocr', 'ultralytics', 'cv2', 'skimage', 
        'scipy.signal', 'PyQt6.sip'
    ],
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
    [],
    exclude_binaries=True,
    name='AegisLens',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='Logo.png'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_app',
)