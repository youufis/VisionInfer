# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['visioninfer.py'],
             pathex=['D:\\MyPython\\visioninfer'],
             binaries=[],
             datas=[
	('D:\\MyPython\\visioninfer\\libs\libblas.dll','.'),
	('D:\\MyPython\\visioninfer\\libs\libgcc_s_seh-1.dll','.'),
	('D:\\MyPython\\visioninfer\\libs\libgfortran-3.dll','.'),
	('D:\\MyPython\\visioninfer\\libs\libiomp5md.dll','.'),
	('D:\\MyPython\\visioninfer\\libs\liblapack.dll','.'),
	('D:\\MyPython\\visioninfer\\libs\libquadmath-0.dll','.'),
	('D:\\MyPython\\visioninfer\\libs\mkldnn.dll','.'),
	('D:\\MyPython\\visioninfer\\libs\mklml.dll','.'),
	('D:\\MyPython\\visioninfer\\libs\warpctc.dll','.')
	],
             hiddenimports=[
	'pandas',
	'pandas._libs.tslibs.base',
	'pandas._libs.tslibs.timedeltas',
	'pandas._libs.tslibs.nattype',
	'pandas._libs.tslibs.np_datetime',
	'pandas._libs.skiplist'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='visioninfer',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , icon='logo.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='visioninfer')
