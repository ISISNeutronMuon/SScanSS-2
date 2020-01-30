import argparse
import json
import os
import shutil
import sys
import fastjsonschema
import PyInstaller
import PyInstaller.__main__ as pyi
from PyInstaller.compat import is_unix, is_win
from sscanss.config import __version__
from test_coverage import run_tests_with_coverage

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
INSTALLER_PATH = os.path.join(PROJECT_PATH, 'installer')


def compile_schema(filename):
    with open(os.path.join(PROJECT_PATH, 'instrument_schema.json'), 'r') as schema_file:
        schema = json.loads(schema_file.read())

    code = fastjsonschema.compile_to_code(schema)
    with open(filename, 'w') as f:
        f.write(code)


def build_exe():
    work_path = os.path.join(INSTALLER_PATH, 'temp')
    dist_path = os.path.join(INSTALLER_PATH, 'bundle')
    main_path = os.path.join(PROJECT_PATH, 'sscanss', 'main.py')
    shutil.rmtree(dist_path, ignore_errors=True)

    pyi_args = ['--name', 'sscanss', '--specpath', work_path, '--workpath', work_path,
                '--windowed', '--noconfirm', '--distpath', dist_path, '--clean', main_path]

    pyi_args.extend(['--exclude-module', 'coverage', '--exclude-module', 'jedi', '--exclude-module', 'tkinter',
                     '--exclude-module', 'IPython', '--exclude-module', 'lib2to3',
                     '--exclude-module', 'PyQt5.QtDBus', '--exclude-module', 'PyQt5.QtDesigner',
                     '--exclude-module', 'PyQt5.QtBluetooth', '--exclude-module', 'PyQt5.QtNetwork',
                     '--exclude-module', 'PyQt5.QtNfc', '--exclude-module', 'PyQt5.QtWebChannel',
                     '--exclude-module', 'PyQt5.QtWebEngine',  '--exclude-module', 'PyQt5.QtWebEngineCore',
                     '--exclude-module', 'PyQt5.QtWebEngineWidgets', '--exclude-module', 'PyQt5.QtWebKit',
                     '--exclude-module', 'PyQt5.QtWebKitWidgets', '--exclude-module', 'PyQt5.QtWebSockets',
                     '--exclude-module', 'PyQt5.QtTest', '--exclude-module', 'PyQt5.QtXml',
                     '--exclude-module', 'PyQt5.QtLocation', '--exclude-module', 'PyQt5.QtMultimediaWidgets',
                     '--exclude-module', 'PyQt5.QtNetworkAuth', '--exclude-module', 'PyQt5.QtPositioning',
                     '--exclude-module', 'PyQt5.QtQuick', '--exclude-module', 'PyQt5.QtSensors',
                     '--exclude-module', 'PyQt5.QtHelp', '--exclude-module', 'PyQt5.QtMultimedia',
                     '--exclude-module', 'PyQt5.QtQml', '--exclude-module', 'PyQt5.QtQuickWidgets',
                     '--exclude-module', 'PyQt5.QtSql', '--exclude-module', 'PyQt5.QtSvg',
                     '--exclude-module', 'PyQt5.QtSerialPort', '--exclude-module', 'PyQt5.QtNetwork',
                     '--exclude-module', 'PyQt5.QtPrintSupport', '--exclude-module', 'PyQt5.sip',
                     '--exclude-module', 'PyQt5.QtScript', '--exclude-module', 'PyQt5.QtXmlPatterns',
                     '--exclude-module', 'scipy.integrate', '--exclude-module', 'scipy.interpolate',
                     '--hidden-import', 'pkg_resources.py2_warn'])

    if is_win:
        pyi_args.extend(['--icon',  os.path.join(INSTALLER_PATH, 'windows', 'logo.ico')])

    pyi.run(pyi_args)

    exe_folder = os.listdir(dist_path)[0]
    os.rename(os.path.join(dist_path, exe_folder), os.path.join(dist_path, 'bin'))
    shutil.rmtree(work_path)

    # Copy resources into installer directory
    resources = ['instruments', 'static', 'LICENSE', 'logging.json']

    for resource in resources:
        dest_path = os.path.join(dist_path, resource)
        src_path = os.path.join(PROJECT_PATH, resource)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
        else:
            shutil.copytree(src_path, dest_path)

    if is_win:
        with open(os.path.join(INSTALLER_PATH, 'windows', 'version.nsh'), 'w') as ver_file:
            ver_file.write(f'!define VERSION "{__version__}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Builds executables for SScanSS-2')
    parser.add_argument('--skip-tests', action='store_true', help='This skips the unit tests.')

    args = parser.parse_args()
    compile_schema(os.path.join(PROJECT_PATH, 'sscanss', 'core', 'instrument', '__validator.py'))
    success = run_tests_with_coverage() if not args.skip_tests else True

    if success:
        # should be safe to build
        build_exe()
    else:
        print('Build was terminated due to failed tests')
        sys.exit(1)
