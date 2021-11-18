import argparse
import json
import os
import pprint
import shutil
import sys
import PyInstaller.__main__ as pyi
from PyQt5.pyrcc_main import processResourceFile
from sscanss.__version import __version__

IS_WINDOWS = sys.platform.startswith('win')
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
INSTALLER_PATH = os.path.join(PROJECT_PATH, 'installer')
EXCLUDED_IMPORT = ['--exclude-module', 'coverage', '--exclude-module', 'jedi', '--exclude-module', 'tkinter',
                   '--exclude-module', 'IPython', '--exclude-module', 'lib2to3', '--exclude-module', 'PyQt5.QtDBus',
                   '--exclude-module', 'PyQt5.QtDesigner', '--exclude-module', 'PyQt5.QtBluetooth',
                   '--exclude-module', 'PyQt5.QtNetwork', '--exclude-module', 'PyQt5.QtNfc',
                   '--exclude-module', 'PyQt5.QtWebChannel', '--exclude-module', 'PyQt5.QtWebEngine',
                   '--exclude-module', 'PyQt5.QtWebEngineCore', '--exclude-module', 'PyQt5.QtWebEngineWidgets',
                   '--exclude-module', 'PyQt5.QtWebKit', '--exclude-module', 'PyQt5.QtWebKitWidgets',
                   '--exclude-module', 'PyQt5.QtWebSockets', '--exclude-module', 'PyQt5.QtTest',
                   '--exclude-module', 'PyQt5.QtTextToSpeech', '--exclude-module', 'PyQt5.QtWinExtras',
                   '--exclude-module', 'PyQt5.QtLocation', '--exclude-module', 'PyQt5.QtMultimediaWidgets',
                   '--exclude-module', 'PyQt5.QtNetworkAuth', '--exclude-module', 'PyQt5.QtPositioning',
                   '--exclude-module', 'PyQt5.QtQuick', '--exclude-module', 'PyQt5.QtQuick3D',
                   '--exclude-module', 'PyQt5.QtSensors', '--exclude-module', 'PyQt5.QtRemoteObjects',
                   '--exclude-module', 'PyQt5.QtMultimedia', '--exclude-module', 'PyQt5.QtQml',
                   '--exclude-module', 'PyQt5.QtQuickWidgets', '--exclude-module', 'PyQt5.QtSql',
                   '--exclude-module', 'PyQt5.QtSvg', '--exclude-module', 'PyQt5.QtSerialPort',
                   '--exclude-module', 'PyQt5.QtNetwork', '--exclude-module', 'PyQt5.QtScript',
                   '--exclude-module', 'PyQt5.QtXml', '--exclude-module', 'PyQt5.QtXmlPatterns']
HIDDEN_IMPORT = ['--hidden-import', 'pkg_resources.py2_warn']


def build_resource_file():
    """Compiles app and editor icons into Qt resource file"""
    resource_file = os.path.join(PROJECT_PATH, 'sscanss', '__resource.py')

    if os.path.isfile(resource_file):
        os.remove(resource_file)

    processResourceFile([os.path.join(INSTALLER_PATH, 'icons', 'images.qrc')], resource_file, False)


def compile_log_config_and_schema():
    """Writes log and instrument schema json into a python file as dictionaries"""
    config_data_path = os.path.join(PROJECT_PATH, 'sscanss', '__config_data.py')

    with open(os.path.join(PROJECT_PATH, 'logging.json'), 'r') as log_file:
        log_config = pprint.pformat(json.loads(log_file.read()))

    with open(os.path.join(PROJECT_PATH, 'instrument_schema.json'), 'r') as schema_file:
        schema = pprint.pformat(json.loads(schema_file.read()))

    with open(config_data_path, 'w') as f:
        f.write(f'LOG_CONFIG = {log_config}\n\n')
        f.write(f'SCHEMA = {schema}\n')


def build_editor():
    """Builds the executable for the instrument editor"""
    work_path = os.path.join(INSTALLER_PATH, 'temp')
    dist_path = os.path.join(INSTALLER_PATH, 'bundle')
    main_path = os.path.join(PROJECT_PATH, 'sscanss', 'editor', 'main.py')

    pyi_args = ['--name', 'editor', '--specpath', work_path, '--workpath', work_path,
                '--windowed', '--noconfirm', '--distpath', dist_path, '--clean', main_path]

    pyi_args.extend(['--exclude-module', 'matplotlib', '--exclude-module', 'hdf5',
                     '--hidden-import', 'PyQt5.QtPrintSupport', *EXCLUDED_IMPORT, *HIDDEN_IMPORT])

    pyi_args.extend(['--icon',  os.path.join(INSTALLER_PATH, 'icons', 'editor-logo.ico')])
    pyi.run(pyi_args)
    shutil.rmtree(work_path)


def build_sscanss():
    """Builds the executable for the sscanss application"""
    work_path = os.path.join(INSTALLER_PATH, 'temp')
    dist_path = os.path.join(INSTALLER_PATH, 'bundle', 'app')
    main_path = os.path.join(PROJECT_PATH, 'sscanss', 'app', 'main.py')
    shutil.rmtree(dist_path, ignore_errors=True)

    pyi_args = ['--name', 'sscanss', '--specpath', work_path, '--workpath', work_path,
                '--windowed', '--noconfirm', '--distpath', dist_path, '--clean', main_path]

    pyi_args.extend(['--exclude-module', 'PyQt5.Qsci', '--hidden-import', 'PyQt5.QtPrintSupport',
                     *EXCLUDED_IMPORT, *HIDDEN_IMPORT])

    if IS_WINDOWS:
        pyi_args.extend(['--icon',  os.path.join(INSTALLER_PATH, 'icons', 'logo.ico')])

    pyi.run(pyi_args)

    exe_folder = os.listdir(dist_path)[0]
    os.rename(os.path.join(dist_path, exe_folder), os.path.join(dist_path, 'bin'))
    shutil.rmtree(work_path)

    # Copy resources into installer directory
    resources = ['instruments', 'static', 'LICENSE']

    for resource in resources:
        dest_path = os.path.join(dist_path, resource)
        src_path = os.path.join(PROJECT_PATH, resource)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
        else:
            shutil.copytree(src_path, dest_path, ignore=shutil.ignore_patterns('__pycache__'))

    if IS_WINDOWS:
        with open(os.path.join(INSTALLER_PATH, 'windows', 'version.nsh'), 'w') as ver_file:
            ver_file.write(f'!define VERSION "{__version__}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Builds executables for SScanSS 2')
    parser.add_argument('--skip-tests', action='store_true', help='This skips the unit tests.')
    parser.add_argument('--skip-editor', action='store_true', help='This builds sscanss without the editor.')
    args = parser.parse_args()

    build_resource_file()
    compile_log_config_and_schema()

    success = True
    if not args.skip_tests:
        from test_coverage import run_tests_with_coverage
        success = run_tests_with_coverage()

    if success:
        # should be safe to build
        build_sscanss()
        if not args.skip_editor:
            build_editor()
    else:
        print('Build was terminated due to failed tests', file=sys.stderr)
        sys.exit(1)
