import os
import shutil
import sys
from sscanss.config import __version__
try:
    import PyInstaller
    import PyInstaller.__main__ as pyi
    from PyInstaller.compat import is_unix, is_win
except ImportError:
    print('\n"PyInstaller" is required to build executable.')
    sys.exit(-1)

from test_coverage import run_tests_with_coverage

INSTALLER_PATH = os.path.abspath('installer')


def build_exe():
    work_path = os.path.join(INSTALLER_PATH, 'temp')
    dist_path = os.path.join(INSTALLER_PATH, 'bundle')
    shutil.rmtree(dist_path, ignore_errors=True)

    pyi_args = ['--name', 'sscanss', '--specpath', work_path, '--workpath', work_path,
                '--windowed', '--noconfirm', '--distpath', dist_path, '--clean', 'sscanss/main.py']

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
                     '--exclude-module', 'scipy.integrate', '--exclude-module', 'scipy.interpolate'
                     ])

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
        if os.path.isfile(resource):
            shutil.copy(resource, dest_path)
        else:
            shutil.copytree(resource, dest_path)

    # if is_unix:
    #     import tarfile
    #     install_script_path = os.path.join(INSTALLER_PATH, 'linux', 'install.sh')
    #     archive_path = os.path.join(INSTALLER_PATH, 'linux', f'SScanSS-2-{__version__}-Linux.tar.gz')
    #     with tarfile.open(archive_path, 'w:gz') as archive:
    #         bundle_dir = os.listdir(dist_path)
    #         for path in bundle_dir:
    #             archive.add(os.path.join(dist_path, path), arcname=f'bundle/{path}')
    #         archive.add(install_script_path, arcname='install.sh')
    # elif is_win:
    if is_win:
        with open(os.path.join(INSTALLER_PATH, 'windows', 'version.nsh'), 'w') as ver_file:
            ver_file.write(f'!define VERSION "{__version__}"')


if __name__ == '__main__':
    # Always run tests before build
    success = run_tests_with_coverage()

    if success:
        # should be safe to build
        build_exe()
    else:
        print('Build was terminated due to failed tests')
        sys.exit(1)
