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

INSTALLER_PATH = './installer'


def build_exe():
    work_path = os.path.join(INSTALLER_PATH, 'temp')
    dist_path = os.path.join(INSTALLER_PATH, 'bundle')
    shutil.rmtree(dist_path, ignore_errors=True)

    icon_path = 'logo.ico'

    pyi_args = ['--name', 'sscanss', '--specpath', work_path, '--workpath', work_path,
                '--windowed', '--noconfirm', '--distpath', dist_path, '--clean', 'sscanss/main.py']

    pyi_args.extend(['--exclude-module', 'coverage', '--exclude-module', 'jedi', '--exclude-module', 'tkinter',
                     '--exclude-module', 'IPython', '--exclude-module', 'lib2to3'])

    if is_win:
        pyi_args.extend(['--icon',  icon_path])

    pyi.run(pyi_args)

    exe_folder = os.listdir(dist_path)[0]
    os.rename(os.path.join(dist_path, exe_folder), os.path.join(dist_path, 'bin'))
    shutil.rmtree(work_path)

    # Copy resources into installer directory
    resources = ['instruments', 'static', 'LICENSE']

    shutil.copy('sscanss/logging.json', os.path.join(dist_path, 'bin', 'logging.json'))
    for resource in resources:
        dest_path = os.path.join(dist_path, resource)
        if os.path.isfile(resource):
            shutil.copy(resource, dest_path)
        else:
            shutil.copytree(resource, dest_path)

    if is_unix:
        shutil.make_archive(os.path.join(INSTALLER_PATH, 'linux', 'sscanss'), 'gztar', dist_path)
    elif is_win:
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
