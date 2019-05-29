import os
import os.path
import shutil
import sys
try:
    import PyInstaller
    import PyInstaller.__main__ as pyi
except ImportError:
    print('\n"PyInstaller" is required to build executable.')
    sys.exit(-1)

from test_coverage import run_tests_with_coverage

INSTALLER_PATH = './installer'


def build_exe():
    work_path = os.path.join(INSTALLER_PATH, 'temp')
    dist_path = os.path.join(INSTALLER_PATH, 'dist')
    icon_path = 'logo.ico'

    pyi_args = ['--name', 'sscanss', '--specpath', INSTALLER_PATH, '--workpath', work_path,
                '--windowed',  '--noconfirm', '--distpath', dist_path, '--clean',  'sscanss/main.py']

    if PyInstaller.is_win:
        pyi_args.extend(['--icon',  icon_path,  ])

    pyi.run(pyi_args)

    exe_folder = os.listdir(dist_path)[0]
    os.rename(os.path.join(dist_path, exe_folder), os.path.join(dist_path, 'bin'))

    # Copy resources into installer directory
    resources = ['static', 'LICENSE']
    for resource in resources:
        dest_path = os.path.join(dist_path, resource)
        if os.path.isfile(resource):
            shutil.copy(resource, dest_path)
        else:
            shutil.copytree(resource, dest_path)

    release = os.path.join(INSTALLER_PATH, 'sscanss')
    archive_format = 'zip' if PyInstaller.is_win else 'gztar'
    shutil.make_archive(release, archive_format, dist_path)


if __name__ == '__main__':
    # clear build directory
    shutil.rmtree(INSTALLER_PATH, ignore_errors=True)

    # Always run tests before build
    success = run_tests_with_coverage()

    if success:
        # should be safe to build
        build_exe()
    else:
        print('Build was terminated due to failed tests')
        sys.exit(1)
