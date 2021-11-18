import argparse
import json
import pathlib
import shutil
import sys
import PyInstaller.__main__ as pyi

MIN_COVERAGE = 70
FILE_PATH = pathlib.Path(__file__).resolve()
PROJECT_PATH = FILE_PATH.parent
INSTALLER_PATH = PROJECT_PATH / 'installer'
EXCLUDED_IMPORT = [
    '--exclude-module', 'coverage', '--exclude-module', 'jedi', '--exclude-module', 'tkinter', '--exclude-module',
    'IPython', '--exclude-module', 'lib2to3', '--exclude-module', 'PyQt5.QtDBus', '--exclude-module',
    'PyQt5.QtDesigner', '--exclude-module', 'PyQt5.QtBluetooth', '--exclude-module', 'PyQt5.QtNetwork',
    '--exclude-module', 'PyQt5.QtNfc', '--exclude-module', 'PyQt5.QtWebChannel', '--exclude-module',
    'PyQt5.QtWebEngine', '--exclude-module', 'PyQt5.QtWebEngineCore', '--exclude-module', 'PyQt5.QtWebEngineWidgets',
    '--exclude-module', 'PyQt5.QtWebKit', '--exclude-module', 'PyQt5.QtWebKitWidgets', '--exclude-module',
    'PyQt5.QtWebSockets', '--exclude-module', 'PyQt5.QtTest', '--exclude-module', 'PyQt5.QtTextToSpeech',
    '--exclude-module', 'PyQt5.QtWinExtras', '--exclude-module', 'PyQt5.QtLocation', '--exclude-module',
    'PyQt5.QtMultimediaWidgets', '--exclude-module', 'PyQt5.QtNetworkAuth', '--exclude-module', 'PyQt5.QtPositioning',
    '--exclude-module', 'PyQt5.QtQuick', '--exclude-module', 'PyQt5.QtQuick3D', '--exclude-module', 'PyQt5.QtSensors',
    '--exclude-module', 'PyQt5.QtRemoteObjects', '--exclude-module', 'PyQt5.QtMultimedia', '--exclude-module',
    'PyQt5.QtQml', '--exclude-module', 'PyQt5.QtQuickWidgets', '--exclude-module', 'PyQt5.QtSql', '--exclude-module',
    'PyQt5.QtSvg', '--exclude-module', 'PyQt5.QtSerialPort', '--exclude-module', 'PyQt5.QtNetwork', '--exclude-module',
    'PyQt5.QtScript', '--exclude-module', 'PyQt5.QtXml', '--exclude-module', 'PyQt5.QtXmlPatterns'
]
HIDDEN_IMPORT = ['--hidden-import', 'pkg_resources.py2_warn']
IS_WINDOWS = sys.platform.startswith('win')


def format_code(check=False):
    """Formats the code with YAPF

    :param check: indicates the formatting should only be checked
    :type check: bool
    """
    try:
        from yapf import main as yapf_main
    except ImportError:
        print('\n"YAPF" is required for code formatting.')
        sys.exit(-1)

    yapf_args = ['make.py']
    if check:
        msg = 'Checking with YAPF'
        yapf_args.extend(['--diff'])
    else:
        msg = 'Reformatting with YAPF'
        yapf_args.extend(['--in-place', '--verbose'])

    print(msg)
    yapf_args.extend([
        '--parallel', '--recursive', '--verify', '--exclude', '*__config_data.py', '--style', 'setup.cfg', 'docs',
        'sscanss', 'tests', 'make.py'
    ])
    exit_code = yapf_main(yapf_args)
    if exit_code != 0:
        sys.exit(exit_code)

    if check:
        print('No re-formatting required!\n')


def linting():
    """Run pylint on the code"""
    try:
        from pylint.lint import Run as pylint_main
    except ImportError:
        print('\n"Pylint" is required for code linting.')
        sys.exit(-1)

    print('Checking with PyLint')
    try:
        pylint_main(['--rcfile', 'setup.cfg', '--score', 'false', 'sscanss'])
    except SystemExit as e:
        if e.code != 0:
            sys.exit(e.code)

    print('Linter completed with no errors or warnings!\n')


def create_pre_commit_hook():
    """Creates pre-commit hook in the .git folder"""
    script = f'#!/usr/bin/sh\n"{pathlib.Path(sys.executable).as_posix()}" "{FILE_PATH.as_posix()}" --check-all'

    try:
        with open(PROJECT_PATH / '.git' / 'hooks' / 'pre-commit', 'w') as f:
            f.write(script)
    except OSError:
        print('error', file=sys.stderr)
        sys.exit(1)

    print('Pre-commit hook installed successfully!')


def run_tests_with_coverage():
    """Runs units tests and checks coverages"""
    import unittest

    try:
        import coverage
    except ImportError:
        print('\n"Coverage" is required for coverage tests.')
        sys.exit(-1)

    cov = coverage.Coverage(source=['sscanss'])
    cov.erase()
    cov.start()

    loader = unittest.TestLoader()
    tests = loader.discover('tests')
    test_runner = unittest.runner.TextTestRunner()
    result = test_runner.run(tests)

    cov.stop()
    cov.save()

    if not result.wasSuccessful():
        sys.exit(1)

    percentage = cov.html_report(omit=['*__init__*'])
    if percentage < MIN_COVERAGE:
        err = 'Coverage of {} is below the expected threshold of {}%'.format(percentage, MIN_COVERAGE)
        print(err, file=sys.stderr)
        sys.exit(1)


def compile_log_config_and_schema():
    """Writes log and instrument schema json into a python file as dictionaries"""
    config_data_path = PROJECT_PATH / 'sscanss' / '__config_data.py'

    with open(PROJECT_PATH / 'logging.json', 'r') as log_file:
        log_config = json.loads(log_file.read())

    with open(PROJECT_PATH / 'instrument_schema.json', 'r') as schema_file:
        schema = json.loads(schema_file.read())

    with open(config_data_path, 'w') as f:
        f.write(f'LOG_CONFIG = {log_config}\n\n')
        f.write(f'SCHEMA = {schema}\n')


def build_editor():
    """Builds the executable for the instrument editor"""
    work_path = INSTALLER_PATH / 'temp'
    dist_path = INSTALLER_PATH / 'bundle'
    main_path = PROJECT_PATH / 'sscanss' / 'editor' / 'main.py'

    pyi_args = [
        '--name', 'editor', '--specpath',
        str(work_path), '--workpath',
        str(work_path), '--windowed', '--noconfirm', '--distpath',
        str(dist_path), '--clean',
        str(main_path)
    ]

    pyi_args.extend([
        '--exclude-module', 'matplotlib', '--exclude-module', 'hdf5', '--hidden-import', 'PyQt5.QtPrintSupport',
        *EXCLUDED_IMPORT, *HIDDEN_IMPORT
    ])

    pyi_args.extend(['--icon', str(INSTALLER_PATH / 'icons' / 'editor-logo.ico')])
    pyi.run(pyi_args)
    shutil.rmtree(work_path)


def build_sscanss():
    """Builds the executable for the sscanss application"""
    work_path = INSTALLER_PATH / 'temp'
    dist_path = INSTALLER_PATH / 'bundle' / 'app'
    main_path = PROJECT_PATH / 'sscanss' / 'app' / 'main.py'
    shutil.rmtree(dist_path, ignore_errors=True)

    pyi_args = [
        '--name', 'sscanss', '--specpath',
        str(work_path), '--workpath',
        str(work_path), '--windowed', '--noconfirm', '--distpath',
        str(dist_path), '--clean',
        str(main_path)
    ]

    pyi_args.extend(
        ['--exclude-module', 'PyQt5.Qsci', '--hidden-import', 'PyQt5.QtPrintSupport', *EXCLUDED_IMPORT, *HIDDEN_IMPORT])

    if IS_WINDOWS:
        pyi_args.extend(['--icon', str(INSTALLER_PATH / 'icons' / 'logo.ico')])

    pyi.run(pyi_args)

    exec_folder = next(dist_path.iterdir())
    exec_folder.rename(dist_path / 'bin')
    shutil.rmtree(work_path)

    # Copy resources into installer directory
    resources = ['instruments', 'static', 'LICENSE']

    for resource in resources:
        dest_path = dist_path / resource
        src_path = PROJECT_PATH / resource
        if src_path.is_file():
            shutil.copy(src_path, dest_path)
        else:
            shutil.copytree(src_path, dest_path, ignore=shutil.ignore_patterns('__pycache__'))

    if IS_WINDOWS:
        with open(INSTALLER_PATH / 'windows' / 'version.nsh', 'w') as ver_file:
            from sscanss.__version import __version__

            ver_file.write(f'!define VERSION "{__version__}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Developer tools for SScanSS 2')
    parser.add_argument('--add-pre-commit-hook', action='store_true', help='Add pre-commit hook')
    parser.add_argument('--test-coverage', action='store_true', help='Run unit test and generate coverage report')
    parser.add_argument('--check-code-format', action='store_true', help='Checks if code formatted correctly')
    parser.add_argument('--format-code', action='store_true', help='Formats code to match style')
    parser.add_argument('--run-linter', action='store_true', help='Run linter on the code')
    parser.add_argument('--check-all', action='store_true', help='Check code style and tests')
    parser.add_argument('--build-sscanss', action='store_true', help='Build the sscanss executable')
    parser.add_argument('--build-editor', action='store_true', help='Build the instrument editor executable')
    parser.add_argument('--build-all', action='store_true', help='Check code style, test and build all executables')

    args = parser.parse_args()

    if args.add_pre_commit_hook:
        create_pre_commit_hook()

    if args.build_all:
        args.check_all = True
        args.build_sscanss = True
        args.build_editor = True

    if args.check_all:
        args.check_code_format = True
        args.run_linter = True
        args.test_coverage = True

    if args.check_code_format or args.format_code:
        format_code(args.check_code_format)

    if args.run_linter:
        linting()

    if args.test_coverage:
        run_tests_with_coverage()

    if args.build_sscanss or args.build_editor:
        compile_log_config_and_schema()

        if args.build_sscanss:
            build_sscanss()

        if args.build_editor:
            build_editor()

    sys.exit(0)
