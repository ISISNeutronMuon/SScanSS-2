import argparse
import os
import pathlib
import sys

MIN_COVERAGE = 70
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))


def format_code(check=False):
    try:
        from black import main as black_main
    except ImportError:
        print('\n"Black" is required for code formatting.')
        sys.exit(-1)

    print("Running black")

    try:
        black_args = ["--check", 'sscanss', 'tests']
        black_main(black_args if check else black_args[1:])
    except SystemExit as exc:
        if exc.code == 1:
            sys.exit(exc.code)


def create_pre_commit_hook():
    script = f'#!/usr/bin/sh\n{pathlib.Path(sys.executable).as_posix()} "{pathlib.Path(os.path.abspath(__file__)).as_posix()}" --check-all'

    try:
        with open(os.path.join(PROJECT_PATH, '.git', 'hooks', 'pre-commit'), 'w') as f:
            f.write(script)
    except OSError:
        print("error", file=sys.stderr)
        sys.exit(1)

    print("Pre-commit hook installed successfully!")


def run_tests_with_coverage():
    import unittest
    try:
        import coverage
    except ImportError:
        print('\n"Coverage.py" is required for coverage tests.')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Builds executables for SScanSS 2')
    parser.add_argument('--add-pre-commit-hook', action='store_true', help='Adds pre-commit hook')
    parser.add_argument('--test-coverage', action='store_true', help='Run unit test and generate coverage report')
    parser.add_argument('--check-code-format', action='store_true', help='Checks if code formatted correctly')
    parser.add_argument('--format-code', action='store_true', help='Formats code to match style')
    parser.add_argument('--check-all', action='store_true', help='Checks code style and tests')
    args = parser.parse_args()

    if args.add_pre_commit_hook:
        create_pre_commit_hook()

    if args.check_all:
        args.format_code = True
        args.test_coverage = True

    if args.check_code_format or args.format_code:
        format_code(args.check_code_format)

    if args.test_coverage:
        run_tests_with_coverage()

    sys.exit(0)
