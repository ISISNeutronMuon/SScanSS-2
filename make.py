import argparse
import os
import pathlib
import sys

MIN_COVERAGE = 70
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))


def create_pre_commit_hook():
    script = f'#!/usr/bin/sh\n{pathlib.Path(sys.executable).as_posix()} "{pathlib.Path(os.path.abspath(__file__)).as_posix()}" --test-coverage'

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
    parser.add_argument('--add-pre-commit-hook', action='store_true', help='')
    parser.add_argument('--test-coverage', action='store_true', help='')
    args = parser.parse_args()

    if args.add_pre_commit_hook:
        create_pre_commit_hook()

    if args.test_coverage:
        run_tests_with_coverage()

    sys.exit(0)
    