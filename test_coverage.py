import sys
import unittest
try:
    import coverage
except ImportError:
    print('\n"Coverage.py" is required for coverage tests.')
    sys.exit(-1)

MIN_COVERAGE = 70


def run_tests_with_coverage():
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
        return False

    percentage = cov.html_report(omit=['*__init__*'])
    if percentage < MIN_COVERAGE:
        err = 'Coverage of {} is below the expected threshold of {}%'.format(percentage, MIN_COVERAGE)
        print(err, file=sys.stderr)
        return False

    return True


if __name__ == '__main__':
    success = run_tests_with_coverage()
    if success:
        sys.exit(0)

    sys.exit(1)
