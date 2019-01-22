import sys
import unittest
try:
    import coverage
except ImportError:
    print('\n"Coverage.py" is required for coverage tests.')
    sys.exit(-1)

MIN_COVERAGE = 70


def run_tests_with_coverage():
    cov = coverage.Coverage()
    cov.erase()
    cov.start()

    loader = unittest.TestLoader()
    tests = loader.discover('tests')
    testRunner = unittest.runner.TextTestRunner()
    result = testRunner.run(tests)

    cov.stop()
    cov.save()

    if not result.wasSuccessful():
        return 1

    percentage = cov.html_report(omit=['test*', '*__init__*'])
    if percentage < MIN_COVERAGE:
        err = 'Coverage of {} is below the expected threshold of {}%'.format(percentage, MIN_COVERAGE)
        print(err, file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(run_tests_with_coverage())
