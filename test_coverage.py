import sys
import unittest
try:
    import coverage
except ImportError:
    print('\n"Coverage.py" is required for coverage tests.')
    sys.exit(-1)


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

    cov.html_report(omit=['test*', '*__init__*'])

    return result


if __name__ == '__main__':
    run_tests_with_coverage()
