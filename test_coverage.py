import sys
import unittest
try:
    import coverage
except ImportError:
    print('\n"Coverage.py" is required for coverage tests.')
    sys.exit(-1)

cov = coverage.Coverage()
cov.erase()
cov.start()

loader = unittest.TestLoader()
tests = loader.discover('tests')
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)

cov.stop()
cov.save()

cov.html_report()
