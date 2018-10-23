import unittest
from sscanss.core.instrument import read_instrument_description_file, get_instrument_list


class TestInstrument(unittest.TestCase):
    def testclass(self):
        instruments = get_instrument_list()
        self.assertIn('ENGIN-X', instruments)

        #instrument = read_instrument_description_file('../sscanss/instruments/engin-x/instrument.json')


if __name__ == '__main__':
    unittest.main()
