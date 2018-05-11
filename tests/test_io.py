import unittest
import shutil
import tempfile
import os
from sscanss.core.io import reader, writer


class TestIO(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testHDFReadWrite(self):
        data = {'name': 'Test Project', 'instrument': 'IMAT'}

        filename = os.path.join(self.test_dir, 'test.h5')

        writer.write_project_hdf(data, filename)
        self.assertTrue(os.path.isfile(filename), 'write_project_hdf failed to write file')

        result = reader.read_project_hdf(filename)

        self.assertEqual(data['name'], result['name'], 'Save and Load data are not Equal')
        self.assertEqual(data['instrument'], result['instrument'], 'Save and Load data are not Equal')


if __name__ == '__main__':
    unittest.main()
