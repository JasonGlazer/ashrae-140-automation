import unittest
from argparse import Namespace
from src.process_input_file import main


class TestInputProcessor(unittest.TestCase):
    """
    Test the input file verification and processing into a cleansed file for vizualizations
    """

    def setup(self):
        return

    def teardown(self):
        return

    def test_no_specified_file_returns_error(self):
        with self.assertRaises(AttributeError):
            main(Namespace())
        return
