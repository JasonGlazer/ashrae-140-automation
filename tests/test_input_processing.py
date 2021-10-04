import unittest
import tempfile
import pathlib
import sys
from argparse import Namespace

this_script_path = pathlib.Path(__file__)
sys.path.append(str(this_script_path.parent.parent.joinpath('src', )))
from src.process_input_file import main
from src.process_input_file import InputProcessor
from src.process_input_file import ASHRAE140FileNotFoundError, ASHRAE140TypeError


class TestInputProcessor(unittest.TestCase):
    """
    Test the input file verification and processing into a cleansed file for vizualizations
    """

    def setup(self):
        return

    def teardown(self):
        return

    def test_version(self):
        main(
            Namespace(
                version=True))
        return

    def test_no_specified_file_returns_error(self):
        with self.assertRaises(AttributeError):
            main(Namespace())
        return

    def test_valid_file_location_is_valicated(self):
        ip = InputProcessor(input_file_location='input/RESULTS5-2A.xlsx')
        self.assertRegex(
            str(ip.input_file_location),
            r'.*/input/RESULTS5-2A\.xlsx$')
        self.assertEqual(
            ip.processing_pipeline,
            'excel')
        return

    def test_bad_file_path_returns_error(self):
        with self.assertRaises(ASHRAE140FileNotFoundError):
            InputProcessor(input_file_location='bad/file/path')
        return

    def test_bad_extension_returns_error(self):
        with self.assertRaises(ASHRAE140TypeError):
            with tempfile.NamedTemporaryFile(suffix='.bad') as tf:
                InputProcessor(input_file_location=tf.name)
        return
