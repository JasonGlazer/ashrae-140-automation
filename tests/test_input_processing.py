import os.path
import unittest
import tempfile
import pathlib
import sys
from argparse import Namespace

this_script_path = pathlib.Path(__file__)
sys.path.append(str(this_script_path.parent.parent.joinpath('src', )))
from src.main import main
from src.input_processor import InputProcessor
from src.descriptors import ASHRAE140FileNotFoundError
from src.input_processor import ASHRAE140TypeError


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

    def test_valid_file_location_is_validated(self):
        cwd_path = os.getcwd()
        root_path, _ = os.path.split(cwd_path)
        file_location = os.path.normpath(os.path.join(root_path, 'input/EnergyPlus/9.0.1/Std140_TF_Output.xlsx'))
        ip = InputProcessor(input_file_location=file_location)
        self.assertEqual(
            str(ip.input_file_location), file_location)
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
            with tempfile.NamedTemporaryFile(suffix='.xlssxm') as tf:
                InputProcessor(input_file_location=tf.name)
        return
