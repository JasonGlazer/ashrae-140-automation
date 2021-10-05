import argparse
import logging
import re
import os
import pathlib
import json
from logger import Logger
from src.descriptors import VerifyInputFile
from src.excel_processor import ExcelProcessor
from custom_exceptions import ASHRAE140TypeError, ASHRAE140ProcessingError

root_directory = pathlib.Path(__file__).parent.parent.resolve()


def get_property(prop):
    """
    Get property value from __init__.py file in src directory
    :param prop: Property name
    :return: Return value for a given property
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), '__init__.py')) as f:
            result = re.search(
                r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                f.read())
            output = result.group(1)
    except AttributeError:
        output = '{} could not be found'.format(prop)
    return output


def build_parser():  # pragma: no cover
    """
    Build argument parser.
    """
    parser = argparse.ArgumentParser(
        prog='ASHRAE 140 Automation',
        description='Automated report generation of ASHRAE 140 Verification')
    parser.add_argument(
        '--logger_level',
        '-l',
        nargs='?',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='Specify logger level')
    parser.add_argument(
        '--version',
        '-v',
        action='store_true',
        help='Display version information')
    parser.add_argument(
        '--write_logs',
        '-wl',
        action='store_true',
        help='Write logs to file')
    parser.add_argument(
        "--files",
        '-f',
        nargs='+',
        help='Path of input file to process')
    return parser


class SetProcessingPipeline:
    """
    Set the processing path based on the input file type
    """
    def __get__(self, obj, owner):
        processing_pipeline = obj._processing_pipeline
        return processing_pipeline

    def __set__(self, obj, value):
        # set processing pipeline type, return error if unexpected extension found
        if re.match(r'.*\.xls((x)?|(m)?)$', value):
            obj._processing_pipeline = 'excel'
        else:
            obj._processing_pipeline = None
            raise ASHRAE140TypeError('File type is not currently supported: {}'.format(value))
        return


class InputProcessor(Logger):
    """
    Verify Input file and create a cleansed epJSON file for data visualizations
    """

    input_file_location = VerifyInputFile()
    processing_pipeline = SetProcessingPipeline()

    def __init__(
            self,
            input_file_location,
            logger_level="WARNING",
            logger_name="console_only_logger"):
        """
        :param logger_level: Logging level
        :param logger_name: Specified logger to use
        :param input_file_location: input file to verify and process
        :param data_sources: dictionary of
        """
        super().__init__(logger_level=logger_level, logger_name=logger_name)
        self.input_processing_map = {
            'excel': ExcelProcessor}
        self.input_file_location = input_file_location
        self.processing_pipeline = str(self.input_file_location)
        try:
            processing_class = self.input_processing_map[self.processing_pipeline](
                file_location=self.input_file_location)
        except KeyError:
            raise ASHRAE140TypeError('Specified processing pipeline does not have processing class implemented {}.'
                                     .format(self.processing_pipeline))
        print(processing_class)
        processing_class.run()
        return

    def __repr__(self):
        rep = 'InputProcessor(input_file_location=' + str(self.input_file_location) + ')'
        return rep

    def run(self) -> None:
        """
        Take class attributes describing input file and perform processing inputs

        :return: class attribute with input data loaded as dictionary object.  Also return it as the function default.
        """
        # Get appropriate pipeline for processing
        valid_pipelines = {'excel', }
        if self.processing_pipeline not in valid_pipelines:
            raise ASHRAE140TypeError('Specified processing pipeline is invalid {}.  The valid pipeline options'
                                     'are {}'.format(self.processing_pipeline, valid_pipelines))
        try:
            processing_function = self.input_processing_map[self.processing_pipeline]
        except KeyError:
            raise ASHRAE140TypeError('Specified processing pipeline does not have processing function implemented {}.'
                                     .format(self.processing_pipeline))
        # execute pipeline class which will return a cleansed data object
        try:
            data_object = processing_function(file_location=self.input_file_location)
            data_object.run()
        except ASHRAE140TypeError:
            raise ASHRAE140ProcessingError('Input file processing failed: {}'.format(self.input_file_location))
        if getattr(data_object, 'test_data'):
            with open(root_directory.joinpath(
                    'processed',
                    '.'.join([self.input_file_location.stem, 'json'])), 'w') as f:
                json.dump(data_object.test_data, f, indent=4, sort_keys=True)
        return


def main(args=None):
    if hasattr(args, 'version') and args.version:
        version = get_property('__version__')
        print('ASHRAE 140 Automation Version: {}'.format(version))
        return
    # set the arg defaults for testing when Namespace is used
    if not hasattr(args, 'logger_level'):
        args.logger_level = 'WARNING'
    if getattr(args, 'write_logs', None):
        logger_name = 'file_logger'
    else:
        logger_name = 'console_only_logger'
    for f in args.files:
        try:
            ip = InputProcessor(
                logger_level=args.logger_level,
                logger_name=logger_name,
                input_file_location=f)
            try:
                if ip.input_file_location:
                    ip.logger.info('Processing file: {}'.format(ip.input_file_location))
                    ip.run()
            except ASHRAE140TypeError:
                ip.logger.error('Failed to process file: {}'.format(str(f)))
                continue
        except ASHRAE140TypeError:
            print('failed to process file: {}'.format(str(f)))
            continue
    return


if __name__ == "__main__":
    ip_parser = build_parser()
    ip_parser_args, unknown_args = ip_parser.parse_known_args()
    # If unknown arguments are passed, and no file specified, then put the arguments
    #  in the file namespace.
    if not ip_parser_args.files and unknown_args:
        ip_parser_args.files = unknown_args
    if not ip_parser_args.files:
        ip_parser.print_help()
        print('------------------------------')
        raise FileNotFoundError('No Files specified for processing')
    main(ip_parser_args)
    logging.shutdown()
