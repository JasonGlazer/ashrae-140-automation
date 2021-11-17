import argparse
import logging
import re
import os
import sys
import pathlib

root_directory = pathlib.Path(__file__).parent.parent.resolve()

if str(root_directory) not in sys.path:
    sys.path.append(str(root_directory))

print(sys.path)

from src.input_processor import InputProcessor
from src.file_renderer import FileRenderer
from src.custom_exceptions import ASHRAE140TypeError


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
        help='Path of input file to process.  Providing a directory will process all files within the target.')
    return parser


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
        # Check files argument input.  If it's a directory then make a list of all files contained within.
        f = pathlib.Path(f).joinpath(root_directory, f)
        if pathlib.Path(f).exists():
            if pathlib.Path(f).is_dir():
                input_files = [str(i) for i in f.rglob('*') if i.is_file()]
            else:
                input_files = [str(f), ]
        else:
            input_files = []
        for input_file in input_files:
            if 'input' in str(input_file):
                try:
                    ip = InputProcessor(
                        logger_level=args.logger_level,
                        logger_name=logger_name,
                        input_file_location=input_file)
                    try:
                        if ip.input_file_location:
                            ip.logger.info('Processing file: {}'.format(ip.input_file_location))
                            ip.run()
                    except ASHRAE140TypeError:
                        ip.logger.error('Failed to process file: {}'.format(str(input_file)))
                        continue
                except ASHRAE140TypeError:
                    print('failed to process file: {}'.format(str(input_file)))
                    continue
            elif 'processed' in str(input_file):
                try:
                    fr = FileRenderer(
                        logger_level=args.logger_level,
                        logger_name=logger_name,
                        file_name=input_file)
                    try:
                        if fr.file_name:
                            fr.logger.info('Rendering file: {}'.format(fr.file_name))
                            fr.run()
                    except ASHRAE140TypeError:
                        print('failed to render file: {}'.format(str(input_file)))
                        continue
                except ASHRAE140TypeError:
                    print('failed to process file: {}'.format(str(input_file)))
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
        raise FileNotFoundError('No Files specified for processing')
    main(ip_parser_args)
    logging.shutdown()
