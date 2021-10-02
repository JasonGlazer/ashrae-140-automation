import argparse
import logging
import re
import os
from logger import Logger


def get_property(prop):
    """
    Get property value from __init__.py file in src directory
    :param prop: Property name
    :return: Return value for a given property
    """
    try:
        result = re.search(
            r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
            open(os.path.join(os.path.dirname(__file__), '__init__.py')).read())
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
    return parser


class InputProcessor(Logger):
    """
    Verify Input file and create a cleansed epJSON file for data visualizations
    """

    def __init__(self, logger_level="WARNING", logger_name="console_only_logger"):
        super().__init__(logger_level=logger_level, logger_name=logger_name)
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
    ip = InputProcessor(logger_level=args.logger_level, logger_name=logger_name)
    ip.logger.info('test')
    return


if __name__ == "__main__":
    ip_parser = build_parser()
    ip_parser_args, unknown_args = ip_parser.parse_known_args()
    main(ip_parser_args)
    logging.shutdown()
