import argparse
import logging
import re
import os
import sys
import pathlib
import inspect
import matplotlib.pyplot as plt

root_directory = pathlib.Path(__file__).parent.parent.resolve()

if str(root_directory) not in sys.path:
    sys.path.append(str(root_directory))

# imports below the system path append operation above are necessary for github workflow
from input_processor import InputProcessor  # noqa: E402
from graphics_renderer import GraphicsRenderer  # noqa: E402
from custom_exceptions import ASHRAE140TypeError  # noqa: E402


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
        '--render_graphics',
        '-rg',
        nargs='+',
        help='Graphic name to render.')
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
    if getattr(args, 'render_graphics'):
        render_from_input = True
    else:
        render_from_input = False
    for f in args.files:
        # Check files argument input.  If it's a directory then make a list of all files contained within.
        f = pathlib.Path(f).joinpath(root_directory, f)
        if pathlib.Path(f).exists():
            if pathlib.Path(f).is_dir():
                input_files = [i for i in f.rglob('*') if i.is_file() and i.suffix not in  ['.py', '.pyc']]
            else:
                input_files = [f, ]
        else:
            input_files = []
        processed_files = []
        for input_file in input_files:
            if 'input' in input_file.parts:
                try:
                    ip = InputProcessor(
                        logger_level=args.logger_level,
                        logger_name=logger_name,
                        input_file_location=str(input_file))
                    try:
                        if ip.input_file_location:
                            ip.logger.info('Processing file: {}'.format(ip.input_file_location))
                            output_file = ip.run()
                            if render_from_input:
                                processed_files.append(output_file)
                    except ASHRAE140TypeError:
                        ip.logger.error('Failed to process file: {}'.format(str(input_file)))
                        continue
                except ASHRAE140TypeError:
                    print('failed to process file: {}'.format(str(input_file)))
                    continue
        for input_file in input_files + processed_files:
            # Ignore base files used as comparisons for renderings
            if 'processed' in input_file.parts and '-'.join([input_file.parts[-3], input_file.parts[-2]]) not in [
                'basecalc-v1.0e',
                'bsimac-9.9.0.7.4',
                'cse-0.861.1',
                'dest-2.0.20190401',
                'energyplus-9.0.1',
                'energyplus-9.0.1',
                'esp-r-13.3',
                'esp-r-13.3',
                'fluent-6.1',
                'ght-2.02',
                'matlab-7.0.4.365-r14-sp2',
                'sunrel-gc-1.14.02',
                'trnsys-18.00.0001',
                'trnsys-18.00.0001',
                'va114-2.20'
            ]:
                try:
                    gr = GraphicsRenderer(
                        input_file,
                        logger_level=args.logger_level,
                        logger_name=logger_name)
                    # get rendering functions from class.  If the 'render_graphics' option was provided then only
                    # render the referenced graphic.  Otherwise, render all graphics
                    if getattr(args, 'render_graphics'):
                        render_function_names = ['_'.join(['render', i]) for i in getattr(args, 'render_graphics')]
                        bad_function_names = [i for i in render_function_names if not hasattr(gr, i)]
                        # print bad function references
                        for bad_function_name in bad_function_names:
                            gr.logger.warning('WARNING: Rendering function (%s) does not exist in GraphicsRenderer',
                                              bad_function_name)
                        render_functions = [(i, getattr(gr, i)) for i in render_function_names if hasattr(gr, i)]
                    else:
                        render_functions = [
                            (i, j) for i, j in
                            inspect.getmembers(gr, predicate=inspect.ismethod)
                            if i.startswith('render') and gr.section_type.lower().replace('-', '_') in i]
                    try:
                        for render_function_name, render_function in render_functions:
                            render_function()
                            plt.close('all')
                            gr.logger.info('%s rendered for %s', render_function_name, str(input_file))
                    except (ValueError, ASHRAE140TypeError):
                        gr.logger.error('Error: failed to render images: {}', str(input_file))
                        continue
                except ASHRAE140TypeError:
                    print('failed to render images: {}'.format(str(input_file)))
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
