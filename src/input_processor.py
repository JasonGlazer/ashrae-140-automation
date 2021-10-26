import re
import pathlib
import json
from logger import Logger
from src.descriptors import VerifyInputFile
from src.excel_processor import ExcelProcessor
from custom_exceptions import ASHRAE140TypeError, ASHRAE140ProcessingError

root_directory = pathlib.Path(__file__).parent.parent.resolve()


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
        """
        super().__init__(logger_level=logger_level, logger_name=logger_name)
        self.input_processing_map = {
            'excel': ExcelProcessor}
        self.input_file_location = input_file_location
        self.processing_pipeline = str(self.input_file_location)
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
            processor_class = self.input_processing_map[self.processing_pipeline]
        except KeyError:
            raise ASHRAE140TypeError('Specified processing pipeline does not have processing function implemented {}.'
                                     .format(self.processing_pipeline))
        # execute pipeline class which will return a cleansed data object
        try:
            data_object = processor_class(file_location=self.input_file_location)
            data_object.run()
        except ASHRAE140TypeError:
            raise ASHRAE140ProcessingError('Input file processing failed: {}'.format(self.input_file_location))
        if getattr(data_object, 'test_data'):
            program_name = self.input_file_location.parts[-3].lower()
            version = self.input_file_location.parts[-2].lower()
            section = self.input_file_location.stem.lower()
            with open(root_directory.joinpath(
                    'processed',
                    '.'.join(['-'.join([program_name, version, section]), 'json'])), 'w') as f:
                json.dump(data_object.test_data, f, indent=4, sort_keys=True)
        return
