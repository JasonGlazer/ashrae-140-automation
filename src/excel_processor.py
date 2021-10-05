import pathlib
import pandas as pd
from descriptors import VerifyInputFile

root_directory = pathlib.Path(__file__).parent.parent.resolve()


class SetDataSources:
    """
    Set the data extraction instructions.  Currently, this is a very simple descriptor, but it is created for
    future versions where data source location may change.
    """
    def __get__(self, obj, owner):
        data_sources = obj._data_sources
        return data_sources

    def __set__(self, obj, value):
        obj._data_sources = [('YourData', 68, 'B:L', 46)]
        return


class ExcelProcessor:
    """
    Extract, Transform, and Load operations for Excel input data

    :param file_location: location of file to be processed
    :param data_sources (Optional): data extraction instructions.
    """

    file_location = VerifyInputFile()
    data_sources = SetDataSources()

    def __init__(self, file_location, data_sources=None):
        self.file_location = file_location
        if not data_sources:
            self.data_sources = file_location
        else:
            self.data_sources = data_sources
        self.test_data = {}
        return

    def __repr__(self):
        rep = 'ExcelProcessor(' \
              'file_location=' + str(self.file_location) + \
              ')'
        return rep

    def _extract_conditioned_zone_loads_non_free_float(self):
        """
        Retrieve and format data from the
        Conditioned Zone Loads (Non-Free-Float Test Cases) table

        :return: dictionary to be merged into main testing output dictionary
        """
        return

    def run(
            self,
            file_location: str = None,
            data_sources: list = None) -> dict:
        """
        Peform operations to convert Excel file into dictionary of dataframes.

        :param file_location: Location of Excel file
        :param data_sources: List of extraction instructions.
        :return: json object of input data
        """
        if not file_location:
            file_location = self.file_location
        if not data_sources:
            data_sources = self.data_sources
        for data_tab, skip_rows, excel_cols, n_rows in data_sources:
            df = pd.read_excel(
                file_location,
                sheet_name=data_tab,
                skiprows=skip_rows,
                usecols=excel_cols,
                nrows=n_rows)
        print(df)
        return
