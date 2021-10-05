import pathlib
import pandas as pd
from descriptors import VerifyInputFile
from logger import Logger

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
        if isinstance(value, dict) and value:
            obj._data_sources = value
        else:
            obj._data_sources = {
                'conditioned_zone_loads_non_free_float': ('YourData', 68, 'B:L', 46)
            }
        return


class ExcelProcessor(Logger):
    """
    Extract, Transform, and Load operations for Excel input data

    :param file_location: location of file to be processed
    :param data_sources (Optional): data extraction instructions.
    """

    file_location = VerifyInputFile()
    data_sources = SetDataSources()

    def __init__(
            self,
            file_location,
            data_sources=None,
            logger_level="WARNING",
            logger_name="console_only_logger"):
        super().__init__(logger_level=logger_level, logger_name=logger_name)
        self.file_location = file_location
        self.data_sources = data_sources
        self.test_data = {}
        return

    def __repr__(self):
        rep = 'ExcelProcessor(' \
              'file_location=' + str(self.file_location) + \
              ')'
        return rep

    def _extract_conditioned_zone_loads_non_free_float(self) -> dict:
        """
        Retrieve and format data from the
        Conditioned Zone Loads (Non-Free-Float Test Cases) table

        :return: dictionary to be merged into main testing output dictionary
        """
        try:
            data_source = self.data_sources['conditioned_zone_loads_non_free_float']
        except KeyError:
            self.logger.error('Test data for Conditioned Zone Loads (Non-Free-Float Test Cases) was not found.  '
                              'Data will not be processed')
            return {}
        data_tab, skip_rows, excel_cols, n_rows = data_source
        df = pd.read_excel(
            self.file_location,
            sheet_name=data_tab,
            skiprows=skip_rows,
            usecols=excel_cols,
            nrows=n_rows)
        # format and verify dataframe
        df.columns = ['case', 'annual_heating_MWh', 'annual_cooling_MWh', 'peak_heating_kW', 'peak_heating_month',
                      'peak_heating_day', 'peak_heating_hour', 'peak_cooling_kW', 'peak_cooling_month',
                      'peak_cooling_day', 'peak_cooling_hour']
        numeric_columns = [1, 2, 3, 5, 6, 7, 9, 10]
        try:
            df.iloc[:, numeric_columns] = df.iloc[:, numeric_columns].apply(pd.to_numeric, errors='raise')
        except ValueError:
            self.logger.error('Failed to verify numeric columns in Conditioned Zone Loads '
                              '(Non-Free-Float Test Cases)')
            return {}
        cases = {'600', '610', '620', '630', '640', '650', '660', '670', '680', '685', '695', '900', '910', '920',
                 '930', '940', '950', '960', '980', '985', '995', '195', '200', '210', '215', '220', '230', '240',
                 '250', '270', '280', '290', '300', '310', '320', '395', '400', '410', '420', '430', '440', '450',
                 '460', '470', '800', '810'}
        failed_cases = df['case'][~df['case'].astype(str).isin(cases)]
        if len(failed_cases) > 0:
            self.logger.error('Invalid Case referenced in Conditioned Zone Loads '
                              '(Non-Free-Float Test Cases): {}'.format(failed_cases))
        try:
            failed_heating_hour = df['peak_heating_hour'].apply(lambda x: 0 < int(x) < 25)
            failed_heating_hour = df['peak_heating_hour'][~failed_heating_hour]
        except ValueError:
            self.logger.error('Invalid peak heating hour in Conditioned Zone Loads '
                              '(Non-Free-Float Test Cases)')
            return {}
        if len(failed_heating_hour) > 0:
            self.logger.error('Invalid peak heating hour in Conditioned Zone Loads '
                              '(Non-Free-Float Test Cases): {}'.format(failed_heating_hour))
            return {}
        # format cleansed dataframe into dictionary
        data_d = {}
        for idx, row in df.iterrows():
            case_number = row[0]
            row_obj = df.iloc[idx, 1:].to_dict()
            data_d.update({
                str(case_number): row_obj})
        return data_d

    def run(self):
        """
        Peform operations to convert Excel file into dictionary of dataframes.

        :return: json object of input data
        """
        conditioned_zone_loads_non_free_float_data = self._extract_conditioned_zone_loads_non_free_float()
        self.test_data.update({'conditioned_zone_loads_non_free_float': conditioned_zone_loads_non_free_float_data})
        return self
