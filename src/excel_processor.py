import pathlib
import re
import pandas as pd
from descriptors import VerifyInputFile
from logger import Logger
from custom_exceptions import ASHRAE140ProcessingError
from src.data_cleanser import DataCleanser

root_directory = pathlib.Path(__file__).parent.parent.resolve()


class SetDataSources:
    """
    Set the data extraction instructions.  Currently, this is a very simple descriptor, but it is created for
    future versions where data source location may change.

    data_sources formatting:
        0 - tab
        1 - start row
        2 - columns
        3 - number of rows to parse
        4 - dictionary of additional arguments to pd.read_excel
    """
    def __get__(self, obj, owner):
        data_sources = obj._data_sources
        return data_sources

    def __set__(self, obj, value):
        if isinstance(value, dict) and value:
            obj._data_sources = value
        else:
            obj._data_sources = {
                'identifying_information': ('YourData', 60, 'B:C', 3, {'header': None}),
                'conditioned_zone_loads_non_free_float': ('YourData', 68, 'B:L', 46),
                'annual_solar_radiation_direct_and_diffuse': ('YourData', 153, 'B:C', 5)
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
        self.software_name = None
        self.software_version = None
        self.software_release_date = None
        return

    def __repr__(self):
        rep = 'ExcelProcessor(' \
              'file_location=' + str(self.file_location) + \
              ')'
        return rep

    def _get_data(self, section_name) -> pd.DataFrame:
        """
        Retrieve section of data and return it as a pandas dataframe

        :param section_name: Named section of data in the data_sources class object.
        :return: Section of excel file converted to dataframe
        """
        try:
            data_source = self.data_sources[section_name]
        except KeyError:
            raise ASHRAE140ProcessingError('Data extraction instructions for Identifying Information section '
                                           'was not found')
        data_tab, skip_rows, excel_cols, n_rows, kwargs = [*list(data_source) + [{}] * 5][:5]
        df = pd.read_excel(
            self.file_location,
            sheet_name=data_tab,
            skiprows=skip_rows,
            usecols=excel_cols,
            nrows=n_rows,
            **kwargs)
        # todo_140: Write simple verifications that data loaded
        return df

    def _extract_identifying_information(self):
        """
        Retrieve information data and store it as class attributes.

        :return: Class attributes identifying software program.
        """
        df = self._get_data('identifying_information')
        if not re.match(r'^Software.*', df.iloc[0, 0]):
            self.logger.error('Software name information not found')
            self.software_name = None
        else:
            self.software_name = df.iloc[0, 1]
        if not re.match(r'^Version.*', df.iloc[1, 0]):
            self.logger.error('Software version information not found')
            self.software_version = None
        else:
            self.software_version = df.iloc[1, 1]
        if not re.match(r'^Date.*', df.iloc[2, 0]):
            self.logger.error('Software release date information not found')
            self.software_release_date = None
        else:
            self.software_release_date = df.iloc[2, 1]
        return

    def _extract_conditioned_zone_loads_non_free_float(self) -> dict:
        """
        Retrieve and format data from the
        Conditioned Zone Loads (Non-Free-Float Test Cases) table

        :return: dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('conditioned_zone_loads_non_free_float')
        # format and verify dataframe
        df.columns = ['case', 'annual_heating_MWh', 'annual_cooling_MWh', 'peak_heating_kW', 'peak_heating_month',
                      'peak_heating_day', 'peak_heating_hour', 'peak_cooling_kW', 'peak_cooling_month',
                      'peak_cooling_day', 'peak_cooling_hour']
        df['case'] = df['case'].astype(str)
        dc = DataCleanser(df)
        df = dc.cleanse_conditioned_zone_loads_non_free_float()
        # format cleansed dataframe into dictionary
        data_d = {}
        for idx, row in df.iterrows():
            case_number = row[0]
            row_obj = df.iloc[idx, 1:].to_dict()
            data_d.update({
                str(case_number): row_obj})
        return data_d

    def _extract_annual_solar_radiation_direct_and_diffuse(self) -> dict:
        """
        Retrieve and format data from the Annual Solar Radiation Section (Direct + Diffuse)

        :return: dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('annual_solar_radiation_direct_and_diffuse')
        df.columns = ['Surface', 'kWh/m2']
        dc = DataCleanser(df)
        df = dc.cleanse_annual_solar_radiation_direct_and_diffuse()
        data_d = {'600': {'Surface': {}}}
        for idx, row in df.iterrows():
            data_d['600']['Surface'].update({
                str(row['Surface']): {'kWh/m2': row['kWh/m2']}})
        return data_d

    def run(self):
        """
        Peform operations to convert Excel file into dictionary of dataframes.

        :return: json object of input data
        """
        self._extract_identifying_information()
        self.test_data.update({
            'identifying_information': {
                'software_name': self.software_name,
                'software_version': self.software_version,
                'software_release_date': str(self.software_release_date)
            }
        })
        self.test_data.update(
            {
                'conditioned_zone_loads_non_free_float': self._extract_conditioned_zone_loads_non_free_float(),
                'annual_solar_radiation_direct_and_diffuse': self._extract_annual_solar_radiation_direct_and_diffuse()})
        return self
