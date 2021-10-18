import pathlib
import pandas as pd
from logger import Logger

root_directory = pathlib.Path(__file__).parent.parent.resolve()


class DataCleanser(Logger):
    """
    Verification and cleansing of data objects from input testing results in dataframe format.
    """

    def __init__(self, df, logger_level='WARNING', logger_name="console_only_logger"):
        super().__init__(logger_level=logger_level, logger_name=logger_name)
        self.input_df = df
        self.df = df
        self.reference_files = [
            'inputs/RESULTS5-2A-EnergyPlus-9.0.1.xlsx',
            'inputs/RESULTS5-2A-BSIMAC-9-9.0.74.xlsx']
        self.valid_cases = {
            '600', '610', '620', '630', '640', '650', '660', '670', '680', '685', '695', '900', '910', '920',
            '930', '940', '950', '960', '980', '985', '995', '195', '200', '210', '215', '220', '230', '240',
            '250', '270', '280', '290', '300', '310', '320', '395', '400', '410', '420', '430', '440', '450',
            '460', '470', '800', '810'}
        return

    def __repr__(self):
        rep = 'DataCleanser(' \
              'df=pd.DataFrame.from_dict(' + str(self.df.to_dict()) + \
              '))'
        return rep

    def _check_cases(self, check_column):
        """
        Verify the case column is using valid values.  Drop rows that do not meet the criteria

        :param check_column: Column containing the test cases.
        :return: boolean series to filter input dataframe.  The internal cleansed dataframe is also updated with
            erroneous entries removed.
        """
        self.logger.info('Cleansing column {}'.format(check_column))
        try:
            failed_cases = ~self.df[check_column].astype(str).isin(self.valid_cases)
            if failed_cases.any():
                self.logger.error('Error: Invalid Case referenced.  These cases will be removed: {}'
                                  .format(self.df['case'][failed_cases]))
                self.df = self.df[~failed_cases].reset_index()
        except (KeyError, ValueError):
            failed_cases = None
            self.logger.error('Error: Case column was improperly referenced ({}).  No validation was performed on this '
                              'column'.format(check_column))
        return failed_cases

    def _check_numeric_with_limits(self, check_column, lower_limit=-float("inf"), upper_limit=float("inf")):
        """
        Verify a column is a valid numeric.  Also, optionally verify the values are within specified limits

        :param check_column: Column to verify
        :param lower_limit: Lowest allowable limit
        :param upper_limit: Highest allowable limit
        :return: boolean series to filter input dataframe.  The internal cleansed dataframe is also updated with
            erroneous entries removed.
        """
        self.logger.info('Cleansing column {}'.format(check_column))
        try:
            failed_numeric = pd.to_numeric(self.df[check_column], errors='coerce').isnull()
            if failed_numeric.any():
                self.logger.error('Info: Values for the {} column did not appear to be numeric and '
                                  'have been removed\n{}'
                                  .format(check_column, self.df[check_column][failed_numeric]))
            self.df[check_column] = self.df[check_column].apply(pd.to_numeric, errors='coerce')
            failed_limit = ~self.df[check_column]\
                .apply(lambda x: False if pd.isnull(x) else lower_limit <= int(x) <= upper_limit)
            if failed_limit.any():
                self.logger.warning(
                    'Info: Values for the {} column were either missing or appeared to be incorrect '
                    'and the values have been removed\n{}'.format(check_column, self.df[check_column][failed_limit]))
                self.df.loc[failed_limit, check_column] = None
        except (KeyError, ValueError):
            import traceback
            print(traceback.print_exc())
            failed_limit = None
            self.logger.error('Error: The {} column was improperly referenced for numeric verification. '
                              'No validation was performed on this column'
                              .format(check_column))
        return failed_limit

    def _check_columns(self, column_check_function, column_list):
        """
        format instructions and iterate a column check function

        :param column_check_function: function to iterate
        :param column_list: list of columns to iterate over. This is a
            tuple of tuple containing numeric check, where inner tuple is:
                0 - column name
                1 - kwargs for numeric check function
        :return: Updated self.df dataframe
        """
        # reformat numeric column list in case it was accidentally passed as string
        if isinstance(column_list, str):
            column_list = [column_list, ]
        for numeric_check_instructions in column_list:
            # process if kwargs were given
            if isinstance(numeric_check_instructions, (tuple, list)):
                column_name, kwargs = [*list(numeric_check_instructions) + [{}] * 2][:2]
            # process if only name was given
            elif isinstance(numeric_check_instructions, str):
                column_name = numeric_check_instructions
                kwargs = {}
            else:
                self.logger.error('Error: Invalid numeric columns input.  This validation was not performed')
                return self.df
            column_check_function(check_column=column_name, **kwargs)
        return

    def cleanse_conditioned_zone_loads_non_free_float(
            self,
            case_column: str = 'case',
            numeric_columns: list = (
                ('peak_heating_hour', {'lower_limit': 0, 'upper_limit': 24}),
                ('peak_cooling_hour', {'lower_limit': 0, 'upper_limit': 24}), ), ):
        """
        Perform operations to cleanse and verify data for the Conditioned Zone Loads (Non-Free-Float Test Cases) table.

        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Conditioned Zone Loads (Non-Free-Float Test Cases) table')
        self._check_cases(case_column)
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_annual_solar_radiation_direct_and_diffuse(
            self,
            case_column: str = 'case',
            numeric_columns: list = (
                ('kWh/m2', {'lower_limit': 0}), )):
        """
        Perform operations to cleanse and verify data for the Annual Solar Radiation (Direct + Diffuse)
        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Annual Solar Radiation (Direct + Diffuse) table')
        # check case column
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_sky_temperature_output(
            self,
            case_column: str = 'case',
            numeric_columns: list = (
                    ('Ann. Hourly Average C', {'lower_limit': -50, 'upper_limit': 50}),
                    ('Minimum C', {'lower_limit': -100, 'upper_limit': 100}),
                    ('Minimum Day', {'lower_limit': 1, 'upper_limit': 31}),
                    ('Minimum Hour', {'lower_limit': 0, 'upper_limit': 24}),
                    ('Maximum C', {'lower_limit': -100, 'upper_limit': 100}),
                    ('Maximum Day', {'lower_limit': 1, 'upper_limit': 31}),
                    ('Maximum Hour', {'lower_limit': 0, 'upper_limit': 24}))):
        """
        Perform operations to cleanse and verify data for the Sky Temperature Output table
        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing sky temperature output')
        # check case column
        self._check_cases(case_column)
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    # todo_140: Make a set of verification test that ensure the data is good for a specific output graphic
