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
        # these following three lines do not seem to be geting used.
        # self.reference_files = [
        #    'inputs/RESULTS5-2A-EnergyPlus-9.0.1.xlsx',
        #    'inputs/RESULTS5-2A-BSIMAC-9-9.0.74.xlsx']
        self.valid_tf_cases = {
            '600', '610', '620', '630', '640', '650', '660', '670', '680', '685', '695', '900', '910', '920',
            '930', '940', '950', '960', '980', '985', '995', '195', '200', '210', '215', '220', '230', '240',
            '250', '270', '280', '290', '300', '310', '320', '395', '400', '410', '420', '430', '440', '450',
            '460', '470', '800', '810', '600FF', '650FF', '680FF', '900FF', '950FF', '980FF'}
        self.valid_he_cases = {
            'CASE HE100', 'CASE HE110', 'CASE HE120', 'CASE HE130', 'CASE HE140', 'CASE HE150', 'CASE HE160',
            'CASE HE170', 'CASE HE210', 'CASE HE220', 'CASE HE230'}
        self.valid_ce_a_cases = {
            'CE100', 'CE110', 'CE120', 'CE130', 'CE140', 'CE150', 'CE160', 'CE165', 'CE170', 'CE180', 'CE185',
            'CE190', 'CE195', 'CE200'}
        self.valid_ce_b_cases = {
            'E300', 'E310', 'E320', 'E330', 'E340', 'E350', 'E360',
            'E400', 'E410', 'E420', 'E430', 'E440',
            'E500', 'E500 May-Sep', 'E510', 'E510 May-Sep', 'E520', 'E522', 'E525', 'E530', 'E540', 'E545'}
        self.valid_months = {
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        }

        return

    def __repr__(self):
        rep = 'DataCleanser(' \
              'df=pd.DataFrame.from_dict(' + str(self.df.to_dict()) + \
              '))'
        return rep

    def _check_cases(self, check_column, test_suite='TF'):
        """
        Verify the case column is using valid values.  Drop rows that do not meet the criteria

        :param check_column: Column containing the test cases.
        :return: boolean series to filter input dataframe.  The internal cleansed dataframe is also updated with
            erroneous entries removed.
        """
        self.logger.info('Cleansing column {}'.format(check_column))
        try:
            if test_suite == 'TF':
                failed_cases = ~self.df[check_column].astype(str).isin(self.valid_tf_cases)
            elif test_suite == 'HE':
                failed_cases = ~self.df[check_column].astype(str).isin(self.valid_he_cases)
            elif test_suite == 'CE_a':
                failed_cases = ~self.df[check_column].astype(str).isin(self.valid_ce_a_cases)
            elif test_suite == 'CE_b':
                failed_cases = ~self.df[check_column].astype(str).isin(self.valid_ce_b_cases)
            if failed_cases.any():
                self.logger.error('Error: Invalid Case referenced.  These cases will be removed: {}'
                                  .format(self.df['case'][failed_cases]))
                self.df = self.df[~failed_cases].reset_index()
        except (KeyError, ValueError):
            failed_cases = None
            self.logger.error('Error: Case column was improperly referenced ({}).  No validation was performed on this '
                              'column'.format(check_column))
        return failed_cases

    def _check_months(self, check_column):
        """
        Verify a column is using valid monthly values.  Drop rows that do not meet the criteria

        :param check_column: Column containing the month values.
        :return: boolean series to filter input dataframe.  The internal cleansed dataframe is also updated with
            erroneous entries removed.
        """
        self.logger.info('Cleansing column {}'.format(check_column))
        try:
            failed_cols = ~self.df[check_column].astype(str).isin(self.valid_months)
            if failed_cols.any():
                self.logger.error('Error: Invalid month referenced.  These cases will be removed: {}'
                                  .format(self.df[failed_cols]))
                self.df = self.df[~failed_cols].reset_index()
        except (KeyError, ValueError):
            failed_cols = None
            self.logger.error('Error: Month column was improperly referenced ({}).  No validation was performed on this'
                              ' column'.format(check_column))
        return failed_cols

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

    def cleanse_solar_radiation_annual(
            self,
            case_column: str = None,
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
        self.logger.info('Cleansing Solar Radiation Annual Incident table')
        if case_column:
            self._check_cases(case_column)
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

    def cleanse_monthly_conditioned_loads(
            self,
            case_column: str = 'case',
            month_column: str = 'month',
            numeric_columns: list = (
                ('total_heating_kwh', {'lower_limit': 0, 'upper_limit': 1000}),
                ('total_cooling_kwh', {'lower_limit': 0, 'upper_limit': 1000}),
                ('peak_heating_kw', {'lower_limit': 0, 'upper_limit': 10}),
                ('peak_heating_day', {'lower_limit': 1, 'upper_limit': 31}),
                ('peak_heating_hour', {'lower_limit': 0, 'upper_limit': 24}),
                ('peak_cooling_kw', {'lower_limit': 0, 'upper_limit': 10}),
                ('peak_cooling_day', {'lower_limit': 1, 'upper_limit': 31}),
                ('peak_cooling_hour', {'lower_limit': 0, 'upper_limit': 24}))):
        """
        Perform operations to cleanse and verify data for the Sky Temperature Output table
        :param case_column: column containing test case identifiers
        :param month_column: column containing month values
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Monthly Conditioned Loads')
        # check case column
        self._check_cases(case_column)
        self._check_months(month_column)
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_specific_day_hourly_output_incident_solar_radiation(
        self,
        numeric_columns: list = (
            ('hour', {'lower_limit': 0, 'upper_limit': 24}),
            ('horizontal', {'lower_limit': 0, 'upper_limit': 10000}),
            ('south', {'lower_limit': 0, 'upper_limit': 10000}),
            ('west', {'lower_limit': 0, 'upper_limit': 10000}))):
        """
        Perform operations to cleanse and verify data for the Sky Temperature Output table
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Specific Day Hourly Output Table')
        # check case column
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_specific_day_hourly_output_free_float_zone_temperatures(
            self,
            numeric_columns: list = (
                ('hour', {'lower_limit': 0, 'upper_limit': 24}),
                ('600FF', {'lower_limit': -100, 'upper_limit': 100}),
                ('900FF', {'lower_limit': -100, 'upper_limit': 100}),
                ('650FF', {'lower_limit': -100, 'upper_limit': 100}),
                ('950FF', {'lower_limit': -100, 'upper_limit': 100}),
                ('680FF', {'lower_limit': -100, 'upper_limit': 100}),
                ('980FF', {'lower_limit': -100, 'upper_limit': 100}))):
        """
        Perform operations to cleanse and verify data for the Sky Temperature Output table
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Specific Day Hourly Output Free Float Temperatures Table')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_specific_day_hourly_output_free_float_zone_loads_feb_1(
        self,
        numeric_columns: list = (
            ('hour', {'lower_limit': 0, 'upper_limit': 24}),
            ('600', {'lower_limit': -100, 'upper_limit': 100}),
            ('640', {'lower_limit': -100, 'upper_limit': 100}),
            ('660', {'lower_limit': -100, 'upper_limit': 100}),
            ('670', {'lower_limit': -100, 'upper_limit': 100}),
            ('680', {'lower_limit': -100, 'upper_limit': 100}),
            ('685', {'lower_limit': -100, 'upper_limit': 100}),
            ('695', {'lower_limit': -100, 'upper_limit': 100}),
            ('900', {'lower_limit': -100, 'upper_limit': 100}),
            ('940', {'lower_limit': -100, 'upper_limit': 100}),
            ('980', {'lower_limit': -100, 'upper_limit': 100}),
            ('985', {'lower_limit': -100, 'upper_limit': 100}),
            ('995', {'lower_limit': -100, 'upper_limit': 100}))):
        """
        Perform operations to cleanse and verify data for the Sky Temperature Output table
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Specific Day Hourly Output Free Float Zone Loads Table')
        # check case column
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_specific_day_hourly_output_free_float_zone_loads_july_14(
            self,
            numeric_columns: list = (
                ('hour', {'lower_limit': 0, 'upper_limit': 24}),
                ('600', {'lower_limit': -100, 'upper_limit': 100}),
                ('660', {'lower_limit': -100, 'upper_limit': 100}),
                ('670', {'lower_limit': -100, 'upper_limit': 100}),
                ('670', {'lower_limit': -100, 'upper_limit': 100}),
                ('680', {'lower_limit': -100, 'upper_limit': 100}),
                ('685', {'lower_limit': -100, 'upper_limit': 100}),
                ('695', {'lower_limit': -100, 'upper_limit': 100}),
                ('900', {'lower_limit': -100, 'upper_limit': 100}),
                ('980', {'lower_limit': -100, 'upper_limit': 100}),
                ('985', {'lower_limit': -100, 'upper_limit': 100}),
                ('995', {'lower_limit': -100, 'upper_limit': 100}))):
        """
        Perform operations to cleanse and verify data for the Sky Temperature Output table
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Specific Day Hourly Output Free Float Zone Loads Table')
        # check case column
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_specific_day_hourly_output_free_float_zone_loads_zone_temps(
            self,
            numeric_columns: list = (
                ('hour', {'lower_limit': 0, 'upper_limit': 24}),
                ('640', {'lower_limit': -100, 'upper_limit': 100}),
                ('940', {'lower_limit': -100, 'upper_limit': 100}))):
        """
        Perform operations to cleanse and verify data for the Sky Temperature Output table
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Specific Day Hourly Output Free Float Zone Loads Table')
        # check case column
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_specific_day_hourly_output_transmitted_total_solar_radiation(
            self,
            numeric_columns: list = (
                ('hour', {'lower_limit': 0, 'upper_limit': 24}),
                ('feb_1', {'lower_limit': -100, 'upper_limit': 1000}),
                ('may_4', {'lower_limit': -100, 'upper_limit': 1000}),
                ('july_14', {'lower_limit': -100, 'upper_limit': 1000}))):
        """
        Perform operations to cleanse and verify data for the Sky Temperature Output table
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing Monthly Conditioned Loads')
        # check case column
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_free_float_case_zone_temperatures(
            self,
            case_column: str = 'case',
            numeric_columns: list = (
                ('average_temperature', {'lower_limit': 0, 'upper_limit': 50}),
                ('minimum_temperature', {'lower_limit': -50, 'upper_limit': 50}),
                ('minimum_day', {'lower_limit': 1, 'upper_limit': 31}),
                ('minimum_hour', {'lower_limit': 0, 'upper_limit': 24}),
                ('maximum_temperature', {'lower_limit': 0, 'upper_limit': 100}),
                ('maximum_day', {'lower_limit': 1, 'upper_limit': 31}),
                ('maximum_hour', {'lower_limit': 0, 'upper_limit': 24}))):
        """
        Perform operations to cleanse and verify data for the free float case zone temperatures table
        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing free float case zone temperature output')
        # check case column
        self._check_cases(case_column)
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_he_furnace_energy(
            self,
            case_column: str = 'case',
            numeric_columns: list = (
                ('GJ', {'lower_limit': 0}), )):
        """
        Perform operations to cleanse and verify data for the heating equipment furnace load
        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing heating equipment furnace energy related')
        if case_column:
            self._check_cases(case_column, 'HE')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_he_temperature(
            self,
            case_column: str = 'case',
            numeric_columns: list = (
                ('C', {'lower_limit': -40, 'upper_limit': 100}), )):
        """
        Perform operations to cleanse and verify data for the heating equipment furnace load
        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing heating equipment furnace energy related')
        if case_column:
            self._check_cases(case_column, 'HE')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_ce_a_main_february_table(
            self,
            case_column: str = 'Cases',
            numeric_columns: list = (
                ('cooling_energy_total_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('cooling_energy_compressor_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('supply_fan_kWh', {'lower_limit': 0, 'upper_limit': 1000}),
                ('condenser_fan_kWh', {'lower_limit': 0, 'upper_limit': 1000}),
                ('evaporator_load_total_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('evaporator_load_sensible_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('evaporator_load_latent_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('envelope_load_total_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('envelope_load_sensible_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('envelope_load_latent_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('feb_mean_cop', {'lower_limit': 0, 'upper_limit': 10}),
                ('feb_mean_idb_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('feb_mean_hum_ratio_kg_kg', {'lower_limit': 0, 'upper_limit': 1.}),
                ('feb_max_cop', {'lower_limit': 0, 'upper_limit': 10}),
                ('feb_max_idb_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('feb_max_hum_ratio_kg_kg', {'lower_limit': 0, 'upper_limit': 1.}),
                ('feb_min_cop', {'lower_limit': 0, 'upper_limit': 10}),
                ('feb_min_idb_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('feb_min_hum_ratio_kg_kg', {'lower_limit': 0, 'upper_limit': 1.}),
            ), ):
        """
        Perform operations to cleanse and verify data for the Conditioned Zone Loads (Non-Free-Float Test Cases) table.

        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing main february table from ce_a')
        self._check_cases(case_column, 'CE_a')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_ce_b_annual_sums_means(
            self,
            case_column: str = 'cases',
            numeric_columns: list = (
                ('cooling_energy_total_kWh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('cooling_energy_compressor_kWh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('condenser_fan_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('indoor_fan_kWh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('evaporator_load_total_kWh', {'lower_limit': 0, 'upper_limit': 200000}),
                ('evaporator_load_sensible_kWh', {'lower_limit': 0, 'upper_limit': 200000}),
                ('evaporator_load_latent_kWh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('cop2', {'lower_limit': 0, 'upper_limit': 10}),
                ('indoor_dry_bulb_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('zone_relative_humidity_perc', {'lower_limit': 0, 'upper_limit': 100}),
            ), ):
        """
        Perform operations to cleanse and verify data for the Conditioned Zone Loads (Non-Free-Float Test Cases) table.

        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing annual sums and means table from ce_b')
        self._check_cases(case_column, 'CE_b')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_ce_b_annual_loads_maxima(
            self,
            case_column: str = 'cases',
            numeric_columns: list = (
                ('compressors_plus_fans_Wh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('evaporator_sensible_Wh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('evaporator_latent_Wh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('evaporator_total_Wh', {'lower_limit': 0, 'upper_limit': 100000}),
            ), ):
        self.logger.info('Cleansing annual maxima table from ce_b')
        self._check_cases(case_column, 'CE_b')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_ce_b_june28(
            self,
            numeric_columns: list = (
                ('compressor_Wh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('condenser_fans_Wh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('evaporator_total_Wh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('evaporator_sensible_Wh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('evaporator_latent_Wh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('zone_humidity_ratio_kg_kg', {'lower_limit': 0, 'upper_limit': 1}),
                ('cop2', {'lower_limit': 0, 'upper_limit': 10}),
                ('outdoor_drybulb_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('entering_drybulb_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('entering_wetbulb_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('outdoor_humidity_ratio_kg_kg', {'lower_limit': 0, 'upper_limit': 1})
            ), ):
        self.logger.info('Cleansing june 28 table from ce_b')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_ce_b_annual_cop_zone(
            self,
            case_column: str = 'cases',
            numeric_columns: list = (
                ('cop2_max_value', {'lower_limit': 0, 'upper_limit': 10}),
                ('cop2_min_value', {'lower_limit': 0, 'upper_limit': 10}),
                ('indoor_db_max_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('indoor_db_min_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('indoor_hum_rat_max_kg_kg', {'lower_limit': 0, 'upper_limit': 1}),
                ('indoor_hum_rat_min_kg_kg', {'lower_limit': 0, 'upper_limit': 1}),
                ('indoor_rel_hum_max_perc', {'lower_limit': 0, 'upper_limit': 100}),
                ('indoor_rel_hum_min_perc', {'lower_limit': 0, 'upper_limit': 100}),
            ), ):
        self.logger.info('Cleansing annual cop and zone min and mx table from ce_b')
        self._check_cases(case_column, 'CE_b')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    def cleanse_ce_b_average_daily(
            self,
            numeric_columns: list = (
                ('cooling_energy_total_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('cooling_energy_compressor_kWh', {'lower_limit': 0, 'upper_limit': 10000}),
                ('condenser_fan_kWh', {'lower_limit': 0, 'upper_limit': 1000}),
                ('indoor_fan_kWh', {'lower_limit': 0, 'upper_limit': 1000}),
                ('evaporator_load_total_kWh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('evaporator_load_sensible_kWh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('evaporator_load_latent_kWh', {'lower_limit': 0, 'upper_limit': 100000}),
                ('zone_humidity_ratio_kg_kg', {'lower_limit': 0, 'upper_limit': 1}),
                ('cop2', {'lower_limit': 0, 'upper_limit': 10}),
                ('outdoor_drybulb_c', {'lower_limit': 0, 'upper_limit': 100}),
                ('entering_drybulb_c', {'lower_limit': 0, 'upper_limit': 100}),
            ), ):
        """
        Perform operations to cleanse and verify data for the Conditioned Zone Loads (Non-Free-Float Test Cases) table.

        :param case_column: column containing test case identifiers
        :param numeric_columns: tuple of tuple containing numeric check, where inner tuple is:
            0 - column name
            1 - kwargs for numeric check function
        :return: Cleansed pandas DataFrame
        """
        self.logger.info('Cleansing annual daily average table from ce_b')
        self._check_columns(
            column_check_function=self._check_numeric_with_limits,
            column_list=numeric_columns)
        return self.df

    # todo_140: Make a set of verification test that ensure the data is good for a specific output graphic
