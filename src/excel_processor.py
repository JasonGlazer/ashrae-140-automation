import re
import pandas as pd
from descriptors import VerifyInputFile
from logger import Logger
from custom_exceptions import ASHRAE140ProcessingError
from src.data_cleanser import DataCleanser


class SectionType:
    """
    Identify Section Type based on input file name
    """
    def __get__(self, obj, owner):
        section_type = obj._section_type
        return section_type

    def __set__(self, obj, value):
        if re.match(r'.*Std140_TF_Output.*', str(value.name), re.IGNORECASE):
            obj._section_type = 'TF'
        elif re.match(r'.*Std140_GC_Output.*', str(value.name), re.IGNORECASE):
            obj._section_type = 'GC'
        elif re.match(r'.*Std140_HE_Output.*', str(value.name), re.IGNORECASE):
            obj._section_type = 'HE'
        else:
            obj.logger.error('Error: The file name ({}) did not match formatting guidelines or '
                             'the referenced section at the beginning of the name is not supported'
                             .format(str(value.name)))
        return


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
            if obj.section_type == 'TF':
                obj._data_sources = {
                    'identifying_information': ('YourData', 60, 'B:C', 3, {'header': None}),
                    'conditioned_zone_loads_non_free_float': ('YourData', 68, 'B:L', 46),
                    'solar_radiation_annual_incident': ('YourData', 153, 'B:C', 5),
                    'solar_radiation_unshaded_annual_transmitted': ('YourData', 161, 'B:C', 4),
                    'solar_radiation_shaded_annual_transmitted': ('YourData', 168, 'B:C', 2),
                    'sky_temperature_output': ('YourData', 176, 'B:K', 1),
                    'annual_hourly_zone_temperature_bin_data': ('YourData', 328, 'B:C', 149),
                    'free_float_case_zone_temperatures': ('YourData', 128, 'B:K', 7),
                    'monthly_conditioned_zone_loads': ('YourData', 188, 'B:R', 12),
                    'specific_day_hourly_output': ('YourData', 228, 'B:T', 24),
                    'specific_day_hourly_output_free_float_zone_temperatures': ('YourData', 292, 'B:H', 24),
                    'specific_day_hourly_output_free_float_zone_loads': ('YourData', 260, 'B:Z', 24)
                }
            elif obj.section_type == 'GC':
                obj._data_sources = {
                    'identifying_information': ('YourData', 4, 'E:I', 4, {'header': None}),
                    'steady_state_cases': ('YourData', 57, 'D:H', 6, {'header': None})
                }
            elif obj.section_type == 'HE':
                obj._data_sources = {
                    'identifying_information': ('Sheet1', 17, 'A:B', 2, {'header': None}),
                    'total_furnace_load': ('Sheet1', 19, 'A:B', 11),
                    'total_furnace_input': ('Sheet1', 35, 'A:B', 11),
                    'fuel_consumption': ('Sheet1', 51, 'A:B', 11),
                    'fan_energy_both_fan': ('Sheet1', 67, 'A:B', 6),
                    'mean_zone_temperature': ('Sheet1', 78, 'A:B', 3),
                    'maximum_zone_temperature': ('Sheet1', 86, 'A:B', 3),
                    'min0imum_zone_temperature': ('Sheet1', 94, 'A:B', 3)
                }
            else:
                obj.logger.error('Error: Section ({}) is not currently supported'.format(obj.section_type))
        return


class SetProcessingFunctions:
    """
    Set the functions to perform for processing.
    """
    def __get__(self, obj, owner):
        processing_functions = obj._processing_functions
        return processing_functions

    def __set__(self, obj, value):
        if value == 'TF':
            obj._processing_functions = {
                'identifying_information': obj._extract_identifying_information_tf(),
                'conditioned_zone_loads_non_free_float': obj._extract_conditioned_zone_loads_non_free_float(),
                'solar_radiation_annual_incident': obj._extract_solar_radiation_annual_incident(),
                'solar_radiation_unshaded_annual_transmitted': obj._extract_solar_radiation_unshaded_annual_transmitted(),
                'solar_radiation_shaded_annual_transmitted': obj._extract_solar_radiation_shaded_annual_transmitted(),
                'sky_temperature_output': obj._extract_sky_temperature_output(),
                'hourly_annual_zone_temperature_bin_data': obj._extract_hourly_annual_zone_temperature_bin_data(),
                'free_float_case_zone_temperatures': obj._extract_free_float_case_zone_temperatures(),
                'monthly_conditioned_zone_loads': obj._extract_monthly_conditioned_zone_loads(),
                'specific_day_hourly_output': obj._extract_specific_day_hourly_output(),
                'specific_day_hourly_output_free_float_zone_temperatures':
                    obj._extract_specific_day_hourly_output_free_float_zone_temperatures(),
                'specific_day_hourly_output_free_float_zone_loads':
                    obj._extract_specific_day_hourly_output_free_float_zone_loads()}
        elif value == 'GC':
            obj._processing_functions = {
                'identifying_information': obj._extract_identifying_information_gc(),
                'steady_state_cases': obj._extract_steady_state_cases()}
        elif value == 'HE':
            obj._processing_functions = {
                'identifying_information': obj._extract_identifying_information_he(),
                'steady_state_cases': obj._extract_steady_state_cases()}
        else:
            obj.logger.error('Error: Section ({}) is not currently supported'.format(obj.section_type))
        return


class ExcelProcessor(Logger):
    """
    Extract, Transform, and Load operations for Excel input data

    :param file_location: location of file to be processed
    :param data_sources (Optional): data extraction instructions.
    """

    file_location = VerifyInputFile()
    section_type = SectionType()
    data_sources = SetDataSources()
    processing_functions = SetProcessingFunctions()

    def __init__(
            self,
            file_location,
            data_sources=None,
            logger_level="WARNING",
            logger_name="console_only_logger"):
        super().__init__(logger_level=logger_level, logger_name=logger_name)
        self.file_location = file_location
        self.section_type = self.file_location
        self.data_sources = data_sources
        self.processing_functions = self.section_type
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

    def _get_data(self, region_name) -> pd.DataFrame:
        """
        Retrieve section of data and return it as a pandas dataframe

        :param region_name: Named section of data in the data_sources class object.
        :return: Section of excel file converted to dataframe
        """
        try:
            data_source = self.data_sources[region_name]
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

    # Section Thermal Fabric TF (was 5-2A) data
    def _extract_identifying_information_tf(self):
        """
        Retrieve information data from section Thermal Fabric submittal and store it as class attributes.

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
        data_d = {
            'software_name': self.software_name,
            'software_version': self.software_version,
            'software_release_date': str(self.software_release_date)
        }
        return data_d

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

    def _extract_solar_radiation_annual_incident(self) -> dict:
        """
        Retrieve and format data from the Solar Radiation ANNUAL INCIDENT (Total Direct-Beam and Diffuse) section

        :return: dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('solar_radiation_annual_incident')
        df.columns = ['Surface', 'kWh/m2']
        dc = DataCleanser(df)
        df = dc.cleanse_solar_radiation_annual()
        data_d = {'600': {'Surface': {}}}
        for idx, row in df.iterrows():
            data_d['600']['Surface'].update({
                str(row['Surface']): {'kWh/m2': row['kWh/m2']}})
        return data_d

    def _extract_solar_radiation_unshaded_annual_transmitted(self) -> dict:
        """
        Retrieve and format data from the Solar Radiation UNSHADED ANNUAL TRANSMITTED
        (Total Direct-Beam and Diffuse) section

        :return: dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('solar_radiation_unshaded_annual_transmitted')
        df.columns = ['Case/Surface', 'kWh/m2']
        df[['Case', 'Surface']] = df['Case/Surface'].str.split(pat='/', expand=True)
        df = df.drop(columns=['Case/Surface', ])
        dc = DataCleanser(df)
        df = dc.cleanse_solar_radiation_annual(case_column='Case')
        data_d = {}
        for idx, row in df.iterrows():
            data_d.update(
                {
                    row['Case']: {
                        'Surface': {
                            row['Surface']: {
                                'kWh/m2': row['kWh/m2']}}}})
        return data_d

    def _extract_solar_radiation_shaded_annual_transmitted(self):
        """
        Retrieve and format data from the Solar Radiation SHADED ANNUAL TRANSMITTED
        (Total Direct-Beam and Diffuse) section

        :return: dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('solar_radiation_shaded_annual_transmitted')
        df.columns = ['Case/Surface', 'kWh/m2']
        df[['Case', 'Surface']] = df['Case/Surface'].str.split(pat='/', expand=True)
        df = df.drop(columns=['Case/Surface', ])
        dc = DataCleanser(df)
        df = dc.cleanse_solar_radiation_annual(case_column='Case')
        data_d = {}
        for idx, row in df.iterrows():
            data_d.update(
                {
                    row['Case']: {
                        'Surface': {
                            row['Surface']: {
                                'kWh/m2': row['kWh/m2']}}}})
        return data_d

    def _extract_sky_temperature_output(self) -> dict:
        """
        Retrieve and format data from the Sky Temperature Output table

        :return: dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('sky_temperature_output')
        df.columns = ['case', 'Ann. Hourly Average C', 'Minimum C', 'Minimum Month', 'Minimum Day', 'Minimum Hour',
                      'Maximum C', 'Maximum Month', 'Maximum Day', 'Maximum Hour']
        dc = DataCleanser(df)
        df = dc.cleanse_sky_temperature_output()
        data_d = {'600': {}}
        for idx, row in df.iterrows():
            data_d['600'].update({'Average': {'C': row['Ann. Hourly Average C']}})
            data_d['600'].update({'Minimum': {
                'C': row['Minimum C'],
                'Month': row['Minimum Month'],
                'Day': row['Minimum Day'],
                'Hour': row['Minimum Hour']}})
            data_d['600'].update({'Maximum': {
                'C': row['Maximum C'],
                'Month': row['Maximum Month'],
                'Day': row['Maximum Day'],
                'Hour': row['Maximum Hour']
            }})
        return data_d

    def _extract_hourly_annual_zone_temperature_bin_data(self) -> dict:
        """
        Retrieve and format data from the Hourly Annual Zone Temperature Bin Data table

        :return: dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('annual_hourly_zone_temperature_bin_data')
        df.columns = ['temperature_bin_c', 'number_of_hours']
        data_d = {'900FF': {'temperature_bin_c': {}}}
        for idx, row in df.iterrows():
            data_d['900FF']['temperature_bin_c'].update(
                {int(row['temperature_bin_c']): {'number_of_hours': int(row['number_of_hours'])}})
        return data_d

    def _extract_free_float_case_zone_temperatures(self):
        """
        Retrieve and format data from the Free Float Case Zone Temperature Table (TF)
        :return:  dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('free_float_case_zone_temperatures')
        df.columns = ['case', 'average_temperature', 'minimum_temperature', 'minimum_month', 'minimum_day', 'minimum_hour',
                      'maximum_temperature', 'maximum_month', 'maximum_day', 'maximum_hour']
        dc = DataCleanser(df)
        df = dc.cleanse_free_float_case_zone_temperatures()
        data_d = {}
        for idx, row in df.iterrows():
            case_number = row[0]
            row_obj = df.iloc[idx, 1:].to_dict()
            data_d.update({
                str(case_number): row_obj})
        return data_d

    def _extract_monthly_conditioned_zone_loads(self):
        """
        Retrieve and format data from the Monthly Conditioned Zone Loads Table
        :return:  dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('monthly_conditioned_zone_loads')
        df_columns = ['month', 'total_heating_kwh', 'total_cooling_kwh', 'peak_heating_kw', 'peak_heating_day',
                      'peak_heating_hour', 'peak_cooling_kw', 'peak_cooling_day', 'peak_cooling_hour']
        df_600 = df.iloc[:, range(9)].copy()
        df_600.columns = df_columns
        df_600['case'] = '600'
        df_900 = df.iloc[:, [0, ] + list(range(9, 17))].copy()
        df_900.columns = df_columns
        df_900['case'] = '900'
        dc_600 = DataCleanser(df_600)
        dc_900 = DataCleanser(df_900)
        df_600 = dc_600.cleanse_monthly_conditioned_loads()
        df_900 = dc_900.cleanse_monthly_conditioned_loads()
        df = pd.concat([df_600, df_900], ignore_index=True)
        data_d = {}
        for idx, row in df.iterrows():
            case_number = str(row['case'])
            col_idx = [i for i, j in enumerate(df.columns) if j not in ['case', 'index', 'month']]
            row_obj = df.iloc[idx, col_idx].to_dict()
            if not data_d.get(case_number):
                data_d[case_number] = {}
            data_d[case_number].update({
                row['month']: row_obj})
        return data_d

    def _extract_specific_day_hourly_output(self):
        """
        Retrieve and format data from the Specific Day Hourly Output Table
        :return:  dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('specific_day_hourly_output')
        df_incident_solar_radiation_may_4 = df.iloc[:, range(4)].copy()
        df_incident_solar_radiation_may_4.columns = ['hour', 'horizontal', 'south', 'west']
        dc = DataCleanser(df_incident_solar_radiation_may_4)
        df_incident_solar_radiation_may_4 = dc.cleanse_specific_day_hourly_output_incident_solar_radiation()
        df_incident_solar_radiation_july_14 = df.iloc[:, [0, ] + list(range(4, 7))].copy()
        df_incident_solar_radiation_july_14.columns = ['hour', 'horizontal', 'south', 'west']
        dc = DataCleanser(df_incident_solar_radiation_july_14)
        df_incident_solar_radiation_july_14 = dc.cleanse_specific_day_hourly_output_incident_solar_radiation()
        df_sky_temperature = df.iloc[:, [0, ] + list(range(7, 10))].copy()
        df_sky_temperature.columns = ['hour', 'feb_1', 'may_4', 'july_14']
        df_transmitted_total_solar_radiation_600 = df.iloc[:, [0, 10, 13, 16]].copy()
        df_transmitted_total_solar_radiation_600.columns = ['hour', 'feb_1', 'may_4', 'july_14']
        dc = DataCleanser(df_transmitted_total_solar_radiation_600)
        df_transmitted_total_solar_radiation_600 = \
            dc.cleanse_specific_day_hourly_output_transmitted_total_solar_radiation()
        df_transmitted_total_solar_radiation_660 = df.iloc[:, [0, 11, 14, 17]].copy()
        df_transmitted_total_solar_radiation_660.columns = ['hour', 'feb_1', 'may_4', 'july_14']
        dc = DataCleanser(df_transmitted_total_solar_radiation_660)
        df_transmitted_total_solar_radiation_660 = \
            dc.cleanse_specific_day_hourly_output_transmitted_total_solar_radiation()
        df_transmitted_total_solar_radiation_670 = df.iloc[:, [0, 12, 15, 18]].copy()
        df_transmitted_total_solar_radiation_670.columns = ['hour', 'feb_1', 'may_4', 'july_14']
        dc = DataCleanser(df_transmitted_total_solar_radiation_670)
        df_transmitted_total_solar_radiation_670 = \
            dc.cleanse_specific_day_hourly_output_transmitted_total_solar_radiation()
        data_d = {}
        if df_incident_solar_radiation_may_4.shape[0] > 0:
            if not data_d.get('600'):
                data_d.update({'600': {}})
            if not data_d['600'].get('incident_solar_radiation'):
                data_d['600'].update({'incident_solar_radiation': {}})
            data_d['600']['incident_solar_radiation'].update({
                'may_4': {'horizontal': {'hour': {}}, 'south': {'hour': {}}, 'west': {'hour': {}}}})
            for idx, row in df_incident_solar_radiation_may_4.iterrows():
                for col_name in ['horizontal', 'south', 'west']:
                    data_d['600']['incident_solar_radiation']['may_4'][col_name]['hour'].update(
                        {int(row['hour']): {'Whm/m2': row[col_name]}})
        if df_incident_solar_radiation_july_14.shape[0] > 0:
            if not data_d.get('600'):
                data_d.update({'600': {}})
            if not data_d['600'].get('incident_solar_radiation'):
                data_d['600'].update({'incident_solar_radiation': {}})
            data_d['600']['incident_solar_radiation'].update({
                'july_14': {'horizontal': {'hour': {}}, 'south': {'hour': {}}, 'west': {'hour': {}}}})
            for idx, row in df_incident_solar_radiation_july_14.iterrows():
                for col_name in ['horizontal', 'south', 'west']:
                    data_d['600']['incident_solar_radiation']['july_14'][col_name]['hour'].update(
                        {int(row['hour']): {'Whm/m2': row[col_name]}})
        if df_sky_temperature.shape[0] > 0:
            if not data_d.get('600'):
                data_d.update({'600': {}})
            if not data_d['600'].get('sky_temperature'):
                data_d['600'].update({'sky_temperature': {}})
            data_d['600']['sky_temperature'].update(
                {'feb_1': {'hour': {}}, 'may_4': {'hour': {}}, 'july_14': {'hour': {}}})
            for idx, row in df_sky_temperature.iterrows():
                for col_name in ['feb_1', 'may_4', 'july_14']:
                    data_d['600']['sky_temperature'][col_name]['hour'].update({int(row['hour']): {'C': row[col_name]}})
        if df_transmitted_total_solar_radiation_600.shape[0] > 0:
            if not data_d.get('600'):
                data_d.update({'600': {}})
            if not data_d['600'].get('transmitted_total_solar_radiation'):
                data_d['600'].update({'transmitted_total_solar_radiation': {}})
            data_d['600']['transmitted_total_solar_radiation'].update(
                {'feb_1': {'hour': {}}, 'may_4': {'hour': {}}, 'july_14': {'hour': {}}})
            for idx, row in df_transmitted_total_solar_radiation_600.iterrows():
                for col_name in ['feb_1', 'may_4', 'july_14']:
                    data_d['600']['transmitted_total_solar_radiation'][col_name]['hour'].update(
                        {int(row['hour']): {'Whm/m2': row[col_name]}})
        if df_transmitted_total_solar_radiation_600.shape[0] > 0:
            if not data_d.get('660'):
                data_d.update({'660': {}})
            if not data_d['660'].get('transmitted_total_solar_radiation'):
                data_d['660'].update({'transmitted_total_solar_radiation': {}})
            data_d['660']['transmitted_total_solar_radiation'].update(
                {'feb_1': {'hour': {}}, 'may_4': {'hour': {}}, 'july_14': {'hour': {}}})
            for idx, row in df_transmitted_total_solar_radiation_660.iterrows():
                for col_name in ['feb_1', 'may_4', 'july_14']:
                    data_d['660']['transmitted_total_solar_radiation'][col_name]['hour'].update(
                        {int(row['hour']): {'Whm/m2': row[col_name]}})
        if df_transmitted_total_solar_radiation_600.shape[0] > 0:
            if not data_d.get('670'):
                data_d.update({'670': {}})
            if not data_d['670'].get('transmitted_total_solar_radiation'):
                data_d['670'].update({'transmitted_total_solar_radiation': {}})
            data_d['670']['transmitted_total_solar_radiation'].update(
                {'feb_1': {'hour': {}}, 'may_4': {'hour': {}}, 'july_14': {'hour': {}}})
            for idx, row in df_transmitted_total_solar_radiation_670.iterrows():
                for col_name in ['feb_1', 'may_4', 'july_14']:
                    data_d['670']['transmitted_total_solar_radiation'][col_name]['hour'].update(
                        {int(row['hour']): {'Whm/m2': row[col_name]}})
        return data_d

    def _extract_specific_day_hourly_output_free_float_zone_temperatures(self):
        """
        Retrieve and format data from the Specific Day Hourly Output Table
        :return:  dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('specific_day_hourly_output_free_float_zone_temperatures')
        df.columns = ['hour', '600FF', '900FF', '650FF', '950FF', '680FF', '980FF']
        dc = DataCleanser(df)
        df = dc.cleanse_specific_day_hourly_output_free_float_zone_temperatures()
        data_d = {
            '600FF': {'feb_1': {'hour': {}}},
            '900FF': {'feb_1': {'hour': {}}},
            '650FF': {'july_14': {'hour': {}}},
            '950FF': {'july_14': {'hour': {}}},
            '680FF': {'feb_1': {'hour': {}}},
            '980FF': {'feb_1': {'hour': {}}}
        }
        if df.shape[0] > 0:
            for idx, row in df.iterrows():
                data_d['600FF']['feb_1']['hour'].update({int(row['hour']): {'C': row['600FF']}})
                data_d['900FF']['feb_1']['hour'].update({int(row['hour']): {'C': row['900FF']}})
                data_d['650FF']['july_14']['hour'].update({int(row['hour']): {'C': row['650FF']}})
                data_d['950FF']['july_14']['hour'].update({int(row['hour']): {'C': row['950FF']}})
                data_d['680FF']['feb_1']['hour'].update({int(row['hour']): {'C': row['680FF']}})
                data_d['980FF']['feb_1']['hour'].update({int(row['hour']): {'C': row['980FF']}})
        return data_d

    def _extract_specific_day_hourly_output_free_float_zone_loads(self):
        """
        Retrieve and format data from the Specific Day Hourly Output Zone Loads Table
        :return:  dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('specific_day_hourly_output_free_float_zone_loads')
        df_feb_1 = df.iloc[:, range(13)].copy()
        df_feb_1.columns = ['hour', '600', '640', '660', '670', '680', '685', '695', '900', '940', '980', '985', '995']
        dc = DataCleanser(df_feb_1)
        df_feb_1 = dc.cleanse_specific_day_hourly_output_free_float_zone_loads_feb_1()
        df_july_14 = df.iloc[:, [0, ] + list(range(13, 23))].copy()
        df_july_14.columns = ['hour', '600', '660', '670', '680', '685', '695', '900', '980', '985', '995']
        dc = DataCleanser(df_july_14)
        df_july_14 = dc.cleanse_specific_day_hourly_output_free_float_zone_loads_july_14()
        df_zone_temps = df.iloc[:, [0, ] + [23, 24]].copy()
        df_zone_temps.columns = ['hour', '640', '940']
        dc = DataCleanser(df_zone_temps)
        df_zone_temps = dc.cleanse_specific_day_hourly_output_free_float_zone_loads_zone_temps()
        data_d = {}
        for col_name in df_feb_1.columns[1:]:
            if not data_d.get(col_name):
                data_d[col_name] = {}
            data_d[col_name]['feb_1'] = {'hour': {}}
            for idx, row in df_feb_1.iterrows():
                data_d[col_name]['feb_1']['hour'].update({int(row['hour']): {'kWh': row[col_name]}})
        for col_name in df_july_14.columns[1:]:
            if not data_d.get(col_name):
                data_d[col_name] = {}
            data_d[col_name]['july_14'] = {'hour': {}}
            for idx, row in df_july_14.iterrows():
                data_d[col_name]['july_14']['hour'].update({int(row['hour']): {'kWh': row[col_name]}})
        for col_name in df_zone_temps.columns[1:]:
            if not data_d.get(col_name):
                data_d[col_name] = {}
            if not data_d[col_name].get('feb_1'):
                data_d[col_name]['feb_1'] = {'hour': {}}
            for idx, row in df_zone_temps.iterrows():
                if not data_d[col_name]['feb_1']['hour'].get(int(row['hour'])):
                    data_d[col_name]['feb_1']['hour'] = {int(row['hour']): {}}
                data_d[col_name]['feb_1']['hour'][row['hour']]['C'] = row[col_name]
        return data_d

    # Section Ground Coupled GC data
    def _extract_identifying_information_gc(self):
        """
        Retrieve information data from section Thermal Fabric submittal and store it as class attributes.

        :return: Class attributes identifying software program.
        """
        df = self._get_data('identifying_information')
        self.software_name = df.iloc[2, 4]
        self.software_version = str(df.iloc[0, 0]).replace(str(df.iloc[2, 4]), '').strip()
        self.software_release_date = str(df.iloc[1, 4])
        data_d = {
            'program_name_and_version': df.iloc[0, 0],
            'program_version_release_date': str(df.iloc[1, 4]),
            'program_name_short': df.iloc[2, 4],
            'results_submittal_date': str(df.iloc[3, 4])}
        return data_d

    def _extract_steady_state_cases(self):
        """
        Retrieve and format data from the Steady State table (GC)

        :return: dictionary to be merged into main testing output dictionary
        """
        df = self._get_data('steady_state_cases')
        df.columns = ['cases', 'qfloor', 'qzone', 'Tzone', 'tsim']
        data_d = {}
        for idx, row in df.iterrows():
            data_d[row['cases']] = {
                'qfloor': row['qfloor'],
                'qzone': row['qzone'],
                'Tzone': row['Tzone'],
                'tsim': row['tsim']}
        return data_d

    def run(self):
        """
        Perform operations to convert Excel file into dictionary of dataframes.

        :return: json object of input data
        """
        self.test_data.update(self.processing_functions)
        return self
