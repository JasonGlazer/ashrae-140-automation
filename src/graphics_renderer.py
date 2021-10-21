import pathlib
import json
import re
import pandas as pd
import numpy as np
import math
from IPython.display import display_html
from itertools import chain, cycle
from textwrap import wrap
import matplotlib.pyplot as plt

from logger import Logger

root_directory = pathlib.Path(__file__).parent.parent.resolve()


class SectionType:
    """
    Identify Section Type based on input file name
    """
    def __get__(self, obj, owner):
        section_type = obj._section_type
        return section_type

    def __set__(self, obj, value):
        if re.match(r'^results5-2a.*', str(value), re.IGNORECASE):
            obj._section_type = '5-2A'
        elif re.match(r'^results5-2b.*', str(value), re.IGNORECASE):
            obj._section_type = '5-2B'
        else:
            obj.logger.error('Error: The file name ({}) did not match formatting guidelines or '
                             'the referenced section at the beginning of the name is not supported'
                             .format(str(value)))
        return


class GraphicsRenderer(Logger):
    """
    Create graphs and tables from a model_results_file.
    """

    section_type = SectionType()

    def __init__(
            self,
            model_results_file,
            processed_file_directory=None,
            base_model_list=None,
            logger_level='WARNING',
            logger_name="console_only_logger"):
        """
        :param model_results_file: file with the model file results to be visualized
        :param logger_level: logger level for reporting
        :param logger_name: logger object to use.
        """
        super().__init__(logger_level=logger_level, logger_name=logger_name)
        # make dataframe of case names and rank orders
        self.section_type = str(model_results_file)
        self.analytical_solutions = {
            'steady_state_cases': {
                'GC10': {
                    'qfloor': 2432.597
                }
            }
        }
        if self.section_type == '5-2A':
            self.case_detailed_df = pd.DataFrame.from_dict(
                {
                    '600': ['600 Base Case, South Windows', 1],
                    '610': ['610 S. Windows + Overhang', 2],
                    '620': ['620 East & West Windows', 3],
                    '630': ['630 E&W Windows + Overhang & Fins', 4],
                    '640': ['640 Case 600 with Htg Temp. Setback', 5],
                    '650': ['650 Case 600 with Night Ventilation', 6],
                    '660': ['660 Low-E Windows', 7],
                    '670': ['670 Single-Pane Windows', 8],
                    '680': ['680 Case 600 with Increased Insulation', 9],
                    '685': ['685 Case 600 with "20/20" Thermostat', 10],
                    '695': ['695 Case 685 with Increased Insulation', 11],
                    '900': ['900 South Windows', 12],
                    '910': ['910 S. Windows + Overhang', 13],
                    '920': ['920 East & West Windows', 14],
                    '930': ['930 E&W Windows + Overhang + Fins', 15],
                    '940': ['940 Case 900 with Htg Temp. Setback', 16],
                    '950': ['950 Case 900 with Night Ventilation', 17],
                    '960': ['960 Sunspace', 18],
                    '980': ['980 Case 900 with Increased Insulation', 19],
                    '985': ['985 Case 900 with "20/20" Thermostat', 20],
                    '995': ['995 Case 985 with Increased Insulation', 21],
                    '195': ['195 Solid Conduction', 22],
                    '200': ['200 Surface Convection (Int & Ext IR="off")', 23],
                    '210': ['210 Infrared Radiation (Int IR="off", Ext IR="on")', 24],
                    '215': ['215 Infrared Radiation (Int IR="on", Ext IR="off")', 25],
                    '220': ['220 In-Depth Base Case', 26],
                    '230': ['230 Infiltration', 27],
                    '240': ['240 Internal Gains', 28],
                    '250': ['250 Exterior Shortwave Absoptance', 29],
                    '270': ['270 South Solar Windows', 30],
                    '280': ['280 Cavity Albedo', 31],
                    '290': ['290 South Shading', 32],
                    '300': ['300 East/West Window', 33],
                    '310': ['310 East/West Shading', 34],
                    '320': ['320 Thermostat', 35],
                    '395': ['395 Low Mass Solid Conduction', 36],
                    '400': ['400 Low Mass High Cond. Wall Elements', 37],
                    '410': ['410 Low Mass Infiltration', 38],
                    '420': ['420 Low Mass Internal Gains', 39],
                    '430': ['430 Low Mass Ext. Shortwave Absoptance', 40],
                    '440': ['440 Low Mass Cavity Albedo', 41],
                    '450': ['450 Constant Interior and Exterior Surf Coeffs', 42],
                    '460': ['460 Constant Interior Surface Coefficients', 43],
                    '470': ['470 Constant Exterior Surface Coefficients', 44],
                    '800': ['800 High Mass Hig Cond. Wall Elements', 45],
                    '810': ['810 HIgh Mass Cavity Albedo', 46]
                },
                orient='index',
                columns=['case_name', 'case_order'])
        if not processed_file_directory:
            self.processed_file_directory = root_directory.joinpath('processed')
        else:
            self.processed_file_directory = processed_file_directory
        if not base_model_list:
            if self.section_type == '5-2A':
                self.baseline_model_list = [
                    'RESULTS5-2A-BSIMAC-9-9.0.74.json',
                    'RESULTS5-2A-CSE-0.861.1.json',
                    'RESULTS5-2A-DeST-2.0-20190401.json',
                    'RESULTS5-2A-EnergyPlus-9.0.1.json',
                    'RESULTS5-2A-ESP-r-13.3.json',
                    'RESULTS5-2A-TRNSYS-18.00.0001.json']
            elif self.section_type == '5-2B':
                self.baseline_model_list = [
                    'RESULTS5-2B-EnergyPlus-9.0.1.json', ]
        else:
            self.baseline_model_list = base_model_list
        self.model_results_file = model_results_file
        # try to extract the model name from the file name for the tested model and base models
        self.baseline_model_names = [i.replace('.json', '') for i in self.baseline_model_list]
        self.model_name = self.model_results_file.replace('.json', '')
        # create an object that keeps the information needed to make the row index for each table object.
        # 0 - json key name
        # 1 - list to make row index
        self.table_lookup = [
            ('conditioned_zone_loads_non_free_float', ['program_name', ])
        ]
        # dictionary to map file names to clean model names.  This dictionary is filled on data loading.
        self.cleansed_model_names = {}
        # instantiate objects to store data as a dictionary of json objects, and a dictionary of pandas dataframes
        self.json_data = {}
        self.df_data = {}
        # set hatches list for visualization objects
        self.hatches = ['/', '-', 'x', '\\', '//', 'o', '||', '+', 'O', '.', '*']
        self.colors = ['blue', 'green', 'red', 'cyan', 'yellow', 'black', 'orange']
        self.markers = ['o', '^', 'h', 'x', 'D', '*', '>']
        self._get_data()
        return

    def _get_data(self):
        """
        Get processed json data and store it in two dictionary objects.  One is the original json file data, the other
        is the converted pandas dataframe with a multiIndex for each json level.
        :return: Updated class objects that represent the data as a json object and pandas dataframe
        """
        table_objects = {}
        for f, model_name in zip(
                self.baseline_model_list + [self.model_results_file, ],
                self.baseline_model_names + [self.model_name, ]):
            with open(self.processed_file_directory.joinpath(f), 'r') as jf:
                data = json.load(jf)
                # load json objects as objects with the file name as the key
                self.json_data.update({model_name: data})
                # make mapping dictionary of file name to cleansed model name
                if data.get('identifying_information') and data[
                        'identifying_information'].get('software_name') and data[
                        'identifying_information'].get('software_version'):
                    self.cleansed_model_names[model_name] = '-'.join([
                        str(data['identifying_information']['software_name']),
                        str(data['identifying_information']['software_version'])])
                # load each table, if exists into a dataframe of the json key name
                for tbl, row_index in self.table_lookup:
                    tbl_data = data.get(tbl)
                    if tbl_data:
                        try:
                            table_objects[tbl]
                        except KeyError:
                            table_objects[tbl] = pd.DataFrame()
                        # Format the json data to a multiIndex table with a meaningful row index
                        # Make the separator something uncommon for easier splitting and re-leveling
                        tmp_df = pd.json_normalize(tbl_data, sep=">")
                        tmp_df.columns = pd.MultiIndex.from_tuples([i.split('>') for i in tmp_df.columns])
                        tmp_df['program_name'] = model_name
                        tmp_df = tmp_df.set_index(row_index)
                        table_objects.update({tbl: pd.concat([table_objects[tbl], tmp_df])})
        self.df_data = table_objects
        return

    @staticmethod
    def _set_theme(fig, ax):
        """
        Set general theme for all graphics by modifying fig and ax objects
        :return: Modified fig and ax objects from matplotlib.subplots
        """
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def _set_html_style():
        """
        Provide CSS styles for html output
        :return: CSS style text
        """
        css_text = """
            <style>
            
            .jp-RenderedImage {
                display: table-cell;
                text-align: center;
                vertical-align: middle;
            }
            
            .placeholder-span {
                visibility: hidden;
            }
            
            .pandas-tbl h2 {
                height: 25px;
                line-height: 18px;
                font-size: 18px;
                text-align: center;
            }
            
            .pandas-tbl caption {
                font-size: 20px;
                font-weight: bold;
                text-align: left;
            }
            
            .dataframe.pandas-sub-tbl th, .dataframe.pandas-sub-tbl-with-cases th {
                height: 50px;
                line-height: 14px;
                font-size: 14px;
                text-align: center;
            }
            
            .dataframe.pandas-sub-tbl td, .dataframe.pandas-sub-tbl-with-cases td {
                font-size: 12px;
            }
            
            .dataframe.pandas-sub-tbl-with-cases td:first-child {
                min-width: 300px;
                text-align: left;
            }
            
            </style>
        """
        return css_text

    @staticmethod
    def display_side_by_side(*args, titles=(), caption=None):
        html_str = ''
        html_str += '<table class="pandas-tbl"><tr>'
        if caption:
            html_str += f'<caption>{caption}</caption>'
        for idx, (df, title) in enumerate(zip(args, chain(titles, cycle(['', ])))):
            html_str += '<th style="text-align:center"><td style="vertical-align:top">'
            if not title:
                html_str += '<h2><span class="placeholder-span">ht</span></h2>'
            else:
                html_str += f'<h2>{title}</h2>'
            if idx == 0:
                class_val = 'pandas-sub-tbl-with-cases'
            else:
                class_val = 'pandas-sub-tbl'
            html_str += df.to_html(index=False, classes=class_val).replace('table', 'table style="display:inline"')
            html_str += '</td></th>'
        html_str += '</tr></table>'
        return display_html(html_str, raw=True)

    def render_section_5_2a_table_b_8_1(
            self,
            output_value='annual_heating_MWh',
            caption='Table B8-1. Annual Heating Loads (MWh)'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-1

        :return: pandas dataframe and output msg for general navigation.
        """
        table_html = None
        msg = None
        try:
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float']\
                .loc[
                    :,
                    self.df_data['conditioned_zone_loads_non_free_float']
                        .columns.get_level_values(1) == output_value]
            df.columns = df.columns.droplevel(level=1)
            df_formatted_table = df.unstack()\
                .reset_index()\
                .rename(columns={0: 'val'})\
                .pivot(index=['level_0'], columns=['program_name', ], values=['val', ])\
                .reset_index()
            df_formatted_table.columns = df_formatted_table.columns.droplevel(level=0)
            df_formatted_table_column_names = [i for i in df_formatted_table.columns]
            df_formatted_table_column_names[0] = 'cases'
            df_formatted_table.columns = df_formatted_table_column_names
            # Create calculated columns df and append them to the base table.
            df_formatted_table['col_min'] = df_formatted_table[self.baseline_model_names].min(axis=1)
            df_formatted_table['col_max'] = df_formatted_table[self.baseline_model_names].max(axis=1)
            df_formatted_table['col_mean'] = df_formatted_table[self.baseline_model_names].mean(axis=1)
            df_formatted_table['(max - min) / mean %'] = df_formatted_table.apply(
                lambda x: np.nan if x.col_mean == 0 else abs((x.col_max - x.col_min) / x.col_mean), axis=1)
            # separate dataframes for side by side visualization
            program_df = df_formatted_table[[self.model_name, ]].copy()
            # change program model column to cleansed name
            program_df.columns = [
                self.cleansed_model_names[i] if i in self.cleansed_model_names.keys()
                else i
                for i in program_df.columns]
            # make statistics dataframe for side by side
            statistics_df = df_formatted_table[['col_min', 'col_max', 'col_mean', '(max - min) / mean %']].rename(
                columns={
                    'col_min': 'min',
                    'col_max': 'max',
                    'col_mean': 'mean'})
            df_formatted_table = df_formatted_table.drop(
                [self.model_name, 'col_min', 'col_max', 'col_mean', '(max - min) / mean %'], axis=1)
            # rename cases by joining the detailed description table and re-order them
            df_formatted_table = df_formatted_table\
                .merge(
                    self.case_detailed_df,
                    how='left',
                    left_on=['cases', ],
                    right_index=True)\
                .sort_values(['case_order'])\
                .drop(['cases', 'case_order'], axis=1)\
                .rename(columns={'case_name': 'cases'})
            # reorder dataframe columns
            column_list = ['cases', ] + [i for i in df_formatted_table.columns if i != 'cases']
            df_formatted_table = df_formatted_table[column_list]
            # Rename model columns to cleansed names
            df_formatted_table.columns = [
                self.cleansed_model_names[i] if i in self.cleansed_model_names.keys()
                else i
                for i in df_formatted_table.columns]
            # Create side by side tables
            table_html = self.display_side_by_side(
                df_formatted_table,
                statistics_df,
                program_df,
                titles=['Simulation Model', 'Statistics for Example Results', ''],
                caption=caption)
        except KeyError:
            msg = 'Section 5-2A B8-1 Failed to be processed'
        return table_html, msg

    def render_section_5_2a_table_b_8_2(self):
        """
        Create dataframe from class dataframe object for table 5-2A B8-2

        :return: pandas dataframe and output msg for general navigation.
        """
        table_html, msg = self.render_section_5_2a_table_b_8_1(
            output_value='annual_cooling_MWh',
            caption='Table B8.2 Annual Sensible Cooling Loads (MWh)'
        )
        return table_html, msg

    def render_section_5_2a_figure_b_8_1(self):
        """
        Render Section 5 2A Figure B8-1 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig, ax = self._set_theme(fig, ax)
        width = 0.1
        data = []
        surfaces = ['HORZ.', 'NORTH', 'EAST', 'SOUTH', 'WEST']
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for surface in surfaces:
                if json_obj.get('annual_solar_radiation_direct_and_diffuse') and json_obj[
                        'annual_solar_radiation_direct_and_diffuse']['600']['Surface'].get(surface):
                    tmp_data.append(
                        json_obj['annual_solar_radiation_direct_and_diffuse']['600']['Surface'][surface].get('kWh/m2'))
                else:
                    tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        ax.set_xticks(np.arange(max([len(i) for i in data])))
        ax.set_title('Figure B8-1.  Annual Incident Solar Radiation', fontsize=30)
        ax.set_xticklabels(['600 ' + i for i in surfaces])
        for idx, (p, d, h) in enumerate(zip(programs, data, self.hatches)):
            x = np.arange(len(d))
            rects = ax.bar(x + (width * idx) - (width / 2 * (len(data) - 1)), d, width, label=p, hatch=h, fill=None)
            ax.bar_label(rects, padding=5, rotation="vertical")
        ax.grid(which='major', axis='y')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(programs), fontsize=14)
        ax.set_ylabel('Diffuse + Direct ($kWh/m^2$)', fontsize=14)
        ax.set_ylim(0, 2000)
        return fig, ax

    def render_section_5_2a_figure_b_8_9(self):
        """
        Render Section 5 2A Figure B8-9 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig, ax = self._set_theme(fig, ax)
        cases = ['395', '430', '600', '610', '620', '630', '640', '650']
        width = 0.1
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                if json_obj.get('conditioned_zone_loads_non_free_float') and json_obj[
                        'conditioned_zone_loads_non_free_float'].get(case):
                    tmp_data.append(json_obj['conditioned_zone_loads_non_free_float'][case].get('peak_heating_kW'))
                else:
                    tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        ax.set_xticks(np.arange(max([len(i) for i in data])))
        ax.set_title('Figure B8-9.  Basic: Low Mass Peak Heating', fontsize=30)
        ax.set_xticklabels(
            [
                '\n'.join(wrap(self.case_detailed_df.loc[i, 'case_name'], 15))
                for i in cases])
        for idx, (p, d, h) in enumerate(zip(programs, data, self.hatches)):
            x = np.arange(len(d))
            rects = ax.bar(x + (width * idx) - (width / 2 * (len(data) - 1)), d, width, label=p, hatch=h, fill=None)
            ax.bar_label(rects, padding=5, rotation="vertical")
        ax.grid(which='major', axis='y')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(programs), fontsize=14)
        ax.set_ylabel('Peak Heating Load (kWh/h)', fontsize=14)
        ax.set_ylim(0, 5)
        return fig, ax

    def render_section_5_2a_figure_b8_17(self):
        """
        Render Section 5 2A Figure B8-17 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        fig, ax = plt.subplots(1, 3, figsize=(18, 8), sharex='none', sharey='all')
        fig, ax = self._set_theme(fig, ax)
        cases = ['600', '610', '620', '900', '920', '930']
        width = 0.1
        data_lists = [[] for n in range(3)]
        programs = []
        xticklabels = [
            [
                '610-600 Low Mass, S. Shade Heating',
                '610-600 Low Mass, S. Shade Cooling',
                '910-900 High Mass, S. Shade Heating',
                '910-900 High Mass, S. Shade Cooling'
            ],
            [
                '620-600 Low Mass, E&W Or., Heating',
                '620-600 Low Mass, E&W Or., Cooling',
                '920-900 High Mass, E&W Or., Heating',
                '920-900 High Mass, E&W Or., Cooling',
            ],
            [
                '630-620 Low Mass. E&W Shd., Heating',
                '630-620 Low Mass. E&W Shd., Cooling',
                '930-920 Low Mass. E&W Shd., Heating',
                '930-920 Low Mass. E&W Shd., Cooling',
            ]
        ]
        sub_title = [
            'South Shading',
            'East/West',
            'E/W Shading'
        ]
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            if json_obj.get('conditioned_zone_loads_non_free_float') and all([
                    case in json_obj['conditioned_zone_loads_non_free_float'].keys() for case in cases]):
                # Left chart data
                tmp_data = []
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['610']
                        .get('annual_heating_MWh') - json_obj['conditioned_zone_loads_non_free_float']['600']
                        .get('annual_heating_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['610']
                        .get('annual_cooling_MWh') - json_obj['conditioned_zone_loads_non_free_float']['600']
                        .get('annual_cooling_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['910']
                        .get('annual_heating_MWh') - json_obj['conditioned_zone_loads_non_free_float']['900']
                        .get('annual_heating_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['910']
                        .get('annual_cooling_MWh') - json_obj['conditioned_zone_loads_non_free_float']['900']
                        .get('annual_cooling_MWh'))
                except TypeError:
                    tmp_data.append(None)
                data_lists[0].insert(idx, tmp_data)
                # Mid chart data
                tmp_data = []
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['620']
                        .get('annual_heating_MWh') - json_obj['conditioned_zone_loads_non_free_float']['600']
                        .get('annual_heating_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['620']
                        .get('annual_cooling_MWh') - json_obj['conditioned_zone_loads_non_free_float']['600']
                        .get('annual_cooling_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['920']
                        .get('annual_heating_MWh') - json_obj['conditioned_zone_loads_non_free_float']['900']
                        .get('annual_heating_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['920']
                        .get('annual_cooling_MWh') - json_obj['conditioned_zone_loads_non_free_float']['900']
                        .get('annual_cooling_MWh'))
                except TypeError:
                    tmp_data.append(None)
                data_lists[1].insert(idx, tmp_data)
                # Right chart data
                tmp_data = []
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['630']
                        .get('annual_heating_MWh') - json_obj['conditioned_zone_loads_non_free_float']['620']
                        .get('annual_heating_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['630']
                        .get('annual_cooling_MWh') - json_obj['conditioned_zone_loads_non_free_float']['620']
                        .get('annual_cooling_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['930']
                        .get('annual_heating_MWh') - json_obj['conditioned_zone_loads_non_free_float']['920']
                        .get('annual_heating_MWh'))
                except TypeError:
                    tmp_data.append(None)
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float']['930']
                        .get('annual_cooling_MWh') - json_obj['conditioned_zone_loads_non_free_float']['920']
                        .get('annual_cooling_MWh'))
                except TypeError:
                    tmp_data.append(None)
                data_lists[2].insert(idx, tmp_data)
                programs.insert(idx, json_obj['identifying_information']['software_name'])
        for didx, data in enumerate(data_lists):
            for idx, (p, d, h) in enumerate(zip(programs, data, self.hatches)):
                x = np.arange(len(d))
                rects = ax[didx].bar(x + (width * idx) - (width / 2 * (len(data) - 1)), d, width, label=p, hatch=h, fill=None)
                ax[didx].bar_label(rects, padding=5, rotation="vertical")
                ax[didx].grid(which='major', axis='y')
                ax[didx].set_xticks(np.arange(max([len(i) for i in data])))
                ax[didx].set_xticklabels(
                    ['\n'.join(wrap(i, 15)) for i in xticklabels[didx]]
                )
                ax[didx].set_title(sub_title[didx], fontsize=18)
        # set legend for all plots
        ax.flatten()[-2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(programs), fontsize=16)
        # Make title, adjust plots, and set y values
        fig.suptitle('Figure B8-17. Basic: Window Shading and Orientation (Delta) '
                     'Annual Heating and Sensible Cooling', fontsize=30, y=0.9)
        fig.subplots_adjust(top=0.8, wspace=0.001)
        ax[0].set_ylim(-2.5, 2.5)
        ax[0].set_yticks(np.arange(-2.5, 2.5, 0.5))
        ax[0].set_ylabel('Load Difference (MWh)', fontsize=14)
        return fig, ax

    def render_section_5_2a_figure_b8_h1(self):
        """
        Render Section 5 2A Figure B8-H1 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig, ax = self._set_theme(fig, ax)
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            if json_obj.get('hourly_annual_zone_temperature_bin_data') and json_obj[
                    'hourly_annual_zone_temperature_bin_data'].get('900FF'):
                try:
                    data_obj = json_obj['hourly_annual_zone_temperature_bin_data']['900FF']['temperature_bin_c']
                    bin_list = []
                    count_list = []
                    # Make ordered lists based on the integer value of the key
                    for k, v in data_obj.items():
                        if not bin_list:
                            bin_list.append(int(k))
                            count_list.append(int(v['number_of_hours']))
                            continue
                        last_item = -float('inf')
                        for bidx, bin_item in enumerate(bin_list):
                            if last_item < int(k) <= bin_item:
                                bin_list.insert(bidx, int(k))
                                count_list.insert(bidx, int(v['number_of_hours']))
                                break
                            if bidx == len(bin_list) - 1:
                                bin_list.append(int(k))
                                count_list.append(int(v['number_of_hours']))
                                break
                            last_item = bin_item
                    data_x.append(bin_list)
                    data_y.append(count_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
        # Add line plots for each program
        for dx, dy, p, c, m in zip(data_x, data_y, programs, self.colors, self.markers):
            ax.plot(dx, dy, color=c, marker=m, label=p)
        # Format plot area
        ax.grid(which='major', axis='y')
        ax.set_yticks(np.arange(0, 500, 100))
        # get minimum/maximum of all x rounded to nearest ten, then increment by 5
        ax.set_xticks(np.arange(
            math.floor(min([min(i) for i in data_x]) / 10) * 10,
            math.ceil(max([max(i) for i in data_x]) / 10) * 10,
            5))
        ax.set_title('Figure B8-H1. Case 900FF Annual Hourly Zone Air Temperature Frequency', fontsize=30)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(programs), fontsize=14)
        ax.set_ylabel('Number of Occurrences', fontsize=14)
        ax.set_xlim(-5, 55)
        ax.set_ylim(0, 500)
        ax.annotate(r'Hourly Occurrences for Each 1 $^\circ$C Bin', (0, 450), fontsize=12)
        return fig, ax
