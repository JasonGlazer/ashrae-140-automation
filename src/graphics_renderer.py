import pathlib
import json
import os
import re
import pandas as pd
import numpy as np
import math
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
        if re.match(r'.*results5-2a\..*$', str(value), re.IGNORECASE):
            obj._section_type = '5-2A'
        elif re.match(r'.*results5-2b\..*$', str(value), re.IGNORECASE):
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
                    'bsimac-9.9.0.7.4-results5-2a.json',
                    'cse-0.861.1-results5-2a.json',
                    'dest-2.0.20190401-results5-2a.json',
                    'energyplus-9.0.1-results5-2a.json',
                    'esp-r-13.3-results5-2a.json',
                    'trnsys-18.00.0001-results5-2a.json']
            elif self.section_type == '5-2B':
                self.baseline_model_list = [
                    'basecalc-v1.0e-results5-2b.json',
                    'energyplus-9.0.1-results5-2b.json',
                    'esp-r-13.3-results5-2b.json',
                    'fluent-6.1-results5-2b.json',
                    'ght-2.02-results5-2b.json',
                    'matlab-7.0.4.365-r14-sp2-results5-2b.json',
                    'sunrel-gc-1.14.02-results5-2b.json',
                    'trnsys-18.00.0001-results5-2b.json',
                    'va114-2.20-results5-2b.json'
                ]
        else:
            self.baseline_model_list = base_model_list
        self.model_results_file = model_results_file
        # try to extract the model name from the file name for the tested model and base models
        self.baseline_model_names = [i.replace('.json', '') for i in self.baseline_model_list]
        self.model_name = self.model_results_file.replace('.json', '')
        # create an object that keeps the information needed to make the row index for each table object.
        # 0 - json key name
        # 1 - list to make row index
        if self.section_type == '5-2A':
            self.table_lookup = [
                ('conditioned_zone_loads_non_free_float', ['program_name', ])
            ]
        elif self.section_type == '5-2B':
            self.table_lookup = [
                ('steady_state_cases', ['program_name', ])
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

    def _create_bar_plot(self, data, programs, title, xticklabels, ylabel, width=0.1, y_plot_pad=0.1, image_name=None):
        """
        Create Bar plot from data input.

        :param data: bar values in nested lists. The number of sublists is the number of programs evaluated.
          The length of each sublist must be the same and is the number of cases evaluated.
        :param programs: list of tested programs
        :param title: plot title
        :param xticklabels: labels to use for each case evaluated
        :param ylabel: y axis plot label
        :param width: width of bars
        :param y_plot_pad: padding between the highest bar and the top of the plot
        :param image_name: unique name to store the plot as a png
        :return: matplotlib fig and ax objects.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig, ax = self._set_theme(fig, ax)
        for idx, (p, d, h) in enumerate(zip(programs, data, self.hatches)):
            x = np.arange(len(d))
            rects = ax.bar(x + (width * idx) - (width / 2 * (len(data) - 1)), d, width, label=p, hatch=h, fill=None)
            ax.bar_label(rects, padding=5, rotation="vertical")
        ax.set_xticks(np.arange(max([len(i) for i in data])))
        ax.grid(which='major', axis='y')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(programs), fontsize=14)
        ax.set_title(title, fontsize=30)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(ylabel, fontsize=14)
        ymin = min([i for i in map(min, data) if not np.isnan(i)])
        ymax = max([i for i in map(max, data) if not np.isnan(i)])
        ax.set_ylim(
            ymin - abs(ymin * y_plot_pad),
            ymax + abs(ymax * y_plot_pad))
        fig.patch.set_facecolor('white')
        if image_name:
            self._make_image_from_plt(image_name)
        return fig, ax

    def _create_line_plot(self, data_x, data_y, programs, title, ylabel, image_name=None):
        """
        Create line plot from data input.

        :param data_x: x values in nested lists. The number of sublists is the number of programs evaluated. The length
            of each sublist is the number of x coordinate data points and must be the same length as the correspoding
            sublist in data_y.
        :param data_y: x values in nested lists.  The number of sublists is the number of programs evaluated. The length
            of each sublist is the number of x coordinate data points and must be the same length as the correspoding
            sublist in data_x.
        :param programs: list of tested programs
        :param title: plot title
        :param ylabel: y axis plot label
        :param image_name: unique name to store the plot as a png
        :return: matplotlib fig and ax objects.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig, ax = self._set_theme(fig, ax)
        # Add line plots for each program
        for dx, dy, p, c, m in zip(data_x, data_y, programs, self.colors, self.markers):
            ax.plot(dx, dy, color=c, marker=m, label=p)
        # Format plot area
        ax.grid(which='major', axis='y')
        # get minimum/maximum of all x rounded to nearest ten, then increment by 5
        ax.set_xticks(np.arange(
            math.floor(min([min(i) for i in data_x]) / 10) * 10,
            math.ceil(max([max(i) for i in data_x]) / 10) * 10,
            5))
        ax.set_title(title, fontsize=30)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(programs), fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if image_name:
            self._make_image_from_plt(image_name)
        return fig, ax

    def _make_image_from_plt(self, figure_name, destionation_directory=('rendered', 'images')):
        """
        make a png file from a matplotlib.pyplot object and save it to a directory

        :param figure_name: name of figure to append to file name
        :param destionation_directory: list of directories leading to the output directory
        :return: saved image in referenced directory
        """
        img_name = root_directory.joinpath(
            *destionation_directory,
            '.'.join(
                [
                    '-'.join(
                        [
                            os.path.splitext(self.model_results_file)[0],
                            figure_name,
                        ]),
                    'png'
                ]))
        plt.savefig(img_name, bbox_inches='tight', facecolor='white')
        return

    def render_section_5_2a_table_b_8_1(
            self,
            output_value='annual_heating_MWh',
            figure_name='section_5_2_a_table_b_8_1',
            caption='Table B8-1. Annual Heating Loads (MWh)'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-1

        :return: pandas dataframe and output msg for general navigation.
        """
        # get and format dataframe into required shape
        df = self.df_data['conditioned_zone_loads_non_free_float']\
            .loc[
                :,
                self.df_data['conditioned_zone_loads_non_free_float']
                    .columns.get_level_values(1) == output_value]
        df.columns = df.columns.droplevel(level=1)
        # round values
        df = df.round(3)
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
        df_formatted_table['col_min'] = df_formatted_table[self.baseline_model_names].min(axis=1).round(3)
        df_formatted_table['col_max'] = df_formatted_table[self.baseline_model_names].max(axis=1).round(3)
        df_formatted_table['col_mean'] = df_formatted_table[self.baseline_model_names].mean(axis=1).round(3)
        df_formatted_table['(max - min) / mean %'] = df_formatted_table.apply(
            lambda x: '' if x.col_mean == 0 else '{:.2%}'.format(abs((x.col_max - x.col_min) / x.col_mean)), axis=1)
        # rename cases by joining the detailed description table and re-order them
        df_formatted_table = df_formatted_table\
            .merge(
                self.case_detailed_df,
                how='left',
                left_on=['cases', ],
                right_index=True)\
            .sort_values(['case_order'])\
            .drop(['cases', 'case_order'], axis=1)\
            .rename(columns={
                'case_name': 'Case',
                'col_min': 'min',
                'col_max': 'max',
                'col_mean': 'mean'})
        # reorder dataframe columns
        column_list = ['Case', ] + \
                      [i for i in df_formatted_table.columns if i != 'Case' and i != self.model_name] + \
                      [self.model_name, ]
        df_formatted_table = df_formatted_table[column_list]
        # Rename model columns to cleansed names
        df_formatted_table.columns = [
            self.cleansed_model_names[i] if i in self.cleansed_model_names.keys()
            else i
            for i in df_formatted_table.columns]
        df_formatted_table.columns = ['\n'.join(wrap(i, 8)) for i in df_formatted_table.columns]
        # set fig size
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(22, 20))
        tab = ax.table(
            cellText=df_formatted_table.values,
            colLabels=df_formatted_table.columns,
            zorder=1,
            bbox=[0, 0, 1, 1],
            edges='LR')
        ax.axis('tight')
        ax.axis('off')
        # set font
        tab.auto_set_font_size(False)
        tab.set_fontsize(12)
        # set cell properties individually
        cell_dict = tab.get_celld()
        for i in range(len(df_formatted_table.columns)):
            cell_dict[(0, i)].set_height(1)
            cell_dict[(0, i)].set_fontsize(16)
            cell_dict[(0, 0)].set_width(2)
            cell_dict[(0, i)].set_width(0.5)
            cell_dict[(0, i)].visible_edges = 'closed'
            for j in range(1, df_formatted_table.shape[0] + 1):
                cell_dict[(j, 0)].set_width(2)
                cell_dict[(j, 0)].set_text_props(ha="left")
                cell_dict[(j, 0)].PAD = 0.02
                cell_dict[(j, i)].set_width(0.5)
                cell_dict[(j, i)].set_height(0.25)
        ax.axis([0, 1, 1, 0])
        # set outer borders
        ax.axhline(y=0, color='black', linewidth=4, zorder=3)
        ax.axhline(y=1, color='black', linewidth=4, zorder=3)
        ax.axvline(x=0, color='black', linewidth=4, zorder=3)
        ax.axvline(x=1, color='black', linewidth=4, zorder=3)
        # set inner borders
        ax.axvline(x=5 / 7.5, color='black', linewidth=2, zorder=3)
        ax.axvline(x=7 / 7.5, color='black', linewidth=3, zorder=3)
        ax.axvline(x=5 / 7.5, color='black', linewidth=2, zorder=3)
        ax.axhline(y=15 / 50, color='black', linewidth=2, zorder=3)
        ax.axhline(y=25 / 50, color='black', linewidth=2, zorder=3)
        ax.axhline(y=39 / 50, color='black', linewidth=2, zorder=3)
        ax.axhline(y=48 / 50, color='black', linewidth=2, zorder=3)
        # Set annotations
        header = [tab.add_cell(-1, h, width=0.5, height=0.35) for h in range(7, 11)]
        header[0].get_text().set_text('Statistics for Example Results')
        header[0].PAD = 0.5
        header[0].set_fontsize(16)
        header[0].set_text_props(ha="left")
        header[0].visible_edges = "open"
        header[1].visible_edges = "open"
        header[2].visible_edges = "open"
        header[3].visible_edges = "open"
        # save the result
        plt.suptitle(caption, fontsize=30)
        self._make_image_from_plt(figure_name)
        plt.subplots_adjust(top=0.92)
        return fig, ax

    def render_section_5_2a_table_b_8_2(self):
        """
        Create dataframe from class dataframe object for table 5-2A B8-2

        :return: pandas dataframe and output msg for general navigation.
        """
        table_html, msg = self.render_section_5_2a_table_b_8_1(
            output_value='annual_cooling_MWh',
            figure_name='section_5_2_a_table_b_8_2',
            caption='Table B8.2 Annual Sensible Cooling Loads (MWh)'
        )
        return table_html, msg

    def render_section_5_2a_figure_b_8_1(self):
        """
        Render Section 5 2A Figure B8-1 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        surfaces = ['HORZ.', 'NORTH', 'EAST', 'SOUTH', 'WEST']
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for surface in surfaces:
                try:
                    tmp_data.append(
                        json_obj['solar_radiation_annual_incident']['600']['Surface'][surface].get('kWh/m2'))
                except (KeyError, ValueError):
                    tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-1.  Annual Incident Solar Radiation',
            xticklabels=['600 ' + i for i in surfaces],
            ylabel='Diffuse + Direct ($kWh/m^2$)',
            image_name='section_5_2_a_figure_b_8_1')
        return fig, ax

    def render_section_5_2a_figure_b_8_2(self):
        """
        Render Section 5 2A Figure B8-2 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        cases = ['600', '620', '660', '670']
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    (_, tmp_d), = \
                        json_obj['solar_radiation_unshaded_annual_transmitted'][case].get('Surface').items()
                    tmp_data.append(tmp_d['kWh/m2'])
                except (KeyError, ValueError):
                    tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-2.  Annual Transmitted Solar Radiation - Unshaded',
            xticklabels=[
                '600 SOUTH', '620 WEST', '660 SOUTH, Low-E', '670 SOUTH, Single Pane'],
            ylabel='Diffuse + Direct ($kWh/m^2$)',
            image_name='section_5_2_a_figure_b_8_2')
        return fig, ax

    def render_section_5_2a_figure_b_8_3(self):
        """
        Render Section 5 2A Figure B8-3 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        cases = ['610', '630']
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    (_, tmp_d), = \
                        json_obj['solar_radiation_shaded_annual_transmitted'][case].get('Surface').items()
                    tmp_data.append(tmp_d['kWh/m2'])
                except (KeyError, ValueError):
                    tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-3.  Annual Transmitted Solar Radiation - Shaded',
            xticklabels=[
                '610 SOUTH', '630 WEST'],
            ylabel='Diffuse + Direct ($kWh/m^2$)',
            image_name='section_5_2_a_figure_b_8_3')
        return fig, ax

    def render_section_5_2a_figure_b_8_4(self):
        """
        Render Section 5 2A Figure B8-4 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        cases = ['600', '620', '660', '670']
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    (surface, unshaded_d), = \
                        json_obj['solar_radiation_unshaded_annual_transmitted'][case].get('Surface').items()
                    for sub_surfaces in json_obj['solar_radiation_annual_incident'].values():
                        for sub_surface, vals in sub_surfaces['Surface'].items():
                            if sub_surface.lower() == surface.lower():
                                incident_surface_value = vals['kWh/m2']
                                tmp_data.append(unshaded_d['kWh/m2'] / incident_surface_value)
                                break
                except (KeyError, ValueError):
                    tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-4.  Annual Transmissivity Coefficient of Windows \n'
                  '(Unshaded Transmitted)/(Incident Solar Radiation)',
            xticklabels=[
                '600 SOUTH', '620 WEST', '660 SOUTH, Low-E', '670 SOUTH, Single Pane'],
            ylabel='Transmissivity Coefficient',
            image_name='section_5_2_a_figure_b_8_4')
        return fig, ax

    def render_section_5_2a_figure_b_8_5(self):
        """
        Render Section 5 2A Figure B8-5 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    1 - (
                        json_obj['solar_radiation_shaded_annual_transmitted']['610']['Surface']['South']
                        ['kWh/m2'] / json_obj['solar_radiation_unshaded_annual_transmitted']['600']['Surface']['South']
                        ['kWh/m2']
                    ))
                tmp_data.append(
                    1 - (
                        json_obj['solar_radiation_shaded_annual_transmitted']['630']['Surface']['West']
                        ['kWh/m2'] / json_obj['solar_radiation_unshaded_annual_transmitted']['620']['Surface']
                        ['West']['kWh/m2']
                    ))
            except (KeyError, ValueError):
                tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-5. \nAnnual Overhang and Fin Shading Coefficients \n'
                  '(1-(Shaded)/(Unshaded)) Transmitted Solar Radiation',
            xticklabels=[
                '(1 - Case610 / Case600) SOUTH', '(1 - Case630 / Case620) WEST'],
            ylabel='Shading Coefficient',
            image_name='section_5_2_a_figure_b_8_5')
        return fig, ax

    def render_section_5_2a_figure_b_8_6(self):
        """
        Render Section 6 2A Figure B8-5 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(json_obj['sky_temperature_output']['600']['Average']['C'])
                tmp_data.append(json_obj['sky_temperature_output']['600']['Minimum']['C'])
                tmp_data.append(json_obj['sky_temperature_output']['600']['Maximum']['C'])
            except (KeyError, ValueError):
                tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-6. \nAverage, Minimum and Maximum Sky Temperature \nCase 600',
            xticklabels=[
                'Average', 'Minimum', 'Maximum'],
            ylabel='Shading Coefficient',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b_8_6')
        return fig, ax

    def render_section_5_2a_figure_b_8_9(self):
        """
        Render Section 5 2A Figure B8-9 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['395', '430', '600', '610', '620', '630', '640', '650']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(json_obj['conditioned_zone_loads_non_free_float'][case].get('peak_heating_kW'))
                except (KeyError, ValueError):
                    tmp_data.append(None)
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-9.  Basic: Low Mass Peak Heating',
            xticklabels=[
                '\n'.join(wrap(self.case_detailed_df.loc[i, 'case_name'], 15))
                for i in cases],
            ylabel='Peak Heating Load (kWh/h)',
            image_name='section_5_2_a_figure_b_8_9')
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
        data_lists = [[] for _ in range(3)]
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
                rects = ax[didx].bar(
                    x + (width * idx) - (width / 2 * (len(data) - 1)),
                    d,
                    width,
                    label=p,
                    hatch=h,
                    fill=None)
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
        self._make_image_from_plt('section_5_2_a_figure_b_8_17')
        return fig, ax

    def render_section_5_2a_figure_b8_h1(self):
        """
        Render Section 5 2A Figure B8-H1 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
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
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H1. Case 900FF Annual Hourly Zone Air Temperature Frequency',
            ylabel='Number of Occurrences',
            image_name='section_5_2_a_figure_b_8_h1')
        ax.set_yticks(np.arange(0, 500, 100))
        ax.set_xlim(-5, 55)
        ax.set_ylim(0, 500)
        ax.annotate(r'Hourly Occurrences for Each 1 $^\circ$C Bin', (0, 450), fontsize=12)
        return fig, ax

    def render_section_5_2b_table_b_8_2_1(
            self,
            output_value='annual_heating_MWh',
            caption='Table B8.2-1 "a"-Series Case Summary, Numerical Model Verification'):
        """
        Create dataframe from class dataframe object for table 5-2B B8.2-1

        :return: pandas dataframe and output msg for general navigation.
        """
        df = self.df_data['steady_state_cases']
        return df
