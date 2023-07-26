import pathlib
import json
import re
import pandas as pd
import numpy as np
import math
from textwrap import wrap
import matplotlib.pyplot as plt
import textwrap as twp

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
                    '810': ['810 HIgh Mass Cavity Albedo', 46],
                    '600FF': ['600FF - Low Mass Building with South Windows', 47],
                    '650FF': ['650FF - Case 600FF with Night Ventilation', 48],
                    '680FF': ['680FF - Case 600FF with More Insulation', 49],
                    '900FF': ['900FF - High Mass Building with South Windows', 50],
                    '950FF': ['950FF - Case 900FF with Night Ventilation', 51],
                    '980FF': ['980FF - Case 900FF with More Insulation', 52]
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
                    root_directory.joinpath('processed', 'bsimac', '9.9.0.7.4', 'results5-2a.json'),
                    root_directory.joinpath('processed', 'cse', '0.861.1', 'results5-2a.json'),
                    root_directory.joinpath('processed', 'dest', '2.0.20190401', 'results5-2a.json'),
                    root_directory.joinpath('processed', 'energyplus', '9.0.1', 'results5-2a.json'),
                    root_directory.joinpath('processed', 'esp-r', '13.3', 'results5-2a.json'),
                    root_directory.joinpath('processed', 'trnsys', '18.00.0001', 'results5-2a.json')]
            elif self.section_type == '5-2B':
                self.baseline_model_list = [
                    root_directory.joinpath('processed', 'basecalc', 'v1.0e', 'results5-2b.json'),
                    root_directory.joinpath('processed', 'energyplus', '9.0.1', 'results5-2b.json'),
                    root_directory.joinpath('processed', 'esp-r', '13.3', 'results5-2b.json'),
                    root_directory.joinpath('processed', 'fluent', '6.1', 'results5-2b.json'),
                    root_directory.joinpath('processed', 'ght', '2.02', 'results5-2b.json'),
                    root_directory.joinpath('processed', 'matlab', '7.0.4.365-r14-sp2', 'results5-2b.json'),
                    root_directory.joinpath('processed', 'sunrel-gc', '1.14.02', 'results5-2b.json'),
                    root_directory.joinpath('processed', 'trnsys', '18.00.0001', 'results5-2b.json'),
                    root_directory.joinpath('processed', 'va114', '2.20', 'results5-2b.json')
                ]
        else:
            self.baseline_model_list = base_model_list
        if isinstance(model_results_file, str):
            self.model_results_file = root_directory / 'processed' / model_results_file
        else:
            self.model_results_file = model_results_file
        # try to extract the model name from the file name for the tested model and base models
        self.baseline_model_names = ['-'.join([i.parts[-3], i.parts[-2]]) for i in self.baseline_model_list]
        self.model_name = '-'.join([self.model_results_file.parts[-3], self.model_results_file.parts[-2]])
        # create an object that keeps the information needed to make the row index for each table object.
        # 0 - json key name
        # 1 - list to make row index
        if self.section_type == '5-2A':
            self.table_lookup = [
                ('conditioned_zone_loads_non_free_float', ['program_name', ]),
                ('free_float_case_zone_temperatures', ['program_name', ])
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

        self.case_map = {
            '600': '600 Base Case, South Windows',
            '610': '610 S. Windows + Overhang',
            '620': '620 East & West Windows',
            '630': '630 E&W Windows + Overhang & Fins',
            '640': '640 Case 600 with Htg Temp. Setback',
            '650': '650 Case 600 with Night Ventilation',
            '660': '660 Low-E Windows',
            '670': '670 Single-Pane Windows',
            '680': '680 Case 600 with Increased Insulation',
            '685': '685 Case 600 with "20/20" Thermostat',
            '695': '695 Case 685 with Increased Insulation',
            '900': '900 South Windows',
            '910': '910 S. Windows + Overhang',
            '920': '920 East & West Windows',
            '930': '930 E&W Windows + Overhang + Fins',
            '940': '940 Case 900 with Htg Temp. Setback',
            '950': '950 Case 900 with Night Ventilation',
            '960': '960 Sunspace',
            '980': '980 Case 900 with Increased Insulation',
            '985': '985 Case 900 with "20/20" Thermostat',
            '995': '995 Case 985 with Increased Insulation',
            '195': '195 Solid Conduction',
            '200': '200 Surface Convection (Int & Ext IR="off")',
            '210': '210 Infrared Radiation (Int IR="off", Ext IR="on")',
            '215': '215 Infrared Radiation (Int IR="on", Ext IR="off")',
            '220': '220 In-Depth Base Case',
            '230': '230 Infiltration',
            '240': '240 Internal Gains',
            '250': '250 Exterior Shortwave Absoptance',
            '270': '270 South Solar Windows',
            '280': '280 Cavity Albedo',
            '290': '290 South Shading',
            '300': '300 East/West Window',
            '310': '310 East/West Shading',
            '320': '320 Thermostat',
            '395': '395 Low Mass Solid Conduction',
            '400': '400 Low Mass High Cond. Wall Elements',
            '410': '410 Low Mass Infiltration',
            '420': '420 Low Mass Internal Gains',
            '430': '430 Low Mass Ext. Shortwave Absoptance',
            '440': '440 Low Mass Cavity Albedo',
            '450': '450 Constant Interior and Exterior Surf Coeffs',
            '460': '460 Constant Interior Surface Coefficients',
            '470': '470 Constant Exterior Surface Coefficients',
            '800': '800 High Mass Hig Cond. Wall Elements',
            '810': '810 HIgh Mass Cavity Albedo'}

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
    def _order_dictionary_with_numeric_keys(input_d, val_key, key_numeric_type=int, val_numeric_type=float):
        """
        Take input dictionary, reorder based in key numeric value, and return two lists.

        key_list: list of keys numerically ordered by input dictionary key
        val_list: list of values numerically ordered by input dictionary key

        :param input_d: dictionary to be sorted
        :param val_key: key for sub-dictionaries of input_d
        :param key_numeric_type: numeric type for the input keys
        :param val_numeric_type: numeric type for the input values
        :return: Two lists (keys and values) sorted by the input dictionary keys (numerically)
        """

        key_list = []
        val_list = []
        # Make ordered lists based on the integer value of the key
        for k, v in input_d.items():
            if not key_list:
                key_list.append(key_numeric_type(k))
                val_list.append(val_numeric_type(v[val_key]))
                continue
            last_item = -float('inf')
            for kidx, bin_item in enumerate(key_list):
                if last_item < key_numeric_type(k) <= bin_item:
                    key_list.insert(kidx, key_numeric_type(k))
                    val_list.insert(kidx, val_numeric_type(v[val_key]))
                    break
                if kidx == len(key_list) - 1:
                    key_list.append(key_numeric_type(k))
                    val_list.append(val_numeric_type(v[val_key]))
                    break
                last_item = bin_item
        return key_list, val_list

    def _create_bar_plot(
            self, data, programs, title, xticklabels, ylabel, width=0.1, y_plot_pad=0.1,
            y_max=None, y_min=None, image_name=None):
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
        :param y_max: maximum override for y axis
        :param y_min: minimum override for y axis
        :param image_name: unique name to store the plot as a png
        :return: matplotlib fig and ax objects.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig, ax = self._set_theme(fig, ax)
        for idx, (p, d, h) in enumerate(zip(programs, data, self.hatches)):
            x = np.arange(len(d))
            # Fill the tested software bar color.  In case there is some duplicate, only fill the last one.
            if p.lower() == \
                    self.json_data[self.model_name]['identifying_information']['software_name'].lower() and idx + 1 == \
                    len(programs):
                bar_color = '#FE7A7C'
            else:
                bar_color = 'w'
            rects = ax.bar(
                x + (width * idx) - (width / 2 * (len(data) - 1)),
                d,
                width,
                label=p,
                hatch=h,
                color=bar_color,
                edgecolor='k')
            ax.bar_label(rects, padding=5, rotation="vertical")
        ax.set_xticks(np.arange(max([len(i) for i in data])))
        ax.grid(which='major', axis='y')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=len(programs), fontsize=14)
        ax.set_title(title, fontsize=30)
        ax.set_xticklabels(['\n'.join(wrap(i, 15)) for i in xticklabels])
        ax.set_ylabel(ylabel, fontsize=14)
        ymin = y_min or min([i for i in map(min, data) if not np.isnan(i)])
        ymax = y_max or max([i for i in map(max, data) if not np.isnan(i)])
        ax.set_ylim(
            ymin - abs(ymin * y_plot_pad),
            ymax + abs(ymax * y_plot_pad))
        fig.patch.set_facecolor('white')
        if image_name:
            self._make_image_from_plt(image_name)
        return fig, ax

    def _create_split_bar_plot(
            self, data, programs, title, xticklabels, ylabel, width=0.1, y_plot_pad=0.1,
            sub_titles=None, y_max=None, y_min=None, image_name=None):
        """
        Create split bar plot from data input.

        :param data: bar values in nested lists.
            sublist level 0 - number of split plots
            sublist level 1 - number of programs evaluated
            sublist level 2 - number of cases evaluated
        :param programs: list of tested programs
        :param title: plot title
        :param sub_titles: subplot titles
        :param xticklabels: labels to use for each case evaluated
        :param ylabel: y axis plot label
        :param width: width of bars
        :param y_plot_pad: padding between the highest bar and the top of the plot
        :param y_max: maximum override for y axis
        :param y_min: minimum override for y axis
        :param image_name: unique name to store the plot as a png
        :return: matplotlib fig and ax objects.
        """
        # make a count of the ticks per subplot, then use it as the width ratio
        ticks_per_split = []
        for d in data:
            tmp_l = []
            for pl in d:
                tmp_l.append(len(pl))
            ticks_per_split.append(max(tmp_l))
        fig, ax = plt.subplots(
            1,
            len(data),
            figsize=(18, 8),
            sharex='none',
            sharey='all',
            gridspec_kw={
                'width_ratios': ticks_per_split
            })
        fig, ax = self._set_theme(fig, ax)
        for didx, sub_data in enumerate(data):
            for idx, (p, d, h) in enumerate(zip(programs, sub_data, self.hatches)):
                x = np.arange(len(d))
                # Fill the tested software bar color.  In case there is some duplicate, only fill the last one.
                if p.lower() == \
                        self.json_data[self.model_name]['identifying_information'][
                            'software_name'].lower() and idx + 1 == len(programs):
                    bar_color = '#FE7A7C'
                else:
                    bar_color = 'w'
                rects = ax[didx].bar(
                    x + (width * idx) - (width / 2 * (len(sub_data) - 1)),
                    d,
                    width,
                    label=p,
                    hatch=h,
                    color=bar_color,
                    edgecolor='k')
                ax[didx].bar_label(rects, padding=5, rotation="vertical")
                ax[didx].grid(which='major', axis='y')
                ax[didx].set_xticks(np.arange(max([len(i) for i in sub_data])))
                ax[didx].set_xticklabels(
                    ['\n'.join(wrap(i, 15)) for i in xticklabels[didx]]
                )
                if sub_titles:
                    ax[didx].set_title(sub_titles[didx], fontsize=18)
        # Get the middle data index and use that to set the legend
        mid_index = math.floor((len(data) - 1) / 2)
        if mid_index == 0:
            legend_d = {
                'loc': 'lower left',
                'bbox_to_anchor': (0.25, -0.25)}
        else:
            legend_d = {
                'loc': 'lower center',
                'bbox_to_anchor': (0.5, -0.25)}
        # set legend for all plots
        ax.flatten()[mid_index].legend(**legend_d, ncol=len(programs), fontsize=16)
        # Make title, adjust plots, and set y values
        fig.suptitle(title, fontsize=30, y=0.99)
        if sub_titles:
            title_lines = len(title.split('\n'))
            fig.subplots_adjust(top=0.9 - (0.05 * title_lines), wspace=0.001)
        ax[0].set_ylabel(ylabel, fontsize=14)
        ymin = y_min or min([j for i in data for j in map(min, i) if not np.isnan(j)])
        ymax = y_max or max([j for i in data for j in map(max, i) if not np.isnan(j)])
        ax[0].set_ylim(
            ymin - abs(ymin * y_plot_pad),
            ymax + abs(ymax * y_plot_pad))
        fig.patch.set_facecolor('white')
        if image_name:
            self._make_image_from_plt(image_name)
        return fig, ax

    def _create_line_plot(
            self, data_x, data_y, programs, title, ylabel, xlabel=None,
            y_plot_pad=0.1, x_min=None, y_min=None, x_max=None, y_max=None, image_name=None, annotations=None,
            x_tick_spacing=1):
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
        :param xlabel: x axis plot label
        :param y_plot_pad: padding between the highest bar and the top of the plot
        :param x_min: minimum override for x axis
        :param x_max: maximum override for x axis
        :param y_min: minimum override for y axis
        :param y_max: maximum override for y axis
        :param image_name: unique name to store the plot as a png
        :return: matplotlib fig and ax objects.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig, ax = self._set_theme(fig, ax)
        # Format labels for plotting such that programs with the same name receive the same colors and markers
        tmp_programs = []
        tmp_colors = []
        tmp_markers = []
        is_unique = []
        line_counter = 0
        for i in range(len(data_x)):
            if programs[i] not in tmp_programs:
                tmp_programs.append(programs[i])
                tmp_markers.append(self.markers[line_counter])
                tmp_colors.append(self.colors[line_counter])
                line_counter += 1
                is_unique.append(True)
            else:
                tmp_programs.append(tmp_programs[tmp_programs.index(programs[i])])
                tmp_markers.append(tmp_markers[tmp_programs.index(programs[i])])
                tmp_colors.append(tmp_colors[tmp_programs.index(programs[i])])
                is_unique.append(False)
        # Add line plots for each program
        for dx, dy, p, c, m, u in zip(data_x, data_y, tmp_programs, tmp_colors, tmp_markers, is_unique):
            ax.plot(dx, dy, color=c, marker=m, label=p if u else '')
        # Format plot area
        ax.grid(which='major', axis='y')
        # get minimum/maximum of all x rounded to nearest ten, then increment by 5
        ax.set_xticks(np.arange(
            math.floor(min([min(i) for i in data_x if i]) / 10) * 10,
            math.ceil(max([max(i) for i in data_x if i]) / 10) * 10,
            x_tick_spacing))
        ax.set_title(title, fontsize=30)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(programs), fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=14)
        ymin = y_min or min([j for i in data_y for j in i if i if not pd.isna(j)])
        ymax = y_max or max([j for i in data_y for j in i if i if not pd.isna(j)])
        xmin = x_min or min([j for i in data_x for j in i if i if not pd.isna(j)])
        xmax = x_max or max([j for i in data_x for j in i if i if not pd.isna(j)])
        ax.set_xlim(
            xmin,
            xmax)
        ax.set_ylim(
            ymin - abs(ymin * y_plot_pad),
            ymax + abs(ymax * y_plot_pad))
        if annotations:
            for annotation in annotations:
                ax.annotate(**annotation)
        fig.patch.set_facecolor('white')
        if image_name:
            self._make_image_from_plt(image_name)
        return fig, ax

    def _make_image_from_plt(self, figure_name, destination_directory=root_directory.joinpath('rendered', 'images')):
        """
        make a png file from a matplotlib.pyplot object and save it to a directory

        :param figure_name: name of figure to append to file name
        :param destination_directory: list of directories leading to the output directory
        :return: saved image in referenced directory
        """
        program_name = self.model_results_file.parts[-3].lower()
        version = self.model_results_file.parts[-2].lower()
        img_directory = destination_directory.joinpath(
            program_name,
            version,
            'images')
        pathlib.Path(img_directory).mkdir(parents=True, exist_ok=True)
        img_name = img_directory.joinpath(
            '.'.join(
                [
                    '-'.join(
                        [
                            self.model_results_file.stem,
                            figure_name,
                        ]),
                    'png'
                ]))
        plt.savefig(img_name, bbox_inches='tight', facecolor='white')
        return

    def _make_markdown_from_table(self, figure_name, caption, table, footnotes=[],
                                  destination_directory=root_directory.joinpath('rendered', 'images')):
        """
        make a markdown .md file from table (list of lists) save it to a directory

        :param figure_name: name of figure to append to file name
        :param caption: the title of the figure appearing above the table
        :param table: a list of lists where the outer list contains rows and each row list contains column values
        :param footnotes: a list of footnotes that appear below the table
        :param destination_directory: list of directories leading to the output directory
        """
        program_name = self.model_results_file.parts[-3].lower()
        version = self.model_results_file.parts[-2].lower()
        img_directory = destination_directory.joinpath(
            program_name,
            version,
            'images')
        pathlib.Path(img_directory).mkdir(parents=True, exist_ok=True)
        md_name = img_directory.joinpath(
            '.'.join(
                [
                    '-'.join(
                        [
                            self.model_results_file.stem,
                            figure_name,
                        ]),
                    'md'
                ]))
        with open(md_name, 'w') as md:
            md.write('# ' + caption + '\n')
            for indx, row in enumerate(table):
                md_string = "| "
                for value in row:
                    md_string += str(value) + " |"
                md.write(md_string + '\n')
                # header row
                if indx == 0:
                    md.write("|-----|" + "-----:|" * (len(row) - 1) + '\n')
            md.write('\n')
            for footnote in footnotes:
                md.write(footnote + '\n')
            md.write('\n')
        return

    @staticmethod
    def _add_stats_to_table(row_headings, column_headings, data_table, digits=1, time_stamps=[]):
        """
        Add statisics to a table as well as merging the headings and time stamps

        :param row_headings: a list of headings for each row of the table
        :param column_headings: a list of headings for each column of the table including the top left
        :param data_table: a list of lists where the outer list contains rows and each row list contains values
        :param digits: the number of digits shown to the right of the decimal point
        :param time_stamps: a list of lists where the outer list contains rows and each row list contains time stamps
        :return: merged table (list of lists) containing text for every column and row formatted and merged
        """
        formatting_string = '{:.' + str(digits) + 'f}'
        final_column_headings = column_headings[:-1]
        final_column_headings.extend(['', 'Min', 'Max', 'Mean', 'Dev % [^1]', ''])
        final_column_headings.append(column_headings[-1])
        text_table_with_stats = [final_column_headings, ]  # list of rows with each row being a list
        for row_index, data_row in enumerate(data_table):
            row = [row_headings[row_index], ]  # first add the heading for the row
            if data_row:  # for blank rows just skip
                for item in data_row[:-1]:
                    row.append(formatting_string.format(item))
                reference_data_row = data_row[:-1]  # remove the last item which is the tested software
                row.append('')
                row_min = min(reference_data_row)
                row.append(formatting_string.format(row_min))
                row_max = max(reference_data_row)
                row.append(formatting_string.format(row_max))
                row_mean = sum(reference_data_row) / len(reference_data_row)
                row.append(formatting_string.format(row_mean))
                if row_mean != 0:
                    row_dev = abs((row_max - row_min) / row_mean) * 100
                    row.append('{:.1f}'.format(row_dev))
                else:
                    row.append('-')
                row.append('')
                row.append(formatting_string.format(data_row[-1]))  # now add the last column back
            text_table_with_stats.append(row)
        # now add the rows with time stamps
        if time_stamps:
            text_table_with_stats.append(['', ])  # add blank line
            text_table_with_stats.append(['Time Stamps', ])  # add blank line
            mid_header = ['Month', ]
            mid_header.extend(['Day-Hr' for x in column_headings[:-2]])
            mid_header.extend(['' for x in range(6)])
            mid_header.append('Day-Hr')
            text_table_with_stats.append(mid_header)
            for row_index, time_row in enumerate(time_stamps):
                row = [row_headings[row_index], ]  # first add the heading for the row
                if time_row:  # for blank rows just skip
                    row.extend(time_row[:-1])
                    row.extend(['' for x in range(6)])
                    row.append(time_row[-1])
                text_table_with_stats.append(row)
        return text_table_with_stats

    def _int_0_if_nan(self, value_in):
        """
        Protected int() and returns zero if not-a-number (NAN)

        :param value_in: a numeric value
        :return: int() of value_in or 0 if value_in is NAN
        """
        if not math.isnan(value_in):
            value_out = int(value_in)
        else:
            value_out = 0
        return value_out

    @staticmethod
    def _make_table_from_df(df, ax, case_col_width=2, cell_text=[]):
        """
        Create a matplotlib table from dataframe

        :param df: Pandas DataFrame object
        :param ax: matplotlib axis to insert table
        :return: matplotlib table object
        """

        if cell_text:
            tab = ax.table(
                cellText=cell_text,
                colLabels=df.columns,
                zorder=1,
                bbox=[0, 0, 1, 1],
                edges='LR')
            num_columns = 2
            num_rows = 2
        else:
            tab = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                zorder=1,
                bbox=[0, 0, 1, 1],
                edges='LR')
            num_columns = len(df.columns)
            num_rows = df.shape[0] + 1

        ax.axis('tight')
        ax.axis('off')
        # set font
        tab.auto_set_font_size(False)
        tab.set_fontsize(12)
        # set cell properties individually
        cell_dict = tab.get_celld()
        for i in range(num_columns):
            cell_dict[(0, i)].set_height(1)
            cell_dict[(0, i)].set_fontsize(16)
            cell_dict[(0, 0)].set_width(case_col_width)
            cell_dict[(0, i)].set_width(0.5)
            cell_dict[(0, i)].visible_edges = 'closed'
            cell_dict[(0, i)].set_facecolor('#EAEEED')
            for j in range(1, num_rows):
                cell_dict[(j, 0)].set_width(case_col_width)
                cell_dict[(j, 0)].set_text_props(ha="left")
                cell_dict[(j, 0)].PAD = 0.02
                cell_dict[(j, i)].set_width(0.5)
                cell_dict[(j, i)].set_height(0.25)
                if j % 2 == 0:
                    cell_dict[(j, i)].visible_edges = 'closed'
                    cell_dict[(j, i)].set_facecolor('#D2F6ED')
        ax.axis([0, 1, 1, 0])
        # set outer borders
        ax.axhline(y=0, color='black', linewidth=4, zorder=3)
        ax.axhline(y=1, color='black', linewidth=4, zorder=3)
        ax.axvline(x=0, color='black', linewidth=4, zorder=3)
        ax.axvline(x=1, color='black', linewidth=4, zorder=3)
        return tab

    def render_section_5_2a_table_b8_1_old(
            self,
            output_value='annual_heating_MWh',
            figure_name='section_5_2_a_table_b8_1',
            caption='Table B8-1. Annual Heating Loads (MWh)'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-1

        :return: pandas dataframe and output msg for general navigation.
        """
        # get and format dataframe into required shape
        df = self.df_data['conditioned_zone_loads_non_free_float'].loc[:,
                                                                       self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1) == output_value]
        df.columns = df.columns.droplevel(level=1)
        # round values
        df = df.round(3)
        df_formatted_table = df.unstack() \
            .reset_index() \
            .rename(columns={0: 'val'}) \
            .pivot(index=['level_0'], columns=['program_name', ], values=['val', ]) \
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
        df_formatted_table = df_formatted_table \
            .merge(
                self.case_detailed_df,
                how='left',
                left_on=['cases', ],
                right_index=True) \
            .sort_values(['case_order']) \
            .drop(['cases', 'case_order'], axis=1) \
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
        tab = self._make_table_from_df(df=df_formatted_table, ax=ax)
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

    def render_section_5_2a_table_b8_2_old(self):
        """
        Create dataframe from class dataframe object for table 5-2A B8-2

        :return: pandas dataframe and output msg for general navigation.
        """
        fig, ax = self.render_section_5_2a_table_b8_1(
            output_value='annual_cooling_MWh',
            figure_name='section_5_2_a_table_b8_2',
            caption='Table B8.2 Annual Sensible Cooling Loads (MWh)'
        )
        return fig, ax

    def render_section_5_2a_table_b8_3_old(
            self,
            output_values=('peak_heating_kW', 'peak_heating_month', 'peak_heating_day', 'peak_heating_hour'),
            figure_name='section_5_2_a_table_b8_3',
            caption='Table B8-3. Annual Hourly Integrated Peak Heating Loads'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-3

        :return: pandas dataframe and output msg for general navigation.
        """
        # get and format dataframe into required shape
        df = self.df_data['conditioned_zone_loads_non_free_float'].loc[:, [i in output_values for i in self.df_data[
            'conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
        df_formatted_table = df.unstack() \
            .reset_index() \
            .rename(columns={0: 'val', 'level_0': 'case'}) \
            .pivot(index=['case', 'level_1'], columns=['program_name', ], values=['val', ]) \
            .unstack() \
            .reset_index()
        df_formatted_table.columns = df_formatted_table.columns.droplevel(level=0)
        kw_cols = [i for i in df_formatted_table.columns if 'kw' in i[1].lower()]
        df_formatted_table[kw_cols] = df_formatted_table[kw_cols].apply(
            lambda x: pd.to_numeric(x, errors='ignore').round(2), axis=0)
        df_stats = pd.DataFrame()
        base_kw_cols = [i for i in kw_cols if i[0] in self.baseline_model_names]
        df_stats['min'] = df_formatted_table[base_kw_cols].min(axis=1).round(2)
        df_stats['max'] = df_formatted_table[base_kw_cols].max(axis=1).round(2)
        df_stats['mean'] = df_formatted_table[base_kw_cols].mean(axis=1).round(2)
        df_stats['(max - min)\n/ mean %'] = df_formatted_table[base_kw_cols].min(axis=1).round(2)
        int_cols = [i for i in df_formatted_table.columns if any(j for j in ['day', 'hour'] if j in i[1].lower())]
        df_formatted_table[int_cols] = df_formatted_table[int_cols].fillna(0).astype(int)
        df_formatted_table = df_formatted_table.reindex(columns=['', *output_values], level=1)
        df_formatted_table = df_formatted_table.fillna('')
        column_formatting_names = {
            'peak_heating_kW': 'kW',
            'peak_heating_month': 'Mo',
            'peak_heating_day': 'Day',
            'peak_heating_hour': 'Hr',
            'peak_cooling_kW': 'kW',
            'peak_cooling_month': 'Mo',
            'peak_cooling_day': 'Day',
            'peak_cooling_hour': 'Hr'
        }
        # reorder columns so test program is last
        column_names = [i for i in df_formatted_table.columns if i[0] != self.model_name] + [
            i for i in df_formatted_table.columns if i[0] == self.model_name]
        df_formatted_table = df_formatted_table[column_names]
        column_names = [column_formatting_names[i[1]] if i[1] in column_formatting_names.keys() else i[1]
                        for i in list(df_formatted_table.columns)]
        column_names[0] = 'cases'
        # make list of program names
        program_list = sorted(
            set(
                [i[0] for i in df_formatted_table.columns if i[0]]),
            key=[i[0] for i in df_formatted_table.columns].index)
        program_rgx = re.compile(r'(^[a-zA-Z]+)')
        program_list_short = []
        for p in program_list:
            result = program_rgx.search(p)
            if result:
                program_list_short.append(result.group(1))
        df_formatted_table.columns = column_names
        df_formatted_name_table = df_formatted_table[['cases']] \
            .merge(
            self.case_detailed_df,
            how='left',
            left_on=['cases', ],
            right_index=True) \
            .sort_values(['case_order']) \
            .drop(['cases', 'case_order'], axis=1) \
            .rename(columns={
                'case_name': 'Case'})
        # set fig size
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(30, 20))
        df_formatted_table = pd.concat(
            [
                df_formatted_name_table,
                df_formatted_table.drop(columns=['cases', ]).iloc[:, range(len(df_formatted_table.columns) - 5)],
                df_stats,
                df_formatted_table.drop(columns=['cases', ]).iloc[
                    :,
                    range(len(df_formatted_table.columns) - 5, len(df_formatted_table.columns) - 1)]],
            axis=1)
        tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=5)
        cell_dict = tab.get_celld()
        for w, i in zip([0.6, 0.6, 0.6, 1.5], [25, 26, 27, 28]):
            for j in range(df_formatted_table.shape[0] + 1):
                cell_dict[(j, i)].set_width(w)
                cell_dict[(j, i)].set_text_props(ha="center")
        # Set annotations
        for idx, hdr in zip([1, 5, 9, 13, 17, 21, 29], program_list_short):
            header = [tab.add_cell(-1, idx, width=0.5, height=0.35), ]
            header[0].get_text().set_text(hdr.upper())
            header[0].PAD = 0.1
            header[0].set_fontsize(16)
            header[0].set_text_props(ha="left")
            header[0].visible_edges = "open"
            if idx != 29:
                ax.axvline(x=(4.5 + idx / 2) / 22.3, color='black', linewidth=4, zorder=3)
            else:
                ax.axvline(x=20 / 22, color='black', linewidth=4, zorder=3)
        ax.axvline(x=(4.5 + 25 / 2) / 22.3, color='black', linewidth=4, zorder=3)
        # save the result
        plt.suptitle(caption, fontsize=30)
        self._make_image_from_plt(figure_name)
        plt.subplots_adjust(top=0.92)
        return fig, ax

    def render_section_5_2a_table_b8_4_old(self):
        """
        Create dataframe from class dataframe object for table 5-2A B8-4

        :return: pandas dataframe and output msg for general navigation.
        """
        fig, ax = self.render_section_5_2a_table_b8_3(
            output_values=('peak_cooling_kW', 'peak_cooling_month', 'peak_cooling_day', 'peak_cooling_hour'),
            figure_name='section_5_2_a_table_b8_4',
            caption='Table B8-4. Annual Hourly Integrated Peak Cooling Loads'
        )
        return fig, ax

    def _make_table_b8_5_df(
            self,
            output_values,
            split_column=5):
        """

        :param output_values:
        :return:
        """
        # get and format dataframe into required shape
        df = self.df_data['free_float_case_zone_temperatures'].loc[:, [i in output_values for i in self.df_data[
            'free_float_case_zone_temperatures'].columns.get_level_values(1)]]
        df_formatted_table = df.unstack() \
            .reset_index() \
            .rename(columns={0: 'val', 'level_0': 'case'}) \
            .pivot(index=['case', 'level_1'], columns=['program_name', ], values=['val', ]) \
            .unstack() \
            .reset_index()
        df_formatted_table.columns = df_formatted_table.columns.droplevel(level=0)
        temperature_cols = [i for i in df_formatted_table.columns if 'temperature' in i[1].lower()]
        df_formatted_table[temperature_cols] = df_formatted_table[temperature_cols].apply(
            lambda x: pd.to_numeric(x, errors='ignore').round(1), axis=0)
        df_stats = pd.DataFrame()
        base_temperature_cols = [i for i in temperature_cols if i[0] in self.baseline_model_names]
        df_stats['min'] = df_formatted_table[base_temperature_cols].min(axis=1).round(2)
        df_stats['max'] = df_formatted_table[base_temperature_cols].max(axis=1).round(2)
        df_stats['mean'] = df_formatted_table[base_temperature_cols].mean(axis=1).round(2)
        df_stats['(max - min)\n/ mean %'] = df_formatted_table[base_temperature_cols].min(axis=1).round(2)
        int_cols = [i for i in df_formatted_table.columns if any(j for j in ['day', 'hour'] if j in i[1].lower())]
        df_formatted_table[int_cols] = df_formatted_table[int_cols].fillna(0).astype(int)
        df_formatted_table = df_formatted_table.reindex(columns=['', *output_values], level=1)
        df_formatted_table = df_formatted_table.fillna('')
        column_formatting_names = {
            'maximum_temperature': 'C',
            'maximum_month': 'Mo',
            'maximum_day': 'Day',
            'maximum_hour': 'Hr',
            'minimum_temperature': 'C',
            'minimum_month': 'Mo',
            'minimum_day': 'Day',
            'minimum_hour': 'Hr',
            'average_temperature': 'C'
        }
        # reorder columns so test program is last
        column_names = [i for i in df_formatted_table.columns if i[0] != self.model_name] + [
            i for i in df_formatted_table.columns if i[0] == self.model_name]
        df_formatted_table = df_formatted_table[column_names]
        column_names = [column_formatting_names[i[1]] if i[1] in column_formatting_names.keys() else i[1]
                        for i in list(df_formatted_table.columns)]
        column_names[0] = 'cases'
        # make list of program names
        program_list = sorted(
            set(
                [i[0] for i in df_formatted_table.columns if i[0]]),
            key=[i[0] for i in df_formatted_table.columns].index)
        program_rgx = re.compile(r'(^[a-zA-Z]+)')
        program_list_short = []
        for p in program_list:
            result = program_rgx.search(p)
            if result:
                program_list_short.append(result.group(1))
        df_formatted_table.columns = column_names
        df_formatted_name_table = df_formatted_table[['cases']] \
            .merge(
            self.case_detailed_df,
            how='left',
            left_on=['cases', ],
            right_index=True) \
            .sort_values(['case_order']) \
            .drop(['cases', 'case_order'], axis=1) \
            .rename(columns={
                'case_name': 'Case'})
        df_formatted_table = pd.concat(
            [
                df_formatted_name_table,
                df_formatted_table.drop(
                    columns=['cases', ]).iloc[:, range(len(df_formatted_table.columns) - split_column)],
                df_stats,
                df_formatted_table.drop(columns=['cases', ]).iloc[
                    :,
                    range(len(df_formatted_table.columns) - split_column, len(df_formatted_table.columns) - 1)]],
            axis=1)
        return df_formatted_table, program_list_short

    def render_section_5_2a_table_b8_5(
            self,
            figure_name='section_5_2_a_table_b8_5',
            caption='Table B8-5. Free Float Temperature Output'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-4

        :return: pandas dataframe and output msg for general navigation.
        """
        df_list = []
        program_list = []
        df, programs = self._make_table_b8_5_df(
            output_values=('maximum_temperature', 'maximum_month', 'maximum_day', 'maximum_hour')
        )
        df_list.append(df)
        program_list.append(programs)
        df, programs = self._make_table_b8_5_df(
            output_values=('minimum_temperature', 'minimum_month', 'minimum_day', 'maximum_hour')
        )
        df_list.append(df)
        program_list.append(programs)
        df, programs = self._make_table_b8_5_df(
            output_values=('average_temperature',),
            split_column=2,
        )
        df_list.append(df)
        program_list.append(programs)
        # set fig size
        fig, axs = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(30, 12))
        for tidx, (
                ax,
                df_formatted_table,
                program_list_short,
                title,
                case_col_width) in \
                enumerate(zip(
                    axs,
                    df_list,
                    program_list,
                    ['MAXIMUM ANNUAL HOURLY INTEGRATED ZONE TEMPERATURE',
                     'MINIMUM ANNUAL HOURLY INTEGRATED ZONE TEMPERATURE',
                     'AVERAGE ANNUAL HOURLY INTEGRATED ZONE TEMPERATURE'],
                    [5, 5, 2])):
            tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=case_col_width)
            cell_dict = tab.get_celld()
            # Set title
            header_title = tab.add_cell(-2, 0, width=1, height=0.2)
            header_title.get_text().set_text(title)
            header_title.PAD = 0.0
            header_title.set_fontsize(16)
            header_title.set_text_props(ha="left")
            header_title.visible_edges = "open"
            if tidx != 2:
                for w, i in zip([0.6, 0.6, 0.6, 1.5], [25, 26, 27, 28]):
                    for j in range(df_formatted_table.shape[0] + 1):
                        cell_dict[(j, i)].set_width(w)
                        cell_dict[(j, i)].set_text_props(ha="center")
                # Set annotations
                for idx, hdr in zip([1, 5, 9, 13, 17, 21, 29], program_list_short):
                    header = [tab.add_cell(-1, idx, width=0.5, height=0.2), ]
                    header[0].get_text().set_text(hdr.upper())
                    header[0].PAD = 0.1
                    header[0].set_fontsize(16)
                    header[0].set_text_props(ha="left")
                    header[0].visible_edges = "open"
                    if idx != 29:
                        ax.axvline(x=(4.5 + idx / 2) / 22.3, color='black', linewidth=4, zorder=3)
                    else:
                        ax.axvline(x=20 / 22, color='black', linewidth=4, zorder=3)
                    ax.axvline(x=(4.5 + 25 / 2) / 22.3, color='black', linewidth=4, zorder=3)
            else:
                for w, i, hdr in zip([0.95, ] * 7, [1, 2, 3, 4, 5, 6, 11], program_list_short):
                    for j in range(df_formatted_table.shape[0] + 1):
                        cell_dict[(j, i)].set_width(w)
                        cell_dict[(j, i)].set_text_props(ha="left")
                        cell_dict[(j, 0)].set_width(2.35)
                        header = [tab.add_cell(-1, i, width=0.5, height=0.2), ]
                        header[0].get_text().set_text(hdr.upper())
                        header[0].PAD = 0.1
                        header[0].set_fontsize(16)
                        header[0].set_text_props(ha="left")
                        header[0].visible_edges = "open"
                for w, i in zip([0.27, 0.27, 0.27, 0.72], [7, 8, 9, 10]):
                    for j in range(df_formatted_table.shape[0] + 1):
                        cell_dict[(j, i)].set_width(w)
                        cell_dict[(j, i)].set_text_props(ha="center")
                for idx in range(7):
                    ax.axvline(
                        x=(2.35 + 0.95 * idx) / (2.35 + 0.95 * 7 + 0.27 * 3 + 0.72),
                        color='black',
                        linewidth=4,
                        zorder=3)
                ax.axvline(
                    x=(2.35 + 0.95 * 7 + 0.27 * 3 + 0.65) / (2.35 + 0.95 * 7 + 0.27 * 3 + 0.72 + 0.95),
                    color='black',
                    linewidth=4,
                    zorder=3)
        # save the result
        plt.suptitle(caption, fontsize=30)
        self._make_image_from_plt(figure_name)
        plt.subplots_adjust(top=0.92)
        return fig, axs

    def render_section_5_2a_table_b8_6(
            self,
            figure_name='section_5_2_a_table_b8_6',
            caption='Table B8-6. Low Mass Basic Sensitivity Tests'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-6

        :return: pandas dataframe and output msg for general navigation.
        """
        table_dfs = [None, ] * 4
        for measurement_type, table_idx in zip(
                ['annual_heating_MWh', 'peak_heating_kW'],
                [0, 2]):
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float'] \
                     .loc[:, [i == measurement_type for i in
                              self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
            df.columns = df.columns.droplevel(level=1)
            df_formatted = pd.DataFrame()
            df_formatted['610 - 600 Heat, S. Shade'] = (df['610'] - df['600']).round(3)
            df_formatted['620 - 600 Heat, E&W Orient'] = (df['620'] - df['600']).round(3)
            df_formatted['630 - 620 Heat, E&W Shade'] = (df['630'] - df['620']).round(3)
            df_formatted['640 - 600 Heat, Htg. Setback'] = (df['640'] - df['600']).round(3)
            df_formatted['660 - 600 Heat, Low-E Win.'] = (df['660'] - df['600']).round(3)
            df_formatted['670 - 600 Heat, 1-Pane Win.'] = (df['670'] - df['600']).round(3)
            df_formatted['680 - 600 Heat, > Ins. 20/27'] = (df['680'] - df['600']).round(3)
            df_formatted['685 - 600 Heat, 20/20 tstat'] = (df['685'] - df['600']).round(3)
            df_formatted['695 - 685 Heat, > Ins. 20/20'] = (df['640'] - df['600']).round(3)
            df_formatted = df_formatted.transpose()
            df_stats = pd.DataFrame()
            stat_cols = [i for i in df_formatted.columns if i in self.baseline_model_names]
            df_stats['min'] = df_formatted[stat_cols].min(axis=1).round(3)
            df_stats['max'] = df_formatted[stat_cols].max(axis=1).round(2)
            df_stats['mean'] = df_formatted[stat_cols].mean(axis=1).round(2)
            df_stats['(max - min)\n/ mean %'] = \
                abs(
                    (
                        df_formatted[stat_cols].max(axis=1) - df_formatted[stat_cols].min(axis=1)) / (
                        df_formatted[stat_cols].mean(axis=1)) * 100).round(2)
            df_formatted = pd.concat([df_formatted.iloc[:, range(len(df_formatted.columns))], df_stats,
                                      df_formatted.iloc[:,
                                      range(len(df_formatted.columns) - 1, len(df_formatted.columns))]], axis=1)
            program_rgx = re.compile(r'(^[a-zA-Z]+)')
            df_formatted.columns = [
                program_rgx.search(i).group(1) if program_rgx.search(i) else i for i in df_formatted.columns]
            df_formatted['(max - min)\n/ mean %'] = df_formatted['(max - min)\n/ mean %'].apply(
                lambda x: '{0:.1f}%'.format(x))
            df_formatted = df_formatted.reset_index().rename(columns={'index': 'Case'})
            table_dfs[table_idx] = df_formatted
        for measurement_type, table_idx in zip(
                ['annual_cooling_MWh', 'peak_cooling_kW'],
                [1, 3]):
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float'] \
                     .loc[:, [i == measurement_type for i in
                              self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
            df.columns = df.columns.droplevel(level=1)
            df_formatted = pd.DataFrame()
            df_formatted['610 - 600 Cool, S. Shade'] = (df['610'] - df['600']).round(3)
            df_formatted['620 - 600 Cool, E&W Orient'] = (df['620'] - df['600']).round(3)
            df_formatted['630 - 620 Cool, E&W Shade'] = (df['630'] - df['620']).round(3)
            df_formatted['640 - 600 Cool, Htg. Setback'] = (df['640'] - df['600']).round(3)
            df_formatted['650 - 600 Cool, Night Vent'] = (df['650'] - df['600']).round(3)
            df_formatted['660 - 600 Heat, Low-E Win.'] = (df['660'] - df['600']).round(3)
            df_formatted['670 - 600 Heat, 1-Pane Win.'] = (df['670'] - df['600']).round(3)
            df_formatted['680 - 600 Heat, > Ins. 20/27'] = (df['680'] - df['600']).round(3)
            df_formatted['685 - 600 Heat, 20/20 tstat'] = (df['685'] - df['600']).round(3)
            df_formatted['695 - 685 Heat, > Ins. 20/20'] = (df['640'] - df['600']).round(3)
            df_formatted = df_formatted.transpose()
            df_stats = pd.DataFrame()
            stat_cols = [i for i in df_formatted.columns if i in self.baseline_model_names]
            df_stats['min'] = df_formatted[stat_cols].min(axis=1).round(3)
            df_stats['max'] = df_formatted[stat_cols].max(axis=1).round(2)
            df_stats['mean'] = df_formatted[stat_cols].mean(axis=1).round(2)
            df_stats['(max - min)\n/ mean %'] = \
                abs(
                    (
                        df_formatted[stat_cols].max(axis=1) - df_formatted[stat_cols].min(axis=1)) / (
                        df_formatted[stat_cols].mean(axis=1)) * 100).round(2)
            df_formatted = pd.concat([df_formatted.iloc[:, range(len(df_formatted.columns))], df_stats,
                                      df_formatted.iloc[:,
                                      range(len(df_formatted.columns) - 1, len(df_formatted.columns))]], axis=1)
            program_rgx = re.compile(r'(^[a-zA-Z]+)')
            df_formatted.columns = [
                program_rgx.search(i).group(1) if program_rgx.search(i) else i for i in df_formatted.columns]
            df_formatted['(max - min)\n/ mean %'] = df_formatted['(max - min)\n/ mean %'].apply(
                lambda x: '{0:.1f}%'.format(x))
            df_formatted = df_formatted.reset_index().rename(columns={'index': 'Case'})
            table_dfs[table_idx] = df_formatted
        fig, axs = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(24, 16))
        for ax, df_t, title in zip(
                axs,
                table_dfs,
                ['ANNUAL HEATING [MWh]', 'ANNUAL SENSIBLE COOLING [MWh]',
                 'PEAK HEATING [kW]', 'PEAK SENSIBLE COOLING [kW]']):
            tab = self._make_table_from_df(df=df_t, ax=ax, case_col_width=1.3)
            # Set title
            header_title = tab.add_cell(-1, 0, width=1, height=0.3)
            header_title.get_text().set_text(title)
            header_title.PAD = 0.0
            header_title.set_fontsize(16)
            header_title.set_text_props(ha="left")
            header_title.visible_edges = "open"
            cell_dict = tab.get_celld()
            for w, i in zip([0.75, 0.75], [4, 11]):
                for j in range(df_t.shape[0] + 1):
                    cell_dict[(j, i)].set_width(w)
                    cell_dict[(j, i)].set_text_props(ha="center")
        # save the result
        plt.suptitle(caption, fontsize=30)
        self._make_image_from_plt(figure_name)
        plt.subplots_adjust(top=0.92)
        return

    def render_section_5_2a_table_b8_7(
            self,
            figure_name='section_5_2_a_table_b8_7',
            caption='Table B8-7. High Mass Basic Sensitivity Tests'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-7

        :return: pandas dataframe and output msg for general navigation.
        """
        table_dfs = [None, ] * 4
        for measurement_type, table_idx in zip(
                ['annual_heating_MWh', 'peak_heating_kW'],
                [0, 2]):
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float'] \
                     .loc[:, [i == measurement_type for i in
                              self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
            df.columns = df.columns.droplevel(level=1)
            df_formatted = pd.DataFrame()
            df_formatted['900 - 600 Mass, Heat'] = (df['900'] - df['600']).round(3)
            df_formatted['910 - 900 Heat, S.Shade'] = (df['910'] - df['900']).round(3)
            df_formatted['920 - 900 Heat, E&W Orient.'] = (df['920'] - df['900']).round(3)
            df_formatted['930 - 920 Heat, E&W Shade'] = (df['930'] - df['920']).round(3)
            df_formatted['940 - 900 Heat, Htg. Setback'] = (df['940'] - df['900']).round(3)
            df_formatted['960 - 900 Heat, Sunspace'] = (df['960'] - df['900']).round(3)
            df_formatted['980 - 900 Heat, > Ins. 20/27'] = (df['980'] - df['900']).round(3)
            df_formatted['985 - 900 Heat, > 20/20 tstat'] = (df['985'] - df['900']).round(3)
            df_formatted['995 - 985 Heat, > Ins. 20/20'] = (df['995'] - df['985']).round(3)
            df_formatted = df_formatted.transpose()
            df_stats = pd.DataFrame()
            stat_cols = [i for i in df_formatted.columns if i in self.baseline_model_names]
            df_stats['min'] = df_formatted[stat_cols].min(axis=1).round(3)
            df_stats['max'] = df_formatted[stat_cols].max(axis=1).round(2)
            df_stats['mean'] = df_formatted[stat_cols].mean(axis=1).round(2)
            df_stats['(max - min)\n/ mean %'] = \
                abs(
                    (
                        df_formatted[stat_cols].max(axis=1) - df_formatted[stat_cols].min(axis=1)) / (
                        df_formatted[stat_cols].mean(axis=1)) * 100).round(2)
            df_formatted = pd.concat([df_formatted.iloc[:, range(len(df_formatted.columns))], df_stats,
                                      df_formatted.iloc[:,
                                      range(len(df_formatted.columns) - 1, len(df_formatted.columns))]], axis=1)
            program_rgx = re.compile(r'(^[a-zA-Z]+)')
            df_formatted.columns = [
                program_rgx.search(i).group(1) if program_rgx.search(i) else i for i in df_formatted.columns]
            df_formatted['(max - min)\n/ mean %'] = df_formatted['(max - min)\n/ mean %'].apply(
                lambda x: '{0:.1f}%'.format(x))
            df_formatted = df_formatted.reset_index().rename(columns={'index': 'Case'})
            table_dfs[table_idx] = df_formatted
        for measurement_type, table_idx in zip(
                ['annual_cooling_MWh', 'peak_cooling_kW'],
                [1, 3]):
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float'] \
                     .loc[:, [i == measurement_type for i in
                              self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
            df.columns = df.columns.droplevel(level=1)
            df_formatted = pd.DataFrame()
            df_formatted['900 - 600 Mass, Cool'] = (df['610'] - df['600']).round(3)
            df_formatted['910 - 900 Cool, S.Shade'] = (df['910'] - df['900']).round(3)
            df_formatted['920 - 900 Cool, E&W Orient.'] = (df['920'] - df['900']).round(3)
            df_formatted['930 - 920 Cool, E&W Shade'] = (df['930'] - df['920']).round(3)
            df_formatted['940 - 900 Cool, Htg. Setback'] = (df['940'] - df['900']).round(3)
            df_formatted['950 - 900 Cool, Night Vent'] = (df['950'] - df['900']).round(3)
            df_formatted['960 - 900 Cool, Sunspace'] = (df['960'] - df['900']).round(3)
            df_formatted['980 - 900 Heat, > Ins. 20/27'] = (df['980'] - df['900']).round(3)
            df_formatted['985 - 900 Heat, > 20/20 tstat'] = (df['985'] - df['900']).round(3)
            df_formatted['995 - 985 Heat, > Ins. 20/20'] = (df['995'] - df['985']).round(3)
            df_formatted = df_formatted.transpose()
            df_stats = pd.DataFrame()
            stat_cols = [i for i in df_formatted.columns if i in self.baseline_model_names]
            df_stats['min'] = df_formatted[stat_cols].min(axis=1).round(3)
            df_stats['max'] = df_formatted[stat_cols].max(axis=1).round(2)
            df_stats['mean'] = df_formatted[stat_cols].mean(axis=1).round(2)
            df_stats['(max - min)\n/ mean %'] = \
                abs(
                    (
                        df_formatted[stat_cols].max(axis=1) - df_formatted[stat_cols].min(axis=1)) / (
                        df_formatted[stat_cols].mean(axis=1)) * 100).round(2)
            df_formatted = pd.concat([df_formatted.iloc[:, range(len(df_formatted.columns))], df_stats,
                                      df_formatted.iloc[:,
                                      range(len(df_formatted.columns) - 1, len(df_formatted.columns))]], axis=1)
            program_rgx = re.compile(r'(^[a-zA-Z]+)')
            df_formatted.columns = [
                program_rgx.search(i).group(1) if program_rgx.search(i) else i for i in df_formatted.columns]
            df_formatted['(max - min)\n/ mean %'] = df_formatted['(max - min)\n/ mean %'].apply(
                lambda x: '{0:.1f}%'.format(x))
            df_formatted = df_formatted.reset_index().rename(columns={'index': 'Case'})
            table_dfs[table_idx] = df_formatted
        fig, axs = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(24, 16))
        for ax, df_t, title in zip(
                axs,
                table_dfs,
                ['ANNUAL HEATING [MWh]', 'ANNUAL SENSIBLE COOLING [MWh]',
                 'PEAK HEATING [kW]', 'PEAK SENSIBLE COOLING [kW]']):
            tab = self._make_table_from_df(df=df_t, ax=ax, case_col_width=1.3)
            # Set title
            header_title = tab.add_cell(-1, 0, width=1, height=0.3)
            header_title.get_text().set_text(title)
            header_title.PAD = 0.0
            header_title.set_fontsize(16)
            header_title.set_text_props(ha="left")
            header_title.visible_edges = "open"
            cell_dict = tab.get_celld()
            for w, i in zip([0.75, 0.75], [4, 11]):
                for j in range(df_t.shape[0] + 1):
                    cell_dict[(j, i)].set_width(w)
                    cell_dict[(j, i)].set_text_props(ha="center")
        # save the result
        plt.suptitle(caption, fontsize=30)
        self._make_image_from_plt(figure_name)
        plt.subplots_adjust(top=0.92)
        return

    def render_section_5_2a_table_b8_8(
            self,
            figure_name='section_5_2_a_table_b8_8',
            caption='Table B8-8. Low Mass In-Depth (Cases 195 Thru 320) Sensitivity Tests'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-8

        :return: pandas dataframe and output msg for general navigation.
        """
        table_dfs = [None, ] * 4
        for measurement_type, table_idx in zip(
                ['annual_heating_MWh', 'peak_heating_kW', 'annual_cooling_MWh', 'peak_cooling_kW'],
                [0, 1, 2, 3]):
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float'] \
                     .loc[:, [i == measurement_type for i in
                              self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
            df.columns = df.columns.droplevel(level=1)
            df_formatted = pd.DataFrame()
            df_formatted['200-195 Surface Convection'] = (df['200'] - df['195']).round(3)
            df_formatted['210-200 Ext IR (Int IR "off")'] = (df['210'] - df['200']).round(3)
            df_formatted['220-215 Ext IR (Int IR "on")'] = (df['220'] - df['215']).round(3)
            df_formatted['215-200 Int IR (Ext IR "off")'] = (df['215'] - df['200']).round(3)
            df_formatted['220-210 Int IR (Ext IR "on")'] = (df['220'] - df['210']).round(3)
            df_formatted['230-220 Infiltration'] = (df['230'] - df['220']).round(3)
            df_formatted['240-220 Internal Gains'] = (df['240'] - df['220']).round(3)
            df_formatted['250-220 Ext Solar Abs.'] = (df['250'] - df['220']).round(3)
            df_formatted['270-220 South Windows'] = (df['270'] - df['220']).round(3)
            df_formatted['280-270 Cavity Albedo'] = (df['280'] - df['270']).round(3)
            df_formatted['320-270 Thermostat'] = (df['320'] - df['270']).round(3)
            df_formatted['290-270 South Shading'] = (df['290'] - df['270']).round(3)
            df_formatted['300-270 E&W Windows'] = (df['300'] - df['270']).round(3)
            df_formatted['310-300 E&W Shading'] = (df['310'] - df['300']).round(3)
            df_formatted = df_formatted.transpose()
            df_stats = pd.DataFrame()
            stat_cols = [i for i in df_formatted.columns if i in self.baseline_model_names]
            df_stats['min'] = df_formatted[stat_cols].min(axis=1).round(3)
            df_stats['max'] = df_formatted[stat_cols].max(axis=1).round(2)
            df_stats['mean'] = df_formatted[stat_cols].mean(axis=1).round(2)
            df_stats['(max - min)\n/ mean %'] = \
                abs(
                    (
                        df_formatted[stat_cols].max(axis=1) - df_formatted[stat_cols].min(axis=1)) / (
                        df_formatted[stat_cols].mean(axis=1)) * 100).round(2)
            df_formatted = pd.concat([df_formatted.iloc[:, range(len(df_formatted.columns))], df_stats,
                                      df_formatted.iloc[:,
                                      range(len(df_formatted.columns) - 1, len(df_formatted.columns))]], axis=1)
            program_rgx = re.compile(r'(^[a-zA-Z]+)')
            df_formatted.columns = [
                program_rgx.search(i).group(1) if program_rgx.search(i) else i for i in df_formatted.columns]
            df_formatted['(max - min)\n/ mean %'] = df_formatted['(max - min)\n/ mean %'].apply(
                lambda x: '{0:.1f}%'.format(x))
            df_formatted = df_formatted.reset_index().rename(columns={'index': 'Case'})
            table_dfs[table_idx] = df_formatted
        fig, axs = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(24, 16))
        for ax, df_t, title in zip(
                axs,
                table_dfs,
                ['ANNUAL HEATING [MWh]', 'ANNUAL SENSIBLE COOLING [MWh]',
                 'PEAK HEATING [kW]', 'PEAK SENSIBLE COOLING [kW]']):
            tab = self._make_table_from_df(df=df_t, ax=ax, case_col_width=1.3)
            # Set title
            header_title = tab.add_cell(-1, 0, width=1, height=0.3)
            header_title.get_text().set_text(title)
            header_title.PAD = 0.0
            header_title.set_fontsize(16)
            header_title.set_text_props(ha="left")
            header_title.visible_edges = "open"
            cell_dict = tab.get_celld()
            for w, i in zip([0.75, 0.75], [4, 11]):
                for j in range(df_t.shape[0] + 1):
                    cell_dict[(j, i)].set_width(w)
                    cell_dict[(j, i)].set_text_props(ha="center")
        # save the result
        plt.suptitle(caption, fontsize=30)
        self._make_image_from_plt(figure_name)
        plt.subplots_adjust(top=0.92)
        return

    def render_section_5_2a_table_b8_9(
            self,
            figure_name='section_5_2_a_table_b8_9',
            caption='Table B8-9. Low Mass In-Depth (Cases 395 Thru 440) Sensitivity Tests'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-9

        :return: pandas dataframe and output msg for general navigation.
        """
        table_dfs = [None, ] * 4
        for measurement_type, table_idx in zip(
                ['annual_heating_MWh', 'peak_heating_kW', 'annual_cooling_MWh', 'peak_cooling_kW'],
                [0, 1, 2, 3]):
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float'] \
                     .loc[:, [i == measurement_type for i in
                              self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
            df.columns = df.columns.droplevel(level=1)
            df_formatted = pd.DataFrame()
            df_formatted['400-395 Surf. Conv. & IR'] = (df['400'] - df['395']).round(3)
            df_formatted['410-400 Infiltration'] = (df['410'] - df['400']).round(3)
            df_formatted['420-410 Internal Gains'] = (df['420'] - df['410']).round(3)
            df_formatted['430-420 Ext Solar Abs.'] = (df['430'] - df['420']).round(3)
            df_formatted['600-430 South Windows'] = (df['600'] - df['430']).round(3)
            df_formatted['440-600 Cavity Albedo'] = (df['440'] - df['600']).round(3)
            df_formatted['450-600 Const Int&Ext Surf Coefs'] = (df['450'] - df['600']).round(3)
            df_formatted['460-600 Const Int Surf Coefs'] = (df['460'] - df['600']).round(3)
            df_formatted['460-450 Auto Ext Surf Heat Transf'] = (df['460'] - df['450']).round(3)
            df_formatted['470-600 Const Ext Surf Coefs'] = (df['470'] - df['600']).round(3)
            df_formatted['470-450 Auto Int Surf Heat Transf'] = (df['470'] - df['450']).round(3)
            df_formatted = df_formatted.transpose()
            df_stats = pd.DataFrame()
            stat_cols = [i for i in df_formatted.columns if i in self.baseline_model_names]
            df_stats['min'] = df_formatted[stat_cols].min(axis=1).round(3)
            df_stats['max'] = df_formatted[stat_cols].max(axis=1).round(2)
            df_stats['mean'] = df_formatted[stat_cols].mean(axis=1).round(2)
            df_stats['(max - min)\n/ mean %'] = \
                abs(
                    (
                        df_formatted[stat_cols].max(axis=1) - df_formatted[stat_cols].min(axis=1)) / (
                        df_formatted[stat_cols].mean(axis=1)) * 100).round(2)
            df_formatted = pd.concat([df_formatted.iloc[:, range(len(df_formatted.columns))], df_stats,
                                      df_formatted.iloc[:,
                                      range(len(df_formatted.columns) - 1, len(df_formatted.columns))]], axis=1)
            program_rgx = re.compile(r'(^[a-zA-Z]+)')
            df_formatted.columns = [
                program_rgx.search(i).group(1) if program_rgx.search(i) else i for i in df_formatted.columns]
            df_formatted['(max - min)\n/ mean %'] = df_formatted['(max - min)\n/ mean %'].apply(
                lambda x: '{0:.1f}%'.format(x))
            df_formatted = df_formatted.reset_index().rename(columns={'index': 'Case'})
            table_dfs[table_idx] = df_formatted
        fig, axs = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(24, 16))
        for ax, df_t, title in zip(
                axs,
                table_dfs,
                ['ANNUAL HEATING [MWh]', 'ANNUAL SENSIBLE COOLING [MWh]',
                 'PEAK HEATING [kW]', 'PEAK SENSIBLE COOLING [kW]']):
            tab = self._make_table_from_df(df=df_t, ax=ax, case_col_width=1.3)
            # Set title
            header_title = tab.add_cell(-1, 0, width=1, height=0.3)
            header_title.get_text().set_text(title)
            header_title.PAD = 0.0
            header_title.set_fontsize(16)
            header_title.set_text_props(ha="left")
            header_title.visible_edges = "open"
            cell_dict = tab.get_celld()
            for w, i in zip([0.75, 0.75], [4, 11]):
                for j in range(df_t.shape[0] + 1):
                    cell_dict[(j, i)].set_width(w)
                    cell_dict[(j, i)].set_text_props(ha="center")
        # save the result
        plt.suptitle(caption, fontsize=30)
        self._make_image_from_plt(figure_name)
        plt.subplots_adjust(top=0.92)
        return

    def render_section_5_2a_table_b8_10(
            self,
            figure_name='section_5_2_a_table_b8_10',
            caption='Table B8-10. High Mass Basic In-Depth Sensitivity Tests'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-10

        :return: pandas dataframe and output msg for general navigation.
        """
        table_dfs = [None, ] * 4
        for measurement_type, table_idx in zip(
                ['annual_heating_MWh', 'peak_heating_kW'],
                [0, 2]):
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float'] \
                     .loc[:, [i == measurement_type for i in
                              self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
            df.columns = df.columns.droplevel(level=1)
            df_formatted = pd.DataFrame()
            df_formatted['800-430 Mass, w/ High Cond. Wall'] = (df['800'] - df['430']).round(3)
            df_formatted['900-800 Himass, S. Win.'] = (df['900'] - df['800']).round(3)
            df_formatted['900-810 Himass, Int. Solar Abs.'] = (df['900'] - df['810']).round(3)
            df_formatted['910-610 Mass, w/ S. Shade'] = (df['910'] - df['610']).round(3)
            df_formatted['920-620 Mass, w/ E&W Win.'] = (df['920'] - df['620']).round(3)
            df_formatted['930-630 Mass, w/ E&W Shade'] = (df['930'] - df['630']).round(3)
            df_formatted['940-640 Mass, w/ Htg. Setback'] = (df['940'] - df['640']).round(3)
            df_formatted['980-680 Mass, w/ Insulation 20/27'] = (df['980'] - df['680']).round(3)
            df_formatted['985-685 Mass, w/ 20/20 Tstat'] = (df['985'] - df['685']).round(3)
            df_formatted['995-695 Mass, w/ Insulation 20/20'] = (df['995'] - df['695']).round(3)
            df_formatted = df_formatted.transpose()
            df_stats = pd.DataFrame()
            stat_cols = [i for i in df_formatted.columns if i in self.baseline_model_names]
            df_stats['min'] = df_formatted[stat_cols].min(axis=1).round(3)
            df_stats['max'] = df_formatted[stat_cols].max(axis=1).round(2)
            df_stats['mean'] = df_formatted[stat_cols].mean(axis=1).round(2)
            df_stats['(max - min)\n/ mean %'] = \
                abs(
                    (
                        df_formatted[stat_cols].max(axis=1) - df_formatted[stat_cols].min(axis=1)) / (
                        df_formatted[stat_cols].mean(axis=1)) * 100).round(2)
            df_formatted = pd.concat([df_formatted.iloc[:, range(len(df_formatted.columns))], df_stats,
                                      df_formatted.iloc[:,
                                      range(len(df_formatted.columns) - 1, len(df_formatted.columns))]], axis=1)
            program_rgx = re.compile(r'(^[a-zA-Z]+)')
            df_formatted.columns = [
                program_rgx.search(i).group(1) if program_rgx.search(i) else i for i in df_formatted.columns]
            df_formatted['(max - min)\n/ mean %'] = df_formatted['(max - min)\n/ mean %'].apply(
                lambda x: '{0:.1f}%'.format(x))
            df_formatted = df_formatted.reset_index().rename(columns={'index': 'Case'})
            table_dfs[table_idx] = df_formatted
        for measurement_type, table_idx in zip(
                ['annual_cooling_MWh', 'peak_cooling_kW'],
                [1, 3]):
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float'] \
                     .loc[:, [i == measurement_type for i in
                              self.df_data['conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
            df.columns = df.columns.droplevel(level=1)
            df_formatted = pd.DataFrame()
            df_formatted['800-430 Mass, w/ High Cond. Wall'] = (df['800'] - df['430']).round(3)
            df_formatted['900-800 Himass, S. Win.'] = (df['900'] - df['800']).round(3)
            df_formatted['900-810 Himass, Int. Solar Abs.'] = (df['900'] - df['810']).round(3)
            df_formatted['910-610 Mass, w/ S. Shade'] = (df['910'] - df['610']).round(3)
            df_formatted['920-620 Mass, w/ E&W Win.'] = (df['920'] - df['620']).round(3)
            df_formatted['930-630 Mass, w/ E&W Shade'] = (df['930'] - df['630']).round(3)
            df_formatted['940-640 Mass, w/ Htg. Setback'] = (df['940'] - df['640']).round(3)
            df_formatted['950-650 Mass, w/ Night Vent'] = (df['950'] - df['650']).round(3)
            df_formatted['980-680 Mass, w/ Insulation 20/27'] = (df['980'] - df['680']).round(3)
            df_formatted['985-685 Mass, w/ 20/20 Tstat'] = (df['985'] - df['685']).round(3)
            df_formatted['995-695 Mass, w/ Insulation 20/20'] = (df['995'] - df['695']).round(3)
            df_formatted = df_formatted.transpose()
            df_stats = pd.DataFrame()
            stat_cols = [i for i in df_formatted.columns if i in self.baseline_model_names]
            df_stats['min'] = df_formatted[stat_cols].min(axis=1).round(3)
            df_stats['max'] = df_formatted[stat_cols].max(axis=1).round(2)
            df_stats['mean'] = df_formatted[stat_cols].mean(axis=1).round(2)
            df_stats['(max - min)\n/ mean %'] = \
                abs(
                    (
                        df_formatted[stat_cols].max(axis=1) - df_formatted[stat_cols].min(axis=1)) / (
                        df_formatted[stat_cols].mean(axis=1)) * 100).round(2)
            df_formatted = pd.concat([df_formatted.iloc[:, range(len(df_formatted.columns))], df_stats,
                                      df_formatted.iloc[:,
                                      range(len(df_formatted.columns) - 1, len(df_formatted.columns))]], axis=1)
            program_rgx = re.compile(r'(^[a-zA-Z]+)')
            df_formatted.columns = [
                program_rgx.search(i).group(1) if program_rgx.search(i) else i for i in df_formatted.columns]
            df_formatted['(max - min)\n/ mean %'] = df_formatted['(max - min)\n/ mean %'].apply(
                lambda x: '{0:.1f}%'.format(x))
            df_formatted = df_formatted.reset_index().rename(columns={'index': 'Case'})
            table_dfs[table_idx] = df_formatted
        fig, axs = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(24, 16))
        for ax, df_t, title in zip(
                axs,
                table_dfs,
                ['ANNUAL HEATING [MWh]', 'ANNUAL SENSIBLE COOLING [MWh]',
                 'PEAK HEATING [kW]', 'PEAK SENSIBLE COOLING [kW]']):
            tab = self._make_table_from_df(df=df_t, ax=ax, case_col_width=1.3)
            # Set title
            header_title = tab.add_cell(-1, 0, width=1, height=0.3)
            header_title.get_text().set_text(title)
            header_title.PAD = 0.0
            header_title.set_fontsize(16)
            header_title.set_text_props(ha="left")
            header_title.visible_edges = "open"
            cell_dict = tab.get_celld()
            for w, i in zip([0.75, 0.75], [4, 11]):
                for j in range(df_t.shape[0] + 1):
                    cell_dict[(j, i)].set_width(w)
                    cell_dict[(j, i)].set_text_props(ha="center")
        # save the result
        plt.suptitle(caption, fontsize=30)
        self._make_image_from_plt(figure_name)
        plt.subplots_adjust(top=0.92)
        return

    def render_section_5_2a_table_b8_11(
            self,
            figure_name='section_5_2_a_table_b8_11',
            caption='Table B8-11. Annual Transmissivity Coefficient of Windows'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-11

        :return: pandas dataframe and output msg for general navigation.
        """

        df_table_b8_13 = pd.DataFrame()
        df_table_b8_14 = pd.DataFrame()
        software_array = []
        # get data
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            df_obj_table_b8_13 = pd.DataFrame.from_dict(json_obj['solar_radiation_annual_incident']['600']['Surface'])
            df_obj_table_b8_13['software'] = json_obj['identifying_information']['software_name']
            software_array.append(json_obj['identifying_information']['software_name'])
            df_table_b8_13 = pd.concat([df_table_b8_13, df_obj_table_b8_13], axis=0)

            df_obj_table_b8_14 = pd.DataFrame.from_dict(json_obj['solar_radiation_unshaded_annual_transmitted'])
            df_obj_table_b8_14['software'] = json_obj['identifying_information']['software_name']
            df_table_b8_14 = pd.concat([df_table_b8_14, df_obj_table_b8_14], axis=0)

        df_table_b8_13 = df_table_b8_13.set_index('software')
        df_table_b8_13 = df_table_b8_13.transpose()
        df_table_b8_14 = df_table_b8_14.set_index('software')
        df_table_b8_14 = df_table_b8_14.transpose()

        for row in range(len(df_table_b8_14)):
            for col in range(len(df_table_b8_14.columns)):
                temp_dict = df_table_b8_14.at[df_table_b8_14.index[row], df_table_b8_14.columns[col]]
                for k, v in temp_dict.items():
                    new_val = v['kWh/m2']
                try:
                    df_table_b8_14.at[df_table_b8_14.index[row], df_table_b8_14.columns[col]] = new_val
                except (KeyError, ValueError):
                    df_table_b8_14.at[df_table_b8_14.index[row], df_table_b8_14.columns[col]] = float('NaN')

        index_dict = {
            '600 South': ['600', 'SOUTH'],
            '620 West': ['620', 'WEST'],
            '660 South, Low-E': ['660', 'SOUTH'],
            '670 South, Single Pane': ['670', 'SOUTH']
        }
        df_table_b8_11 = pd.DataFrame()
        df_table_b8_11 = pd.DataFrame(columns=df_table_b8_14.columns)
        for key in index_dict:
            df_table_b8_11.loc[key] = df_table_b8_14.loc[index_dict[key][0]].div(
                df_table_b8_13.loc[index_dict[key][1]]).where(df_table_b8_13.loc[index_dict[key][1]] != 0, np.nan)

        # calculate stats
        df_stats = pd.DataFrame()
        df_stats['min'] = df_table_b8_11.iloc[:, 0:-1].min(axis=1)
        df_stats['max'] = df_table_b8_11.iloc[:, 0:-1].max(axis=1)
        df_stats['mean'] = df_table_b8_11.iloc[:, 0:-1].mean(axis=1)
        df_stats['(max-min)\n/mean*(%)'] = abs(df_stats['max'] - df_stats['min']).div(
            df_stats['mean'].where(df_stats['mean'] != 0, np.nan))

        # merge dataframes
        df_merged = pd.concat(
            [
                df_table_b8_11.iloc[:, range(len(df_table_b8_11.columns) - 1)],
                df_stats,
                df_table_b8_11.iloc[:, range(len(df_table_b8_11.columns) - 1, len(df_table_b8_11.columns))]
            ],
            axis=1)

        # format dataframe
        df_formatted_table = df_merged.reset_index(drop=False).rename(columns={'index': 'Cases'})
        df_formatted_table.fillna('', inplace=True)
        for col in df_formatted_table:
            if col in software_array:
                df_formatted_table[col] = df_formatted_table[col].apply(lambda x: round(x, 3) if x != '' else x)

        df_formatted_table['min'] = df_formatted_table['min'].apply(lambda x: round(x, 3) if x != '' else x)
        df_formatted_table['max'] = df_formatted_table['max'].apply(lambda x: round(x, 3) if x != '' else x)
        df_formatted_table['mean'] = df_formatted_table['mean'].apply(lambda x: round(x, 3) if x != '' else x)
        df_formatted_table['(max-min)\n/mean*(%)'] = df_formatted_table['(max-min)\n/mean*(%)'].apply(
            lambda x: '{0:.1f}%'.format(round(x * 100, 3)))

        # wrap text in table
        wrapped_col = {}
        for col in df_formatted_table:
            wrapped_col[col] = twp.fill(col, break_long_words=False, width=5)
        df_formatted_table.rename(columns=wrapped_col, inplace=True)

        # save results
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(24, 6))
        tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=1.2)

        # Set annotations
        header_stats = [tab.add_cell(-1, h, width=0.5, height=0.20) for h in range(7, 11)]
        header_stats[0].get_text().set_text('Statistics for Example Results')
        header_stats[0].PAD = 0.5
        header_stats[0].set_fontsize(16)
        header_stats[0].set_text_props(ha="left")
        header_stats[0].visible_edges = "open"
        header_stats[1].visible_edges = "open"
        header_stats[2].visible_edges = "open"
        header_stats[3].visible_edges = "open"
        plt.figtext(0.15, 0.08, "*ABS[(Max-Min)/(Mean of Example Simulation Results)", ha="left", fontsize=12)

        plt.suptitle(caption, fontsize=30, y=1.05)
        self._make_image_from_plt(figure_name)

        return fig, ax

    def render_section_5_2a_table_b8_12(
            self,
            figure_name='section_5_2_a_table_b8_12',
            caption='Table B8-12. Annual Shading Coefficient of Window Shading Devices: Overhangs & Fins'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-12

        :return: pandas dataframe and output msg for general navigation.
        """

        df_table_b8_14 = pd.DataFrame()
        df_table_b8_15 = pd.DataFrame()
        software_array = []

        # get data
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            df_obj_table_b8_14 = pd.DataFrame.from_dict(json_obj['solar_radiation_unshaded_annual_transmitted'])
            df_obj_table_b8_14['software'] = json_obj['identifying_information']['software_name']
            software_array.append(json_obj['identifying_information']['software_name'])
            df_table_b8_14 = pd.concat([df_table_b8_14, df_obj_table_b8_14], axis=0)

            df_obj_table_b8_15 = pd.DataFrame.from_dict(json_obj['solar_radiation_shaded_annual_transmitted'])
            df_obj_table_b8_15['software'] = json_obj['identifying_information']['software_name']
            df_table_b8_15 = pd.concat([df_table_b8_15, df_obj_table_b8_15], axis=0)

        df_table_b8_14 = df_table_b8_14.set_index('software')
        df_table_b8_14 = df_table_b8_14.transpose()
        df_table_b8_15 = df_table_b8_15.set_index('software')
        df_table_b8_15 = df_table_b8_15.transpose()

        for row in range(len(df_table_b8_14)):
            for col in range(len(df_table_b8_14.columns)):
                temp_dict = df_table_b8_14.at[df_table_b8_14.index[row], df_table_b8_14.columns[col]]
                for k, v in temp_dict.items():
                    new_val = v['kWh/m2']
                try:
                    df_table_b8_14.at[df_table_b8_14.index[row], df_table_b8_14.columns[col]] = new_val
                except (KeyError, ValueError):
                    df_table_b8_14.at[df_table_b8_14.index[row], df_table_b8_14.columns[col]] = float('NaN')

        for row in range(len(df_table_b8_15)):
            for col in range(len(df_table_b8_15.columns)):
                temp_dict = df_table_b8_15.at[df_table_b8_15.index[row], df_table_b8_15.columns[col]]
                for k, v in temp_dict.items():
                    new_val = v['kWh/m2']
                try:
                    df_table_b8_15.at[df_table_b8_15.index[row], df_table_b8_15.columns[col]] = new_val
                except (KeyError, ValueError):
                    df_table_b8_15.at[df_table_b8_15.index[row], df_table_b8_15.columns[col]] = float('NaN')

        index_dict = {
            '610/600 South': ['610', '600'],
            '630/620 West': ['630', '620']
        }
        df_table_b8_12 = pd.DataFrame()
        df_table_b8_12 = pd.DataFrame(columns=df_table_b8_14.columns)

        for key in index_dict:
            df_table_b8_12.loc[key] = 1 - df_table_b8_15.loc[index_dict[key][0]].div(
                df_table_b8_14.loc[index_dict[key][1]]) \
                .where(df_table_b8_14.loc[index_dict[key][1]] != 0, np.nan)

        # calculate stats
        df_stats = pd.DataFrame()
        df_stats['min'] = df_table_b8_12.iloc[:, 0:-1].min(axis=1)
        df_stats['max'] = df_table_b8_12.iloc[:, 0:-1].max(axis=1)
        df_stats['mean'] = df_table_b8_12.iloc[:, 0:-1].mean(axis=1)
        df_stats['(max-min)\n/mean*(%)'] = abs(df_stats['max'] - df_stats['min']).div(
            df_stats['mean'].where(df_stats['mean'] != 0, np.nan))

        # merge dataframes
        df_merged = pd.concat(
            [
                df_table_b8_12.iloc[:, range(len(df_table_b8_12.columns) - 1)],
                df_stats,
                df_table_b8_12.iloc[:, range(len(df_table_b8_12.columns) - 1, len(df_table_b8_12.columns))]
            ],
            axis=1)

        # format dataframe
        df_formatted_table = df_merged.reset_index(drop=False).rename(columns={'index': 'Cases'})
        df_formatted_table.fillna('', inplace=True)
        for col in df_formatted_table:
            if col in software_array:
                df_formatted_table[col] = df_formatted_table[col].apply(lambda x: round(x, 3) if x != '' else x)
        df_formatted_table['min'] = df_formatted_table['min'].apply(lambda x: round(x, 3) if x != '' else x)
        df_formatted_table['max'] = df_formatted_table['max'].apply(lambda x: round(x, 3) if x != '' else x)
        df_formatted_table['mean'] = df_formatted_table['mean'].apply(lambda x: round(x, 3) if x != '' else x)
        df_formatted_table['(max-min)\n/mean*(%)'] = df_formatted_table['(max-min)\n/mean*(%)'].apply(
            lambda x: '{0:.1f}%'.format(round(x * 100, 3)))

        # wrap text in table
        wrapped_col = {}
        for col in df_formatted_table:
            wrapped_col[col] = twp.fill(col, break_long_words=False, width=5)
        df_formatted_table.rename(columns=wrapped_col, inplace=True)

        # save results
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(24, 6))
        tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=1.2)

        # Set annotations
        header_stats = [tab.add_cell(-1, h, width=0.5, height=0.20) for h in range(7, 11)]
        header_stats[0].get_text().set_text('Statistics for Example Results')
        header_stats[0].PAD = 0.5
        header_stats[0].set_fontsize(16)
        header_stats[0].set_text_props(ha="left")
        header_stats[0].visible_edges = "open"
        header_stats[1].visible_edges = "open"
        header_stats[2].visible_edges = "open"
        header_stats[3].visible_edges = "open"
        plt.figtext(0.15, 0.08, "*ABS[(Max-Min)/(Mean of Example Simulation Results)", ha="left", fontsize=12)

        plt.suptitle(caption, fontsize=30, y=1.05)
        self._make_image_from_plt(figure_name)

        return fig, ax

    def render_section_5_2a_table_b8_13(
            self,
            figure_name='section_5_2_a_table_b8_13',
            caption='Table B8-13. Case 600 Annual Incident Solar Radiation ($kWh/m^{2}$)'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-13

        :return: pandas dataframe and output msg for general navigation.
        """

        df = pd.DataFrame()
        software_array = []
        # get data
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            df_obj = pd.DataFrame.from_dict(json_obj['solar_radiation_annual_incident']['600']['Surface'])
            df_obj['software'] = json_obj['identifying_information']['software_name']
            software_array.append(json_obj['identifying_information']['software_name'])
            df = pd.concat([df, df_obj], axis=0)
        df = df.set_index('software')
        df = df.transpose()

        for row in range(len(df)):
            for col in range(len(df.columns)):
                try:
                    df.at[df.index[row], df.columns[col]]
                except (KeyError, ValueError):
                    df.at[df.index[row], df.columns[col]] = float('NaN')

        # calculate stats
        df_stats = pd.DataFrame()
        df_stats['min'] = df.iloc[:, 0:-1].min(axis=1)
        df_stats['max'] = df.iloc[:, 0:-1].max(axis=1)
        df_stats['mean'] = df.iloc[:, 0:-1].mean(axis=1)
        df_stats['(max-min)\n/mean*(%)'] = abs(df_stats['max'] - df_stats['min']).div(
            df_stats['mean'].where(df_stats['mean'] != 0, np.nan))

        # merge dataframes
        df_merged = pd.concat(
            [
                df.iloc[:, range(len(df.columns) - 1)],
                df_stats,
                df.iloc[:, range(len(df.columns) - 1, len(df.columns))]
            ],
            axis=1)

        index_dict = {
            'HORZ.': 'Horizontal',
            'NORTH': 'North',
            'EAST': 'East',
            'SOUTH': 'South',
            'WEST': 'West',
        }
        df_formatted_table = df_merged.rename(index=index_dict)
        df_formatted_table = df_formatted_table.reindex(['Horizontal', 'North', 'East', 'South', 'West'])
        df_formatted_table = df_formatted_table.reset_index(drop=False).rename(columns={'index': 'Cases'})
        df_formatted_table.fillna('', inplace=True)

        for col in df_formatted_table:
            if col in software_array:
                df_formatted_table[col] = df_formatted_table[col].apply(lambda x: int(round(x)) if x != '' else x)
        df_formatted_table['min'] = df_formatted_table['min'].apply(lambda x: int(round(x)) if x != '' else x)
        df_formatted_table['max'] = df_formatted_table['max'].apply(lambda x: int(round(x)) if x != '' else x)
        df_formatted_table['mean'] = df_formatted_table['mean'].apply(lambda x: int(round(x)) if x != '' else x)
        df_formatted_table['(max-min)\n/mean*(%)'] = df_formatted_table['(max-min)\n/mean*(%)'].apply(
            lambda x: '{0:.1f}%'.format(round(x * 100, 3)))

        # wrap text in table
        wrapped_col = {}
        for col in df_formatted_table:
            wrapped_col[col] = twp.fill(col, break_long_words=False, width=5)
        df_formatted_table.rename(columns=wrapped_col, inplace=True)

        # save results
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(24, 6))
        tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=1.2)

        # Set annotations
        header_stats = [tab.add_cell(-1, h, width=0.5, height=0.20) for h in range(7, 11)]
        header_stats[0].get_text().set_text('Statistics for Example Results')
        header_stats[0].PAD = 0.5
        header_stats[0].set_fontsize(16)
        header_stats[0].set_text_props(ha="left")
        header_stats[0].visible_edges = "open"
        header_stats[1].visible_edges = "open"
        header_stats[2].visible_edges = "open"
        header_stats[3].visible_edges = "open"
        plt.figtext(0.15, 0.08, "*ABS[(Max-Min)/(Mean of Example Simulation Results)", ha="left", fontsize=12)

        plt.suptitle(caption, fontsize=30, y=1.05)
        self._make_image_from_plt(figure_name)

        return fig, ax

    def render_section_5_2a_table_b8_14(
            self,
            output_value='solar_radiation_unshaded_annual_transmitted',
            figure_name='section_5_2_a_table_b8_14',
            caption='Table B8-14. Annual Transmitted Solar Radiation - Unshaded ($kWh/m^{2}$)'):
        """
        Create dataframe from class dataframe object for table 5-2A B8-14

        :return: pandas dataframe and output msg for general navigation.
        """

        df = pd.DataFrame()
        software_array = []
        # get data
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            df_obj = pd.DataFrame.from_dict(json_obj[output_value])
            df_obj['software'] = json_obj['identifying_information']['software_name']
            software_array.append(json_obj['identifying_information']['software_name'])
            df = pd.concat([df, df_obj], axis=0)
        df = df.set_index('software')
        df = df.transpose()

        for row in range(len(df)):
            for col in range(len(df.columns)):
                temp_dict = df.at[df.index[row], df.columns[col]]
                for k, v in temp_dict.items():
                    new_val = v['kWh/m2']
                try:
                    df.at[df.index[row], df.columns[col]] = new_val
                except (KeyError, ValueError):
                    df.at[df.index[row], df.columns[col]] = float('NaN')

        # calculate stats
        df_stats = pd.DataFrame()
        df_stats['min'] = df.iloc[:, 0:-1].min(axis=1)
        df_stats['max'] = df.iloc[:, 0:-1].max(axis=1)
        df_stats['mean'] = df.iloc[:, 0:-1].mean(axis=1)
        df_stats['(max-min)\n/mean*(%)'] = abs(df_stats['max'] - df_stats['min']).div(
            df_stats['mean'].where(df_stats['mean'] != 0, np.nan))

        # merge dataframes
        df_merged = pd.concat(
            [
                df.iloc[:, range(len(df.columns) - 1)],
                df_stats,
                df.iloc[:, range(len(df.columns) - 1, len(df.columns))]
            ],
            axis=1)

        index_dict = {
            '600': '600 South',
            '610': '610 South',
            '620': '620 West',
            '630': '630 West',
            '660': '660 South',
            '670': '670 South',
        }
        df_formatted_table = df_merged.rename(index=index_dict)
        df_formatted_table = df_formatted_table.reset_index(drop=False).rename(columns={'index': 'Cases'})

        df_formatted_table.fillna('', inplace=True)

        for col in df_formatted_table:
            if col in software_array:
                df_formatted_table[col] = df_formatted_table[col].apply(lambda x: int(round(x)) if x != '' else x)
        df_formatted_table['min'] = df_formatted_table['min'].apply(lambda x: int(round(x)) if x != '' else x)
        df_formatted_table['max'] = df_formatted_table['max'].apply(lambda x: int(round(x)) if x != '' else x)
        df_formatted_table['mean'] = df_formatted_table['mean'].apply(lambda x: int(round(x)) if x != '' else x)
        df_formatted_table['(max-min)\n/mean*(%)'] = df_formatted_table['(max-min)\n/mean*(%)'].apply(
            lambda x: '{0:.1f}%'.format(round(x * 100, 3)))

        # wrap text in table
        wrapped_col = {}
        for col in df_formatted_table:
            wrapped_col[col] = twp.fill(col, break_long_words=False, width=5)
        df_formatted_table.rename(columns=wrapped_col, inplace=True)

        # save results
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(24, 6))
        tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=1.2)

        # Set annotations
        header_stats = [tab.add_cell(-1, h, width=0.5, height=0.20) for h in range(7, 11)]
        header_stats[0].get_text().set_text('Statistics for Example Results')
        header_stats[0].PAD = 0.5
        header_stats[0].set_fontsize(16)
        header_stats[0].set_text_props(ha="left")
        header_stats[0].visible_edges = "open"
        header_stats[1].visible_edges = "open"
        header_stats[2].visible_edges = "open"
        header_stats[3].visible_edges = "open"
        plt.figtext(0.15, 0.08, "*ABS[(Max-Min)/(Mean of Example Simulation Results)", ha="left", fontsize=12)

        plt.suptitle(caption, fontsize=30, y=1.05)
        self._make_image_from_plt(figure_name)

        return fig, ax

    def render_section_5_2a_table_b8_15(self):
        """
        Create dataframe from class dataframe object for table 5-2A B8-15

        :return: pandas dataframe and output msg for general navigation.
        """
        fig, ax = self.render_section_5_2a_table_b8_14(
            output_value='solar_radiation_shaded_annual_transmitted',
            figure_name='section_5_2_a_table_b8_15',
            caption='Table B8.15 Annual Transmitted Solar Radiation - Shaded ($kWh/m^{2}$)'
        )

        return fig, ax

    # def render_section_5_2a_table_b8_16(
    #         self,
    #         figure_name='section_5_2_a_table_b8_16',
    #         caption='Table B8-16. Sky Temperature Output, Case 600'):
    #     """
    #     Create dataframe from class dataframe object for table 5-2A B8-16
    #
    #     :return: pandas dataframe and output msg for general navigation.
    #     """
    #     dfo = pd.DataFrame()
    #     software_arrayo = []
    #     # get data
    #     data = {}
    #     data1 = {}
    #     data2 = {}
    #     data3 = {}
    #     for idx, (tst, json_obj) in enumerate(self.json_data.items()):
    #         data = json_obj['solar_radiation_annual_incident']['600']['Surface']
    #         df_obj = pd.DataFrame.from_dict(data)
    #         df_obj['software'] = json_obj['identifying_information']['software_name']
    #         software_arrayo.append(json_obj['identifying_information']['software_name'])
    #         dfo = pd.concat([dfo, df_obj], axis=0)
    #     dfo = dfo.set_index('software')
    #     dfo = dfo.transpose()
    #
    #     df = pd.DataFrame()
    #     df_float = pd.DataFrame()
    #     software_array = []
    #     # get data
    #     data = {}
    #     data_float = {}
    #     for idx, (tst, json_obj) in enumerate(self.json_data.items()):
    #         # data['Average'] = {'C': json_obj['sky_temperature_output']['600']['Average']['C']}
    #         # data['Minimum'] = {'C': json_obj['sky_temperature_output']['600']['Minimum']['C']}
    #         # data['Maximum'] = {'C': json_obj['sky_temperature_output']['600']['Maximum']['C']}
    #         # data['Average'] = {'C': json_obj['sky_temperature_output']['600']['Average']['C'],
    #         # 'Hour': '', 'Month': ''}
    #         # data['Minimum'] = {'C': json_obj['sky_temperature_output']['600']['Minimum']['C'],
    #         # 'Hour': json_obj['sky_temperature_output']['600']['Minimum']['Hour'],
    #         # 'Month': json_obj['sky_temperature_output']['600']['Minimum']['Month']}
    #         # data['Maximum'] = {'C': json_obj['sky_temperature_output']['600']['Maximum']['C'],
    #         # 'Hour': json_obj['sky_temperature_output']['600']['Maximum']['Hour'],
    #         # 'Month': json_obj['sky_temperature_output']['600']['Maximum']['Month']}
    #         # data['Average'] = [{'C': json_obj['sky_temperature_output']['600']['Average']['C']}, {'Hour': ''},
    #         #  {'Month': ''}]
    #         # data['Minimum'] = [{'C': json_obj['sky_temperature_output']['600']['Minimum']['C']},
    #         # {'Hour': json_obj['sky_temperature_output']['600']['Minimum']['Hour']},
    #         # {'Month': json_obj['sky_temperature_output']['600']['Minimum']['Month']}]
    #         # data['Maximum'] = [{'C': json_obj['sky_temperature_output']['600']['Maximum']['C']},
    #         # {'Hour': json_obj['sky_temperature_output']['600']['Maximum']['Hour']},
    #         # {'Month': json_obj['sky_temperature_output']['600']['Maximum']['Month']}]
    #         data['Average'] = {'C': json_obj['sky_temperature_output']['600']['Average']['C']}
    #         data['Minimum'] = {'C': json_obj['sky_temperature_output']['600']['Minimum']['C']}
    #         data['Maximum'] = {'C': json_obj['sky_temperature_output']['600']['Maximum']['C']}
    #         data_float['Average'] = {'C': json_obj['sky_temperature_output']['600']['Average']['C']}
    #         data_float['Minimum'] = {'C': json_obj['sky_temperature_output']['600']['Minimum']['C']}
    #         data_float['Maximum'] = {'C': json_obj['sky_temperature_output']['600']['Maximum']['C']}
    #         df_obj = pd.DataFrame.from_dict(data)
    #         df_float_obj = pd.DataFrame.from_dict(data_float)
    #         df_obj['software'] = json_obj['identifying_information']['software_name']
    #         df_float_obj['software'] = json_obj['identifying_information']['software_name']
    #         software_array.append(json_obj['identifying_information']['software_name'])
    #         df = pd.concat([df, df_obj], axis=0)
    #         df_float = pd.concat([df_float, df_float_obj], axis=0)
    #
    #         data1['Average'] = {'Month': ''}
    #         data1['Minimum'] = {'Month': json_obj['sky_temperature_output']['600']['Minimum']['Month']}
    #         data1['Maximum'] = {'Month': json_obj['sky_temperature_output']['600']['Maximum']['Month']}
    #         df_obj = pd.DataFrame.from_dict(data1)
    #         df_obj['software'] = json_obj['identifying_information']['software_name']
    #         software_array.append(json_obj['identifying_information']['software_name'])
    #         df = pd.concat([df, df_obj], axis=0)
    #
    #         data2['Average'] = {'Day': ''}
    #         data2['Minimum'] = {'Day': json_obj['sky_temperature_output']['600']['Minimum']['Day']}
    #         data2['Maximum'] = {'Day': json_obj['sky_temperature_output']['600']['Maximum']['Day']}
    #         df_obj = pd.DataFrame.from_dict(data2)
    #         df_obj['software'] = json_obj['identifying_information']['software_name']
    #         software_array.append(json_obj['identifying_information']['software_name'])
    #         df = pd.concat([df, df_obj], axis=0)
    #
    #         data3['Average'] = {'Hour': ''}
    #         data3['Minimum'] = {'Hour': json_obj['sky_temperature_output']['600']['Minimum']['Hour']}
    #         data3['Maximum'] = {'Hour': json_obj['sky_temperature_output']['600']['Maximum']['Hour']}
    #         df_obj = pd.DataFrame.from_dict(data3)
    #         df_obj['software'] = json_obj['identifying_information']['software_name']
    #         software_array.append(json_obj['identifying_information']['software_name'])
    #         df = pd.concat([df, df_obj], axis=0)
    #
    #     df = df.set_index('software')
    #     df_float = df_float.set_index('software')
    #     df = df.transpose()
    #     df_float = df_float.transpose()
    #
    #     # for row in range(len(df)):
    #     #    for col in range(len(df.columns)):
    #     #        try:
    #     #            df.at[df.index[row], df.columns[col]]
    #     #        except (KeyError, ValueError):
    #     #            df.at[df.index[row], df.columns[col]] = float('NaN')
    #
    #     for row in range(len(df_float)):
    #         for col in range(len(df_float.columns)):
    #             try:
    #                 df_float.at[df_float.index[row], df_float.columns[col]]
    #             except (KeyError, ValueError):
    #                 df_float.at[df_float.index[row], df_float.columns[col]] = float('NaN')
    #
    #     # calculate stats
    #     df_stats = pd.DataFrame()
    #     df_stats['min'] = df_float.iloc[:, 0:-1].min(axis=1)
    #     df_stats['max'] = df_float.iloc[:, 0:-1].max(axis=1)
    #     df_stats['mean'] = df_float.iloc[:, 0:-1].mean(axis=1)
    #     df_stats['(max-min)\n/mean*(%)'] = abs(df_stats['max'] - df_stats['min']).div(
    #         df_stats['mean'].where(df_stats['mean'] != 0, np.nan))
    #
    #     # merge dataframes
    #     df_merged = pd.concat(
    #         [
    #             df.iloc[:, range(len(df.columns) - 1)],
    #             df_stats,
    #             df.iloc[:, range(len(df.columns) - 1, len(df.columns))]
    #         ],
    #         axis=1)
    #
    #     index_dict = {
    #         'Average': 'Average Annual Hourly Integrated',
    #         'Minimum': 'Minimum Annual Hourly Integrated',
    #         'Maximum': 'Maximum Annual Hourly Integrated'
    #     }
    #     df_formatted_table = df_merged.rename(index=index_dict)
    #     df_formatted_table = df_formatted_table.reindex(
    #         ['Average Annual Hourly Integrated', 'Minimum Annual Hourly Integrated',
    #          'Maximum Annual Hourly Integrated'])
    #     df_formatted_table = df_formatted_table.reset_index(drop=False).rename(columns={'index': 'Cases'})
    #     df_formatted_table.fillna('', inplace=True)
    #
    #     # for col in df_formatted_table:
    #     #     if col in software_array:
    #     #         df_formatted_table[col] = df_formatted_table[col].apply(lambda x: round(x, 1) if x != '' else x)
    #     df_formatted_table['min'] = df_formatted_table['min'].apply(lambda x: round(x, 1) if x != '' else x)
    #     df_formatted_table['max'] = df_formatted_table['max'].apply(lambda x: round(x, 1) if x != '' else x)
    #     df_formatted_table['mean'] = df_formatted_table['mean'].apply(lambda x: round(x, 1) if x != '' else x)
    #     df_formatted_table['(max-min)\n/mean*(%)'] = df_formatted_table['(max-min)\n/mean*(%)'].apply(
    #         lambda x: '{0:.1f}%'.format(round(x * 100, 3)))
    #
    #     # wrap text in table
    #     wrapped_col = {}
    #     for col in df_formatted_table:
    #         wrapped_col[col] = twp.fill(col, break_long_words=False, width=5)
    #     df_formatted_table.rename(columns=wrapped_col, inplace=True)
    #
    #     # save results
    #     fig, ax = plt.subplots(
    #         nrows=1,
    #         ncols=1,
    #         figsize=(24, 6))
    #     tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=1.2)
    #
    #     # Set annotations
    #     header_stats = [tab.add_cell(-1, h, width=0.5, height=0.20) for h in range(7, 11)]
    #     header_stats[0].get_text().set_text('Statistics for Example Results')
    #     header_stats[0].PAD = 0.5
    #     header_stats[0].set_fontsize(16)
    #     header_stats[0].set_text_props(ha="left")
    #     header_stats[0].visible_edges = "open"
    #     header_stats[1].visible_edges = "open"
    #     header_stats[2].visible_edges = "open"
    #     header_stats[3].visible_edges = "open"
    #     plt.figtext(0.15, 0.08, "*ABS[(Max-Min)/(Mean of Example Simulation Results)", ha="left", fontsize=12)
    #
    #     # Set annotations (additional for extra group headers similar to Table 5-2A B8-3 and B8-4)
    #     for idx, hdr in zip([1, 5, 9, 13, 17, 21, 29], software_array):  # program_list_short):
    #         header = [tab.add_cell(-1, idx, width=0.5, height=0.35), ]
    #         header[0].get_text().set_text(hdr.upper())
    #         header[0].PAD = 0.1
    #         header[0].set_fontsize(16)
    #         header[0].set_text_props(ha="left")
    #         header[0].visible_edges = "open"
    #         # if idx != 29:
    #         #    ax.axvline(x=(4.5 + idx / 2) / 22.3, color='black', linewidth=4, zorder=3)
    #         # else:
    #         #    ax.axvline(x=20 / 22, color='black', linewidth=4, zorder=3)
    #     # ax.axvline(x=(4.5 + 25 / 2) / 22.3, color='black', linewidth=4, zorder=3)
    #
    #     plt.suptitle(caption, fontsize=30, y=1.05)
    #     self._make_image_from_plt(figure_name)
    #
    #     """
    #     Calling using similar way to B8-4 to create dataframe from class dataframe object for table 5-2A B8-16
    #
    #         :return: pandas dataframe and output msg for general navigation.
    #     """
    #     # # An alternative way to create the table using a structure similiar to 5-2A B8-3 and B8-4
    #     # fig, ax = self.render_section_5_2a_table_b8_16_new_call(
    #     #     output_values=('case600_sky_TC', 'case600_sky_month', 'cae600_sky_day', 'case600_sky_hour'),
    #     #     figure_name='section_5_2_a_table_b8_16',
    #     #     caption='Table B8-16. Sky Temperature Output, Case 600'
    #     # )
    #
    #     return fig, ax

    # def render_section_5_2a_table_b8_16b(
    #         self,
    #         output_values=('peak_heating_kW', 'peak_heating_month', 'peak_heating_day', 'peak_heating_hour'),
    #         figure_name='section_5_2_a_table_b8_16b',
    #         caption='Table B8-16b. Sky Temperature Output, Case 600'):
    #     """
    #     Create dataframe from class dataframe object for table 5-2A B8-16
    #
    #     :return: pandas dataframe and output msg for general navigation.
    #     """
    #     # get and format dataframe into required shape
    #     df = self.df_data['conditioned_zone_loads_non_free_float'].loc[:, [i in output_values for i in self.df_data[
    #         'conditioned_zone_loads_non_free_float'].columns.get_level_values(1)]]
    #     df_formatted_table = df.unstack() \
    #         .reset_index() \
    #         .rename(columns={0: 'val', 'level_0': 'case'}) \
    #         .pivot(index=['case', 'level_1'], columns=['program_name', ], values=['val', ]) \
    #         .unstack() \
    #         .reset_index()
    #     df_formatted_table.columns = df_formatted_table.columns.droplevel(level=0)
    #     kw_cols = [i for i in df_formatted_table.columns if 'kw' in i[1].lower()]
    #     df_formatted_table[kw_cols] = df_formatted_table[kw_cols].apply(
    #         lambda x: pd.to_numeric(x, errors='ignore').round(2), axis=0)
    #     df_stats = pd.DataFrame()
    #     base_kw_cols = [i for i in kw_cols if i[0] in self.baseline_model_names]
    #     df_stats['min'] = df_formatted_table[base_kw_cols].min(axis=1).round(2)
    #     df_stats['max'] = df_formatted_table[base_kw_cols].max(axis=1).round(2)
    #     df_stats['mean'] = df_formatted_table[base_kw_cols].mean(axis=1).round(2)
    #     df_stats['(max - min)\n/ mean %'] = df_formatted_table[base_kw_cols].min(axis=1).round(2)
    #     int_cols = [i for i in df_formatted_table.columns if any(j for j in ['day', 'hour'] if j in i[1].lower())]
    #     df_formatted_table[int_cols] = df_formatted_table[int_cols].fillna(0).astype(int)
    #     df_formatted_table = df_formatted_table.reindex(columns=['', *output_values], level=1)
    #     df_formatted_table = df_formatted_table.fillna('')
    #     column_formatting_names = {
    #         'peak_heating_kW': 'kW',
    #         'peak_heating_month': 'Mo',
    #         'peak_heating_day': 'Day',
    #         'peak_heating_hour': 'Hr',
    #         'peak_cooling_kW': 'kW',
    #         'peak_cooling_month': 'Mo',
    #         'peak_cooling_day': 'Day',
    #         'peak_cooling_hour': 'Hr'
    #     }
    #     # reorder columns so test program is last
    #     column_names = [i for i in df_formatted_table.columns if i[0] != self.model_name] + [
    #         i for i in df_formatted_table.columns if i[0] == self.model_name]
    #     df_formatted_table = df_formatted_table[column_names]
    #     column_names = [column_formatting_names[i[1]] if i[1] in column_formatting_names.keys() else i[1]
    #                     for i in list(df_formatted_table.columns)]
    #     column_names[0] = 'cases'
    #     # make list of program names
    #     program_list = sorted(
    #         set(
    #             [i[0] for i in df_formatted_table.columns if i[0]]),
    #         key=[i[0] for i in df_formatted_table.columns].index)
    #     program_rgx = re.compile(r'(^[a-zA-Z]+)')
    #     program_list_short = []
    #     for p in program_list:
    #         result = program_rgx.search(p)
    #         if result:
    #             program_list_short.append(result.group(1))
    #     df_formatted_table.columns = column_names
    #     df_formatted_name_table = df_formatted_table[['cases']] \
    #         .merge(
    #         self.case_detailed_df,
    #         how='left',
    #         left_on=['cases', ],
    #         right_index=True) \
    #         .sort_values(['case_order']) \
    #         .drop(['cases', 'case_order'], axis=1) \
    #         .rename(columns={
    #         'case_name': 'Case'})
    #     # set fig size
    #     fig, ax = plt.subplots(
    #         nrows=1,
    #         ncols=1,
    #         figsize=(30, 20))
    #     df_formatted_table = pd.concat(
    #         [
    #             df_formatted_name_table,
    #             df_formatted_table.drop(columns=['cases', ]).iloc[:, range(len(df_formatted_table.columns) - 5)],
    #             df_stats,
    #             df_formatted_table.drop(columns=['cases', ]).iloc[
    #             :,
    #             range(len(df_formatted_table.columns) - 5, len(df_formatted_table.columns) - 1)]],
    #         axis=1)
    #     gridvalues = [
    #         ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a',
    #          'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', ],
    #         ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a',
    #          'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', ]
    #     ]
    #     tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=5, cell_text=[gridvalues])
    #
    #     # this experiment did not do anything
    #     #        cell_data = [['a','b'],['c','d']]
    #     #        tab.cellText = cell_data
    #
    #     cell_dict = tab.get_celld()
    #     for w, i in zip([0.6, 0.6, 0.6, 1.5], [25, 26, 27, 28]):
    #         for j in range(df_formatted_table.shape[0] + 1):
    #             cell_dict[(j, i)].set_width(w)
    #             cell_dict[(j, i)].set_text_props(ha="center")
    #     # Set annotations
    #     for idx, hdr in zip([1, 5, 9, 13, 17, 21, 29], program_list_short):
    #         header = [tab.add_cell(-1, idx, width=0.5, height=0.35), ]
    #         header[0].get_text().set_text(hdr.upper())
    #         header[0].PAD = 0.1
    #         header[0].set_fontsize(16)
    #         header[0].set_text_props(ha="left")
    #         header[0].visible_edges = "open"
    #         if idx != 29:
    #             ax.axvline(x=(4.5 + idx / 2) / 22.3, color='black', linewidth=4, zorder=3)
    #         else:
    #             ax.axvline(x=20 / 22, color='black', linewidth=4, zorder=3)
    #     ax.axvline(x=(4.5 + 25 / 2) / 22.3, color='black', linewidth=4, zorder=3)
    #     # save the result
    #     plt.suptitle(caption, fontsize=30)
    #     self._make_image_from_plt(figure_name)
    #     plt.subplots_adjust(top=0.92)
    #     return fig, ax

    # def render_section_5_2a_table_b8_16_new_call(
    #         self,
    #         output_values=('case600_skyTC', 'case600_sky_month', 'case600_sky_day', 'case600_sky_hour'),
    #         figure_name='section_5_2_a_table_b8_16',
    #         caption='Table B8-16. Sky Temperature Output, Case 600'):
    #     """
    #     Create dataframe from class dataframe object for table 5-2A B8-16
    #     an alternative way based on similar style from Table 5-2A B8-3 and B8-4
    #
    #     :return: pandas dataframe and output msg for general navigation.
    #     """
    #     # get and format dataframe into required shape
    #     df = self.df_data['solar_radiation_annual_incident'].loc[:, [i in output_values for i in
    #                                                                  self.df_data['solar_radiation_annual_incident']
    #                                                                  .columns.get_level_values(1)]]
    #     df_formatted_table = df.unstack() \
    #         .reset_index() \
    #         .rename(columns={0: 'val', 'level_0': 'case'}) \
    #         .pivot(index=['case', 'level_1'], columns=['program_name', ], values=['val', ]) \
    #         .unstack() \
    #         .reset_index()
    #     df_formatted_table.columns = df_formatted_table.columns.droplevel(level=0)
    #     kw_cols = [i for i in df_formatted_table.columns if 'C' in i[1].lower()]
    #     df_formatted_table[kw_cols] = df_formatted_table[kw_cols].apply(
    #         lambda x: pd.to_numeric(x, errors='ignore').round(2), axis=0)
    #     df_stats = pd.DataFrame()
    #     sky_TC_cols = [i for i in kw_cols if i[0] in self.baseline_model_names]
    #     df_stats['min'] = df_formatted_table[sky_TC_cols].min(axis=1).round(2)
    #     df_stats['max'] = df_formatted_table[sky_TC_cols].max(axis=1).round(2)
    #     df_stats['mean'] = df_formatted_table[sky_TC_cols].mean(axis=1).round(2)
    #     df_stats['(max - min)\n/ mean %'] = df_formatted_table[sky_TC_cols].min(axis=1).round(2)
    #     int_cols = [i for i in df_formatted_table.columns if any(j for j in ['day', 'hour'] if j in i[1].lower())]
    #     df_formatted_table[int_cols] = df_formatted_table[int_cols].fillna(0).astype(int)
    #     df_formatted_table = df_formatted_table.reindex(columns=['', *output_values], level=1)
    #     df_formatted_table = df_formatted_table.fillna('')
    #     column_formatting_names = {
    #         'case600_sky_TC': 'C',
    #         'case600_sky_month': 'Mo',
    #         'case600_sky_day': 'Day',
    #         'case600_sky_hour': 'Hr',
    #     }
    #     # reorder columns so test program is last
    #     column_names = [i for i in df_formatted_table.columns if i[0] != self.model_name] + [
    #         i for i in df_formatted_table.columns if i[0] == self.model_name]
    #     df_formatted_table = df_formatted_table[column_names]
    #     column_names = [column_formatting_names[i[1]] if i[1] in column_formatting_names.keys() else i[1]
    #                     for i in list(df_formatted_table.columns)]
    #     column_names[0] = 'cases'
    #     # make list of program names
    #     program_list = sorted(
    #         set(
    #             [i[0] for i in df_formatted_table.columns if i[0]]),
    #         key=[i[0] for i in df_formatted_table.columns].index)
    #     program_rgx = re.compile(r'(^[a-zA-Z]+)')
    #     program_list_short = []
    #     for p in program_list:
    #         result = program_rgx.search(p)
    #         if result:
    #             program_list_short.append(result.group(1))
    #     df_formatted_table.columns = column_names
    #     df_formatted_name_table = df_formatted_table[['cases']] \
    #         .merge(
    #         self.case_detailed_df,
    #         how='left',
    #         left_on=['cases', ],
    #         right_index=True) \
    #         .sort_values(['case_order']) \
    #         .drop(['cases', 'case_order'], axis=1) \
    #         .rename(columns={
    #         'case_name': 'Case'})
    #     # set fig size
    #     fig, ax = plt.subplots(
    #         nrows=1,
    #         ncols=1,
    #         figsize=(30, 20))
    #     df_formatted_table = pd.concat(
    #         [df_formatted_name_table,
    #          df_formatted_table.drop(columns=['cases', ]).iloc[:, range(len(df_formatted_table.columns) - 5)],
    #          df_stats,
    #          df_formatted_table.drop(columns=['cases', ]).iloc[:, range(len(df_formatted_table.columns) - 5,
    #                                                                     len(df_formatted_table.columns) - 1)]],
    #         axis=1)
    #     tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=5)
    #     cell_dict = tab.get_celld()
    #     for w, i in zip([0.6, 0.6, 0.6, 1.5], [25, 26, 27, 28]):
    #         for j in range(df_formatted_table.shape[0] + 1):
    #             cell_dict[(j, i)].set_width(w)
    #             cell_dict[(j, i)].set_text_props(ha="center")
    #     # Set annotations
    #     for idx, hdr in zip([1, 5, 9, 13, 17, 21, 29], program_list_short):
    #         header = [tab.add_cell(-1, idx, width=0.5, height=0.35), ]
    #         header[0].get_text().set_text(hdr.upper())
    #         header[0].PAD = 0.1
    #         header[0].set_fontsize(16)
    #         header[0].set_text_props(ha="left")
    #         header[0].visible_edges = "open"
    #         if idx != 29:
    #             ax.axvline(x=(4.5 + idx / 2) / 22.3, color='black', linewidth=4, zorder=3)
    #         else:
    #             ax.axvline(x=20 / 22, color='black', linewidth=4, zorder=3)
    #     ax.axvline(x=(4.5 + 25 / 2) / 22.3, color='black', linewidth=4, zorder=3)
    #     # save the result
    #     plt.suptitle(caption, fontsize=30)
    #     self._make_image_from_plt(figure_name)
    #     plt.subplots_adjust(top=0.92)
    #     return fig, ax

    def render_section_5_2a_figure_b8_1(self):
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
                        json_obj['solar_radiation_annual_incident']['600']['Surface'][surface]['kWh/m2'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-1.  Annual Incident Solar Radiation',
            xticklabels=['600 ' + i for i in surfaces],
            ylabel='Diffuse + Direct ($kWh/m^2$)',
            image_name='section_5_2_a_figure_b8_1')
        return fig, ax

    def render_section_5_2a_figure_b8_2(self):
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
                        json_obj['solar_radiation_unshaded_annual_transmitted'][case]['Surface'].items()
                    if tmp_d:
                        tmp_data.append(tmp_d['kWh/m2'])
                    else:
                        tmp_data.append(float('NaN'))
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-2.  Annual Transmitted Solar Radiation - Unshaded',
            xticklabels=[
                '600 SOUTH', '620 WEST', '660 SOUTH, Low-E', '670 SOUTH, Single Pane'],
            ylabel='Diffuse + Direct ($kWh/m^2$)',
            image_name='section_5_2_a_figure_b8_2')
        return fig, ax

    def render_section_5_2a_figure_b8_3(self):
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
                        json_obj['solar_radiation_shaded_annual_transmitted'][case]['Surface'].items()
                    if tmp_d:
                        tmp_data.append(tmp_d['kWh/m2'])
                    else:
                        tmp_data.append(float('NaN'))
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-3.  Annual Transmitted Solar Radiation - Shaded',
            xticklabels=[
                '610 SOUTH', '630 WEST'],
            ylabel='Diffuse + Direct ($kWh/m^2$)',
            image_name='section_5_2_a_figure_b8_3')
        return fig, ax

    def render_section_5_2a_figure_b8_4(self):
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
                        json_obj['solar_radiation_unshaded_annual_transmitted'][case]['Surface'].items()
                    for sub_surfaces in json_obj['solar_radiation_annual_incident'].values():
                        for sub_surface, vals in sub_surfaces['Surface'].items():
                            if sub_surface.lower() == surface.lower():
                                incident_surface_value = vals['kWh/m2']
                                tmp_data.append(unshaded_d['kWh/m2'] / incident_surface_value)
                                break
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
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
            image_name='section_5_2_a_figure_b8_4')
        return fig, ax

    def render_section_5_2a_figure_b8_5(self):
        """
        Render Section 5 2A Figure B8-5 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(1 - (json_obj['solar_radiation_shaded_annual_transmitted']['610']['Surface']['South']
                                     ['kWh/m2'] / json_obj['solar_radiation_unshaded_annual_transmitted']['600']
                                     ['Surface']['South']['kWh/m2']))
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(1 - (json_obj['solar_radiation_shaded_annual_transmitted']['630']['Surface']['West']
                                     ['kWh/m2'] / json_obj['solar_radiation_unshaded_annual_transmitted']['620']
                                     ['Surface']['West']['kWh/m2']))
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
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
            image_name='section_5_2_a_figure_b8_5')
        return fig, ax

    def render_section_5_2a_figure_b8_6(self):
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
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(json_obj['sky_temperature_output']['600']['Minimum']['C'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(json_obj['sky_temperature_output']['600']['Maximum']['C'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
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
            image_name='section_5_2_a_figure_b8_6')
        return fig, ax

    def render_section_5_2a_figure_b8_7(self):
        """
        Render Section 5 2A Figure B8-7 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['395', '430', '600', '610', '620', '630', '640', '650']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_heating_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-7.  Basic: Low Mass Annual Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Annual Heating Load (MWh)',
            image_name='section_5_2_a_figure_b8_7')
        return fig, ax

    def render_section_5_2a_figure_b8_8(self):
        """
        Render Section 5 2A Figure B8-8 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['395', '430', '600', '610', '620', '630', '640', '650']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_cooling_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-8.  Basic: Low Mass Annual Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel='Annual Heating Load (MWh)',
            image_name='section_5_2_a_figure_b8_8')
        return fig, ax

    def render_section_5_2a_figure_b8_9(self):
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
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_heating_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-9.  Basic: Low Mass Peak Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel='Peak Heating Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_9')
        return fig, ax

    def render_section_5_2a_figure_b8_10(self):
        """
        Render Section 5 2A Figure B8-10 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['395', '430', '600', '610', '620', '630', '640', '650']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_cooling_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-10.  Basic: Low Mass Peak Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel='Peak Cooling Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_10')
        return fig, ax

    def render_section_5_2a_figure_b8_11(self):
        """
        Render Section 5 2A Figure B8-11 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['800', '900', '910', '920', '930', '940', '950', '960']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_heating_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-11.  Basic: High Mass Annual Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel='Annual Heating Load (MWh)',
            image_name='section_5_2_a_figure_b8_11')
        return fig, ax

    def render_section_5_2a_figure_b8_12(self):
        """
        Render Section 5 2A Figure B8-12 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['800', '900', '910', '920', '930', '940', '950', '960']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_cooling_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-12.  Basic: High Mass Annual Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel='Annual Cooling Load (MWh)',
            image_name='section_5_2_a_figure_b8_12')
        return fig, ax

    def render_section_5_2a_figure_b8_13(self):
        """
        Render Section 5 2A Figure B8-13 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['800', '900', '910', '920', '930', '940', '950', '960']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_heating_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-13.  Basic: High Mass Peak Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel='Peak Heating Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_13')
        return fig, ax

    def render_section_5_2a_figure_b8_14(self):
        """
        Render Section 5 2A Figure B8-14 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['800', '900', '910', '920', '930', '940', '950', '960']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_cooling_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-14.  Basic: High Mass Peak Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel='Peak Cooling Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_14')
        return fig, ax

    def render_section_5_2a_figure_b8_15(self):
        """
        Render Section 5 2A Figure B8-15 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['430'][
                        'annual_heating_MWh']
                )
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['430'][
                        'annual_cooling_MWh']
                )
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['800'][
                        'annual_heating_MWh']
                )
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['800'][
                        'annual_cooling_MWh']
                )
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-15. Basic and In-Depth:\nSouth Window (Delta)\n'
                  'Annual Heating and Sensible Cooling',
            xticklabels=[
                '600-430 Low Mass, Heating S. Window', '600-430 Low Mass, Cooling S. Window',
                '900-800 High Mass, Heating S. Window', '900-800 High Mass, Cooling S. Window'
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_15')
        return fig, ax

    def render_section_5_2a_figure_b8_16(self):
        """
        Render Section 5 2A Figure B8-16 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['430'][
                        'peak_heating_kW']
                )
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['430'][
                        'peak_cooling_kW']
                )
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['800'][
                        'peak_heating_kW']
                )
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['800'][
                        'peak_cooling_kW']
                )
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-16. Basic and In-Depth:\nSouth Window (Delta)\n'
                  'Peak Heating and Sensible Cooling',
            xticklabels=[
                '600-430 Low Mass, Heating S. Window', '600-430 Low Mass, Cooling S. Window',
                '900-800 High Mass, Heating S. Window', '900-800 High Mass, Cooling S. Window'
            ],
            ylabel='Load Difference (kWh/h)',
            y_plot_pad=0.3,
            y_min=-1,
            image_name='section_5_2_a_figure_b8_16')
        return fig, ax

    def render_section_5_2a_figure_b8_17(self):
        """
        Render Section 5 2A Figure B8-17 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_lists = [[] for _ in range(3)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['610'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['610'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['910'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['910'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # Mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['620'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['620'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['920'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['920'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            # Right chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['630'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['620'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['630'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['620'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['930'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['920'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['930'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['920'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[2].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-17. Basic: Window Shading and Orientation (Delta) '
                  'Annual Heating and Sensible Cooling',
            sub_titles=[
                'South Shading',
                'East/West',
                'E/W Shading'
            ],
            xticklabels=[
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
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_17')
        return fig, ax

    def render_section_5_2a_figure_b8_18(self):
        """
        Render Section 5 2A Figure B8-18 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_lists = [[] for _ in range(3)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['610'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['610'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['910'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['910'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # Mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['620'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['620'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['920'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['920'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            # Right chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['630'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['620'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['630'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['620'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['930'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['920'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['930'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['920'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[2].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-18. Basic: Window Shading and Orientation (Delta) '
                  'Peak Heating and Sensible Cooling',
            sub_titles=[
                'South Shading',
                'East/West',
                'E/W Shading'
            ],
            xticklabels=[
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
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            y_max=1.0,
            image_name='section_5_2_a_figure_b8_18')
        return fig, ax

    def render_section_5_2a_figure_b8_19(self):
        """
        Render Section 5 2A Figure B8-19 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_lists = [[] for _ in range(3)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['640'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['940'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['650'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['950'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            # right chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['960'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['960'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[2].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-19. Basic: Thermostat Setback, Vent Cooling, and Sunspace (Delta) '
                  'Annual Heating and Sensible Cooling',
            sub_titles=[
                'Tstat Setback',
                'Vent Cooling',
                'Sunspace'
            ],
            xticklabels=[
                [
                    '640-600 Low Mass, Tstat Setback, Heating',
                    '940-900 High Mass, Tstat Setback, Heating'
                ],
                [
                    '650-600 Low Mass, Night Vent, Cooling',
                    '950-900 High Mass, Night Vent, Cooling'
                ],
                [
                    '960-900 High Mass, Sunspace, Heating',
                    '960-900 High Mass, Sunspace, Cooling'
                ]
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            y_max=1.5,
            image_name='section_5_2_a_figure_b8_19')
        return fig, ax

    def render_section_5_2a_figure_b8_20(self):
        """
        Render Section 5 2A Figure B8-20 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_lists = [[] for _ in range(3)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['640'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['940'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['650'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['950'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            # right chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['960'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['960'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[2].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-20. Basic: Thermostat Setback, Vent Cooling, and Sunspace (Delta) Peak '
                  'Heating and Sensible Cooling',
            sub_titles=[
                'Tstat Setback',
                'Vent Cooling',
                'Sunspace'
            ],
            xticklabels=[
                [
                    '640-600 Low Mass, Tstat Setback, Heating',
                    '940-900 High Mass, Tstat Setback, Heating'
                ],
                [
                    '650-600 Low Mass, Night Vent, Cooling',
                    '950-900 High Mass, Night Vent, Cooling'
                ],
                [
                    '960-900 High Mass, Sunspace, Heating',
                    '960-900 High Mass, Sunspace, Cooling'
                ]
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_20')
        return fig, ax

    def render_section_5_2a_figure_b8_21(self):
        """
        Render Section 5 2A Figure B8-21 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['800'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['430'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['800'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['430'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['940'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['640'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['950'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['650'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-21. Basic and In-Depth: Mass Effect (Delta) Annual Heating and Sensible Cooling',
            xticklabels=[
                '800-430 Mass, Heating w/ High Cond. Wall El.',
                '800-430 Mass, Cooling w/ High Cond. Wall El.',
                '900-600 Mass, Heating',
                '900-600 Mass, Cooling',
                '940-640 Mass, Heating w/ Heating Setback',
                '950-650 Mass, Cooling w/ Cooling Setback'
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_21')
        return fig, ax

    def render_section_5_2a_figure_b8_22(self):
        """
        Render Section 5 2A Figure B8-22 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['800'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['430'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['800'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['430'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['940'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['640'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['950'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['650'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-22. Basic and In-Depth: Mass Effect (Delta) Peak Heating and Sensible Cooling',
            xticklabels=[
                '800-430 Mass, Heating w/ High Cond. Wall El.',
                '800-430 Mass, Cooling w/ High Cond. Wall El.',
                '900-600 Mass, Heating',
                '900-600 Mass, Cooling',
                '940-640 Mass, Heating w/ Heating Setback',
                '950-650 Mass, Cooling w/ Cooling Setback'
            ],
            ylabel='Load Difference (KWh/h)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_22')
        return fig, ax

    def render_section_5_2a_figure_b8_23(self):
        """
        Render Section 5 2A Figure B8-23 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['600', '660', '670', '680', '685', '695', '900', '980', '985', '995']
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['660']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['670']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['680']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['685']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['695']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995']['annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-23. Basic: Cases 660 to 695 and 980 to 995 Annual Heating',
            sub_titles=[
                'Low Mass',
                'High Mass'
            ],
            xticklabels=[
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases][:6],
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases][6:]
            ],
            ylabel='Annual Heating Load (MWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_23')
        return fig, ax

    def render_section_5_2a_figure_b8_24(self):
        """
        Render Section 5 2A Figure B8-24 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['600', '660', '670', '680', '685', '695', '900', '980', '985', '995']
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['660']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['670']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['680']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['685']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['695']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995']['annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-24. Basic: Cases 660 to 695 and 980 to 995 Annual Cooling',
            sub_titles=[
                'Low Mass',
                'High Mass'
            ],
            xticklabels=[
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases][:6],
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases][6:]
            ],
            ylabel='Annual Heating Load (MWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_24')
        return fig, ax

    def render_section_5_2a_figure_b8_25(self):
        """
        Render Section 5 2A Figure B8-25 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['600', '660', '670', '680', '685', '695', '900', '980', '985', '995']
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['660']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['670']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['680']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['685']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['695']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995']['peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-25. Basic: Cases 660 to 695 and 980 to 995 Peak Heating',
            sub_titles=[
                'Low Mass',
                'High Mass'
            ],
            xticklabels=[
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases][:6],
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases][6:]
            ],
            ylabel='Peak Heating Load (kWh/h)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_25')
        return fig, ax

    def render_section_5_2a_figure_b8_26(self):
        """
        Render Section 5 2A Figure B8-26 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['600', '660', '670', '680', '685', '695', '900', '980', '985', '995']
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['660']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['670']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['680']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['685']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['695']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['900']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995']['peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-26. Basic: Cases 660 to 695 and 980 to 995 Peak Cooling',
            sub_titles=[
                'Low Mass',
                'High Mass'
            ],
            xticklabels=[
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases][:6],
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases][6:]
            ],
            ylabel='Peak Cooling Load (kWh/h)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_26')
        return fig, ax

    def render_section_5_2a_figure_b8_27(self):
        """
        Render Section 5 2A Figure B8-27 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['660'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['660'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['670'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['670'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-27. Basic: Window Types (Delta) Annual Heating and Sensible Cooling',
            sub_titles=[
                'Low-E',
                'Single Pane'
            ],
            xticklabels=[
                [
                    '660-600 Low-E Windows, Heating', '660-600 Low-E Windows, Cooling'
                ],
                [
                    '670-600 Single-Pane Windows, Heating', '670-600 Single-Pane Windows, Cooling'
                ]
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            y_max=2.0,
            image_name='section_5_2_a_figure_b8_27')
        return fig, ax

    def render_section_5_2a_figure_b8_28(self):
        """
        Render Section 5 2A Figure B8-28 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['660'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['660'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['670'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['670'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-28. Basic: Window Types (Delta) Peak Heating and Sensible Cooling',
            sub_titles=[
                'Low-E',
                'Single Pane'
            ],
            xticklabels=[
                [
                    '660-600 Low-E Windows, Heating', '660-600 Low-E Windows, Cooling'
                ],
                [
                    '670-600 Single-Pane Windows, Heating', '670-600 Single-Pane Windows, Cooling'
                ]
            ],
            ylabel='Load Difference (kWh/h)',
            y_plot_pad=0.3,
            y_max=2.0,
            image_name='section_5_2_a_figure_b8_28')
        return fig, ax

    def render_section_5_2a_figure_b8_29(self):
        """
        Render Section 5 2A Figure B8-29 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['680'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['680'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['695'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['695'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-29. Basic: Window Types (Delta) Annual Heating and Sensible Cooling',
            sub_titles=[
                'Low Mass',
                'High Mass'
            ],
            xticklabels=[
                [
                    '680-600\n600 with\nGreater\nInsulation,\nHeating',
                    '680-600\n600 with\nGreater\nInsulation,\nCooling',
                    '685-600\n600 with\n20/20\nTstat,\nHeating',
                    '685-600\n600 with\n20/20\nTstat,\nCooling',
                    '695-685\n685 with\nGreater\nInsulation,\nHeating',
                    '695-685\n685 with\nGreater\nInsulation,\nCooling',
                ],
                [
                    '980-900\n900 with\nGreater\nInsulation,\nHeating',
                    '980-900\n900 with\nGreater\nInsulation,\nCooling',
                    '985-900\n900 with\n20/20\nTstat,\nHeating',
                    '985-900\n900 with\n20/20\nTstat,\nCooling',
                    '995-985\n985 with\nGreater\nInsulation,\nHeating',
                    '995-985\n985 with\nGreater\nInsulation,\nCooling',
                ],
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            y_min=-3.0,
            image_name='section_5_2_a_figure_b8_29')
        return fig, ax

    def render_section_5_2a_figure_b8_30(self):
        """
        Render Section 5 2A Figure B8-30 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['680'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['680'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['600'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['695'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['695'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['900'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-30. Basic: Window Types (Delta) Peak Heating and Sensible Cooling',
            sub_titles=[
                'Low Mass',
                'High Mass'
            ],
            xticklabels=[
                [
                    '680-600\n600 with\nGreater\nInsulation,\nHeating',
                    '680-600\n600 with\nGreater\nInsulation,\nCooling',
                    '685-600\n600 with\n20/20\nTstat,\nHeating',
                    '685-600\n600 with\n20/20\nTstat,\nCooling',
                    '695-685\n685 with\nGreater\nInsulation,\nHeating',
                    '695-685\n685 with\nGreater\nInsulation,\nCooling',
                ],
                [
                    '980-900\n900 with\nGreater\nInsulation,\nHeating',
                    '980-900\n900 with\nGreater\nInsulation,\nCooling',
                    '985-900\n900 with\n20/20\nTstat,\nHeating',
                    '985-900\n900 with\n20/20\nTstat,\nCooling',
                    '995-985\n985 with\nGreater\nInsulation,\nHeating',
                    '995-985\n985 with\nGreater\nInsulation,\nCooling',
                ],
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            y_max=1.25,
            y_min=-1.5,
            image_name='section_5_2_a_figure_b8_30')
        return fig, ax

    def render_section_5_2a_figure_b8_31(self):
        """
        Render Section 5 2A Figure B8-31 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['680'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['680'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995'][
                        'annual_heating_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['695'][
                        'annual_heating_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995'][
                        'annual_cooling_MWh'] - json_obj['conditioned_zone_loads_non_free_float']['695'][
                        'annual_cooling_MWh'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-31. Basic and In-Depth: Insulation, Mass Effect (Delta) '
                  'Annual Heating and Sensible Cooling',
            xticklabels=[
                '980-680 Mass, Heating with Greater Insulation',
                '980-680 Mass, Cooling with Greater Insulation',
                '985-685 Mass, Heating with 20/20 Tstat',
                '985-685 Mass, Cooling with 20/20 Tstat',
                '995-695 Mass, Heating w/ > Insul. & 20/20 Tstat',
                '995-695 Mass, Cooling w/ > Insul. & 20/20 Tstat',
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.1,
            image_name='section_5_2_a_figure_b8_31')
        return fig, ax

    def render_section_5_2a_figure_b8_32(self):
        """
        Render Section 5 2A Figure B8-32 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['680'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['980'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['680'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['985'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['685'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995'][
                        'peak_heating_kW'] - json_obj['conditioned_zone_loads_non_free_float']['695'][
                        'peak_heating_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['995'][
                        'peak_cooling_kW'] - json_obj['conditioned_zone_loads_non_free_float']['695'][
                        'peak_cooling_kW'])
            except (KeyError, TypeError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-32. Basic and In-Depth: Insulation, Mass Effect (Delta) Peak Heating and Sensible Cooling',
            xticklabels=[
                '980-680 Mass, Heating with Greater Insulation',
                '980-680 Mass, Cooling with Greater Insulation',
                '985-685 Mass, Heating with 20/20 Tstat',
                '985-685 Mass, Cooling with 20/20 Tstat',
                '995-695 Mass, Heating w/ > Insul. & 20/20 Tstat',
                '995-695 Mass, Cooling w/ > Insul. & 20/20 Tstat',
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.1,
            image_name='section_5_2_a_figure_b8_32')
        return fig, ax

    def render_section_5_2a_figure_b8_33(self):
        """
        Render Section 5 2A Figure B8-33 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        cases = ['600FF', '900FF', '650FF', '950FF', '680FF', '980FF', '960']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(json_obj['free_float_case_zone_temperatures'][case]['average_temperature'])
                except (KeyError, TypeError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-33. Basic: Average Hourly Annual Temperature Free-Float Cases',
            xticklabels=[self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel=r'Average Temperature ($^\circ$C)',
            y_plot_pad=0.1,
            image_name='section_5_2_a_figure_b8_33')
        return fig, ax

    def render_section_5_2a_figure_b8_34(self):
        """
        Render Section 5 2A Figure B8-34 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        cases = ['600FF', '900FF', '650FF', '950FF', '680FF', '980FF', '960']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(json_obj['free_float_case_zone_temperatures'][case]['maximum_temperature'])
                except (KeyError, TypeError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-34. Basic: Maximum Hourly Annual Temperature Free-Float Cases',
            xticklabels=[self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel=r'Maximum Temperature ($^\circ$C)',
            y_plot_pad=0.1,
            image_name='section_5_2_a_figure_b8_34')
        return fig, ax

    def render_section_5_2a_figure_b8_35(self):
        """
        Render Section 5 2A Figure B8-35 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        cases = ['600FF', '900FF', '650FF', '950FF', '680FF', '980FF', '960']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(json_obj['free_float_case_zone_temperatures'][case]['minimum_temperature'])
                except (KeyError, TypeError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-35. Basic: Minimum Hourly Annual Temperature Free-Float Cases',
            xticklabels=[self.case_detailed_df.loc[i, 'case_name'] for i in cases],
            ylabel=r'Minimum Temperature ($^\circ$C)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_35')
        return fig, ax

    def render_section_5_2a_figure_b8_36(self):
        """
        Render Section 5 2A Figure B8-36 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['195', '200', '210', '215', '220', '230', '240', '250']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_heating_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-36. In-Depth: Low Mass Cases 195 to 250 Annual Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Annual Heating Load (MWh)',
            image_name='section_5_2_a_figure_b8_36')
        return fig, ax

    def render_section_5_2a_figure_b8_37(self):
        """
        Render Section 5 2A Figure B8-37 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['195', '200', '210', '215', '220', '230', '240', '250']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_cooling_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-37. In-Depth: Low Mass Cases 195 to 250 Annual Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Annual Cooling Load (MWh)',
            image_name='section_5_2_a_figure_b8_37')
        return fig, ax

    def render_section_5_2a_figure_b8_38(self):
        """
        Render Section 5 2A Figure B8-38 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['195', '200', '210', '215', '220', '230', '240', '250']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_heating_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-38. In-Depth: Low Mass Cases 195 to 250 Peak Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Peak Heating Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_38')
        return fig, ax

    def render_section_5_2a_figure_b8_39(self):
        """
        Render Section 5 2A Figure B8-39 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['195', '200', '210', '215', '220', '230', '240', '250']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_cooling_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-39. In-Depth: Low Mass Cases 195 to 250 Peak Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Peak Cooling Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_39')
        return fig, ax

    def render_section_5_2a_figure_b8_40(self):
        """
        Render Section 5 2A Figure B8-40 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['270', '280', '290', '300', '310', '320']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_heating_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-40. In-Depth: Low Mass Cases 270 to 320 Annual Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Annual Heating Load (MWh)',
            image_name='section_5_2_a_figure_b8_40')
        return fig, ax

    def render_section_5_2a_figure_b8_41(self):
        """
        Render Section 5 2A Figure B8-41 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['270', '280', '290', '300', '310', '320']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_cooling_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-41. In-Depth: Low Mass Cases 270 to 320 Annual Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Annual Cooling Load (MWh)',
            image_name='section_5_2_a_figure_b8_41')
        return fig, ax

    def render_section_5_2a_figure_b8_42(self):
        """
        Render Section 5 2A Figure B8-42 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['270', '280', '290', '300', '310', '320']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_heating_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-42. In-Depth: Low Mass Cases 270 to 320 Peak Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Peak Heating Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_42')
        return fig, ax

    def render_section_5_2a_figure_b8_43(self):
        """
        Render Section 5 2A Figure B8-43 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['270', '280', '290', '300', '310', '320']
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_cooling_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-43. In-Depth: Low Mass Cases 270 to 320 Peak Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Peak Cooling Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_43')
        return fig, ax

    def render_section_5_2a_figure_b8_44(self):
        """
        Render Section 5 2A Figure B8-44 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['200']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['195']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['200']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['195']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['210']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['200']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['210']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['200']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['220']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['215']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['220']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['215']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['215']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['200']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['220']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['210']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-44. In-Depth: Cases 195 to 220 (Delta) Annual Heating and Sensible Cooling',
            xticklabels=[
                '200-195\nHeating\nSurface\nConvection',
                '200-195\nCooling\nSurface\nConvection',
                '210-200\nHeating\nExt IR\n(Int IR "off")',
                '220-215\nCooling\nExt IR\n(Int IR "off")',
                '215-200\nHeating\nExt IR\n(Int IR "on")',
                '215-200\nCooling\nExt IR\n(Int IR "on")',
                '215-200\nHeating\nInt IR\n(Ext IR "off")',
                '220-210\nHeating\nInt IR\n(Ext IR "on")'
            ],
            ylabel='Load Difference (MWh)',
            y_min=-0.5,
            image_name='section_5_2_a_figure_b8_44')
        return fig, ax

    def render_section_5_2a_figure_b8_45(self):
        """
        Render Section 5 2A Figure B8-45 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['200']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['195']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['200']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['195']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['210']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['200']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['220']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['215']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['215']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['200']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['215']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['200']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['220']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['210']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['220']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['210']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-45. Cases 195 to 220 (Delta) Peak Heating and Sensible Cooling',
            xticklabels=[
                '200-195\nHeating\nSurface\nConvection',
                '200-195\nCooling\nSurface\nConvection',
                '210-200\nCooling\nExt IR\n(Int IR "off")',
                '220-215\nCooling\nExt IR\n(Int IR "on")',
                '215-200\nHeating\nInt IR\n(Ext IR "off")',
                '215-200\nCooling\nInt IR\n(Ext IR "off")',
                '220-210\nHeating\nInt IR\n(Ext IR "on")',
                '220-210\nCooling\nInt IR\n(Ext IR "on")'
            ],
            ylabel='Load Difference (kWh/h)',
            y_plot_pad=0.3,
            y_min=-0.5,
            image_name='section_5_2_a_figure_b8_45')
        return fig, ax

    def render_section_5_2a_figure_b8_46(self):
        """
        Render Section 5 2A Figure B8-46 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['230']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['230']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['240']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['240']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['250']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['250']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['270']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['270']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-46. Cases 220 to 270 (Delta) Annual Heating and Sensible Cooling',
            xticklabels=[
                '230-220\nHeating\nInfiltration',
                '230-220\nCooling\nInfiltration',
                '240-220\nHeating\nInternal\nGains',
                '240-220\nCooling\nInternal\nGains',
                '250-220\nHeating\nExt Solar\nAbsorptance',
                '250-220\nCooling\nExt Solar Abs.',
                '270-220\nHeating\nSouth\nWindows',
                '270-220\nCooling\nSouth\nWindows',
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            y_min=-0.5,
            image_name='section_5_2_a_figure_b8_46')
        return fig, ax

    def render_section_5_2a_figure_b8_47(self):
        """
        Render Section 5 2A Figure B8-47 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['230']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['230']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['240']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['240']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['250']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['270']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['220']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-47. Cases 220 to 270 (Delta) Peak Heating and Sensible Cooling',
            xticklabels=[
                '230-220\nHeating\nInfiltration',
                '230-220\nCooling\nInfiltration',
                '240-220\nHeating\nInternal\nGains',
                '240-220\nCooling\nInternal\nGains',
                '250-220\nHeating\nExt Solar\nAbsorptance',
                '270-220\nCooling\nSouth\nWindows',
            ],
            ylabel='Load Difference (kWh/h)',
            y_plot_pad=0.3,
            y_min=-1.0,
            image_name='section_5_2_a_figure_b8_47')
        return fig, ax

    def render_section_5_2a_figure_b8_48(self):
        """
        Render Section 5 2A Figure B8-48 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['280']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['320']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['320']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['290']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['300']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['310']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['300']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['310']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['300']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-48. Cases 270 to 320 (Delta) Annual Heating and Sensible Cooling',
            xticklabels=[
                '280-270\nCooling\nCavity Albedo',
                '320-270\nHeating\nThermostat',
                '320-270\nCooling\nThermostat',
                '290-270\nCooling\nCooling\nSouth Shading',
                '300-270\nCooling\nE&W Windows',
                '310-300\nHeating\nE&W Shading',
                '310-300\nCooling\nE&W Shading'
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            y_max=1.0,
            image_name='section_5_2_a_figure_b8_48')
        return fig, ax

    def render_section_5_2a_figure_b8_49(self):
        """
        Render Section 5 2A Figure B8-49 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['280']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['320']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['290']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['300']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['270']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['310']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['300']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-49. In-Depth: Cases 270 to 320 (Delta) Peak Sensible Cooling',
            xticklabels=[
                '280-270\nCavity Albedo',
                '320-270\nThermostat',
                '290-270\nSouth Shading',
                '300-270\nE&W Windows',
                '310-300\nE&W Shading'
            ],
            ylabel='Load Difference (kWh/h)',
            image_name='section_5_2_a_figure_b8_49')
        return fig, ax

    def render_section_5_2a_figure_b8_50(self):
        """
        Render Section 5 2A Figure B8-50 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        cases = ['395', '400', '410', '420', '430', '440', '800', '810']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_heating_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-50. In-Depth: Cases 395 to 440, 800, 810 Annual Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Load Difference (MWh)',
            image_name='section_5_2_a_figure_b8_50')
        return fig, ax

    def render_section_5_2a_figure_b8_51(self):
        """
        Render Section 5 2A Figure B8-51 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        cases = ['395', '400', '410', '420', '430', '440', '800', '810']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_cooling_MWh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-51. In-Depth: Cases 395 to 440, 800, 810 Annual Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Load Difference (MWh)',
            image_name='section_5_2_a_figure_b8_51')
        return fig, ax

    def render_section_5_2a_figure_b8_52(self):
        """
        Render Section 5 2A Figure B8-52 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        cases = ['395', '400', '410', '420', '430', '440', '800', '810']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_heating_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-52. In-Depth: Cases 395 to 440, 800, 810 Peak Heating',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Load Difference (kWh/h)',
            image_name='section_5_2_a_figure_b8_52')
        return fig, ax

    def render_section_5_2a_figure_b8_53(self):
        """
        Render Section 5 2A Figure B8-53 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        cases = ['395', '400', '410', '420', '430', '440', '800', '810']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_cooling_kW'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-53. In-Depth: Cases 395 to 440, 800, 810 Peak Sensible Cooling',
            xticklabels=[
                self.case_detailed_df.loc[i, 'case_name']
                for i in cases],
            ylabel='Load Difference (kWh/h)',
            image_name='section_5_2_a_figure_b8_53')
        return fig, ax

    def render_section_5_2a_figure_b8_54(self):
        """
        Render Section 5 2A Figure B8-54 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['400']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['395']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['410']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['400']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['420']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['410']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['430']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['420']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['430']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['420']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['430']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['430']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['440']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['440']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['810']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['900']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['810']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['900']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-54. In-Depth: Cases 395 to 600, 810 to 900 (Delta) Annual Heating and Sensible Cooling',
            xticklabels=[
                '400-395\nLow Mass,\nHeating\nSurf. Conv.\n& IR',
                '410-400\nLow Mass,\nHeating\nInfiltration',
                '420-410\nLow Mass,\nHeating\nInt. Gains',
                '430-420\nLow Mass,\nHeating\nExt. Solar\nAbs.',
                '430-420\nLow Mass,\nCooling\nExt. Solar\nAbs.',
                '600-430\nLow Mass,\nHeating\nS. Window',
                '600-430\nLow Mass,\nCooling\nS. Window',
                '440-600\nLow Mass,\nHeating\nCavity\nAlbedo',
                '440-600\nLow Mass,\nCooling\nCavity\nAlbedo',
                '810-900\nHigh Mass,\nHeating\nCavity\nAlbedo',
                '810-900\nHigh Mass,\nCooling\nCavity\nAlbedo',
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            y_min=-2.5,
            image_name='section_5_2_a_figure_b8_54')
        return fig, ax

    def render_section_5_2a_figure_b8_55(self):
        """
        Render Section 5 2A Figure B8-55 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['400']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['395']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['400']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['395']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['410']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['400']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['410']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['400']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['420']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['410']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['420']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['410']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['430']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['420']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['600']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['430']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['440']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['810']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['900']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-55. In-Depth: Cases 395 to 600, 810 to 900 (Delta) Peak Heating and Sensible Cooling',
            xticklabels=[
                '400-395\nLow Mass,\nHeating\nSurf. Conv.\n& IR',
                '400-395\nLow Mass,\nCooling\nSurf. Conv.\n& IR',
                '410-400\nLow Mass,\nHeating\nInfiltration',
                '410-400\nLow Mass,\nCooling\nInfiltration',
                '420-410\nLow Mass,\nHeating\nHeating\nInt. Gains',
                '420-410\nLow Mass,\nCooling\nHeating\nInt. Gains',
                '430-420\nLow Mass,\nCooling\nExt. Solar\nAbs.',
                '600-430\nLow Mass,\nCooling\nS. Window',
                '440-600\nLow Mass,\nCooling\nCavity\nAlbedo',
                '810-900\nHigh Mass,\nCooling\nCavity\nAlbedo',
            ],
            ylabel='Load Difference (kWh/h)',
            y_plot_pad=0.3,
            y_min=-2.5,
            image_name='section_5_2_a_figure_b8_55')
        return fig, ax

    def render_section_5_2a_figure_b8_56(self):
        """
        Render Section 5 2A Figure B8-56 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['600', '450', '460', '470']
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_heating_MWh'])
                except (KeyError, TypeError):
                    tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['annual_cooling_MWh'])
                except (KeyError, TypeError):
                    tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-56. In-Depth: Surface Heat Transfer\n'
                  'Cases 600, 450, 460, 470\n'
                  'Annual Heating and Sensible Cooling',
            sub_titles=[
                'Heating',
                'Cooling'
            ],
            xticklabels=[
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases],
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases]
            ],
            ylabel='Annual Heating Load (MWh)',
            image_name='section_5_2_a_figure_b8_56')
        return fig, ax

    def render_section_5_2a_figure_b8_57(self):
        """
        Render Section 5 2A Figure B8-57 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        cases = ['600', '450', '460', '470']
        data_lists = [[] for _ in range(2)]
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            # Left chart data
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_heating_kW'])
                except (KeyError, TypeError):
                    tmp_data.append(float('NaN'))
            data_lists[0].insert(idx, tmp_data)
            # mid chart data
            tmp_data = []
            for case in cases:
                try:
                    tmp_data.append(
                        json_obj['conditioned_zone_loads_non_free_float'][case]['peak_cooling_kW'])
                except (KeyError, TypeError):
                    tmp_data.append(float('NaN'))
            data_lists[1].insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_split_bar_plot(
            data=data_lists,
            programs=programs,
            title='Figure B8-57. In-Depth: Surface Heat Transfer\n'
                  'Cases 600, 450, 460, 470\n'
                  'Peak Heating and Sensible Cooling',
            sub_titles=[
                'Heating',
                'Cooling'
            ],
            xticklabels=[
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases],
                [self.case_detailed_df.loc[i, 'case_name'] for i in cases]
            ],
            ylabel='Peak Heating Load (kWh/h)',
            image_name='section_5_2_a_figure_b8_57')
        return fig, ax

    def render_section_5_2a_figure_b8_58(self):
        """
        Render Section 5 2A Figure B8-58 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['450']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['450']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['460']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['460']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['460']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['450']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['460']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['450']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['470']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['470']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['470']['annual_heating_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['450']['annual_heating_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['470']['annual_cooling_MWh'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['450']['annual_cooling_MWh'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-58. In-Depth: Surface Heat Transfer\n'
                  'Cases 450 to 600 (Delta)\n'
                  'Annual Heating and Sensible Cooling',
            xticklabels=[
                '450-600\nConst.\nInt. & Ext.\nSurf. Coeffs.,\nHeating',
                '450-600\nConst.\nInt. & Ext.\nSurf. Coeffs.,\nCooling',
                '460-600\nConst.\nInterior\nSurf. Coeffs.,\nHeating',
                '460-600\nConst.\nInterior\nSurf. Coeffs.,\nCooling',
                '460-450\nAuto\nExterior\nSurf. H.T.,\nHeating',
                '460-450\nAuto\nExterior\nSurf. H.T.,\nCooling',
                '470-600\nConst.\nExterior\nSurf. Coeffs.,\nHeating',
                '470-600\nConst.\nExterior\nSurf. Coeffs.,\nCooling',
                '470-450\nAuto\nInterior\nSurf. H.T.,\nHeating',
                '470-450\nAuto\nInterior\nSurf. H.T.,\nCooling',
            ],
            ylabel='Load Difference (MWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_58')
        return fig, ax

    def render_section_5_2a_figure_b8_59(self):
        """
        Render Section 5 2A Figure B8-59 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['450']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['450']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['460']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['460']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['460']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['450']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['460']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['450']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['470']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['470']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['600']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['470']['peak_heating_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['450']['peak_heating_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            try:
                tmp_data.append(
                    json_obj['conditioned_zone_loads_non_free_float']['470']['peak_cooling_kW'] - json_obj[
                        'conditioned_zone_loads_non_free_float']['450']['peak_cooling_kW'])
            except (KeyError, ValueError):
                tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-59. In-Depth: Surface Heat Transfer\n'
                  'Cases 450 to 600 (Delta)\n'
                  'Peak Heating and Sensible Cooling',
            xticklabels=[
                '450-600\nConst.\nInt. & Ext.\nSurf. Coeffs.,\nHeating',
                '450-600\nConst.\nInt. & Ext.\nSurf. Coeffs.,\nCooling',
                '460-600\nConst.\nInterior\nSurf. Coeffs.,\nHeating',
                '460-600\nConst.\nInterior\nSurf. Coeffs.,\nCooling',
                '460-450\nAuto\nExterior\nSurf. H.T.,\nHeating',
                '460-450\nAuto\nExterior\nSurf. H.T.,\nCooling',
                '470-600\nConst.\nExterior\nSurf. Coeffs.,\nHeating',
                '470-600\nConst.\nExterior\nSurf. Coeffs.,\nCooling',
                '470-450\nAuto\nInterior\nSurf. H.T.,\nHeating',
                '470-450\nAuto\nInterior\nSurf. H.T.,\nCooling',
            ],
            ylabel='Load Difference (kWh/h)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_59')
        return fig, ax

    def render_section_5_2a_figure_b8_m1(self):
        """
        Render Section 5 2A Figure B8-M1 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['600'][month]['total_heating_kwh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M1.\n'
                  'Monthly Heating\n'
                  'Case 600',
            xticklabels=months,
            ylabel='Monthly Heating Load (kWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m1')
        return fig, ax

    def render_section_5_2a_figure_b8_m2(self):
        """
        Render Section 5 2A Figure B8-M2 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['600'][month]['total_cooling_kwh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M2.\n'
                  'Monthly Sensible Cooling\n'
                  'Case 600',
            xticklabels=months,
            ylabel='Monthly Cooling Load (kWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m2')
        return fig, ax

    def render_section_5_2a_figure_b8_m3(self):
        """
        Render Section 5 2A Figure B8-M3 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['600'][month]['peak_heating_kw'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M3.\n'
                  'Monthly Peak Heating\n'
                  'Case 600',
            xticklabels=months,
            ylabel='Monthly Peak Heating Load (kW)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m3')
        return fig, ax

    def render_section_5_2a_figure_b8_m4(self):
        """
        Render Section 5 2A Figure B8-M4 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['600'][month]['peak_cooling_kw'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M4.\n'
                  'Monthly Peak Cooling\n'
                  'Case 600',
            xticklabels=months,
            ylabel='Monthly Peak Cooling Load (kW)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m4')
        return fig, ax

    def render_section_5_2a_figure_b8_m5(self):
        """
        Render Section 5 2A Figure B8-M5 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['900'][month]['total_heating_kwh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M5.\n'
                  'Monthly Heating\n'
                  'Case 900',
            xticklabels=months,
            ylabel='Monthly Heating Load (kWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m5')
        return fig, ax

    def render_section_5_2a_figure_b8_m6(self):
        """
        Render Section 5 2A Figure B8-M6 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['900'][month]['total_cooling_kwh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M6.\n'
                  'Monthly Heating\n'
                  'Case 900',
            xticklabels=months,
            ylabel='Monthly Cooling Load (kWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m6')
        return fig, ax

    def render_section_5_2a_figure_b8_m7(self):
        """
        Render Section 5 2A Figure B8-M7 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['900'][month]['peak_heating_kw'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M7.\n'
                  'Monthly Peak Heating\n'
                  'Case 900',
            xticklabels=months,
            ylabel='Monthly Peak Heating Load (kW)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m7')
        return fig, ax

    def render_section_5_2a_figure_b8_m8(self):
        """
        Render Section 5 2A Figure B8-M8 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['900'][month]['peak_cooling_kw'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M8.\n'
                  'Monthly Peak Sensible Cooling\n'
                  'Case 900',
            xticklabels=months,
            ylabel='Monthly Peak Cooling Load (kW)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m8')
        return fig, ax

    def render_section_5_2a_figure_b8_m9(self):
        """
        Render Section 5 2A Figure B8-M9 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['600'][month][
                            'total_heating_kwh'] - json_obj['monthly_conditioned_zone_loads'][
                            '900'][month]['total_heating_kwh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M9.\n'
                  'Monthly Heating Sensitivity (Delta)\n'
                  'Case 600-900',
            xticklabels=months,
            ylabel='Monthly Heating Load (kWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m9')
        return fig, ax

    def render_section_5_2a_figure_b8_m10(self):
        """
        Render Section 5 2A Figure B8-M10 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['600'][month][
                            'total_cooling_kwh'] - json_obj['monthly_conditioned_zone_loads'][
                            '900'][month]['total_cooling_kwh'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M10.\n'
                  'Monthly Cooling Sensitivity (Delta)\n'
                  'Case 600-900',
            xticklabels=months,
            ylabel='Monthly Cooling Load (kWh)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m10')
        return fig, ax

    def render_section_5_2a_figure_b8_m11(self):
        """
        Render Section 5 2A Figure B8-M11 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['600'][month][
                            'peak_heating_kw'] - json_obj['monthly_conditioned_zone_loads'][
                            '900'][month]['peak_heating_kw'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M11.\n'
                  'Monthly Peak Heating Sensitivity (Delta)\n'
                  'Case 600-900',
            xticklabels=months,
            ylabel='Monthly Peak Heating Load (kW)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m11')
        return fig, ax

    def render_section_5_2a_figure_b8_m12(self):
        """
        Render Section 5 2A Figure B8-M12 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        programs = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for month in months:
                try:
                    tmp_data.append(
                        json_obj['monthly_conditioned_zone_loads']['600'][month][
                            'peak_cooling_kw'] - json_obj['monthly_conditioned_zone_loads'][
                            '900'][month]['peak_cooling_kw'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        fig, ax = self._create_bar_plot(
            data=data,
            programs=programs,
            title='Figure B8-M12.\n'
                  'Monthly Peak Cooling Sensitivity (Delta)\n'
                  'Case 600-900',
            xticklabels=months,
            ylabel='Monthly Peak Cooling Load (kW)',
            y_plot_pad=0.3,
            image_name='section_5_2_a_figure_b8_m12')
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
            try:
                data_obj = json_obj['hourly_annual_zone_temperature_bin_data']['900FF']['temperature_bin_c']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='number_of_hours')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H1. Case 900FF Annual Hourly Zone Air Temperature Frequency',
            xlabel=r'Hour of Day',
            ylabel='Number of Occurrences',
            image_name='section_5_2_a_figure_b8_h1',
            x_min=-5,
            x_max=55,
            x_tick_spacing=5,
            annotations=[
                {
                    'text': r'Hourly Occurrences for Each 1 $^\circ$C Bin',
                    'xy': (-4, 410),
                    'fontsize': 12}
            ])
        ax.annotate(r'Hourly Occurrences for Each 1 $^\circ$C Bin', (0, 450), fontsize=12)
        return fig, ax

    def render_section_5_2a_figure_b8_h2(self):
        """
        Render Section 5 2A Figure B8-H2 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        specific_days = ['may_4', 'july_14']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for specific_day in specific_days:
                try:
                    data_obj = json_obj['specific_day_hourly_output']['600']['incident_solar_radiation'][specific_day][
                        'horizontal']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='Whm/m2')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H2. Case 600\n'
                  'Cloudy & Clear Day Hourly Incident Solar\n'
                  'Horizontal (Upward) Facing Surface',
            xlabel=r'Hour of Day',
            ylabel='Incident Solar Radiation (Wh/m2)',
            image_name='section_5_2_a_figure_b8_h2',
            annotations=[
                {
                    'text': r'Clear Day (July 14)',
                    'xy': (17, 900),
                    'fontsize': 18},
                {
                    'text': r'Cloudy Day (May 4)',
                    'xy': (10, 100),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h3(self):
        """
        Render Section 5 2A Figure B8-H3 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        specific_days = ['may_4', 'july_14']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for specific_day in specific_days:
                try:
                    data_obj = json_obj['specific_day_hourly_output']['600']['incident_solar_radiation'][specific_day][
                        'south']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='Whm/m2')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H3. Case 600\n'
                  'Cloudy & Clear Day Hourly Incident Solar\n'
                  'South Facing Surface',
            xlabel=r'Hour of Day',
            ylabel='Incident Solar Radiation (Wh/m2)',
            image_name='section_5_2_a_figure_b8_h3',
            annotations=[
                {
                    'text': r'Clear Day (July 14)',
                    'xy': (17, 360),
                    'fontsize': 18},
                {
                    'text': r'Cloudy Day (May 4)',
                    'xy': (10, 60),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h4(self):
        """
        Render Section 5 2A Figure B8-H4 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        specific_days = ['may_4', 'july_14']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for specific_day in specific_days:
                try:
                    data_obj = json_obj['specific_day_hourly_output']['600']['incident_solar_radiation'][specific_day][
                        'west']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='Whm/m2')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H4. Case 600\n'
                  'Cloudy & Clear Day Hourly Incident Solar\n'
                  'West Facing Surface',
            xlabel=r'Hour of Day',
            ylabel='Incident Solar Radiation (Wh/m2)',
            image_name='section_5_2_a_figure_b8_h4',
            annotations=[
                {
                    'text': r'Clear Day (July 14)',
                    'xy': (19.5, 750),
                    'fontsize': 18},
                {
                    'text': r'Cloudy Day (May 4)',
                    'xy': (11, 50),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h5(self):
        """
        Render Section 5 2A Figure B8-H5 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        cases = ['600', '660', '670']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for case in cases:
                try:
                    data_obj = json_obj['specific_day_hourly_output'][case]['transmitted_total_solar_radiation'][
                        'feb_1']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='Whm/m2')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            x_min=5,
            x_max=20,
            title='Figure B8-H5. Case 600, 660, 670\n'
                  'Hourly Transmitted Solar, Clear/Cold Day (Feb 1)\n'
                  'Double-Pane, Low-E, Single-Pane Windows',
            xlabel=r'Hour of Day',
            ylabel='Transmitted Solar Radiation (Wh/m2)',
            image_name='section_5_2_a_figure_b8_h5',
            annotations=[
                {
                    'text': r'Low-E (Case 600)',
                    'xy': (11, 210),
                    'fontsize': 18},
                {
                    'text': r'Double-Pane (Case 660)',
                    'xy': (11, 510),
                    'fontsize': 18},
                {
                    'text': r'Single-Pane (Case 670)',
                    'xy': (11, 820),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h6(self):
        """
        Render Section 5 2A Figure B8-H6 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output']['600']['transmitted_total_solar_radiation'][
                    'may_4']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='Whm/m2')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            x_min=4,
            x_max=21,
            title='Figure B8-H6. Case 600\n'
                  'Hourly Transmitted Solar, Cloudy Day (May 4)\n'
                  'Double-Pane Windows',
            xlabel=r'Hour of Day',
            ylabel='Transmitted Solar Radiation (Wh/m2)',
            image_name='section_5_2_a_figure_b8_h6',
            annotations=[
                {
                    'text': 'Double-Pane\n(Case 600)',
                    'xy': (11.5, 70),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h7(self):
        """
        Render Section 5 2A Figure B8-H7 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        cases = ['660', '670']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for case in cases:
                try:
                    data_obj = json_obj['specific_day_hourly_output'][case]['transmitted_total_solar_radiation'][
                        'may_4']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='Whm/m2')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            x_min=4,
            x_max=21,
            title='Figure B8-H7. Case 660, 670\n'
                  'Hourly Transmitted Solar, Cloudy Day (May 4)\n'
                  'Low-E and Single-Pane Windows',
            xlabel=r'Hour of Day',
            ylabel='Transmitted Solar Radiation (Wh/m2)',
            image_name='section_5_2_a_figure_b8_h7',
            annotations=[
                {
                    'text': 'Low-E (Case 660)',
                    'xy': (11.5, 30),
                    'fontsize': 18},
                {
                    'text': 'Single-Pane (Case 670)',
                    'xy': (8, 130),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h8(self):
        """
        Render Section 5 2A Figure B8-H8 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output']['600']['transmitted_total_solar_radiation'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='Whm/m2')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            x_min=2,
            x_max=22,
            title='Figure B8-H8. Case 600\n'
                  'Hourly Transmitted Solar, Clear/Hot Day (July 14)\n'
                  'Double-Pane Windows',
            xlabel=r'Hour of Day',
            ylabel='Transmitted Solar Radiation (Wh/m2)',
            image_name='section_5_2_a_figure_b8_h8',
            annotations=[
                {
                    'text': 'Double-Pane\n(Case 600)',
                    'xy': (11.5, 140),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h9(self):
        """
        Render Section 5 2A Figure B8-H9 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        cases = ['660', '670']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for case in cases:
                try:
                    data_obj = json_obj['specific_day_hourly_output'][case]['transmitted_total_solar_radiation'][
                        'july_14']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='Whm/m2')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            x_min=2,
            x_max=22,
            title='Figure B8-H9. Case 660, 670\n'
                  'Hourly Transmitted Solar, Clear/Hot Day (July 14)\n'
                  'Double-Pane Windows',
            xlabel=r'Hour of Day',
            ylabel='Transmitted Solar Radiation (Wh/m2)',
            image_name='section_5_2_a_figure_b8_h9',
            annotations=[
                {
                    'text': 'Single-Pane (Case 670)',
                    'xy': (7, 310),
                    'fontsize': 18},
                {
                    'text': 'Low-E (Case 660)',
                    'xy': (11, 60),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h10(self):
        """
        Render Section 5 2A Figure B8-H10 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        specific_days = ['feb_1', 'may_4']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for specific_day in specific_days:
                try:
                    data_obj = json_obj['specific_day_hourly_output']['600'][
                        'sky_temperature'][specific_day]['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='C')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H10.\n'
                  'Hourly Sky Temperatures\n'
                  'Case 600: Clear/Cold, Cloudy Days',
            xlabel=r'Hour of Day',
            ylabel=r'Temperature ($^\circ$C)',
            image_name='section_5_2_a_figure_b8_h10',
            annotations=[
                {
                    'text': 'Cloudy (May 4)',
                    'xy': (2, -5),
                    'fontsize': 18},
                {
                    'text': 'Clear/Cold (Feb 1)',
                    'xy': (13, -28),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h11(self):
        """
        Render Section 5 2A Figure B8-H11 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        specific_days = ['feb_1', 'july_14']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for specific_day in specific_days:
                try:
                    data_obj = json_obj['specific_day_hourly_output']['600'][
                        'sky_temperature'][specific_day]['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='C')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H11.\n'
                  'Hourly Sky Temperatures\n'
                  'Case 600: Clear/Cold, Clear/Hot Days',
            ylabel=r'Temperature ($^\circ$C)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h11',
            annotations=[
                {
                    'text': 'Clear/Hot (July 14)',
                    'xy': (2, -8),
                    'fontsize': 18},
                {
                    'text': 'Clear/Cold (Feb 1)',
                    'xy': (13, -28),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h12(self):
        """
        Render Section 5 2A Figure B8-H12 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        cases = ['600FF', '900FF']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for case in cases:
                try:
                    data_obj = json_obj['specific_day_hourly_output_free_float_zone_temperatures'][case][
                        'feb_1']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='C')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H12.\n'
                  'Hourly Free-Float Temperatures\n'
                  'Clear Cold Day (Feb 1), Cases 600FF and 900FF',
            ylabel=r'Temperature ($^\circ$C)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h12',
            annotations=[
                {
                    'text': 'Case 600FF',
                    'xy': (10, 0),
                    'fontsize': 18},
                {
                    'text': 'Case 900FF',
                    'xy': (4, 18),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h13(self):
        """
        Render Section 5 2A Figure B8-H13 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        cases = ['650FF', '950FF']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for case in cases:
                try:
                    data_obj = json_obj['specific_day_hourly_output_free_float_zone_temperatures'][case][
                        'july_14']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='C')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H13.\n'
                  'Hourly Free-Float Temperatures\n'
                  'Clear Hot Day (Jul 14), Cases 650FF and 950FF',
            ylabel=r'Temperature ($^\circ$C)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h13',
            annotations=[
                {
                    'text': 'Case 650FF',
                    'xy': (19.5, 43),
                    'fontsize': 18},
                {
                    'text': 'Case 950FF',
                    'xy': (15, 26),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h14(self):
        """
        Render Section 5 2A Figure B8-H14 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        cases = ['680FF', '980FF']
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            for case in cases:
                try:
                    data_obj = json_obj['specific_day_hourly_output_free_float_zone_temperatures'][case][
                        'feb_1']['hour']
                    key_list, val_list = self._order_dictionary_with_numeric_keys(
                        input_d=data_obj,
                        val_key='C')
                    data_x.append(key_list)
                    data_y.append(val_list)
                    programs.append(json_obj['identifying_information']['software_name'])
                except (TypeError, KeyError):
                    data_x.append([])
                    data_y.append([])
                    programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H14.\n'
                  'Hourly Free-Float Temperatures\n'
                  'Clear Cold Day (Feb 1), Cases 680FF and 980FF',
            ylabel=r'Temperature ($^\circ$C)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h14',
            annotations=[
                {
                    'text': 'Case 680FF',
                    'xy': (11, 55),
                    'fontsize': 18},
                {
                    'text': 'Case 980FF',
                    'xy': (15, 25),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h15(self):
        """
        Render Section 5 2A Figure B8-H15 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['600'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H15. Hourly Loads\n'
                  'Clear Cold Day, Cases 600 (Low Mass, Double-Clear Window)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h15',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 1.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h16(self):
        """
        Render Section 5 2A Figure B8-H16 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['600'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H16. Hourly Loads\n'
                  'Clear Hot Day, Cases 600 (Low Mass, Double-Clear Window)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h16',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h17(self):
        """
        Render Section 5 2A Figure B8-H17 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['640'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H17. Hourly Loads\n'
                  'Clear Cold Day, Cases 640 (Low Mass, Night Setback)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h17',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 2.5),
                    'fontsize': 18},
                {
                    'text': 'Tstat ramp-up specified for Hour 8 (0700 to 0800)',
                    'xy': (1.5, -3.5),
                    'fontsize': 16}])
        return fig, ax

    def render_section_5_2a_figure_b8_h18(self):
        """
        Render Section 5 2A Figure B8-H18 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['640'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='C')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H18.\n'
                  'Hourly Conditioned Zone Temperatures\n'
                  'Clear Cold Day, Case 640',
            ylabel=r'Temperature ($^\circ$C)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h18',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (12, 17),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h19(self):
        """
        Render Section 5 2A Figure B8-H19 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['940'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H19. Hourly Loads\n'
                  'Clear Cold Day, Cases 940 (High Mass, Night Setback)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h19',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (13, 1.7),
                    'fontsize': 18},
                {
                    'text': 'Tstat ramp-up specified for Hour 8 (0700 to 0800)',
                    'xy': (13, 1.3),
                    'fontsize': 16}])
        return fig, ax

    def render_section_5_2a_figure_b8_h20(self):
        """
        Render Section 5 2A Figure B8-H20 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['940'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='C')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H20.\n'
                  'Hourly Conditioned Zone Temperatures\n'
                  'Clear Cold Day, Case 940',
            ylabel=r'Temperature ($^\circ$C)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h20',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (13, 17),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h21(self):
        """
        Render Section 5 2A Figure B8-H21 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['660'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H21. Hourly Loads\n'
                  'Clear Cold Day, Cases 660 (Low-E Window)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h21',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h22(self):
        """
        Render Section 5 2A Figure B8-H22 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['660'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H22. Hourly Loads\n'
                  'Clear Hot Day, Cases 660 (Low-E Window)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h22',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h23(self):
        """
        Render Section 5 2A Figure B8-H23 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['670'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H23. Hourly Loads\n'
                  'Clear Cold Day, Cases 670 (Single-Pane Window)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h23',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 2.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h24(self):
        """
        Render Section 5 2A Figure B8-H24 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['670'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H24. Hourly Loads\n'
                  'Clear Hot Day, Cases 670 (Single-Pane Window)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h24',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h25(self):
        """
        Render Section 5 2A Figure B8-H25 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['680'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H25. Hourly Loads\n'
                  'Clear Cold Day, Cases 680 (Insulation)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h25',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 0.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h26(self):
        """
        Render Section 5 2A Figure B8-H26 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['680'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H26. Hourly Loads\n'
                  'Clear Hot Day, Cases 680 (Insulation)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h26',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h27(self):
        """
        Render Section 5 2A Figure B8-H27 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['685'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H27. Hourly Loads\n'
                  'Clear Cold Day, Cases 685 (20/20 Tstat)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h27',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 1.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h28(self):
        """
        Render Section 5 2A Figure B8-H28 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['685'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H28. Hourly Loads\n'
                  'Clear Hot Day, Cases 685 (20/20 Tstat)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h28',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h29(self):
        """
        Render Section 5 2A Figure B8-H29 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['695'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H29. Hourly Loads\n'
                  'Clear Cold Day, Cases 695 (20/20, Insulation)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h29',
            annotations=[
                {
                    'text': 'Cold Day (Feb 14)',
                    'xy': (11, 1.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h30(self):
        """
        Render Section 5 2A Figure B8-H30 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['695'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H30. Hourly Loads\n'
                  'Clear Hot Day, Cases 695 (20/20, Insulation)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h30',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h31(self):
        """
        Render Section 5 2A Figure B8-H31 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['900'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H31. Hourly Loads\n'
                  'Clear Cold Day, Cases 900 (High Mass)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h31',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (13, 1.1),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h32(self):
        """
        Render Section 5 2A Figure B8-H32 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['900'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H32. Hourly Loads\n'
                  'Clear Hot Day, Cases 900 (High Mass)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h32',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (3, -0.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h33(self):
        """
        Render Section 5 2A Figure B8-H33 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['980'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H33. Hourly Loads\n'
                  'Clear Cold Day, Cases 980 (High Mass, Insulation)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h33',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (12, 0.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h34(self):
        """
        Render Section 5 2A Figure B8-H34 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['980'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H34. Hourly Loads\n'
                  'Clear Hour Day, Cases 980 (High Mass, Insulation)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h34',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -0.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h35(self):
        """
        Render Section 5 2A Figure B8-H35 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['985'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H35. Hourly Loads\n'
                  'Clear Cold Day, Cases 985 (High Mass, 20/20 Tstat)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h35',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (12, 0.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h36(self):
        """
        Render Section 5 2A Figure B8-H36 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['985'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H36. Hourly Loads\n'
                  'Clear Hot Day, Cases 985 (High Mass, 20/20 Tstat)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h36',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h37(self):
        """
        Render Section 5 2A Figure B8-H37 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['995'][
                    'feb_1']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H37. Hourly Loads\n'
                  'Clear Cold Day, Cases 995 (High Mass, 20/20, Insulation)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h37',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (12, 0.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_5_2a_figure_b8_h38(self):
        """
        Render Section 5 2A Figure B8-H38 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data_x = []
        data_y = []
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            try:
                data_obj = json_obj['specific_day_hourly_output_free_float_zone_loads']['995'][
                    'july_14']['hour']
                key_list, val_list = self._order_dictionary_with_numeric_keys(
                    input_d=data_obj,
                    val_key='kWh')
                data_x.append(key_list)
                data_y.append(val_list)
                programs.append(json_obj['identifying_information']['software_name'])
            except (TypeError, KeyError):
                data_x.append([])
                data_y.append([])
                programs.append('NA')
        fig, ax = self._create_line_plot(
            data_x=data_x,
            data_y=data_y,
            programs=programs,
            title='Figure B8-H38. Hourly Loads\n'
                  'Clear Hot Day, Cases 995 (High Mass, 20/20, Insulation)\n'
                  'Heating (+), Sensible Cooling (-)',
            ylabel=r'Heating or Sensible Cooling Load (kWh/h)',
            xlabel=r'Hour of Day',
            image_name='section_5_2_a_figure_b8_h38',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (3, -1.3),
                    'fontsize': 18}])
        return fig, ax

    # def render_section_5_2b_table_b8_2_1(
    #         self,
    #         output_value='annual_heating_MWh',
    #         caption='Table B8.2-1 "a"-Series Case Summary, Numerical Model Verification'):
    #     """
    #     Create dataframe from class dataframe object for table 5-2B B8.2-1
    #
    #     :return: pandas dataframe and output msg for general navigation.
    #     """
    #     df = self.df_data['steady_state_cases']
    #     return df

    def render_section_5_4_table_b16_6_5(
            self,
            output_value='mean_zone_temperature',
            figure_name='section_5_4_table_b16_6_5',
            caption=r'Table B16.6-5. Mean Zone Temperature ($^\circ$C)'):
        """
        Create dataframe from class dataframe object for table 5-4 B16.6-5

        :return: pandas dataframe and output msg for general navigation.
        """

        df = pd.DataFrame()
        software_array = []
        index_dict = {
            'CASE HE210': 'HE210: Realistic Weather',
            'CASE HE220': 'HE220: Setback Thermostat',
            'CASE HE230': 'HE230: Undersized Furnace'}
        # get data
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            df_obj = pd.DataFrame.from_dict(json_obj[output_value])
            df_obj['software'] = json_obj['identifying_information']['program_name_and_version'] + '\n' + \
                json_obj['identifying_information']['program_organization']
            software_array.append(json_obj['identifying_information']['program_name_and_version'])
            df = pd.concat([df, df_obj], axis=0)
        df = df.set_index('software')
        df = df.rename(index_dict, axis=1)
        df = df.groupby(level=0, axis=1).sum()
        df = df.transpose()

        for row in range(len(df)):
            for col in range(len(df.columns)):
                try:
                    df.at[df.index[row], df.columns[col]]
                except (KeyError, ValueError):
                    df.at[df.index[row], df.columns[col]] = float('NaN')

        # calculate stats
        df_stats = pd.DataFrame()
        df_stats['min'] = df.iloc[:, 0:-1].min(axis=1)
        df_stats['max'] = df.iloc[:, 0:-1].max(axis=1)
        df_stats['mean'] = df.iloc[:, 0:-1].mean(axis=1)
        df_stats['(max-min)\n/mean*'] = abs(df_stats['max'] - df_stats['min']).div(
            df_stats['mean'].where(df_stats['mean'] != 0, np.nan))

        # merge dataframes
        df_merged = pd.concat(
            [
                df.iloc[:, range(len(df.columns) - 1)],
                df_stats,
                df.iloc[:, range(len(df.columns) - 1, len(df.columns))]
            ],
            axis=1)

        df_formatted_table = df_merged.reset_index(drop=False).rename(columns={'index': 'Cases'})
        df_formatted_table.fillna('', inplace=True)

        for col in df_formatted_table:
            if col in software_array:
                df_formatted_table[col] = df_formatted_table[col].apply(lambda x: round(x, 2) if x != '' else x)

        df_formatted_table['min'] = df_formatted_table['min'].apply(lambda x: round(x, 2) if x != '' else x)
        df_formatted_table['max'] = df_formatted_table['max'].apply(lambda x: round(x, 2) if x != '' else x)
        df_formatted_table['mean'] = df_formatted_table['mean'].apply(lambda x: round(x, 2) if x != '' else x)
        df_formatted_table['(max-min)\n/mean*'] = df_formatted_table['(max-min)\n/mean*'].apply(
            lambda x: '{0:.1f}%'.format(math.floor(x * 1000) / 10) if x * 1000 - math.floor(
                x * 1000) < 0.5 else '{0:.1f}%'.format(
                math.ceil(x * 1000) / 10))  # check all rounds, 1.5 is round to 1 currently

        wrapped_col = {}
        for col in df_formatted_table:
            wrapped_col[col] = twp.fill(col, break_long_words=False, width=5)
        df_formatted_table.rename(columns=wrapped_col, inplace=True)

        # save results
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(24, 6))
        tab = self._make_table_from_df(df=df_formatted_table, ax=ax, case_col_width=1.2)

        # Set annotations
        header_stats = [tab.add_cell(-1, h, width=0.5, height=0.20) for h in range(4, 9)]
        header_stats[0].get_text().set_text('Statistics, All Results')
        header_stats[0].PAD = 0.5
        header_stats[0].set_fontsize(16)
        header_stats[0].set_text_props(ha="left")
        header_stats[0].visible_edges = "open"
        header_stats[1].visible_edges = "open"
        header_stats[2].visible_edges = "open"
        header_stats[3].visible_edges = "open"
        header_stats[4].visible_edges = "open"

        plt.figtext(0.15, 0.08, "*ABS[(Max-Min)/(Mean of Example Results)", ha="left", fontsize=12)

        plt.suptitle(caption, fontsize=30, y=1.05)
        self._make_image_from_plt(figure_name)

        return fig, ax

    def render_section_5_4_table_b16_6_6(self):
        """
        Create dataframe from class dataframe object for table 5-4 B16.6-6

        :return: pandas dataframe and output msg for general navigation.
        """
        fig, ax = self.render_section_5_4_table_b16_6_5(
            output_value='maximum_zone_temperature',
            figure_name='section_5_4_table_b16_6_6',
            caption=r'Table B16.6-6. Maximum Zone Temperature ($^\circ$C)')

        return fig, ax

    def render_section_5_4_table_b16_6_7(self):
        """
        Create dataframe from class dataframe object for table 5-4 B16.6-7

        :return: pandas dataframe and output msg for general navigation.
        """
        fig, ax = self.render_section_5_4_table_b16_6_5(
            output_value='minimum_zone_temperature',
            figure_name='section_5_4_table_b16_6_7',
            caption=r'Table B16.6-7. Minimum Zone Temperature ($^\circ$C)')

        return fig, ax

    def render_section_5_4_figure_b16_6_5(
            self,
            output_value='mean_zone_temperature',
            figure_name='section_5_4_figure_b16_6_5',
            caption='Figure B16.6-5. Comparison of the Mean Zone Temperature \n'
                    'for the Fuel-Fired Furnace Comparative Test Cases'):
        """
        Render Section 5-4 Figure B16.6-5 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        data = []
        cases = ['HE210\nRealistic Weather', 'HE220\nSetback Thermostat', 'HE230\nUndersize Furnace']
        programs = []
        for idx, (tst, json_obj) in enumerate(self.json_data.items()):
            tmp_data = []
            for key in json_obj[output_value]:
                try:
                    tmp_data.append(
                        json_obj[output_value][key]['degC'])
                except (KeyError, ValueError):
                    tmp_data.append(float('NaN'))
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['program_name_and_version'] + '\n' + json_obj['identifying_information']['program_organization'])

            fig, ax = self._create_bar_plot(
                data=data,
                programs=programs,
                title=caption,
                xticklabels=[i for i in cases],
                ylabel=output_value.replace('_', ' ').title() + r' ($^\circ$C)',
                y_min=0,
                y_max=23,  # y max needs to be the same across 3 charts?
                image_name=figure_name)
        return fig, ax

    def render_section_5_4_figure_b16_6_6(self):
        """
        Render Section 5-4 Figure B16.6-6 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        fig, ax = self.render_section_5_4_figure_b16_6_5(
            output_value='maximum_zone_temperature',
            figure_name='section_5_4_figure_b16_6_6',
            caption='Figure B16.6-6.  Comparison of the Max Zone Temperature \n'
                    'for the Fuel-Fired Furnace Comparative Test Cases')

        return fig, ax

    def render_section_5_4_figure_b16_6_7(self):
        """
        Render Section 5-4 Figure B16.6-7 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
        fig, ax = self.render_section_5_4_figure_b16_6_5(
            output_value='minimum_zone_temperature',
            figure_name='section_5_4_figure_b16_6_7',
            caption='Figure B16.6-7.  Comparison of the Min Zone Temperature \n'
                    'for the Fuel-Fired Furnace Comparative Test Cases')

        return fig, ax

    def render_section_5_2a_table_b8_16_md(self):
        figure_name = 'section_5_2_figure_b8_16'
        caption = 'Table B8-16. Sky Temperatures Output, Case 600'
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]

        top_table = []
        bottom_table = []
        averages = []
        minimums = []
        maximums = []

        header_row = ['Case', 'Parameter', 'Annual Hourly<br>Integrated Average', 'Annual Hourly<br>Integrated Minimum',
                      'Annual Hourly<br>Integrated Maximum']
        top_table.append(header_row)
        for index, (tst, json_obj) in enumerate(self.json_data.items()):
            sky_600 = json_obj['sky_temperature_output']['600']
            data_row = [json_obj['identifying_information']['software_name'], 'T(C)']
            if not math.isnan(sky_600['Average']['C']):
                average = sky_600['Average']['C']
                averages.append(average)
                data_row.append(round(average, 1))

                minimum = sky_600['Minimum']['C']
                minimums.append(minimum)
                data_row.append(round(minimum, 1))

                maximum = sky_600['Maximum']['C']
                maximums.append(maximum)
                data_row.append(round(maximum, 1))

            # now work on the timestamp rows
            timestamp_row = [json_obj['identifying_information']['software_name'], 'Mo Day Hr',
                             '']  # include extra space for average column that doesn't have timestamp
            if not math.isnan(sky_600['Average']['C']):
                timestamp_row.append(sky_600['Minimum']['Month'] + ' ' + str(sky_600['Minimum']['Day']) + ' ' + str(
                    sky_600['Minimum']['Hour']))
                timestamp_row.append(sky_600['Maximum']['Month'] + ' ' + str(sky_600['Maximum']['Day']) + ' ' + str(
                    sky_600['Maximum']['Hour']))

            # if it is not the tested program, add it to the table
            if index < (len(self.json_data) - 1):
                top_table.append(data_row)
                bottom_table.append(timestamp_row)
            else:
                tested_program_data_row = data_row
                tested_program_timestamp_row = timestamp_row

        # remove last item since it is the tested program
        minimums.pop()
        maximums.pop()
        averages.pop()

        # add test spec alt
        row = ['TestSpec-Alt', 'T(C)', -5.9, -46.9, 24.6]
        averages.append(-5.9)
        minimums.append(-46.9)
        maximums.append(24.6)
        top_table.append(row)
        top_table.append(['', ])  # add blank row

        bottom_table.append(['', 'Mo Day Hr'])

        # now do the statistic rows
        minimum_of_averages = min(averages)
        minimum_of_minimums = min(minimums)
        minimum_of_maximums = min(maximums)
        row = ['Min', 'T(C)', round(minimum_of_averages, 1), round(minimum_of_minimums, 1),
               round(minimum_of_maximums, 1)]
        top_table.append(row)

        maximum_of_averages = max(averages)
        maximum_of_minimums = max(minimums)
        maximum_of_maximums = max(maximums)
        row = ['Max', 'T(C)', round(maximum_of_averages, 1), round(maximum_of_minimums, 1),
               round(maximum_of_maximums, 1)]
        top_table.append(row)

        mean_of_averages = sum(averages) / len(averages)
        mean_of_minimums = sum(minimums) / len(minimums)
        mean_of_maximums = sum(maximums) / len(maximums)
        row = ['Mean', 'T(C)', round(mean_of_averages, 1), round(mean_of_minimums, 1),
               round(mean_of_maximums, 1)]
        top_table.append(row)
        row = ['(Max-Min)/Mean [^1]', ' % ',
               round(100 * abs((maximum_of_averages - minimum_of_averages) / mean_of_averages), 1),
               round(100 * abs((maximum_of_minimums - minimum_of_minimums) / mean_of_minimums), 1),
               round(100 * abs((maximum_of_maximums - minimum_of_maximums) / mean_of_maximums), 1)]
        top_table.append(row)
        # now put in the tested program rows
        top_table.append(['', ])  # add blank row
        top_table.append(tested_program_data_row)
        bottom_table.append(['', ])  # add blank row
        bottom_table.append(tested_program_timestamp_row)

        table = top_table
        table.append(['', ])  # add blank row
        table.extend(bottom_table)
        self._make_markdown_from_table(figure_name, caption, table, footnotes)
        return

    def render_section_5_2a_table_b8_1(self):
        figure_name = 'section_5_2_table_b8_1'
        caption = 'Table B8-1. Annual Heating Loads (kWh)'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['conditioned_zone_loads_non_free_float'][case]['annual_heating_MWh'])
            data_table.append(row)
        for blank_row in [44,35,21,11]:
            data_table.insert(blank_row,[])  # add blank line as separator
            row_headings.insert(blank_row, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_2(self):
        figure_name = 'section_5_2_table_b8_2'
        caption = 'Table B8-2. Annual Sensible Cooling Loads (kWh)'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['conditioned_zone_loads_non_free_float'][case]['annual_cooling_MWh'])
            data_table.append(row)
        for blank_row in [44,35,21,11]:
            data_table.insert(blank_row,[])  # add blank line as separator
            row_headings.insert(blank_row, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_3(self):
        figure_name = 'section_5_2_table_b8_3'
        caption = 'Table B8-3. Annual Hourly Integrated Peak Heating Loads (kWh)'
        data_table = []
        time_stamp_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                case_json = json_obj['conditioned_zone_loads_non_free_float'][case]
                row.append(case_json['peak_heating_kW'])
                month = case_json['peak_heating_month']
                day = self._int_0_if_nan(case_json['peak_heating_day'])
                hour = self._int_0_if_nan(case_json['peak_heating_hour'])
                time_stamp_row.append(f'{month} {day}-{hour}')
            data_table.append(row)
            time_stamp_table.append(time_stamp_row)
        for blank_row in [44,35,21,11]:
            data_table.insert(blank_row,[])  # add blank line as separator
            time_stamp_table.insert(blank_row,[])
            row_headings.insert(blank_row, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_4(self):
        figure_name = 'section_5_2_table_b8_4'
        caption = 'Table B8-4. Annual Hourly Integrated Peak Sensible Cooling Loads (kWh)'
        data_table = []
        time_stamp_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                case_json = json_obj['conditioned_zone_loads_non_free_float'][case]
                row.append(case_json['peak_cooling_kW'])
                month = case_json['peak_cooling_month']
                day = self._int_0_if_nan(case_json['peak_cooling_day'])
                hour = self._int_0_if_nan(case_json['peak_cooling_hour'])
                time_stamp_row.append(f'{month} {day}-{hour}')
            data_table.append(row)
            time_stamp_table.append(time_stamp_row)
        for blank_row in [44,35,21,11]:
            data_table.insert(blank_row,[])  # add blank line as separator
            time_stamp_table.insert(blank_row,[])
            row_headings.insert(blank_row, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m1a(self):  # case 600
        figure_name = 'section_5_2_table_b8_m1a'
        caption = 'Table B8-M1a. Monthly Heating Loads (kWh), Case 600'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['monthly_conditioned_zone_loads']['600'][month]['total_heating_kwh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m1b(self):  # case 900
        figure_name = 'section_5_2_table_b8_m1b'
        caption = 'Table B8-M1b. Monthly Heating Loads (kWh), Case 900'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['monthly_conditioned_zone_loads']['900'][month]['total_heating_kwh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m2a(self):  # case 600
        figure_name = 'section_5_2_table_b8_m2a'
        caption = 'Table B8-M2a. Monthly Sensible Cooling Loads (kWh), Case 600'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['monthly_conditioned_zone_loads']['600'][month]['total_cooling_kwh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m2b(self):
        figure_name = 'section_5_2_table_b8_m2b'
        caption = 'Table B8-M2b. Monthly Sensible Cooling Loads (kWh), Case 900'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['monthly_conditioned_zone_loads']['900'][month]['total_cooling_kwh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m3a(self):  # case 900
        figure_name = 'section_5_2_table_b8_m3a'
        caption = 'Table B8-M3a. Monthly Hourly Integrated Peak Heating Loads (kW), Case 600'
        data_table = []
        time_stamp_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            data_row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                monthly_results = json_obj['monthly_conditioned_zone_loads']['600'][month]
                data_row.append(monthly_results['peak_heating_kw'])
                day = int(monthly_results['peak_heating_day'])
                hour = int(monthly_results['peak_heating_hour'])
                time_stamp_row.append(f'{day}-{hour}')
            data_table.append(data_row)
            time_stamp_table.append(time_stamp_row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m3b(self):  # case 900
        figure_name = 'section_5_2_table_b8_m3b'
        caption = 'Table B8-M3b. Monthly Hourly Integrated Peak Heating Loads (kW), Case 900'
        data_table = []
        time_stamp_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            data_row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                monthly_results = json_obj['monthly_conditioned_zone_loads']['900'][month]
                data_row.append(monthly_results['peak_heating_kw'])
                day = self._int_0_if_nan(monthly_results['peak_heating_day'])
                hour = self._int_0_if_nan(monthly_results['peak_heating_hour'])
                time_stamp_row.append(f'{day}-{hour}')
            data_table.append(data_row)
            time_stamp_table.append(time_stamp_row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m4a(self):  # case 900
        figure_name = 'section_5_2_table_b8_m4a'
        caption = 'Table B8-M4a. Monthly Hourly Integrated Peak Sensible Cooling Loads (kW), Case 600'
        data_table = []
        time_stamp_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            data_row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                monthly_results = json_obj['monthly_conditioned_zone_loads']['600'][month]
                data_row.append(monthly_results['peak_cooling_kw'])
                day = int(monthly_results['peak_cooling_day'])
                hour = int(monthly_results['peak_cooling_hour'])
                time_stamp_row.append(f'{day}-{hour}')
            data_table.append(data_row)
            time_stamp_table.append(time_stamp_row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m4b(self):  # case 900
        figure_name = 'section_5_2_table_b8_m4b'
        caption = 'Table B8-M4b. Monthly Hourly Integrated Peak Sensible Cooling Loads (kW), Case 900'
        data_table = []
        time_stamp_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            data_row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                monthly_results = json_obj['monthly_conditioned_zone_loads']['900'][month]
                data_row.append(monthly_results['peak_cooling_kw'])
                day = self._int_0_if_nan(monthly_results['peak_cooling_day'])
                hour = self._int_0_if_nan(monthly_results['peak_cooling_hour'])
                time_stamp_row.append(f'{day}-{hour}')
            data_table.append(data_row)
            time_stamp_table.append(time_stamp_row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m5a(self):  # case 600
        figure_name = 'section_5_2_table_b8_m5a'
        caption = 'Table B8-M5a. Monthly Load 600-900 Sensitivity Tests - Annual Heating (kWh)'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['monthly_conditioned_zone_loads']['600'][month][
                    'total_heating_kwh'] - json_obj['monthly_conditioned_zone_loads'][
                    '900'][month]['total_heating_kwh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m5b(self):
        figure_name = 'section_5_2_table_b8_m5b'
        caption = 'Table B8-M5b. Monthly Load 600-900 Sensitivity Tests - Annual Sensible Cooling (kWh)'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['monthly_conditioned_zone_loads']['600'][month][
                    'total_cooling_kwh'] - json_obj['monthly_conditioned_zone_loads'][
                    '900'][month]['total_cooling_kwh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m5c(self):
        figure_name = 'section_5_2_table_b8_m5c'
        caption = 'Table B8-M5c. Monthly Load 600-900 Sensitivity Tests - Peak Heating (kW)'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            row = []
            for tst, json_obj in self.json_data.items():
                value_600 = float(json_obj['monthly_conditioned_zone_loads']['600'][month]['peak_heating_kw'])
                if math.isnan(value_600):
                    value_600 = 0
                value_900 = float(json_obj['monthly_conditioned_zone_loads']['900'][month]['peak_heating_kw'])
                if math.isnan(value_900):
                    value_900 = 0
                row.append(value_600 - value_900)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, 3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_5_2a_table_b8_m5d(self):
        figure_name = 'section_5_2_table_b8_m5d'
        caption = 'Table B8-M5d. Monthly Load 600-900 Sensitivity Tests - Peak Sensible Cooling (kW)'
        data_table = []
        footnotes = ['[^1]: ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        column_headings = ['Month']
        # create column headings
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        # create table of values
        for month in row_headings:
            row = []
            for tst, json_obj in self.json_data.items():
                value_600 = float(json_obj['monthly_conditioned_zone_loads']['600'][month]['peak_cooling_kw'])
                if math.isnan(value_600):
                    value_600 = 0
                value_900 = float(json_obj['monthly_conditioned_zone_loads']['900'][month]['peak_cooling_kw'])
                if math.isnan(value_900):
                    value_900 = 0
                row.append(value_600 - value_900)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, 3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return
