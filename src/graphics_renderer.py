import pathlib
import json
import os
import pandas as pd
import numpy as np

from logger import Logger

root_directory = pathlib.Path(__file__).parent.parent.resolve()


class GraphicsRenderer(Logger):
    """
    Create graphs and tables from a model_results_file.
    """

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
        if not processed_file_directory:
            self.processed_file_directory = root_directory.joinpath('processed')
        else:
            self.processed_file_directory = processed_file_directory
        if not base_model_list:
            self.baseline_model_list = [
                'RESULTS5-2A-BSIMAC-9-9.0.74.json',
                'RESULTS5-2A-CSE-0.861.1.json',
                'RESULTS5-2A-DeST-2.0-20190401.json',
                'RESULTS5-2A-EnergyPlus-9.0.1.json',
                'RESULTS5-2A-ESP-r-13.3.json',
                'RESULTS5-2A-TRNSYS-18.00.0001.json']
        else:
            self.baseline_model_list = base_model_list
        self.model_results_file = model_results_file
        # create an object that keeps the information needed to make the row index for each table object.
        # 0 - json key name
        # 1 - list to make row index
        self.table_lookup = [
            ('conditioned_zone_loads_non_free_float', ['program_name', ])
        ]
        # instantiate objects to store data as a dictionary of json objects, and a dictionary of pandas dataframes
        self.json_data = {}
        self.df_data = {}
        # set hatches list for visualization objects
        self.hatches = ['/', '-', 'x', '\\', '//', 'o', '||', '+', 'O', '.', '*']
        self._get_data()
        return

    def _get_data(self):
        """
        Get processed json data and store it in two dictionary objects.  One is the original json file data, the other
        is the converted pandas dataframe with a multiIndex for each json level.
        :return: Updated class objects that represent the data as a json object and pandas dataframe
        """
        table_objects = {}
        for f in self.baseline_model_list + [self.model_results_file, ]:
            base_name = str(os.path.basename(f)).replace('.json', '')
            with open(self.processed_file_directory.joinpath(f), 'r') as jf:
                data = json.load(jf)
                # load json objects as objects with the file name as the key
                self.json_data.update({base_name: data})
                # load each table, if exists into a dataframe of the json key name
                for tbl, row_index in self.table_lookup:
                    tbl_data = data.get(tbl)
                    if tbl_data:
                        try:
                            table_objects[tbl]
                        except KeyError:
                            table_objects[tbl] = pd.DataFrame()
                        # Format the json data to a multiIndex table with a meaningful row index
                        tmp_df = pd.json_normalize(tbl_data)
                        tmp_df.columns = pd.MultiIndex.from_tuples([i.split('.') for i in tmp_df.columns])
                        tmp_df['program_name'] = str(os.path.basename(f)).replace('.json', '')
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

    def render_section_5_2a_table_b_8_1(self):
        """
        Create dataframe from class dataframe object for table 5-2A B8-1

        :return: pandas dataframe
        """
        df_formatted_table = pd.DataFrame()
        msg = None
        try:
            # get and format dataframe into required shape
            df = self.df_data['conditioned_zone_loads_non_free_float']\
                .loc[
                    :,
                    self.df_data['conditioned_zone_loads_non_free_float']
                        .columns.get_level_values(1) == 'annual_heating_MWh']
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
            base_file_names = [i.replace('.json', '') for i in self.baseline_model_list]
            calculated_df = pd.DataFrame()
            calculated_df['min'] = df_formatted_table[base_file_names].min(axis=1)
            df_formatted_table = pd.concat([df_formatted_table, calculated_df], axis=1)
        except KeyError:
            msg = 'Section 5-2A B8-1 Failed to be processed'
        return df_formatted_table, msg

    def render_section_5_2a_figure_b_8_9(self, fig, ax):
        """
        Render Section 5 2A Figure B8-9 by modifying fig an ax inputs from matplotlib
        :return: modified fig and ax objects from matplotlib.subplots()
        """
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
            data.insert(idx, tmp_data)
            programs.insert(idx, json_obj['identifying_information']['software_name'])
        ax.set_xticks(np.arange(max([len(i) for i in data])))
        ax.set_title('Figure B8-9.  Basic: Low Mass Peak Heating', fontsize=30)
        ax.set_xticklabels(cases)
        for idx, (p, d, h) in enumerate(zip(programs, data, self.hatches)):
            x = np.arange(len(d))
            rects = ax.bar(x + (width * idx) - (width / 2 * (len(data) - 1)), d, width, label=p, hatch=h, fill=None)
            ax.bar_label(rects, padding=5, rotation="vertical")
        ax.grid(which='major', axis='y')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(programs), fontsize=14)
        ax.set_ylabel('Peak Heating Load (kWh/h)', fontsize=14)
        ax.set_ylim(0, 5)
        return fig, ax
