import pathlib
import json
import re
import pandas as pd
import numpy as np
import math
from textwrap import wrap
import matplotlib.pyplot as plt
import plotly.express as px
# import kaleido

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
        if re.match(r'.*Std140_TF_Output\..*$', str(value), re.IGNORECASE):
            obj._section_type = 'TF'
        elif re.match(r'.*Std140_GC_Output\..*$', str(value), re.IGNORECASE):
            obj._section_type = 'GC'
        elif re.match(r'.*Std140_HE_Output\..*$', str(value), re.IGNORECASE):
            obj._section_type = 'HE'
        elif re.match(r'.*Std140_CE_a_Output\..*$', str(value), re.IGNORECASE):
            obj._section_type = 'CE_a'
        elif re.match(r'.*Std140_CE_b_Output\..*$', str(value), re.IGNORECASE):
            obj._section_type = 'CE_b'
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
        if self.section_type == 'TF':
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
        elif self.section_type == 'HE':
            self.case_detailed_df = pd.DataFrame.from_dict(
                {
                    'HE100': ['HE100 100% eff.', 1],
                    'HE110': ['HE110 80% eff.', 2],
                    'HE120': ['HE120 80% eff., PLR=0.4', 3],
                    'HE130': ['HE130 No Load', 4],
                    'HE140': ['HE140 Periodic PLR', 5],
                    'HE150': ['HE150 Continuous Circ. Fan', 6],
                    'HE160': ['HE160 Cycling Circ. Fan', 7],
                    'HE170': ['HE170 Draft Fan', 8],
                    'HE210': ['HE210 Realistic Weather', 9],
                    'HE220': ['HE220 Setback Thermostat', 10],
                    'HE230': ['HE230 Undersized Furnace', 11]
                },
                orient='index',
                columns=['case_name', 'case_order'])
        elif self.section_type == 'CE_a':
            self.case_detailed_df = pd.DataFrame.from_dict(
                {
                    'CE100': ['CE100 dry lo IDB hi ODB', 1],
                    'CE110': ['CE110 as 100 lo ODB', 2],
                    'CE120': ['CE120 as 100 hi IDB', 3],
                    'CE130': ['CE130 as 100 lo PLR', 4],
                    'CE140': ['CE140 as 130 lo ODB', 5],
                    'CE150': ['CE150 as 110 hi SHR', 6],
                    'CE160': ['CE160 as 150 hi IDB', 7],
                    'CE165': ['CE165 as 150 m IDB m ODB', 8],
                    'CE170': ['CE170 as 150 m SHR m PLR', 9],
                    'CE180': ['CE180 as 150 lo SHR', 10],
                    'CE185': ['CE185 lo SHR hi ODB', 11],
                    'CE190': ['CE190 as 180 lo PLR', 12],
                    'CE195': ['CE195 as 185 lo PLR', 13],
                    'CE200': ['CE200 ARI  PLR=1 hi SHR', 14]
                },
                orient='index',
                columns=['case_name', 'case_order'])
        elif self.section_type == 'CE_b':
            self.case_detailed_df = pd.DataFrame.from_dict(
                {
                    'E300': ['CE300 Base, 15% OA', 1],
                    'E310': ['CE310 High Latent', 2],
                    'E320': ['CE320 High Infiltration', 3],
                    'E330': ['CE330 100% OA', 4],
                    'E340': ['CE340 50% OA, 50% Infl', 5],
                    'E350': ['CE350 Tstat Set Up', 6],
                    'E360': ['CE360 Undersized System', 7],
                    'E400': ['CE400 Ec. Temp. Ctrl.', 8],
                    'E410': ['CE410 Ec. Comp. Lockout', 9],
                    'E420': ['CE420 Ec. ODB Limit', 10],
                    'E430': ['CE430 Ec. Enthalpy Ctrl.', 11],
                    'E440': ['CE440 Ec. Enthalpy Limit', 12],
                    'E500': ['CE500 Base w/ 0%OA', 13],
                    'E500 May-Sep': ['CE500 May-Sep', 14],
                    'E510': ['CE510 May-Sep, High PLR', 15],
                    'E520': ['CE520 EDB = 15°C', 16],
                    'E522': ['CE522 EDB = 20°C', 17],
                    'E525': ['CE525 EDB = 35°C', 18],
                    'E530': ['CE530 Dry Coil', 19],
                    'E540': ['CE540 Dry, EDB = 15°C', 20],
                    'E545': ['CE545 Dry, EDB = 35°C', 21]
                },
                orient='index',
                columns=['case_name', 'case_order'])
        if not processed_file_directory:
            self.processed_file_directory = root_directory.joinpath('processed')
        else:
            self.processed_file_directory = processed_file_directory
        if not base_model_list:
            if self.section_type == 'TF':
                self.baseline_model_list = [
                    root_directory.joinpath('processed', 'bsimac', '9.9.0.7.4', 'std140_tf_output.json'),
                    root_directory.joinpath('processed', 'cse', '0.861.1', 'std140_tf_output.json'),
                    root_directory.joinpath('processed', 'dest', '2.0.20190401', 'std140_tf_output.json'),
                    root_directory.joinpath('processed', 'energyplus', '9.0.1', 'std140_tf_output.json'),
                    root_directory.joinpath('processed', 'esp-r', '13.3', 'std140_tf_output.json'),
                    root_directory.joinpath('processed', 'trnsys', '18.00.0001', 'std140_tf_output.json')]
            elif self.section_type == 'GC':
                self.baseline_model_list = [
                    root_directory.joinpath('processed', 'basecalc', 'v1.0e', 'std140_gc_output.json'),
                    root_directory.joinpath('processed', 'energyplus', '9.0.1', 'std140_gc_output.json'),
                    root_directory.joinpath('processed', 'esp-r', '13.3', 'std140_gc_output.json'),
                    root_directory.joinpath('processed', 'fluent', '6.1', 'std140_gc_output.json'),
                    root_directory.joinpath('processed', 'ght', '2.02', 'std140_gc_output.json'),
                    root_directory.joinpath('processed', 'matlab', '7.0.4.365-r14-sp2', 'std140_gc_output.json'),
                    root_directory.joinpath('processed', 'sunrel-gc', '1.14.02', 'std140_gc_output.json'),
                    root_directory.joinpath('processed', 'trnsys', '18.00.0001', 'std140_gc_output.json'),
                    root_directory.joinpath('processed', 'va114', '2.20', 'std140_gc_output.json')
                ]
            elif self.section_type == 'HE':
                self.baseline_model_list = [
                    root_directory.joinpath('processed', 'esp-r-hot3000', '1.7', 'std140_he_output.json'),
                    root_directory.joinpath('processed', 'energyplus', '1.0.2', 'std140_he_output.json'),
                    root_directory.joinpath('processed', 'doe21e', 'c133', 'std140_he_output.json'),
                    root_directory.joinpath('processed', 'analytical', '0', 'std140_he_output.json')
                ]
            elif self.section_type == 'CE_a':
                self.baseline_model_list = [
                    root_directory.joinpath('processed', 'ca-sis', 'v1', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'clim2000', '2.1.6', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'doe21e', '88', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'doe21e', 'c133', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'energyplus', '1.0.0.023', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'trnsys', '14.02.id', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'trnsys', '14.02.re', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'analytical-tud', '0', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'analytical-htal1', '0', 'std140_ce_a_output.json'),
                    root_directory.joinpath('processed', 'analytical-htal2', '0', 'std140_ce_a_output.json')
                ]
            elif self.section_type == 'CE_b':
                self.baseline_model_list = [
                    root_directory.joinpath('processed', 'trnsys', '14.02.re', 'std140_ce_b_output.json'),
                    root_directory.joinpath('processed', 'doe22', '42', 'std140_ce_b_output.json'),
                    root_directory.joinpath('processed', 'doe21e', '120', 'std140_ce_b_output.json'),
                    root_directory.joinpath('processed', 'energyplus', '1.1.0.020', 'std140_ce_b_output.json'),
                    root_directory.joinpath('processed', 'codyrun', '1', 'std140_ce_b_output.json'),
                    root_directory.joinpath('processed', 'hot3000', '1', 'std140_ce_b_output.json')
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
        if self.section_type == 'TF':
            self.table_lookup = [
                ('conditioned_zone_loads_non_free_float', ['program_name', ]),
                ('free_float_case_zone_temperatures', ['program_name', ])
            ]
        elif self.section_type == 'GC':
            self.table_lookup = [
                ('steady_state_cases', ['program_name', ])
            ]
        elif self.section_type == 'HE':
            self.table_lookup = [
                ('furnace_input', ['program_name', ])
            ]
        elif self.section_type == 'CE_a':
            self.table_lookup = [
                ('february_results', ['program_name', ])
            ]
        elif self.section_type == 'CE_b':
            self.table_lookup = [
                ('annual_sums_means', ['program_name', ])
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

        if self.section_type == 'TF':
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
                '810': '810 High Mass Cavity Albedo'}
        elif self.section_type == 'HE':
            self.case_map = {
                'HE100': 'HE100 100% eff.',
                'HE110': 'HE110 80% eff.',
                'HE120': 'HE120 80% eff., PLR=0.4',
                'HE130': 'HE130 No Load',
                'HE140': 'HE140 Periodic PLR',
                'HE150': 'HE150 Continuous Circ. Fan',
                'HE160': 'HE160 Cycling Circ. Fan',
                'HE170': 'HE170 Draft Fan',
                'HE210': 'HE210 Realistic Weather',
                'HE220': 'HE220 Setback Thermostat',
                'HE230': 'HE230 Undersized Furnace'
            }
        elif self.section_type == 'CE_a':
            self.case_map = {
                'CE100': 'CE100 dry lo IDB hi ODB',
                'CE110': 'CE110 as 100 lo ODB',
                'CE120': 'CE120 as 100 hi IDB',
                'CE130': 'CE130 as 100 lo PLR',
                'CE140': 'CE140 as 130 lo ODB',
                'CE150': 'CE150 as 110 hi SHR',
                'CE160': 'CE160 as 150 hi IDB',
                'CE165': 'CE165 as 150 m IDB m ODB',
                'CE170': 'CE170 as 150 m SHR m PLR',
                'CE180': 'CE180 as 150 lo SHR',
                'CE185': 'CE185 lo SHR hi ODB',
                'CE190': 'CE190 as 180 lo PLR',
                'CE195': 'CE195 as 185 lo PLR',
                'CE200': 'CE200 ARI  PLR=1 hi SHR'
            }
        elif self.section_type == 'CE_b':
            self.case_map = {
                'E300': 'CE300',
                'E310': 'CE310',
                'E320': 'CE320',
                'E330': 'CE330',
                'E340': 'CE340',
                'E350': 'CE350',
                'E360': 'CE360',
                'E400': 'CE400',
                'E410': 'CE410',
                'E420': 'CE420',
                'E430': 'CE430',
                'E440': 'CE440',
                'E500': 'CE500',
                'E500 May-Sep': 'CE500 May-Sep',
                'E510 May-Sep': 'CE510 May-Sep',
                'E520': 'CE520',
                'E522': 'CE522',
                'E525': 'CE525',
                'E530': 'CE530',
                'E540': 'CE540',
                'E545': 'CE545',
            }
        self.case_map_max = {
            'E300': 'CE300',
            'E310': 'CE310',
            'E320': 'CE320',
            'E330': 'CE330',
            'E340': 'CE340',
            'E350': 'CE350',
            'E360': 'CE360',
            'E400': 'CE400',
            'E410': 'CE410',
            'E420': 'CE420',
            'E430': 'CE430',
            'E440': 'CE440',
            'E500': 'CE500',
            'E510': 'CE510',
            'E520': 'CE520',
            'E522': 'CE522',
            'E525': 'CE525',
            'E530': 'CE530',
            'E540': 'CE540',
            'E545': 'CE545',
        }
        self.case_map_charts = {
            'E300': 'CE300 Base, 15% OA',
            'E310': 'CE310 High Latent',
            'E320': 'CE320 High Infiltration',
            'E330': 'CE330 100% OA',
            'E340': 'CE340 50% OA, 50% Infl',
            'E350': 'CE350 Tstat Set Up',
            'E360': 'CE360 Undersized System',
            'E400': 'CE400 Ec. Temp. Ctrl.',
            'E410': 'CE410 Ec. Comp. Lockout',
            'E420': 'CE420 Ec. ODB Limit',
            'E430': 'CE430 Ec. Enthalpy Ctrl.',
            'E440': 'CE440 Ec. Enthalpy Limit',
            'E500': 'CE500 Base w/ 0%OA',
            'E500 May-Sep': 'CE500 May-Sep',
            'E510': 'CE510 High PLR',
            'E510 May-Sep': 'CE510 May-Sep High PLR',
            'E520': 'CE520 EDB = 15°C',
            'E522': 'CE522 EDB = 20°C',
            'E525': 'CE525 EDB = 35°C',
            'E530': 'CE530 Dry Coil',
            'E540': 'CE540 Dry, EDB = 15°C',
            'E545': 'CE545 Dry, EDB = 35°C',
        }
        # some test suites do not include the software names of the reference cases using the same term as the column headings
        if self.section_type == 'CE_a':
            software_column_name_map = {
                'doe21e-88': 'DOE-2.1E/CIEMAT',
                'doe21e-c133': 'DOE-2.1E/NREL',
                'trnsys-14.02.id': 'TRNSYS-ideal/TUD',
                'trnsys-14.02.re': 'TRNSYS-real/TUD',
                'clim2000-2.1.6': 'clim2000/EDF',
                'ca-sis-v1': 'CA-SIS/EDF',
                'energyplus-1.0.0.023': 'EnergyPlus/GARD',
                'analytical-tud-0': 'Analytical/TUD',
                'analytical-htal1-0': 'Analytical/HTAL1',
                'analytical-htal2-0': 'Analytical/HTAL2'
            }
            for name, json_obj in self.json_data.items():
                id_info = json_obj['identifying_information']
                if name in software_column_name_map:
                    id_info['software_column_name'] = software_column_name_map[name]
                elif id_info['software_name'] != 'None':
                    id_info['software_column_name'] = id_info['software_name']
                else:
                    id_info['software_column_name'] = name
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
            # first find the maximum width for each column
            column_widths = [0] * len(table[0])
            for row in table:
                for column_index, cell in enumerate(row):
                    cell_width = len(cell)
                    if cell_width > column_widths[column_index]:
                        column_widths[column_index] = cell_width
            column_format_strings = []
            separator_row = '|'
            for column_index, width in enumerate(column_widths):
                min_width = max(width, 3)
                if column_index == 0:
                    column_format_strings.append('{:<' + str(min_width) + '}')
                    separator_row += ':' + '-' * min_width + ' | '
                else:
                    column_format_strings.append('{:>' + str(min_width) + '}')
                    separator_row += '-' * min_width + ':| '
            for row_index, row in enumerate(table):
                md_string = "| "
                for column_index, cell in enumerate(row):
                    string_cell = str(cell)
                    if string_cell == 'nan':
                        string_cell = ''
                    md_string += column_format_strings[column_index].format(string_cell) + " | "
                md.write(md_string + '\n')
                # header row
                if row_index == 0:
                    md.write(separator_row + '\n')
            md.write('\n')
            for footnote in footnotes:
                md.write(footnote + '\n\n')
            md.write('\n')
        return

    def _add_stats_to_table(self, row_headings, column_headings, data_table, digits=1, time_stamps=[]):
        """
        Add statistics to a table as well as merging the headings and time stamps

        :param row_headings: a list of headings for each row of the table
        :param column_headings: a list of headings for each column of the table including the top left
        :param data_table: a list of lists where the outer list contains rows and each row list contains values
        :param digits: the number of digits shown to the right of the decimal point
        :param time_stamps: a list of lists where the outer list contains rows and each row list contains time stamps
        :return: merged table (list of lists) containing text for every column and row formatted and merged
        """
        formatting_string = '{:.' + str(digits) + 'f}'
        if self.section_type == 'TF':
            final_column_headings = column_headings[:-1]
            final_column_headings.extend(['', 'Min', 'Max', 'Mean', 'Dev % $$', ''])
            final_column_headings.append(column_headings[-1])
        elif self.section_type == 'HE':
            final_column_headings = column_headings[:-2]
            final_column_headings.extend(['', 'Min', 'Max', 'Mean', 'Dev % $$', ''])
            final_column_headings.append(column_headings[-2])
            final_column_headings.append(column_headings[-1])
        elif self.section_type == 'CE_a':
            final_column_headings = column_headings[:-4]
            final_column_headings.extend(['', 'Min', 'Max', 'Dev % $$', ''])
            final_column_headings.append(column_headings[-4])
            final_column_headings.append(column_headings[-3])
            final_column_headings.append(column_headings[-2])
            final_column_headings.append('')
            final_column_headings.append(column_headings[-1])
        elif self.section_type == 'CE_b':
            final_column_headings = column_headings[:-1]
            final_column_headings.extend(['', 'Min', 'Max', 'Mean', 'Dev % $$', ''])
            final_column_headings.append(column_headings[-1])
        text_table_with_stats = [final_column_headings, ]  # list of rows with each row being a list
        for row_index, data_row in enumerate(data_table):
            row = [row_headings[row_index], ]  # first add the heading for the row
            if data_row:  # for blank rows just skip
                if self.section_type == 'TF':
                    for item in data_row[:-1]:
                        row.append(formatting_string.format(item))
                    reference_data_row = self._scrub_number_list(data_row[:-1])  # remove the last item which is the tested software
                elif self.section_type == 'HE':
                    for item in data_row[:-2]:
                        row.append(formatting_string.format(item))
                    reference_data_row = self._scrub_number_list(data_row[:-2])  # remove the last item which is the tested software
                elif self.section_type == 'CE_a':
                    for item in data_row[:-4]:
                        row.append(formatting_string.format(item))
                    reference_data_row = self._scrub_number_list(data_row[:-4])  # remove the last item which is the tested software
                elif self.section_type == 'CE_b':
                    for item in data_row[:-1]:
                        row.append(formatting_string.format(item))
                    reference_data_row = self._scrub_number_list(data_row[:-1])  # remove the last item which is the tested software
                row.append('')
                row_min = min(reference_data_row)
                row.append(formatting_string.format(row_min))
                row_max = max(reference_data_row)
                row.append(formatting_string.format(row_max))
                row_mean = sum(reference_data_row) / len(reference_data_row)
                if self.section_type == 'HE' and 'HE1' in row_headings[row_index]:
                    row_mean = data_row[-2]  # substitute the analytical value for mean
                    row.append('')  # leave the "mean" column empty
                elif self.section_type == 'CE_a':
                    row_mean = sum(data_row[-4:-1]) / 3  # use the average of the three analytical test results
                else:
                    row.append(formatting_string.format(row_mean))
                if row_mean != 0:
                    row_dev = abs((row_max - row_min) / row_mean) * 100
                    row.append('{:.1f}'.format(row_dev))
                else:
                    row.append('-')
                row.append('')
                if self.section_type == 'TF':
                    row.append(formatting_string.format(data_row[-1]))  # now add the last column back
                elif self.section_type == 'HE':
                    row.append(formatting_string.format(data_row[-2]))  # now add the last column back
                    row.append(formatting_string.format(data_row[-1]))  # now add the last column back
                elif self.section_type == 'CE_a':
                    row.append(formatting_string.format(data_row[-4]))
                    row.append(formatting_string.format(data_row[-3]))
                    row.append(formatting_string.format(data_row[-2]))
                    row.append('')
                    row.append(formatting_string.format(data_row[-1]))
                elif self.section_type == 'CE_b':
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
                    row.extend(self._scrub_timestamp_list(time_row[:-1]))
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

    def _scrub_number_list(self, list_in):
        """
        Remove NAN and non-number in a list of number

        :param list_in: a list containing numeric values
        :return: the input list but without non-numbers or NAN
        """
        list_out = []
        for item in list_in:
            if isinstance(item, (int, float)):
                if not math.isnan(item):
                    list_out.append(item)
            elif isinstance(item, str):
                if item.isnumeric():
                    list_out.append(float(item))
        return list_out

    def _scrub_timestamp_list(self, list_in):
        """
        Remove NAN and non-number in a list of timestamps

        :param list_in: a list containing numeric values
        :return: the input list but without non-numbers or NAN
        """
        list_out = []
        for item in list_in:
            if item != 'nan 0-0':
                list_out.append(item)
            else:
                list_out.append('')
        return list_out

    def _create_plotly_bar(self, file_name, table, row_headings, column_headings, yaxis_title, caption):
        """"
        Create a plotly bar chart from a data table.

        :param file_name: name of the figure to be used as a file name without any extension
        :param table: a list of list data without headings
        :param row_headings: a list of headings for each row of the table
        :param column_headings: a list of headings for each colum of the table
        :param yaxis_title: a string for the title of the y-axis
        :param caption: the caption to be used as the title of the chart
        """
        program_name = self.model_results_file.parts[-3].lower()
        version = self.model_results_file.parts[-2].lower()
        destination_directory = root_directory.joinpath('rendered', 'images')
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
                            file_name,
                        ]),
                    'png'
                ]))
        table_with_row_headings = []
        for count, row in enumerate(table):
            row_with_heading = [row_headings[count], ]
            row_with_heading.extend(row)
            table_with_row_headings.append(row_with_heading)
        df = pd.DataFrame(table_with_row_headings, columns=column_headings)
        fig = px.bar(df, x="Case", y=column_headings[1:], text_auto='.2f')
        fig.update_layout(barmode='group', title=dict(text=caption, font=dict(size=25), xanchor='center', x=0.5),
                          yaxis_title=yaxis_title, xaxis_title="")
        # fig.show() # for debugging purposes shows the figure in the browser
        # fig.write_html(file_name + '.html') # save the interactive version of the chart
        fig.write_image(img_name, engine='kaleido', width=1400, height=1000)

    def _create_plotly_line(self, file_name, table, row_headings, column_headings, yaxis_title, caption):
        """"
        Create a plotly bar chart from a data table.

        :param file_name: name of the figure to be used as a file name without any extension
        :param table: a list of list data without headings
        :param row_headings: a list of headings for each row of the table
        :param column_headings: a list of headings for each colum of the table
        :param yaxis_title: a string for the title of the y-axis
        :param caption: the caption to be used as the title of the chart
        """
        program_name = self.model_results_file.parts[-3].lower()
        version = self.model_results_file.parts[-2].lower()
        destination_directory = root_directory.joinpath('rendered', 'images')
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
                            file_name,
                        ]),
                    'png'
                ]))
        table_with_row_headings = []
        for count, row in enumerate(table):
            row_with_heading = [row_headings[count], ]
            row_with_heading.extend(row)
            table_with_row_headings.append(row_with_heading)
        df = pd.DataFrame(table_with_row_headings, columns=column_headings)
        fig = px.line(df, x="Hour", y=column_headings[1:], markers=True)
        fig.update_layout(barmode='group', title=dict(text=caption, font=dict(size=25), xanchor='center', x=0.5),
                          yaxis_title=yaxis_title, xaxis_title="")
        # fig.show() # for debugging purposes shows the figure in the browser
        # fig.write_html(file_name + '.html') # save the interactive version of the chart
        fig.write_image(img_name, engine='kaleido', width=1400, height=1000)

    def render_section_tf_figure_b8_1(self):
        """
        Render Section Thermal Fabric Figure B8-1 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_1')
        return fig, ax

    def render_section_tf_figure_b8_2(self):
        """
        Render Section Thermal Fabric Figure B8-2 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_2')
        return fig, ax

    def render_section_tf_figure_b8_3(self):
        """
        Render Section Thermal Fabric Figure B8-3 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_3')
        return fig, ax

    def render_section_tf_figure_b8_4(self):
        """
        Render Section Thermal Fabric Figure B8-4 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_4')
        return fig, ax

    def render_section_tf_figure_b8_5(self):
        """
        Render Section Thermal Fabric Figure B8-5 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_5')
        return fig, ax

    def render_section_tf_figure_b8_6(self):
        """
        Render Section Thermal Fabric Figure B8-5 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_6')
        return fig, ax

    def render_section_tf_figure_b8_7(self):
        """
        Render Section Thermal Fabric Figure B8-7 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_7')
        return fig, ax

    def render_section_tf_figure_b8_8(self):
        """
        Render Section Thermal Fabric Figure B8-8 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_8')
        return fig, ax

    def render_section_tf_figure_b8_9(self):
        """
        Render Section Thermal Fabric Figure B8-9 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_9')
        return fig, ax

    def render_section_tf_figure_b8_10(self):
        """
        Render Section Thermal Fabric Figure B8-10 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_10')
        return fig, ax

    def render_section_tf_figure_b8_11(self):
        """
        Render Section Thermal Fabric Figure B8-11 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_11')
        return fig, ax

    def render_section_tf_figure_b8_12(self):
        """
        Render Section Thermal Fabric Figure B8-12 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_12')
        return fig, ax

    def render_section_tf_figure_b8_13(self):
        """
        Render Section Thermal Fabric Figure B8-13 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_13')
        return fig, ax

    def render_section_tf_figure_b8_14(self):
        """
        Render Section Thermal Fabric Figure B8-14 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_14')
        return fig, ax

    def render_section_tf_figure_b8_15(self):
        """
        Render Section Thermal Fabric Figure B8-15 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_15')
        return fig, ax

    def render_section_tf_figure_b8_16(self):
        """
        Render Section Thermal Fabric Figure B8-16 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_16')
        return fig, ax

    def render_section_tf_figure_b8_17(self):
        """
        Render Section Thermal Fabric Figure B8-17 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_17')
        return fig, ax

    def render_section_tf_figure_b8_18(self):
        """
        Render Section Thermal Fabric Figure B8-18 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_18')
        return fig, ax

    def render_section_tf_figure_b8_19(self):
        """
        Render Section Thermal Fabric Figure B8-19 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_19')
        return fig, ax

    def render_section_tf_figure_b8_20(self):
        """
        Render Section Thermal Fabric Figure B8-20 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_20')
        return fig, ax

    def render_section_tf_figure_b8_21(self):
        """
        Render Section Thermal Fabric Figure B8-21 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_21')
        return fig, ax

    def render_section_tf_figure_b8_22(self):
        """
        Render Section Thermal Fabric Figure B8-22 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_22')
        return fig, ax

    def render_section_tf_figure_b8_23(self):
        """
        Render Section Thermal Fabric Figure B8-23 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_23')
        return fig, ax

    def render_section_tf_figure_b8_24(self):
        """
        Render Section Thermal Fabric Figure B8-24 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_24')
        return fig, ax

    def render_section_tf_figure_b8_25(self):
        """
        Render Section Thermal Fabric Figure B8-25 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_25')
        return fig, ax

    def render_section_tf_figure_b8_26(self):
        """
        Render Section Thermal Fabric Figure B8-26 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_26')
        return fig, ax

    def render_section_tf_figure_b8_27(self):
        """
        Render Section Thermal Fabric Figure B8-27 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_27')
        return fig, ax

    def render_section_tf_figure_b8_28(self):
        """
        Render Section Thermal Fabric Figure B8-28 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_28')
        return fig, ax

    def render_section_tf_figure_b8_29(self):
        """
        Render Section Thermal Fabric Figure B8-29 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_29')
        return fig, ax

    def render_section_tf_figure_b8_30(self):
        """
        Render Section Thermal Fabric Figure B8-30 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_30')
        return fig, ax

    def render_section_tf_figure_b8_31(self):
        """
        Render Section Thermal Fabric Figure B8-31 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_31')
        return fig, ax

    def render_section_tf_figure_b8_32(self):
        """
        Render Section Thermal Fabric Figure B8-32 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_32')
        return fig, ax

    def render_section_tf_figure_b8_33(self):
        """
        Render Section Thermal Fabric Figure B8-33 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_33')
        return fig, ax

    def render_section_tf_figure_b8_34(self):
        """
        Render Section Thermal Fabric Figure B8-34 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_34')
        return fig, ax

    def render_section_tf_figure_b8_35(self):
        """
        Render Section Thermal Fabric Figure B8-35 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_35')
        return fig, ax

    def render_section_tf_figure_b8_36(self):
        """
        Render Section Thermal Fabric Figure B8-36 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_36')
        return fig, ax

    def render_section_tf_figure_b8_37(self):
        """
        Render Section Thermal Fabric Figure B8-37 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_37')
        return fig, ax

    def render_section_tf_figure_b8_38(self):
        """
        Render Section Thermal Fabric Figure B8-38 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_38')
        return fig, ax

    def render_section_tf_figure_b8_39(self):
        """
        Render Section Thermal Fabric Figure B8-39 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_39')
        return fig, ax

    def render_section_tf_figure_b8_40(self):
        """
        Render Section Thermal Fabric Figure B8-40 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_40')
        return fig, ax

    def render_section_tf_figure_b8_41(self):
        """
        Render Section Thermal Fabric Figure B8-41 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_41')
        return fig, ax

    def render_section_tf_figure_b8_42(self):
        """
        Render Section Thermal Fabric Figure B8-42 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_42')
        return fig, ax

    def render_section_tf_figure_b8_43(self):
        """
        Render Section Thermal Fabric Figure B8-43 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_43')
        return fig, ax

    def render_section_tf_figure_b8_44(self):
        """
        Render Section Thermal Fabric Figure B8-44 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_44')
        return fig, ax

    def render_section_tf_figure_b8_45(self):
        """
        Render Section Thermal Fabric Figure B8-45 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_45')
        return fig, ax

    def render_section_tf_figure_b8_46(self):
        """
        Render Section Thermal Fabric Figure B8-46 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_46')
        return fig, ax

    def render_section_tf_figure_b8_47(self):
        """
        Render Section Thermal Fabric Figure B8-47 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_47')
        return fig, ax

    def render_section_tf_figure_b8_48(self):
        """
        Render Section Thermal Fabric Figure B8-48 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_48')
        return fig, ax

    def render_section_tf_figure_b8_49(self):
        """
        Render Section Thermal Fabric Figure B8-49 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_49')
        return fig, ax

    def render_section_tf_figure_b8_50(self):
        """
        Render Section Thermal Fabric Figure B8-50 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_50')
        return fig, ax

    def render_section_tf_figure_b8_51(self):
        """
        Render Section Thermal Fabric Figure B8-51 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_51')
        return fig, ax

    def render_section_tf_figure_b8_52(self):
        """
        Render Section Thermal Fabric Figure B8-52 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_52')
        return fig, ax

    def render_section_tf_figure_b8_53(self):
        """
        Render Section Thermal Fabric Figure B8-53 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_53')
        return fig, ax

    def render_section_tf_figure_b8_54(self):
        """
        Render Section Thermal Fabric Figure B8-54 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_54')
        return fig, ax

    def render_section_tf_figure_b8_55(self):
        """
        Render Section Thermal Fabric Figure B8-55 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_55')
        return fig, ax

    def render_section_tf_figure_b8_56(self):
        """
        Render Section Thermal Fabric Figure B8-56 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_56')
        return fig, ax

    def render_section_tf_figure_b8_57(self):
        """
        Render Section Thermal Fabric Figure B8-57 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_57')
        return fig, ax

    def render_section_tf_figure_b8_58(self):
        """
        Render Section Thermal Fabric Figure B8-58 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_58')
        return fig, ax

    def render_section_tf_figure_b8_59(self):
        """
        Render Section Thermal Fabric Figure B8-59 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_59')
        return fig, ax

    def render_section_tf_figure_b8_m1(self):
        """
        Render Section Thermal Fabric Figure B8-M1 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m1')
        return fig, ax

    def render_section_tf_figure_b8_m2(self):
        """
        Render Section Thermal Fabric Figure B8-M2 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m2')
        return fig, ax

    def render_section_tf_figure_b8_m3(self):
        """
        Render Section Thermal Fabric Figure B8-M3 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m3')
        return fig, ax

    def render_section_tf_figure_b8_m4(self):
        """
        Render Section Thermal Fabric Figure B8-M4 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m4')
        return fig, ax

    def render_section_tf_figure_b8_m5(self):
        """
        Render Section Thermal Fabric Figure B8-M5 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m5')
        return fig, ax

    def render_section_tf_figure_b8_m6(self):
        """
        Render Section Thermal Fabric Figure B8-M6 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m6')
        return fig, ax

    def render_section_tf_figure_b8_m7(self):
        """
        Render Section Thermal Fabric Figure B8-M7 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m7')
        return fig, ax

    def render_section_tf_figure_b8_m8(self):
        """
        Render Section Thermal Fabric Figure B8-M8 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m8')
        return fig, ax

    def render_section_tf_figure_b8_m9(self):
        """
        Render Section Thermal Fabric Figure B8-M9 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m9')
        return fig, ax

    def render_section_tf_figure_b8_m10(self):
        """
        Render Section Thermal Fabric Figure B8-M10 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m10')
        return fig, ax

    def render_section_tf_figure_b8_m11(self):
        """
        Render Section Thermal Fabric Figure B8-M11 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m11')
        return fig, ax

    def render_section_tf_figure_b8_m12(self):
        """
        Render Section Thermal Fabric Figure B8-M12 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_m12')
        return fig, ax

    def render_section_tf_figure_b8_h1(self):
        """
        Render Section Thermal Fabric Figure B8-H1 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h1',
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

    def render_section_tf_figure_b8_h2(self):
        """
        Render Section Thermal Fabric Figure B8-H2 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h2',
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

    def render_section_tf_figure_b8_h3(self):
        """
        Render Section Thermal Fabric Figure B8-H3 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h3',
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

    def render_section_tf_figure_b8_h4(self):
        """
        Render Section Thermal Fabric Figure B8-H4 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h4',
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

    def render_section_tf_figure_b8_h5(self):
        """
        Render Section Thermal Fabric Figure B8-H5 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h5',
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

    def render_section_tf_figure_b8_h6(self):
        """
        Render Section Thermal Fabric Figure B8-H6 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h6',
            annotations=[
                {
                    'text': 'Double-Pane\n(Case 600)',
                    'xy': (11.5, 70),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h7(self):
        """
        Render Section Thermal Fabric Figure B8-H7 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h7',
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

    def render_section_tf_figure_b8_h8(self):
        """
        Render Section Thermal Fabric Figure B8-H8 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h8',
            annotations=[
                {
                    'text': 'Double-Pane\n(Case 600)',
                    'xy': (11.5, 140),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h9(self):
        """
        Render Section Thermal Fabric Figure B8-H9 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h9',
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

    def render_section_tf_figure_b8_h10(self):
        """
        Render Section Thermal Fabric Figure B8-H10 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h10',
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

    def render_section_tf_figure_b8_h11(self):
        """
        Render Section Thermal Fabric Figure B8-H11 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h11',
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

    def render_section_tf_figure_b8_h12(self):
        """
        Render Section Thermal Fabric Figure B8-H12 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h12',
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

    def render_section_tf_figure_b8_h13(self):
        """
        Render Section Thermal Fabric Figure B8-H13 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h13',
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

    def render_section_tf_figure_b8_h14(self):
        """
        Render Section Thermal Fabric Figure B8-H14 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h14',
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

    def render_section_tf_figure_b8_h15(self):
        """
        Render Section Thermal Fabric Figure B8-H15 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h15',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 1.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h16(self):
        """
        Render Section Thermal Fabric Figure B8-H16 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h16',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h17(self):
        """
        Render Section Thermal Fabric Figure B8-H17 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h17',
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

    def render_section_tf_figure_b8_h18(self):
        """
        Render Section Thermal Fabric Figure B8-H18 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h18',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (12, 17),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h19(self):
        """
        Render Section Thermal Fabric Figure B8-H19 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h19',
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

    def render_section_tf_figure_b8_h20(self):
        """
        Render Section Thermal Fabric Figure B8-H20 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h20',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (13, 17),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h21(self):
        """
        Render Section Thermal Fabric Figure B8-H21 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h21',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h22(self):
        """
        Render Section Thermal Fabric Figure B8-H22 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h22',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h23(self):
        """
        Render Section Thermal Fabric Figure B8-H23 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h23',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 2.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h24(self):
        """
        Render Section Thermal Fabric Figure B8-H24 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h24',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h25(self):
        """
        Render Section Thermal Fabric Figure B8-H25 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h25',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 0.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h26(self):
        """
        Render Section Thermal Fabric Figure B8-H26 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h26',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h27(self):
        """
        Render Section Thermal Fabric Figure B8-H27 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h27',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (11, 1.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h28(self):
        """
        Render Section Thermal Fabric Figure B8-H28 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h28',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h29(self):
        """
        Render Section Thermal Fabric Figure B8-H29 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h29',
            annotations=[
                {
                    'text': 'Cold Day (Feb 14)',
                    'xy': (11, 1.5),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h30(self):
        """
        Render Section Thermal Fabric Figure B8-H30 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h30',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h31(self):
        """
        Render Section Thermal Fabric Figure B8-H31 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h31',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (13, 1.1),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h32(self):
        """
        Render Section Thermal Fabric Figure B8-H32 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h32',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (3, -0.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h33(self):
        """
        Render Section Thermal Fabric Figure B8-H33 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h33',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (12, 0.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h34(self):
        """
        Render Section Thermal Fabric Figure B8-H34 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h34',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -0.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h35(self):
        """
        Render Section Thermal Fabric Figure B8-H35 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h35',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (12, 0.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h36(self):
        """
        Render Section Thermal Fabric Figure B8-H36 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h36',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (4, -1.3),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h37(self):
        """
        Render Section Thermal Fabric Figure B8-H37 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h37',
            annotations=[
                {
                    'text': 'Cold Day (Feb 1)',
                    'xy': (12, 0.7),
                    'fontsize': 18}])
        return fig, ax

    def render_section_tf_figure_b8_h38(self):
        """
        Render Section Thermal Fabric Figure B8-H38 by modifying fig an ax inputs from matplotlib
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
            image_name='section_7_figure_b8_h38',
            annotations=[
                {
                    'text': 'Hot Day (July 14)',
                    'xy': (3, -1.3),
                    'fontsize': 18}])
        return fig, ax

    # def render_section_7b_table_b8_2_1(
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

    def render_section_tf_table_b8_16_alt(self):
        figure_name = 'section_7_table_b8_16_alt'
        caption = 'Table B8-16. Sky Temperatures Output, Case 600'
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]

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
        # convert entire table to strings
        string_table = []
        for row in table:
            string_table.append([str(item) for item in row])
        self._make_markdown_from_table(figure_name, caption, string_table, footnotes)
        return

    def render_section_tf_table_b8_1(self):
        figure_name = 'section_7_table_b8_01'
        caption = 'Table B8-1. Annual Heating Loads (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['conditioned_zone_loads_non_free_float'][case]['annual_heating_MWh'])
            data_table.append(row)
        for blank_row in [44, 35, 21, 11]:
            data_table.insert(blank_row, [])  # add blank line as separator
            row_headings.insert(blank_row, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_2(self):
        figure_name = 'section_7_table_b8_02'
        caption = 'Table B8-2. Annual Sensible Cooling Loads (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['conditioned_zone_loads_non_free_float'][case]['annual_cooling_MWh'])
            data_table.append(row)
        for blank_row in [44, 35, 21, 11]:
            data_table.insert(blank_row, [])  # add blank line as separator
            row_headings.insert(blank_row, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_3(self):
        figure_name = 'section_7_table_b8_03'
        caption = 'Table B8-3. Annual Hourly Integrated Peak Heating Loads (kWh)'
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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
        for blank_row in [44, 35, 21, 11]:
            data_table.insert(blank_row, [])  # add blank line as separator
            time_stamp_table.insert(blank_row, [])
            row_headings.insert(blank_row, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_4(self):
        figure_name = 'section_7_table_b8_04'
        caption = 'Table B8-4. Annual Hourly Integrated Peak Sensible Cooling Loads (kWh)'
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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
        for blank_row in [44, 35, 21, 11]:
            data_table.insert(blank_row, [])  # add blank line as separator
            time_stamp_table.insert(blank_row, [])
            row_headings.insert(blank_row, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_5a(self):
        figure_name = 'section_7_table_b8_05a'
        caption = 'Table B8-5a. Free-Float Temperature Output Maximum Annual Hourly Integrated Zone Temperature (C)'
        free_float_cases = {
            '600FF': '600FF - Low Mass Building with South Windows',
            '900FF': '900FF - High Mass Building with South Windows',
            '650FF': '650FF - Case 600FF with Night Ventilation',
            '950FF': '950FF - Case 900FF with Night Ventilation',
            '680FF': '680FF - Case 600FF with More Insulation',
            '980FF': '980FF - Case 900FF with More Insulation',
            '960': '960 - Sunspace'}
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(free_float_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in free_float_cases.keys():
            row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                case_json = json_obj['free_float_case_zone_temperatures'][case]
                row.append(case_json['maximum_temperature'])
                month = case_json['maximum_month']
                day = self._int_0_if_nan(case_json['maximum_day'])
                hour = self._int_0_if_nan(case_json['maximum_hour'])
                time_stamp_row.append(f'{month} {day}-{hour}')
            data_table.append(row)
            time_stamp_table.append(time_stamp_row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=1, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_5b(self):
        figure_name = 'section_7_table_b8_05b'
        caption = 'Table B8-5b. Free-Float Temperature Output Minimum Annual Hourly Integrated Zone Temperature (C)'
        free_float_cases = {
            '600FF': '600FF - Low Mass Building with South Windows',
            '900FF': '900FF - High Mass Building with South Windows',
            '650FF': '650FF - Case 600FF with Night Ventilation',
            '950FF': '950FF - Case 900FF with Night Ventilation',
            '680FF': '680FF - Case 600FF with More Insulation',
            '980FF': '980FF - Case 900FF with More Insulation',
            '960': '960 - Sunspace'}
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(free_float_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in free_float_cases.keys():
            row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                case_json = json_obj['free_float_case_zone_temperatures'][case]
                row.append(case_json['minimum_temperature'])
                month = case_json['minimum_month']
                day = self._int_0_if_nan(case_json['minimum_day'])
                hour = self._int_0_if_nan(case_json['minimum_hour'])
                time_stamp_row.append(f'{month} {day}-{hour}')
            data_table.append(row)
            time_stamp_table.append(time_stamp_row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=1, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_5c(self):
        figure_name = 'section_7_table_b8_05c'
        caption = 'Table B8-5c. Free-Float Temperature Output Average Annual Hourly Integrated Zone Temperature (C)'
        free_float_cases = {
            '600FF': '600FF - Low Mass Building with South Windows',
            '900FF': '900FF - High Mass Building with South Windows',
            '650FF': '650FF - Case 600FF with Night Ventilation',
            '950FF': '950FF - Case 900FF with Night Ventilation',
            '680FF': '680FF - Case 600FF with More Insulation',
            '980FF': '980FF - Case 900FF with More Insulation',
            '960': '960 - Sunspace'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(free_float_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in free_float_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_json = json_obj['free_float_case_zone_temperatures'][case]
                row.append(case_json['average_temperature'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=1)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_6a(self):
        figure_name = 'section_7_table_b8_06a'
        caption = 'Table B8-6a. Low Mass Basic Sensitivity Tests - Annual Heating (MWh)'
        sensitivity_cases = {
            ('610', '600'): '610 - 600 Heat, S. Shade',
            ('620', '600'): '620 - 600 Heat, E&W Orient',
            ('630', '620'): '630 - 620 Heat, E&W Shade',
            ('640', '600'): '640 - 600 Heat, Htg. Setback',
            ('660', '600'): '660 - 600 Heat, Low-E Win.',
            ('670', '600'): '670 - 600 Heat, 1-Pane Win.',
            ('680', '600'): '680 - 600 Heat, > Ins. 20/27',
            ('685', '600'): '685 - 600 Heat, 20/20 tstat',
            ('695', '685'): '695 - 685 Heat, > Ins. 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_heating_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_heating_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_6b(self):
        figure_name = 'section_7_table_b8_06b'
        caption = 'Table B8-6b. Low Mass Basic Sensitivity Tests - Annual Sensible Cooling (MWh)'
        sensitivity_cases = {
            ('610', '600'): '610 - 600 Cool, S. Shade',
            ('620', '600'): '620 - 600 Cool, E&W Orient',
            ('630', '620'): '630 - 620 Cool, E&W Shade',
            ('640', '600'): '640 - 600 Cool, Htg. Setback',
            ('650', '600'): '650 - 600 Cool, Night Vent',
            ('660', '600'): '660 - 600 Heat, Low-E Win.',
            ('670', '600'): '670 - 600 Heat, 1-Pane Win.',
            ('680', '600'): '680 - 600 Heat, > Ins. 20/27',
            ('685', '600'): '685 - 600 Heat, 20/20 tstat',
            ('695', '685'): '695 - 685 Heat, > Ins. 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_cooling_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_cooling_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_6c(self):
        figure_name = 'section_7_table_b8_06c'
        caption = 'Table B8-6c. Low Mass Basic Sensitivity Tests - Peak Heating (kW)'
        sensitivity_cases = {
            ('610', '600'): '610 - 600 Heat, S. Shade',
            ('620', '600'): '620 - 600 Heat, E&W Orient',
            ('630', '620'): '630 - 620 Heat, E&W Shade',
            ('640', '600'): '640 - 600 Heat, Htg. Setback',
            ('660', '600'): '660 - 600 Heat, Low-E Win.',
            ('670', '600'): '670 - 600 Heat, 1-Pane Win.',
            ('680', '600'): '680 - 600 Heat, > Ins. 20/27',
            ('685', '600'): '685 - 600 Heat, 20/20 tstat',
            ('695', '685'): '695 - 685 Heat, > Ins. 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_heating_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_heating_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_6d(self):
        figure_name = 'section_7_table_b8_06d'
        caption = 'Table B8-6d. Low Mass Basic Sensitivity Tests - Peak Sensible Cooling (kW)'
        sensitivity_cases = {
            ('610', '600'): '610 - 600 Cool, S. Shade',
            ('620', '600'): '620 - 600 Cool, E&W Orient',
            ('630', '620'): '630 - 620 Cool, E&W Shade',
            ('640', '600'): '640 - 600 Cool, Htg. Setback',
            ('650', '600'): '650 - 600 Cool, Night Vent',
            ('660', '600'): '660 - 600 Heat, Low-E Win.',
            ('670', '600'): '670 - 600 Heat, 1-Pane Win.',
            ('680', '600'): '680 - 600 Heat, > Ins. 20/27',
            ('685', '600'): '685 - 600 Heat, 20/20 tstat',
            ('695', '685'): '695 - 685 Heat, > Ins. 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_cooling_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_cooling_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_7a(self):
        figure_name = 'section_7_table_b8_07a'
        caption = 'Table B8-7a. High Mass Basic Sensitivity Tests - Annual Heating (MWh)'
        sensitivity_cases = {
            ('900', '600'): '900 - 600 Mass, Heat',
            ('910', '900'): '910 - 900 Heat, S.Shade',
            ('920', '900'): '920 - 900 Heat, E&W Orient.',
            ('930', '920'): '930 - 920 Heat, E&W Shade',
            ('940', '900'): '940 - 900 Heat, Htg. Setback',
            ('960', '900'): '960 - 900 Heat, Sunspace',
            ('980', '900'): '980 - 900 Heat, > Ins. 20/27',
            ('985', '900'): '985 - 900 Heat, > 20/20 tstat',
            ('995', '985'): '995 - 985 Heat, > Ins. 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_heating_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_heating_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_7b(self):
        figure_name = 'section_7_table_b8_07b'
        caption = 'Table B8-7b. High Mass Basic Sensitivity Tests - Annual Sensible Cooling (MWh)'
        sensitivity_cases = {
            ('900', '600'): '900 - 600 Mass, Cool',
            ('910', '900'): '910 - 900 Cool, S.Shade',
            ('920', '900'): '920 - 900 Cool, E&W Orient.',
            ('930', '920'): '930 - 920 Cool, E&W Shade',
            ('940', '900'): '940 - 900 Cool, Htg. Setback',
            ('950', '900'): '950 - 900 Cool, Night Vent',
            ('960', '900'): '960 - 900 Cool, Sunspace',
            ('980', '900'): '980 - 900 Heat, > Ins. 20/27',
            ('985', '900'): '985 - 900 Heat, > 20/20 tstat',
            ('995', '985'): '995 - 985 Heat, > Ins. 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_cooling_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_cooling_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_7c(self):
        figure_name = 'section_7_table_b8_07c'
        caption = 'Table B8-7c. High Mass Basic Sensitivity Tests - Peak Heating (kW)'
        sensitivity_cases = {
            ('900', '600'): '900 - 600 Mass, Heat',
            ('910', '900'): '910 - 900 Heat, S.Shade',
            ('920', '900'): '920 - 900 Heat, E&W Orient.',
            ('930', '920'): '930 - 920 Heat, E&W Shade',
            ('940', '900'): '940 - 900 Heat, Htg. Setback',
            ('960', '900'): '960 - 900 Heat, Sunspace',
            ('980', '900'): '980 - 900 Heat, > Ins. 20/27',
            ('985', '900'): '985 - 900 Heat, > 20/20 tstat',
            ('995', '985'): '995 - 985 Heat, > Ins. 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_heating_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_heating_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_7d(self):
        figure_name = 'section_7_table_b8_07d'
        caption = 'Table B8-7d. High Mass Basic Sensitivity Tests - Peak Sensible Cooling (kW)'
        sensitivity_cases = {
            ('900', '600'): '900 - 600 Mass, Cool',
            ('910', '900'): '910 - 900 Cool, S.Shade',
            ('920', '900'): '920 - 900 Cool, E&W Orient.',
            ('930', '920'): '930 - 920 Cool, E&W Shade',
            ('940', '900'): '940 - 900 Cool, Htg. Setback',
            ('950', '900'): '950 - 900 Cool, Night Vent',
            ('960', '900'): '960 - 900 Cool, Sunspace',
            ('980', '900'): '980 - 900 Heat, > Ins. 20/27',
            ('985', '900'): '985 - 900 Heat, > 20/20 tstat',
            ('995', '985'): '995 - 985 Heat, > Ins. 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_cooling_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_cooling_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_8a(self):
        figure_name = 'section_7_table_b8_08a'
        caption = 'Table B8-8a. Low Mass In-Depth (Cases 195 thru 320) Sensitivity Tests - Annual Heating (MWh)'
        sensitivity_cases = {
            ('200', '195'): '200-195 Surface Convection',
            ('210', '200'): '210-200 Ext IR (Int IR "off")',
            ('220', '215'): '220-215 Ext IR (Int IR "on")',
            ('215', '200'): '215-200 Int IR (Ext IR "off")',
            ('220', '210'): '220-210 Int IR (Ext IR "on")',
            ('230', '220'): '230-220 Infiltration',
            ('240', '220'): '240-220 Internal Gains',
            ('250', '220'): '250-220 Ext Solar Abs.',
            ('270', '220'): '270-220 South Windows',
            ('280', '270'): '280-270 Cavity Albedo',
            ('320', '270'): '320-270 Thermostat',
            ('290', '270'): '290-270 South Shading',
            ('300', '270'): '300-270 E&W Windows',
            ('310', '300'): '310-300 E&W Shading'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_heating_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_heating_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_8b(self):
        figure_name = 'section_7_table_b8_08b'
        caption = 'Table B8-8b. Low Mass In-Depth (Cases 195 thru 320) Sensitivity Tests - Annual Sensible Cooling (MWh)'
        sensitivity_cases = {
            ('200', '195'): '200-195 Surface Convection',
            ('210', '200'): '210-200 Ext IR (Int IR "off")',
            ('220', '215'): '220-215 Ext IR (Int IR "on")',
            ('215', '200'): '215-200 Int IR (Ext IR "off")',
            ('220', '210'): '220-210 Int IR (Ext IR "on")',
            ('230', '220'): '230-220 Infiltration',
            ('240', '220'): '240-220 Internal Gains',
            ('250', '220'): '250-220 Ext Solar Abs.',
            ('270', '220'): '270-220 South Windows',
            ('280', '270'): '280-270 Cavity Albedo',
            ('320', '270'): '320-270 Thermostat',
            ('290', '270'): '290-270 South Shading',
            ('300', '270'): '300-270 E&W Windows',
            ('310', '300'): '310-300 E&W Shading'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_cooling_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_cooling_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_8c(self):
        figure_name = 'section_7_table_b8_08c'
        caption = 'Table B8-8c. Low Mass In-Depth (Cases 195 thru 320) Sensitivity Tests - Peak Heating (kW)'
        sensitivity_cases = {
            ('200', '195'): '200-195 Surface Convection',
            ('210', '200'): '210-200 Ext IR (Int IR "off")',
            ('220', '215'): '220-215 Ext IR (Int IR "on")',
            ('215', '200'): '215-200 Int IR (Ext IR "off")',
            ('220', '210'): '220-210 Int IR (Ext IR "on")',
            ('230', '220'): '230-220 Infiltration',
            ('240', '220'): '240-220 Internal Gains',
            ('250', '220'): '250-220 Ext Solar Abs.',
            ('270', '220'): '270-220 South Windows',
            ('280', '270'): '280-270 Cavity Albedo',
            ('320', '270'): '320-270 Thermostat',
            ('290', '270'): '290-270 South Shading',
            ('300', '270'): '300-270 E&W Windows',
            ('310', '300'): '310-300 E&W Shading'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_heating_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_heating_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_8d(self):
        figure_name = 'section_7_table_b8_08d'
        caption = 'Table B8-8d. Low Mass In-Depth (Cases 195 thru 320) Sensitivity Tests - Peak Sensible Cooling (kW)'
        sensitivity_cases = {
            ('200', '195'): '200-195 Surface Convection',
            ('210', '200'): '210-200 Ext IR (Int IR "off")',
            ('220', '215'): '220-215 Ext IR (Int IR "on")',
            ('215', '200'): '215-200 Int IR (Ext IR "off")',
            ('220', '210'): '220-210 Int IR (Ext IR "on")',
            ('230', '220'): '230-220 Infiltration',
            ('240', '220'): '240-220 Internal Gains',
            ('250', '220'): '250-220 Ext Solar Abs.',
            ('270', '220'): '270-220 South Windows',
            ('280', '270'): '280-270 Cavity Albedo',
            ('320', '270'): '320-270 Thermostat',
            ('290', '270'): '290-270 South Shading',
            ('300', '270'): '300-270 E&W Windows',
            ('310', '300'): '310-300 E&W Shading'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_cooling_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_cooling_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_9a(self):
        figure_name = 'section_7_table_b8_09a'
        caption = 'Table B8-9a. Low Mass In-Depth (Cases 395 thru 440) sensitivity Tests - Annual Heating (MWh)'
        sensitivity_cases = {
            ('400', '395'): '400-395 Surf. Conv. & IR',
            ('410', '400'): '410-400 Infiltration',
            ('420', '410'): '420-410 Internal Gains',
            ('430', '420'): '430-420 Ext Solar Abs.',
            ('600', '430'): '600-430 South Windows',
            ('440', '600'): '440-600 Cavity Albedo',
            ('450', '600'): '450-600 Const Int&Ext Surf Coefs',
            ('460', '600'): '460-600 Const Int Surf Coefs',
            ('460', '450'): '460-450 Auto Ext Surf Heat Transf',
            ('470', '600'): '470-600 Const Ext Surf Coefs',
            ('470', '450'): '470-450 Auto Int Surf Heat Transf'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_heating_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_heating_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_9b(self):
        figure_name = 'section_7_table_b8_09b'
        caption = 'Table B8-9b. Low Mass In-Depth (Cases 395 thru 440) Sensitivity Tests - Annual Sensible Cooling (MWh)'
        sensitivity_cases = {
            ('400', '395'): '400-395 Surf. Conv. & IR',
            ('410', '400'): '410-400 Infiltration',
            ('420', '410'): '420-410 Internal Gains',
            ('430', '420'): '430-420 Ext Solar Abs.',
            ('600', '430'): '600-430 South Windows',
            ('440', '600'): '440-600 Cavity Albedo',
            ('450', '600'): '450-600 Const Int&Ext Surf Coefs',
            ('460', '600'): '460-600 Const Int Surf Coefs',
            ('460', '450'): '460-450 Auto Ext Surf Heat Transf',
            ('470', '600'): '470-600 Const Ext Surf Coefs',
            ('470', '450'): '470-450 Auto Int Surf Heat Transf'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_cooling_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_cooling_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_9c(self):
        figure_name = 'section_7_table_b8_09c'
        caption = 'Table B8-9c. Low Mass In-Depth (Cases 395 thru 440) Sensitivity Tests - Peak Heating (kW)'
        sensitivity_cases = {
            ('400', '395'): '400-395 Surf. Conv. & IR',
            ('410', '400'): '410-400 Infiltration',
            ('420', '410'): '420-410 Internal Gains',
            ('430', '420'): '430-420 Ext Solar Abs.',
            ('600', '430'): '600-430 South Windows',
            ('440', '600'): '440-600 Cavity Albedo',
            ('450', '600'): '450-600 Const Int&Ext Surf Coefs',
            ('460', '600'): '460-600 Const Int Surf Coefs',
            ('460', '450'): '460-450 Auto Ext Surf Heat Transf',
            ('470', '600'): '470-600 Const Ext Surf Coefs',
            ('470', '450'): '470-450 Auto Int Surf Heat Transf'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_heating_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_heating_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_9d(self):
        figure_name = 'section_7_table_b8_09d'
        caption = 'Table B8-9d. Low Mass In-Depth (Cases 395 thru 440) Sensitivity Tests - Peak Sensible Cooling (kW)'
        sensitivity_cases = {
            ('400', '395'): '400-395 Surf. Conv. & IR',
            ('410', '400'): '410-400 Infiltration',
            ('420', '410'): '420-410 Internal Gains',
            ('430', '420'): '430-420 Ext Solar Abs.',
            ('600', '430'): '600-430 South Windows',
            ('440', '600'): '440-600 Cavity Albedo',
            ('450', '600'): '450-600 Const Int&Ext Surf Coefs',
            ('460', '600'): '460-600 Const Int Surf Coefs',
            ('460', '450'): '460-450 Auto Ext Surf Heat Transf',
            ('470', '600'): '470-600 Const Ext Surf Coefs',
            ('470', '450'): '470-450 Auto Int Surf Heat Transf'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_cooling_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_cooling_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_10a(self):
        figure_name = 'section_7_table_b8_10a'
        caption = 'Table B8-10a. High Mass Basic and In-Depth Sensitivity Tests - Annual Heating (MWh)'
        sensitivity_cases = {
            ('800', '430'): '800-430 Mass, w/ High Cond. Wall',
            ('900', '800'): '900-800 Himass, S. Win.',
            ('900', '810'): '900-810 Himass, Int. Solar Abs.',
            ('910', '610'): '910-610 Mass, w/ S. Shade',
            ('920', '620'): '920-620 Mass, w/ E&W Win.',
            ('930', '630'): '930-630 Mass, w/ E&W Shade',
            ('940', '640'): '940-640 Mass, w/ Htg. Setback',
            ('980', '680'): '980-680 Mass, w/ Insulation 20/27',
            ('985', '685'): '985-685 Mass, w/ 20/20 Tstat',
            ('995', '695'): '995-695 Mass, w/ Insulation 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_heating_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_heating_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_10b(self):
        figure_name = 'section_7_table_b8_10b'
        caption = 'Table B8-10b. High Mass Basic and In-Depth Sensitivity Tests - Annual Sensible Cooling (MWh)'
        sensitivity_cases = {
            ('800', '430'): '800-430 Mass, w/ High Cond. Wall',
            ('900', '800'): '900-800 Himass, S. Win.',
            ('900', '810'): '900-810 Himass, Int. Solar Abs.',
            ('910', '610'): '910-610 Mass, w/ S. Shade',
            ('920', '620'): '920-620 Mass, w/ E&W Win.',
            ('930', '630'): '930-630 Mass, w/ E&W Shade',
            ('940', '640'): '940-640 Mass, w/ Htg. Setback',
            ('950', '650'): '940-640 Mass, w/ Night Vent',
            ('980', '680'): '980-680 Mass, w/ Insulation 20/27',
            ('985', '685'): '985-685 Mass, w/ 20/20 Tstat',
            ('995', '695'): '995-695 Mass, w/ Insulation 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['annual_cooling_MWh']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['annual_cooling_MWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_10c(self):
        figure_name = 'section_7_table_b8_10c'
        caption = 'Table B8-10c. High Mass Basic and In-Depth Sensitivity Tests - Peak Heating (kW)'
        sensitivity_cases = {
            ('800', '430'): '800-430 Mass, w/ High Cond. Wall',
            ('900', '800'): '900-800 Himass, S. Win.',
            ('900', '810'): '900-810 Himass, Int. Solar Abs.',
            ('910', '610'): '910-610 Mass, w/ S. Shade',
            ('920', '620'): '920-620 Mass, w/ E&W Win.',
            ('930', '630'): '930-630 Mass, w/ E&W Shade',
            ('940', '640'): '940-640 Mass, w/ Htg. Setback',
            ('980', '680'): '980-680 Mass, w/ Insulation 20/27',
            ('985', '685'): '985-685 Mass, w/ 20/20 Tstat',
            ('995', '695'): '995-695 Mass, w/ Insulation 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_heating_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_heating_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_10d(self):
        figure_name = 'section_7_table_b8_10d'
        caption = 'Table B8-10d. High Mass Basic and In-Depth Sensitivity Tests - Peak Sensible Cooling (kW)'
        sensitivity_cases = {
            ('800', '430'): '800-430 Mass, w/ High Cond. Wall',
            ('900', '800'): '900-800 Himass, S. Win.',
            ('900', '810'): '900-810 Himass, Int. Solar Abs.',
            ('910', '610'): '910-610 Mass, w/ S. Shade',
            ('920', '620'): '920-620 Mass, w/ E&W Win.',
            ('930', '630'): '930-630 Mass, w/ E&W Shade',
            ('940', '640'): '940-640 Mass, w/ Htg. Setback',
            ('950', '650'): '940-640 Mass, w/ Night Vent',
            ('980', '680'): '980-680 Mass, w/ Insulation 20/27',
            ('985', '685'): '985-685 Mass, w/ 20/20 Tstat',
            ('995', '695'): '995-695 Mass, w/ Insulation 20/20'}
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(sensitivity_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_a, case_b) in sensitivity_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['conditioned_zone_loads_non_free_float'][case_a]['peak_cooling_kW']
                case_b_value = json_obj['conditioned_zone_loads_non_free_float'][case_b]['peak_cooling_kW']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_11(self):
        figure_name = 'section_7_table_b8_11'
        caption = 'Table B8-11. Annual Transmissivity Coefficient of Windows'
        transmitted_cases = {
            '600 South': ('600', 'South'),
            '620 West': ('620', 'West'),
            '660 South, Low-E': ('660', 'South'),
            '670 South, Single Pane': ('670', 'South')
        }
        data_table = []
        footnotes = ['Annual Unshaded Transmitted Solar Radiation/Annual Unshaded Incident Solar Radiation',
                     '$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]']
        row_headings = list(transmitted_cases.keys())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_num, case_direction) in transmitted_cases.values():
            row = []
            for tst, json_obj in self.json_data.items():
                incident_600 = json_obj['solar_radiation_annual_incident']['600']['Surface'][case_direction.upper()]['kWh/m2']
                transmitted_value = json_obj['solar_radiation_unshaded_annual_transmitted'][case_num]['Surface'][case_direction]['kWh/m2']
                if incident_600 != 0:
                    row.append(float(transmitted_value) / float(incident_600))
                else:
                    row.append('')
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_12(self):
        figure_name = 'section_7_table_b8_12'
        caption = 'Table B8-12. Annual Shading Coefficient of Window Shading Devices: Overhangs & Fins'
        coefficient_cases = {
            '610/600 South': ('610', '600', 'South'),
            '630/620 West': ('630', '620', 'West')}
        data_table = []
        footnotes = ['(1-(Annual Shaded Transmitted Solar Radiation)/(Annual Unshaded Transmitted Solar Radiation))'
                     '$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]']
        row_headings = list(coefficient_cases.keys())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_shaded, case_unshaded, case_direction) in coefficient_cases.values():
            row = []
            for tst, json_obj in self.json_data.items():
                shaded = json_obj['solar_radiation_shaded_annual_transmitted'][case_shaded]['Surface'][case_direction]['kWh/m2']
                unshaded = json_obj['solar_radiation_unshaded_annual_transmitted'][case_unshaded]['Surface'][case_direction]['kWh/m2']
                if unshaded != 0:
                    row.append(1 - (float(shaded) / float(unshaded)))
                else:
                    row.append('')
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_13(self):
        figure_name = 'section_7_table_b8_13'
        caption = 'Table B8-13. Case 600 Annual Incident Solar Radiation (kWh/m2)'
        directions = {
            'HORZ.': 'Horizontal',
            'NORTH': 'North',
            'EAST': 'East',
            'SOUTH': 'South',
            'WEST': 'West',
        }
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(directions.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case_direction in directions.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['solar_radiation_annual_incident']['600']['Surface'][case_direction]['kWh/m2'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_14(self):
        figure_name = 'section_7_table_b8_14'
        caption = 'Table B8-14. Annual Transmitted Solar Radiation - Unshaded (kWh/m2)'
        transmitted_cases = {
            '600 South': ('600', 'South'),
            '620 West': ('620', 'West'),
            '660 South': ('660', 'South'),
            '670 South': ('670', 'South')
        }
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(transmitted_cases.keys())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_num, case_direction) in transmitted_cases.values():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['solar_radiation_unshaded_annual_transmitted'][case_num]['Surface'][case_direction]['kWh/m2'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_15(self):
        figure_name = 'section_7_table_b8_15'
        caption = 'Table B8-15. Annual Transmitted Solar Radiation - Shaded (kWh/m2)'
        transmitted_cases = {
            '610 South': ('610', 'South'),
            '630 West': ('630', 'West'),
        }
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(transmitted_cases.keys())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for (case_num, case_direction) in transmitted_cases.values():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['solar_radiation_shaded_annual_transmitted'][case_num]['Surface'][case_direction]['kWh/m2'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_16(self):
        figure_name = 'section_7_table_b8_16'
        caption = 'Table B8-16. Sky Temperatures Output, Case 600'
        cases = {
            'Annual Hourly Integrated Average': 'Average',
            'Annual Hourly Integrated Minimum': 'Minimum',
            'Annual Hourly Integrated Maximum': 'Maximum'
        }
        test_spec_alt = {
            'Average': -5.9,
            'Minimum': -46.9,
            'Maximum': 24.6
        }
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(cases.keys())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        column_headings.insert(len(column_headings) - 1, 'TestSpec-Alt')
        for case in cases.values():
            row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                sky_600 = json_obj['sky_temperature_output']['600'][case]
                row.append(sky_600['C'])
                if case != 'Average':
                    month = sky_600['Month']
                    day = self._int_0_if_nan(sky_600['Day'])
                    hour = self._int_0_if_nan(sky_600['Hour'])
                    time_stamp_row.append(f'{month} {day}-{hour}')
                else:
                    time_stamp_row.append('')
            # add the test spec column
            row.insert(len(row) - 1, test_spec_alt[case])
            time_stamp_row.insert(len(time_stamp_row) - 1, '')
            data_table.append(row)
            time_stamp_table.append(time_stamp_row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=1, time_stamps=time_stamp_table)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_m1a(self):  # case 600
        figure_name = 'section_7_table_b8_m1a'
        caption = 'Table B8-M1a. Monthly Heating Loads (kWh), Case 600'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m1b(self):  # case 900
        figure_name = 'section_7_table_b8_m1b'
        caption = 'Table B8-M1b. Monthly Heating Loads (kWh), Case 900'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m2a(self):  # case 600
        figure_name = 'section_7_table_b8_m2a'
        caption = 'Table B8-M2a. Monthly Sensible Cooling Loads (kWh), Case 600'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m2b(self):
        figure_name = 'section_7_table_b8_m2b'
        caption = 'Table B8-M2b. Monthly Sensible Cooling Loads (kWh), Case 900'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m3a(self):  # case 900
        figure_name = 'section_7_table_b8_m3a'
        caption = 'Table B8-M3a. Monthly Hourly Integrated Peak Heating Loads (kW), Case 600'
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m3b(self):  # case 900
        figure_name = 'section_7_table_b8_m3b'
        caption = 'Table B8-M3b. Monthly Hourly Integrated Peak Heating Loads (kW), Case 900'
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m4a(self):  # case 900
        figure_name = 'section_7_table_b8_m4a'
        caption = 'Table B8-M4a. Monthly Hourly Integrated Peak Sensible Cooling Loads (kW), Case 600'
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m4b(self):  # case 900
        figure_name = 'section_7_table_b8_m4b'
        caption = 'Table B8-M4b. Monthly Hourly Integrated Peak Sensible Cooling Loads (kW), Case 900'
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m5a(self):  # case 600
        figure_name = 'section_7_table_b8_m5a'
        caption = 'Table B8-M5a. Monthly Load 600-900 Sensitivity Tests - Annual Heating (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m5b(self):
        figure_name = 'section_7_table_b8_m5b'
        caption = 'Table B8-M5b. Monthly Load 600-900 Sensitivity Tests - Annual Sensible Cooling (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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

    def render_section_tf_table_b8_m5c(self):
        figure_name = 'section_7_table_b8_m5c'
        caption = 'Table B8-M5c. Monthly Load 600-900 Sensitivity Tests - Peak Heating (kW)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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
                value_900 = float(json_obj['monthly_conditioned_zone_loads']['900'][month]['peak_heating_kw'])
                if math.isnan(value_600) or math.isnan(value_900):
                    row.append(math.nan)
                else:
                    row.append(value_600 - value_900)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, 3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_tf_table_b8_m5d(self):
        figure_name = 'section_7_table_b8_m5d'
        caption = 'Table B8-M5d. Monthly Load 600-900 Sensitivity Tests - Peak Sensible Cooling (kW)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
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
                value_900 = float(json_obj['monthly_conditioned_zone_loads']['900'][month]['peak_cooling_kw'])
                if math.isnan(value_600) or math.isnan(value_900):
                    row.append(math.nan)
                else:
                    row.append(value_600 - value_900)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, 3)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        return

    def render_section_he_table_b16_6_1(self):
        figure_name = 'section_10_table_b16_6_01'
        caption = 'Table B16.6-1. Total Furnace Load (GJ)'
        figure_caption = 'Figure B16.6-1. Comparison of the Energy Delivered for the Fuel-Fired Furnace Test Cases'
        yaxis_name = 'Total Furnace Load (GJ)'
        data_table = []
        footnotes = ['$$ For HE1xx cases ABS[ (Max-Min) / (Analytic Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['furnace_loads'][case])
            data_table.append(row)
        data_table.insert(8, [])  # add blank line as separator
        row_headings.insert(8, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(figure_name, data_table, row_headings, column_headings, yaxis_name, figure_caption)
        return

    def render_section_he_table_b16_6_2(self):
        figure_name = 'section_10_table_b16_6_02'
        caption = 'Table B16.6-2. Total Furnace Input (GJ)'
        figure_caption = 'Figure B16.6-2. Comparison of the Energy Consumed for the Fuel-Fired Furnace Test Cases'
        yaxis_name = 'Total Furnace Input (GJ)'
        data_table = []
        footnotes = ['$$ For HE1xx cases ABS[ (Max-Min) / (Analytic Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['furnace_input'][case])
            data_table.append(row)
        data_table.insert(8, [])  # add blank line as separator
        row_headings.insert(8, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(figure_name, data_table, row_headings, column_headings, yaxis_name, figure_caption)
        return

    def render_section_he_table_b16_6_3(self):
        figure_name = 'section_10_table_b16_6_03'
        caption = 'Table B16.6-3. Fuel Consumption (m3/2)'
        figure_caption = 'Figure B16.6-3. Comparison of the Fuel Consumed for the Fuel-Fired Furnace Test Cases'
        yaxis_name = 'Fuel Consumption (m3/s)'
        data_table = []
        footnotes = ['$$ For HE1xx cases ABS[ (Max-Min) / (Analytic Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['fuel_consumption'][case])
            data_table.append(row)
        data_table.insert(8, [])  # add blank line as separator
        row_headings.insert(8, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=6)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(figure_name, data_table, row_headings, column_headings, yaxis_name, figure_caption)
        return

    def render_section_he_table_b16_6_4(self):
        figure_name = 'section_10_table_b16_6_04'
        caption = 'Table B16.6-4. Fan Energy, both fans (kWh)'
        figure_caption = 'Figure B16.6-4. Comparison of the Fan Energy for the Fuel-Fired Furnace Test Cases'
        yaxis_name = 'Fuel Consumption (m3/s)'
        fan_energy_cases = {
            'HE150': 'HE150 Continuous Circ. Fan',
            'HE160': 'HE160 Cycling Circ. Fan',
            'HE170': 'HE170 Draft Fan',
            'HE210': 'HE210 Realistic Weather',
            'HE220': 'HE220 Setback Thermostat',
            'HE230': 'HE230 Undersized Furnace'
        }
        data_table = []
        footnotes = ['$$ For HE1xx cases ABS[ (Max-Min) / (Analytic Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(fan_energy_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in fan_energy_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['fan_energy'][case])
            data_table.append(row)
        data_table.insert(3, [])  # add blank line as separator
        row_headings.insert(3, '')
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=1)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(figure_name, data_table, row_headings, column_headings, yaxis_name, figure_caption)
        return

    def render_section_he_table_b16_6_5(self):
        figure_name = 'section_10_table_b16_6_05'
        caption = 'Table B16.6-5. Mean Zone Temperature (C)'
        figure_caption = ('Figure B16.6-5. Comparison of the Mean Zone Temperature for the Fuel-Fired Furnace '
                          'Comparative Test Cases')
        yaxis_name = "Mean Zone Temperature (C)"
        two_hundred_cases = {
            'HE210': 'HE210 Realistic Weather',
            'HE220': 'HE220 Setback Thermostat',
            'HE230': 'HE230 Undersized Furnace'
        }
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(two_hundred_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case, case_description in two_hundred_cases.items():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['mean_zone_temperature'][case])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(figure_name, data_table, row_headings, column_headings, yaxis_name, figure_caption)
        return

    def render_section_he_table_b16_6_6(self):
        figure_name = 'section_10_table_b16_6_06'
        caption = 'Table B16.6-6. Maximum Zone Temperature (C)'
        figure_caption = ('Figure B16.6-6. Comparison of the Maximum Zone Temperature for the Fuel-Fired Furnace '
                          'Comparative Test Cases')
        yaxis_name = "Maximum Zone Temperature (C)"
        two_hundred_cases = {
            'HE210': 'HE210 Realistic Weather',
            'HE220': 'HE220 Setback Thermostat',
            'HE230': 'HE230 Undersized Furnace'
        }
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(two_hundred_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in two_hundred_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['maximum_zone_temperature'][case])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(figure_name, data_table, row_headings, column_headings, yaxis_name, figure_caption)
        return

    def render_section_he_table_b16_6_7(self):
        figure_name = 'section_10_table_b16_6_07'
        caption = 'Table B16.6-7. Minimum Zone Temperature (C)'
        figure_caption = ('Figure B16.6-7. Comparison of the Minimum Zone Temperature for the Fuel-Fired Furnace '
                          'Comparative Test Cases')
        yaxis_name = "Minimum Zone Temperature (C)"
        two_hundred_cases = {
            'HE210': 'HE210 Realistic Weather',
            'HE220': 'HE220 Setback Thermostat',
            'HE230': 'HE230 Undersized Furnace'
        }
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(two_hundred_cases.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in two_hundred_cases.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['minimum_zone_temperature'][case])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(figure_name, caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(figure_name, data_table, row_headings, column_headings, yaxis_name, figure_caption)
        return

    def render_section_ce_a_table_b16_5_1_01a(self):
        table_name = 'section_9_table_b16_5_1_01a'
        table_caption = 'Table B16.5.1-1a. Space Cooling Energy Consumption - Total (kWh,e)'
        chart_name = 'section_9_figure_b16_5_1_04'
        chart_caption = 'Figure B16.5.1-4. HVAC BESTEST: Total Space Cooling Electricity Consumption'
        yaxis_name = 'Electricity Consumption  (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['cooling_energy_total_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_01b(self):
        table_name = 'section_9_table_b16_5_1_01b'
        table_caption = 'Table B16.5.1-1b. Space Cooling Energy Consumption - Compressor (kWh,e)'
        chart_name = 'section_9_figure_b16_5_1_06'
        chart_caption = 'Figure B16.5.1-6. HVAC BESTEST: Compressor Electricity Consumption'
        yaxis_name = 'Electricity Consumption  (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['cooling_energy_compressor_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_01c(self):
        table_name = 'section_9_table_b16_5_1_01c'
        table_caption = 'Table B16.5.1-1c. Space Cooling Energy Consumption - Supply Fan (kWh,e)'
        chart_name = 'section_9_figure_b16_5_1_08'
        chart_caption = 'Figure B16.5.1-8. HVAC BESTEST: Total Indoor (Supply) Fan Electricity Consumption'
        yaxis_name = 'Electricity Consumption  (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['supply_fan_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_01d(self):
        table_name = 'section_9_table_b16_5_1_01d'
        table_caption = 'Table B16.5.1-1d. Space Cooling Energy Consumption - Condenser Fan (kWh,e)'
        chart_name = 'section_9_figure_b16_5_1_10'
        chart_caption = 'Figure B16.5.1-10. HVAC BESTEST: Outdoor (Condenser) Fan Electricity Consumption'
        yaxis_name = 'Electricity Consumption  (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['condenser_fan_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_02a(self):
        table_name = 'section_9_table_b16_5_1_02a'
        table_caption = 'Table B16.5.1-2a. COP Mean'
        chart_name = 'section_9_figure_b16_5_1_01'
        chart_caption = 'Figure B16.5.1-1. HVAC BESTEST: Mean COP'
        yaxis_name = 'COP'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['feb_mean_cop'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_02b(self):
        table_name = 'section_9_table_b16_5_1_02b'
        table_caption = 'Table B16.5.1-2b. COP (Max-Min)/Mean'
        chart_name = 'section_9_figure_b16_5_1_02'
        chart_caption = 'Figure B16.5.1-2. HVAC BESTEST: (Maximum - Minimum)/Mean COP'
        yaxis_name = 'Fractional Variation'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                min_value = json_obj['main_table'][case]['feb_min_cop']
                max_value = json_obj['main_table'][case]['feb_max_cop']
                mean_value = json_obj['main_table'][case]['feb_mean_cop']
                if math.isnan(min_value) or math.isnan(max_value) or math.isnan(mean_value):
                    row.append(math.nan)
                else:
                    row.append((max_value - min_value) / mean_value)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_03a(self):
        table_name = 'section_9_table_b16_5_1_03a'
        table_caption = 'Table B16.5.1-3a. Coil Loads, Total (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_1_12'
        chart_caption = 'Figure B16.5.1-12. HVAC BESTEST: Total Coil Load'
        yaxis_name = 'Load (kWh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['evaporator_load_total_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_03b(self):
        table_name = 'section_9_table_b16_5_1_03b'
        table_caption = 'Table B16.5.1-3b. Coil Loads, Sensible (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_1_14'
        chart_caption = 'Figure B16.5.1-14. HVAC BESTEST: Sensible Coil Load'
        yaxis_name = 'Load (kWh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['evaporator_load_sensible_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_03c(self):
        table_name = 'section_9_table_b16_5_1_03c'
        table_caption = 'Table B16.5.1-3c. Coil Loads, Latent (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_1_16'
        chart_caption = 'Figure B16.5.1-16. HVAC BESTEST: Latent Coil Load'
        yaxis_name = 'Load (kWh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['evaporator_load_latent_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_04(self):
        table_name = 'section_9_table_b16_5_1_04'
        table_caption = 'Table B16.5.1-4. Sensible Coil Load minus Zone Load (Fan Heat) (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_1_25'
        chart_caption = 'Figure B16.5.1-25. HVAC BESTEST: Sensible Coil Load - Zone Load (Fan Heat)'
        yaxis_name = 'Load (kWh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                sensible_coil_value = json_obj['main_table'][case]['evaporator_load_sensible_kWh']
                zone_load_value = json_obj['main_table'][case]['envelope_load_sensible_kWh']
                if math.isnan(sensible_coil_value) or math.isnan(zone_load_value):
                    row.append(math.nan)
                else:
                    row.append(sensible_coil_value - zone_load_value)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_05a(self):
        table_name = 'section_9_table_b16_5_1_05a'
        table_caption = 'Table B16.5.1-5a. Zone Loads, Total (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_1_22'
        chart_caption = 'Figure B16.5.1-22. HVAC BESTEST: Total Zone Load'
        yaxis_name = 'Load (Wh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['envelope_load_total_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_05b(self):
        table_name = 'section_9_table_b16_5_1_05b'
        table_caption = 'Table B16.5.1-5b. Zone Loads, Sensible (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_1_23'
        chart_caption = 'Figure B16.5.1-23. HVAC BESTEST: Sensible Zone Load'
        yaxis_name = 'Load (Wh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['envelope_load_sensible_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_05c(self):
        table_name = 'section_9_table_b16_5_1_05c'
        table_caption = 'Table B16.5.1-5c. Zone Loads, Latent (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_1_24'
        chart_caption = 'Figure B16.5.1-24. HVAC BESTEST: Latent Zone Load'
        yaxis_name = 'Load (Wh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['envelope_load_latent_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_06(self):
        table_name = 'section_9_table_b16_5_1_06'
        table_caption = 'Table B16.5.1-6. Latent Coil Load minus Zone Load (Should be 0) (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_1_26'
        chart_caption = 'Figure B16.5.1-26. HVAC BESTEST: Latent Coil Load - Latent Zone Load (Should = 0)'
        yaxis_name = 'Load (kWh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                latent_coil_value = json_obj['main_table'][case]['evaporator_load_latent_kWh']
                zone_load_value = json_obj['main_table'][case]['envelope_load_latent_kWh']
                if math.isnan(latent_coil_value) or math.isnan(zone_load_value):
                    row.append(math.nan)
                else:
                    row.append(latent_coil_value - zone_load_value)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_07a(self):
        table_name = 'section_9_table_b16_5_1_07a'
        table_caption = 'Table B16.5.1-7a. Sensitivities for Space Cooling Electricty Consumption Delta Qtot (kWh,e)'
        chart_name = 'section_9_figure_b16_5_1_05'
        chart_caption = 'Figure B16.5.1-5. HVAC BESTEST: Total Space Cooling Electricity Sensitivies'
        yaxis_name = 'delta Electricity Consumption  (kWh)'
        sensitivity_cases = [
            ('CE110', 'CE100', 'ODB'),
            ('CE120', 'CE110', 'IDB'),
            ('CE120', 'CE100', 'IDB+ODB'),
            ('CE130', 'CE100', 'PLR'),
            ('CE140', 'CE130', 'ODB @lowPLR'),
            ('CE140', 'CE110', 'PLR @loODB'),
            ('CE150', 'CE110', 'hiSHR v. dry'),
            ('CE160', 'CE150', 'IDB @hiSHR'),
            ('CE165', 'CE160', 'IDB+ODB @hiSH'),
            ('CE170', 'CE150', 'sens x 0.39'),
            ('CE180', 'CE150', 'SHR'),
            ('CE180', 'CE170', 'lat x 4'),
            ('CE185', 'CE180', 'ODB @loSHR'),
            ('CE190', 'CE180', 'PLR @loSHR'),
            ('CE190', 'CE140', 'lat @loPLR'),
            ('CE195', 'CE190', 'ODB @loPLloSH'),
            ('CE195', 'CE185', 'PLR @loSHR'),
            ('CE195', 'CE130', 'lat @loPLR'),
            ('CE200', 'CE100', 'ARI v dry'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in sensitivity_cases]
        chart_row_headings = [c[0] + '-' + c[1] + ' ' + c[2] for c in sensitivity_cases]
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for (case_a, case_b, _) in sensitivity_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['main_table'][case_a]['cooling_energy_total_kWh']
                case_b_value = json_obj['main_table'][case_b]['cooling_energy_total_kWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_07b(self):
        table_name = 'section_9_table_b16_5_1_07b'
        table_caption = 'Table B16.5.1-7b. Sensitivities for Space Cooling Electricity Consumption Delta Qcomp (kWh,e)'
        chart_name = 'section_9_figure_b16_5_1_07'
        chart_caption = 'Figure B16.5.1-7. HVAC BESTEST: Total Compressor Electricity Sensitivities'
        yaxis_name = 'delta Electricity Consumption  (kWh)'
        sensitivity_cases = [
            ('CE110', 'CE100', 'ODB'),
            ('CE120', 'CE110', 'IDB'),
            ('CE120', 'CE100', 'IDB+ODB'),
            ('CE130', 'CE100', 'PLR'),
            ('CE140', 'CE130', 'ODB @lowPLR'),
            ('CE140', 'CE110', 'PLR @loODB'),
            ('CE150', 'CE110', 'hiSHR v. dry'),
            ('CE160', 'CE150', 'IDB @hiSHR'),
            ('CE165', 'CE160', 'IDB+ODB @hiSH'),
            ('CE170', 'CE150', 'sens x 0.39'),
            ('CE180', 'CE150', 'SHR'),
            ('CE180', 'CE170', 'lat x 4'),
            ('CE185', 'CE180', 'ODB @loSHR'),
            ('CE190', 'CE180', 'PLR @loSHR'),
            ('CE190', 'CE140', 'lat @loPLR'),
            ('CE195', 'CE190', 'ODB @loPLloSH'),
            ('CE195', 'CE185', 'PLR @loSHR'),
            ('CE195', 'CE130', 'lat @loPLR'),
            ('CE200', 'CE100', 'ARI v dry'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in sensitivity_cases]
        chart_row_headings = [c[0] + '-' + c[1] + ' ' + c[2] for c in sensitivity_cases]
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for (case_a, case_b, _) in sensitivity_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['main_table'][case_a]['cooling_energy_compressor_kWh']
                case_b_value = json_obj['main_table'][case_b]['cooling_energy_compressor_kWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_07c(self):
        table_name = 'section_9_table_b16_5_1_07c'
        table_caption = 'Table B16.5.1-7c. Sensitivities for Space Cooling Electricity Consumption Delta Q IDfan (kWh,e)'
        chart_name = 'section_9_figure_b16_5_1_09'
        chart_caption = 'Figure B16.5.1-9. HVAC BESTEST: Indoor (Supply) Fan Electricity Sensitivities'
        yaxis_name = 'delta Electricity Consumption  (kWh)'
        sensitivity_cases = [
            ('CE110', 'CE100', 'ODB'),
            ('CE120', 'CE110', 'IDB'),
            ('CE120', 'CE100', 'IDB+ODB'),
            ('CE130', 'CE100', 'PLR'),
            ('CE140', 'CE130', 'ODB @lowPLR'),
            ('CE140', 'CE110', 'PLR @loODB'),
            ('CE150', 'CE110', 'hiSHR v. dry'),
            ('CE160', 'CE150', 'IDB @hiSHR'),
            ('CE165', 'CE160', 'IDB+ODB @hiSH'),
            ('CE170', 'CE150', 'sens x 0.39'),
            ('CE180', 'CE150', 'SHR'),
            ('CE180', 'CE170', 'lat x 4'),
            ('CE185', 'CE180', 'ODB @loSHR'),
            ('CE190', 'CE180', 'PLR @loSHR'),
            ('CE190', 'CE140', 'lat @loPLR'),
            ('CE195', 'CE190', 'ODB @loPLloSH'),
            ('CE195', 'CE185', 'PLR @loSHR'),
            ('CE195', 'CE130', 'lat @loPLR'),
            ('CE200', 'CE100', 'ARI v dry'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in sensitivity_cases]
        chart_row_headings = [c[0] + '-' + c[1] + ' ' + c[2] for c in sensitivity_cases]
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for (case_a, case_b, _) in sensitivity_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['main_table'][case_a]['supply_fan_kWh']
                case_b_value = json_obj['main_table'][case_b]['supply_fan_kWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_07d(self):
        table_name = 'section_9_table_b16_5_1_07d'
        table_caption = 'Table B16.5.1-7d. Sensitivities for Space Cooling Electricty Consumption Delta Q ODfan (kWh,e)'
        chart_name = 'section_9_figure_b16_5_1_11'
        chart_caption = 'Figure B16.5.1-11. HVAC BESTEST: Outdoor (Condenser) Fan Electricity Sensitivities'
        yaxis_name = 'delta Electricity Consumption  (kWh)'
        sensitivity_cases = [
            ('CE110', 'CE100', 'ODB'),
            ('CE120', 'CE110', 'IDB'),
            ('CE120', 'CE100', 'IDB+ODB'),
            ('CE130', 'CE100', 'PLR'),
            ('CE140', 'CE130', 'ODB @lowPLR'),
            ('CE140', 'CE110', 'PLR @loODB'),
            ('CE150', 'CE110', 'hiSHR v. dry'),
            ('CE160', 'CE150', 'IDB @hiSHR'),
            ('CE165', 'CE160', 'IDB+ODB @hiSH'),
            ('CE170', 'CE150', 'sens x 0.39'),
            ('CE180', 'CE150', 'SHR'),
            ('CE180', 'CE170', 'lat x 4'),
            ('CE185', 'CE180', 'ODB @loSHR'),
            ('CE190', 'CE180', 'PLR @loSHR'),
            ('CE190', 'CE140', 'lat @loPLR'),
            ('CE195', 'CE190', 'ODB @loPLloSH'),
            ('CE195', 'CE185', 'PLR @loSHR'),
            ('CE195', 'CE130', 'lat @loPLR'),
            ('CE200', 'CE100', 'ARI v dry'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in sensitivity_cases]
        chart_row_headings = [c[0] + '-' + c[1] + ' ' + c[2] for c in sensitivity_cases]
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for (case_a, case_b, _) in sensitivity_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['main_table'][case_a]['condenser_fan_kWh']
                case_b_value = json_obj['main_table'][case_b]['condenser_fan_kWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_08a(self):
        table_name = 'section_9_table_b16_5_1_08a'
        table_caption = 'Table B16.5.1-8a. Sensitivities Delta COP (kWh,t)'
        chart_name = 'section_9_figure_b16_5_1_03'
        chart_caption = 'Figure B16.5.1-3. HVAC BESTEST: Mean COP Sensitivities'
        sensitivity_cases = [
            ('CE110', 'CE100', 'ODB'),
            ('CE120', 'CE110', 'IDB'),
            ('CE120', 'CE100', 'IDB+ODB'),
            ('CE130', 'CE100', 'PLR'),
            ('CE140', 'CE130', 'ODB @lowPLR'),
            ('CE140', 'CE110', 'PLR @loODB'),
            ('CE150', 'CE110', 'hiSHR v. dry'),
            ('CE160', 'CE150', 'IDB @hiSHR'),
            ('CE165', 'CE160', 'IDB+ODB @hiSH'),
            ('CE170', 'CE150', 'sens x 0.39'),
            ('CE180', 'CE150', 'SHR'),
            ('CE180', 'CE170', 'lat x 4'),
            ('CE185', 'CE180', 'ODB @loSHR'),
            ('CE190', 'CE180', 'PLR @loSHR'),
            ('CE190', 'CE140', 'lat @loPLR'),
            ('CE195', 'CE190', 'ODB @loPLloSH'),
            ('CE195', 'CE185', 'PLR @loSHR'),
            ('CE195', 'CE130', 'lat @loPLR'),
            ('CE200', 'CE100', 'ARI v dry'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in sensitivity_cases]
        chart_row_headings = [c[0] + '-' + c[1] + ' ' + c[2] for c in sensitivity_cases]
        yaxis_name = 'delta COP'
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for (case_a, case_b, _) in sensitivity_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['main_table'][case_a]['feb_mean_cop']
                case_b_value = json_obj['main_table'][case_b]['feb_mean_cop']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_08b(self):
        table_name = 'section_9_table_b16_5_1_08b'
        table_caption = 'Table B16.5.1-8b. Sensitivities Coil Loads, Delta Total (kWh,t)'
        chart_name = 'section_9_figure_b16_5_1_13'
        chart_caption = 'Figure B16.5.1-13. HVAC BESTEST: Total Coil Load Sensitivities'
        yaxis_name = 'delta Load  (kWh thermal)'
        sensitivity_cases = [
            ('CE110', 'CE100', 'ODB'),
            ('CE120', 'CE110', 'IDB'),
            ('CE120', 'CE100', 'IDB+ODB'),
            ('CE130', 'CE100', 'PLR'),
            ('CE140', 'CE130', 'ODB @lowPLR'),
            ('CE140', 'CE110', 'PLR @loODB'),
            ('CE150', 'CE110', 'hiSHR v. dry'),
            ('CE160', 'CE150', 'IDB @hiSHR'),
            ('CE165', 'CE160', 'IDB+ODB @hiSH'),
            ('CE170', 'CE150', 'sens x 0.39'),
            ('CE180', 'CE150', 'SHR'),
            ('CE180', 'CE170', 'lat x 4'),
            ('CE185', 'CE180', 'ODB @loSHR'),
            ('CE190', 'CE180', 'PLR @loSHR'),
            ('CE190', 'CE140', 'lat @loPLR'),
            ('CE195', 'CE190', 'ODB @loPLloSH'),
            ('CE195', 'CE185', 'PLR @loSHR'),
            ('CE195', 'CE130', 'lat @loPLR'),
            ('CE200', 'CE100', 'ARI v dry'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in sensitivity_cases]
        chart_row_headings = [c[0] + '-' + c[1] + ' ' + c[2] for c in sensitivity_cases]
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for (case_a, case_b, _) in sensitivity_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['main_table'][case_a]['evaporator_load_total_kWh']
                case_b_value = json_obj['main_table'][case_b]['evaporator_load_total_kWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_08c(self):
        table_name = 'section_9_table_b16_5_1_08c'
        table_caption = 'Table B16.5.1-8c. Sensitivities Coil Loads, Delta Sensible (kWh,t)'
        chart_name = 'section_9_figure_b16_5_1_15'
        chart_caption = 'Figure B16.5.1-15. HVAC BESTEST: Sensible Coil Load Sensitivities'
        yaxis_name = 'delta Load  (kWh thermal)'
        sensitivity_cases = [
            ('CE110', 'CE100', 'ODB'),
            ('CE120', 'CE110', 'IDB'),
            ('CE120', 'CE100', 'IDB+ODB'),
            ('CE130', 'CE100', 'PLR'),
            ('CE140', 'CE130', 'ODB @lowPLR'),
            ('CE140', 'CE110', 'PLR @loODB'),
            ('CE150', 'CE110', 'hiSHR v. dry'),
            ('CE160', 'CE150', 'IDB @hiSHR'),
            ('CE165', 'CE160', 'IDB+ODB @hiSH'),
            ('CE170', 'CE150', 'sens x 0.39'),
            ('CE180', 'CE150', 'SHR'),
            ('CE180', 'CE170', 'lat x 4'),
            ('CE185', 'CE180', 'ODB @loSHR'),
            ('CE190', 'CE180', 'PLR @loSHR'),
            ('CE190', 'CE140', 'lat @loPLR'),
            ('CE195', 'CE190', 'ODB @loPLloSH'),
            ('CE195', 'CE185', 'PLR @loSHR'),
            ('CE195', 'CE130', 'lat @loPLR'),
            ('CE200', 'CE100', 'ARI v dry'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in sensitivity_cases]
        chart_row_headings = [c[0] + '-' + c[1] + ' ' + c[2] for c in sensitivity_cases]
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for (case_a, case_b, _) in sensitivity_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['main_table'][case_a]['evaporator_load_sensible_kWh']
                case_b_value = json_obj['main_table'][case_b]['evaporator_load_sensible_kWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_08d(self):
        table_name = 'section_9_table_b16_5_1_08d'
        table_caption = 'Table B16.5.1-8d. Sensitivities Coil Loads, Delta Latent (kWh,t)'
        chart_name = 'section_9_figure_b16_5_1_17'
        chart_caption = 'Figure B16.5.1-17. HVAC BESTEST: Latent Coil Load Sensitivities'
        yaxis_name = 'delta Load  (kWh thermal)'
        sensitivity_cases = [
            ('CE110', 'CE100', 'ODB'),
            ('CE120', 'CE110', 'IDB'),
            ('CE120', 'CE100', 'IDB+ODB'),
            ('CE130', 'CE100', 'PLR'),
            ('CE140', 'CE130', 'ODB @lowPLR'),
            ('CE140', 'CE110', 'PLR @loODB'),
            ('CE150', 'CE110', 'hiSHR v. dry'),
            ('CE160', 'CE150', 'IDB @hiSHR'),
            ('CE165', 'CE160', 'IDB+ODB @hiSH'),
            ('CE170', 'CE150', 'sens x 0.39'),
            ('CE180', 'CE150', 'SHR'),
            ('CE180', 'CE170', 'lat x 4'),
            ('CE185', 'CE180', 'ODB @loSHR'),
            ('CE190', 'CE180', 'PLR @loSHR'),
            ('CE190', 'CE140', 'lat @loPLR'),
            ('CE195', 'CE190', 'ODB @loPLloSH'),
            ('CE195', 'CE185', 'PLR @loSHR'),
            ('CE195', 'CE130', 'lat @loPLR'),
            ('CE200', 'CE100', 'ARI v dry'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in sensitivity_cases]
        chart_row_headings = [c[0] + '-' + c[1] + ' ' + c[2] for c in sensitivity_cases]
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for (case_a, case_b, _) in sensitivity_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                case_a_value = json_obj['main_table'][case_a]['evaporator_load_latent_kWh']
                case_b_value = json_obj['main_table'][case_b]['evaporator_load_latent_kWh']
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_09a(self):
        table_name = 'section_9_table_b16_5_1_09a'
        table_caption = 'Table B16.5.1-9a. Indoor Drybulb Temperature: Mean (C)'
        chart_name = 'section_9_figure_b16_5_1_18'
        chart_caption = 'Figure B16.5.1-18. HVAC BESTEST: Mean Indoor Drybulb Temperature'
        yaxis_name = 'Temperature (C)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['feb_mean_idb_c'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=1)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_09b(self):
        table_name = 'section_9_table_b16_5_1_09b'
        table_caption = 'Table B16.5.1-9b. Indoor Drybulb Temperature (Max-Min)/Mean'
        chart_name = 'section_9_figure_b16_5_1_19'
        chart_caption = 'Figure B16.5.1-19. HVAC BESTEST: (Maximum - Minimum)/Mean Indoor Drybulb Temperature'
        yaxis_name = 'Fractional Variation'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                min_value = json_obj['main_table'][case]['feb_min_idb_c']
                max_value = json_obj['main_table'][case]['feb_max_idb_c']
                mean_value = json_obj['main_table'][case]['feb_mean_idb_c']
                if math.isnan(min_value) or math.isnan(max_value) or math.isnan(mean_value):
                    row.append(math.nan)
                else:
                    row.append((max_value - min_value) / mean_value)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_10a(self):
        table_name = 'section_9_table_b16_5_1_10a'
        table_caption = 'Table B16.5.1-10a. Humidity Ratio: Mean'
        chart_name = 'section_9_figure_b16_5_1_20'
        chart_caption = 'Figure B16.5.1-20. HVAC BESTEST: Mean Indoor Humidity Ratio'
        yaxis_name = 'Humidity Ratio (kg/kg)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['main_table'][case]['feb_mean_hum_ratio_kg_kg'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=4)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_a_table_b16_5_1_10b(self):
        table_name = 'section_9_table_b16_5_1_10b'
        table_caption = 'Table B16.5.1-10b. Humidity Ratio (Max-Min)/Mean'
        chart_name = 'section_9_figure_b16_5_1_21'
        chart_caption = 'Figure B16.5.1-21. HVAC BESTEST: (Maximum - Minimum)/Mean Indoor Humidity Ratio'
        yaxis_name = 'Fractional Variation'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Analytical Solutions)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_column_name'])
        for case in self.case_map.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                min_value = json_obj['main_table'][case]['feb_min_hum_ratio_kg_kg']
                max_value = json_obj['main_table'][case]['feb_max_hum_ratio_kg_kg']
                mean_value = json_obj['main_table'][case]['feb_mean_hum_ratio_kg_kg']
                if math.isnan(min_value) or math.isnan(max_value) or math.isnan(mean_value):
                    row.append(math.nan)
                else:
                    row.append((max_value - min_value) / mean_value)
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_01a(self):
        table_name = 'section_9_table_b16_5_2_01a'
        table_caption = 'Table B16.5.2-1a. Annual Space Cooling Electricity Consumption - Total (kWh,e)'
        chart_name = 'section_9_figure_b16_5_2_01'
        chart_caption = 'Figure B16.5.2-1. HVAC BESTEST: CE300 - CE545 Annual Total Electricity Consumption'
        yaxis_name = 'Electricity Consumption  (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['cooling_energy_total_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_01b(self):
        table_name = 'section_9_table_b16_5_2_01b'
        table_caption = 'Table B16.5.2-1b. Annual Space Cooling Electricity Consumption - Compressor (kWh,e)'
        chart_name = 'section_9_figure_b16_5_2_05'
        chart_caption = 'Figure B16.5.2-5. HVAC BESTEST: CE300 - CE545 Annual Compressor Electricity Consumption'
        yaxis_name = 'Electricity Consumption  (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['cooling_energy_compressor_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_02a(self):
        table_name = 'section_9_table_b16_5_2_02a'
        table_caption = 'Table B16.5.2-2a. Annual Space Cooling Electricity Consumption - Supply Fan (kWh,e)'
        chart_name = 'section_9_figure_b16_5_2_07'
        chart_caption = 'Figure B16.5.2-7. HVAC BESTEST: CE300 - CE545 Annual Indoor (Supply) Fan Electricity Consumption'
        yaxis_name = 'Electricity Consumption  (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['indoor_fan_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_02b(self):
        table_name = 'section_9_table_b16_5_2_02b'
        table_caption = 'Table B16.5.2-2b. Annual Space Cooling Electricity Consumption - Condenser Fan (kWh,e)'
        chart_name = 'section_9_figure_b16_5_2_09'
        chart_caption = 'Figure B16.5.2-9. HVAC BESTEST: CE300 - CE545 Annual Outdoor (Condenser) Fan Electricity Consumption'
        yaxis_name = 'Electricity Consumption  (kWh)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['condenser_fan_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_03a(self):
        table_name = 'section_9_table_b16_5_2_03a'
        table_caption = 'Table B16.5.2-3a. Weather Data Checks, CE300 Only, Annual Outdoor Dry Bulb'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_info = {
            'ODB-Mean':
                {'heading': 'Mean (C)',
                 'table': 'annual_means_ce300',
                 'keyname': 'outdoor_drybulb_c'},
            'ODB-Max':
                {'heading': 'Hourly Integrated Maxima (C)',
                 'table': 'annual_weather_data_ce300',
                 'keyname': 'outdoor_drybulb_max_c'},
        }
        row_headings = []
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for k, v in row_info.items():
            row = []
            row_headings.append(v['heading'])
            for tst, json_obj in self.json_data.items():
                row.append(json_obj[v['table']][v['keyname']])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        return

    def render_section_ce_b_table_b16_5_2_03b(self):
        table_name = 'section_9_table_b16_5_2_03b'
        table_caption = 'Table B16.5.2-3b. Weather Data Checks, CE300 Only, Annual Outdoor Humidity Ratio'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_info = {
            'OHR-Mean':
                {'heading': 'Mean (kg/kg)',
                 'table': 'annual_means_ce300',
                 'keyname': 'outdoor_humidity_ratio_kg_kg'},
            'OHR-Max':
                {'heading': 'Hourly Integrated Maxima (kg/kg)',
                 'table': 'annual_weather_data_ce300',
                 'keyname': 'outdoor_humidity_ratio_max_kg_kg'},
        }
        row_headings = []
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for k, v in row_info.items():
            row = []
            row_headings.append(v['heading'])
            for tst, json_obj in self.json_data.items():
                row.append(json_obj[v['table']][v['keyname']])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=5)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        return

    def render_section_ce_b_table_b16_5_2_04a(self):
        table_name = 'section_9_table_b16_5_2_04a'
        table_caption = 'Table B16.5.2-4a. Annual Space Cooling Coil Loads - Total Sensible + Latent (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_2_11'
        chart_caption = 'Figure B16.5.2-11. HVAC BESTEST: CE300 - CE545 Annual Total Coil Load'
        yaxis_name = 'Load  (kWh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['evaporator_load_total_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_04b(self):
        table_name = 'section_9_table_b16_5_2_04b'
        table_caption = 'Table B16.5.2-4b. Annual Space Cooling Coil Loads - Sensible (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_2_14'
        chart_caption = 'Figure B16.5.2-14. HVAC BESTEST: CE300 - CE545 Annual Sensible Coil Load'
        yaxis_name = 'Load  (kWh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['evaporator_load_sensible_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_05(self):
        table_name = 'section_9_table_b16_5_2_05'
        table_caption = 'Table B16.5.2-5. Annual Space Cooling Coil Loads - Latent (kWh,thermal)'
        chart_name = 'section_9_figure_b16_5_2_17'
        chart_caption = 'Figure B16.5.2-17. HVAC BESTEST: CE300 - CE545 Annual Latent Coil Load'
        yaxis_name = 'Load  (kWh thermal)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['evaporator_load_latent_kWh'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=0)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_06a(self):
        table_name = 'section_9_table_b16_5_2_06a'
        table_caption = 'Table B16.5.2-6a. Various Annual Means - COP2'
        chart_name = 'section_9_figure_b16_5_2_21'
        chart_caption = 'Figure B16.5.2-21. HVAC BESTEST: CE300 - CE545 Annual Mean COP2'
        yaxis_name = 'COP2'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['cop2'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=3)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_06b(self):
        table_name = 'section_9_table_b16_5_2_06b'
        table_caption = 'Table B16.5.2-6b. Various Annual Means - Indoor Dry Bulb (C)'
        chart_name = 'section_9_figure_b16_5_2_27'
        chart_caption = 'Figure B16.5.2-27. HVAC BESTEST: CE300 - CE545 Annual Mean Indoor Dry-Bulb Temperature'
        yaxis_name = 'Temperature (C)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['indoor_dry_bulb_c'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_07a(self):
        table_name = 'section_9_table_b16_5_2_07a'
        table_caption = 'Table B16.5.2-7a. Various Annual Means - Zone Humidity Ratio (kg/kg)'
        chart_name = 'section_9_figure_b16_5_2_32'
        chart_caption = 'Figure B16.5.2-32. HVAC BESTEST: CE300 - CE545 Annual Mean Zone Humidity Ratio'
        yaxis_name = 'Humidity Ratio (kg/kg)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['zone_humidity_ratio_kg_kg'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=4)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_07b(self):
        table_name = 'section_9_table_b16_5_2_07b'
        table_caption = 'Table B16.5.2-7b. Various Annual Means - Zone Relative Humidity (%)'
        chart_name = 'section_9_figure_b16_5_2_37'
        chart_caption = 'Figure B16.5.2-37. HVAC BESTEST: CE300 - CE545 Annual Mean Relative Humidity'
        yaxis_name = 'Relative Humidity (%)'
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map.keys():
            if case == 'E510':
                continue
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['annual_sums_means'][case]['zone_relative_humidity_perc'])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE510 High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        # self._create_plotly_line('section_9_line_b16_5_2_37', data_table, chart_row_headings, column_headings, yaxis_name, chart_caption)
        return

    def general_ce_b_table_08_09(self, table_letter, caption_end, json_key, sig_digits, chart_code='', yaxis=''):
        """Function that handles any table from ce_b 8 or 9 """
        table_name = f'section_9_table_b16_5_2_0{table_letter}'
        chart_name = f'section_9_figure_b16_5_2_{chart_code}'
        table_caption = (f'Table B16.5.2-{table_letter}. f(ODB) Sensitivity CE500 and CE530, April 30 and July 25, '
                         f'{caption_end}')
        chart_caption = (f'Figure B16.5.2-{chart_code}. HVAC BESTEST: f(ODB) for CE500, CE530'
                         f'<br>Specific Day {caption_end}')
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = []
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for test_name in ['ce500', 'ce530']:
            april_30_row = []
            june_25_row = []
            diff_row = []
            for tst, json_obj in self.json_data.items():
                april_30_value = json_obj[test_name + '_avg_daily']['April 30'][json_key]
                june_25_value = json_obj[test_name + '_avg_daily']['June 25'][json_key]
                diff_value = june_25_value - april_30_value
                april_30_row.append(april_30_value)
                june_25_row.append(june_25_value)
                diff_row.append(diff_value)
            data_table.append(april_30_row)
            data_table.append(june_25_row)
            data_table.append(diff_row)
            row_headings.append(test_name.upper() + ' April 30')
            row_headings.append(test_name.upper() + ' June 25')
            row_headings.append('Delta ' + test_name.upper())
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=sig_digits)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        if chart_code and yaxis:
            self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis, chart_caption)
        return

    def render_section_ce_b_table_b16_5_2_08a(self):
        self.general_ce_b_table_08_09('8a', 'Energy Consumption, Compressor + Both Fans (Wh,e)',
                                      'cooling_energy_total_kWh', 0,
                                      '42a', 'Consumption (Wh/h)')
        return

    def render_section_ce_b_table_b16_5_2_08b(self):
        self.general_ce_b_table_08_09('8b', 'Energy Consumption, Compressor (Wh,e)',
                                      'cooling_energy_compressor_kWh', 0)
        return

    def render_section_ce_b_table_b16_5_2_08c(self):
        self.general_ce_b_table_08_09('8c', 'Energy Consumption, Condenser Fan (Wh,e)',
                                      'condenser_fan_kWh', 0,
                                      '42b', 'Consumption (Wh/h)')
        return

    def render_section_ce_b_table_b16_5_2_08d(self):
        self.general_ce_b_table_08_09('8d', 'Energy Consumption, Supply Fan (Wh,e)',
                                      'indoor_fan_kWh', 0,
                                      '42c', 'Consumption (Wh/h)')
        return

    def render_section_ce_b_table_b16_5_2_08e(self):
        self.general_ce_b_table_08_09('8e', 'Sensible + Latent Coil Load (Wh,th)',
                                      'evaporator_load_total_kWh', 0,
                                      '43a', 'Daily Load (Wh/h thermal)')
        return

    def render_section_ce_b_table_b16_5_2_08f(self):
        self.general_ce_b_table_08_09('8f', 'Sensible Coil Load (Wh,th)',
                                      'evaporator_load_sensible_kWh', 0,
                                      '43b', 'Daily Load (Wh/h thermal)')
        return

    def render_section_ce_b_table_b16_5_2_08g(self):
        self.general_ce_b_table_08_09('8g', 'Latent Coil Load (Wh,th)',
                                      'evaporator_load_latent_kWh', 0,
                                      '43c', 'Daily Load (Wh/h thermal)')
        return

    def render_section_ce_b_table_b16_5_2_09a(self):
        self.general_ce_b_table_08_09('9a', 'Humidity Ratio (kg/kg)',
                                      'zone_humidity_ratio_kg_kg', 4,
                                      '45', 'Humidity Ratio (kg/kg)')
        return

    def render_section_ce_b_table_b16_5_2_09b(self):
        self.general_ce_b_table_08_09('9b', 'COP2',
                                      'cop2', 3,
                                      '44', 'COP2')
        return

    def render_section_ce_b_table_b16_5_2_09c(self):
        self.general_ce_b_table_08_09('9c', 'ODB (C)',
                                      'outdoor_drybulb_c', 2)
        return

    def render_section_ce_b_table_b16_5_2_09d(self):
        self.general_ce_b_table_08_09('9d', 'EDB (C)',
                                      'entering_drybulb_c', 2)
        return

    def general_ce_b_table_max_min(self, include_min, table_letter, caption_end, json_dict, json_key, key_suffix,
                                   sig_digits):
        table_name = f'section_9_table_b16_5_2_{table_letter}'
        maxmin = 'Maxima'
        if include_min:
            maxmin = 'Maxima and Minima'
        table_caption = f'Table B16.5.2-{table_letter}. Hourly Integrated {maxmin} {caption_end}'
        data_table = []
        time_stamp_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = list(self.case_map_max.values())
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map_max.keys():
            row = []
            time_stamp_row = []
            for tst, json_obj in self.json_data.items():
                case_json = json_obj[json_dict][case]
                row.append(case_json[f'{json_key}_{key_suffix}'])
                month = self._int_0_if_nan(case_json[f'{json_key}_month'])
                day = self._int_0_if_nan(case_json[f'{json_key}_day'])
                hour = self._int_0_if_nan(case_json[f'{json_key}_hour'])
                time_stamp_row.append(f'{month} {day}-{hour}')
            data_table.append(row)
            time_stamp_table.append(time_stamp_row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=sig_digits,
                                                         time_stamps=time_stamp_table)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        return

    def render_section_ce_b_table_b16_5_2_10a(self):
        self.general_ce_b_table_max_min(False, '10a',
                                        'Total Cooling Energy Consumption, Compressor + Both Fans (Wh,e)',
                                        'annual_load_maxima', 'compressors_plus_fans', 'Wh', 0)

    def render_section_ce_b_table_b16_5_2_10b(self):
        self.general_ce_b_table_max_min(False, '10b',
                                        'Total Coil Load - Sensible + Latent Coil Load  (Wh,th)',
                                        'annual_load_maxima', 'evaporator_total', 'Wh', 0)

    def render_section_ce_b_table_b16_5_2_11a(self):
        self.general_ce_b_table_max_min(False, '11a',
                                        'Sensible Coil Load  (Wh,th)',
                                        'annual_load_maxima', 'evaporator_sensible', 'Wh', 0)

    def render_section_ce_b_table_b16_5_2_11b(self):
        self.general_ce_b_table_max_min(False, '11b',
                                        'Latent Coil Load  (Wh,th)',
                                        'annual_load_maxima', 'evaporator_latent', 'Wh', 0)

    def render_section_ce_b_table_b16_5_2_12a(self):
        self.general_ce_b_table_max_min(True, '12a',
                                        '- Maximum COP2',
                                        'annual_cop_zone', 'cop2_max', 'value', 3)

    def render_section_ce_b_table_b16_5_2_12b(self):
        self.general_ce_b_table_max_min(True, '12b',
                                        '- Minimum COP2',
                                        'annual_cop_zone', 'cop2_min', 'value', 3)

    def render_section_ce_b_table_b16_5_2_13a(self):
        self.general_ce_b_table_max_min(True, '13a',
                                        '- Maximum Indoor Dry Bulb (C)',
                                        'annual_cop_zone', 'indoor_db_max', 'c', 2)

    def render_section_ce_b_table_b16_5_2_13b(self):
        self.general_ce_b_table_max_min(True, '13b',
                                        '- Minimum Indoor Dry Bulb (C)',
                                        'annual_cop_zone', 'indoor_db_min', 'c', 2)

    def render_section_ce_b_table_b16_5_2_14a(self):
        self.general_ce_b_table_max_min(True, '14a',
                                        '- Maximum Zone Humidity Ratio (kg/kg)',
                                        'annual_cop_zone', 'indoor_hum_rat_max', 'kg_kg', 4)

    def render_section_ce_b_table_b16_5_2_14b(self):
        self.general_ce_b_table_max_min(True, '14b',
                                        '- Minimum Zone Humidity Ratio (kg/kg)',
                                        'annual_cop_zone', 'indoor_hum_rat_min', 'kg_kg', 4)

    def render_section_ce_b_table_b16_5_2_15a(self):
        self.general_ce_b_table_max_min(True, '15a',
                                        '- Maximum Relative Humidity (%)',
                                        'annual_cop_zone', 'indoor_rel_hum_max', 'perc', 2)

    def render_section_ce_b_table_b16_5_2_15b(self):
        self.general_ce_b_table_max_min(True, '15b',
                                        '- Minimum Relative Humidity (%)',
                                        'annual_cop_zone', 'indoor_rel_hum_min', 'perc', 2)

    def render_section_ce_b_table_b16_5_2_16(self):
        # unlike almost all other table rendering methods, this one generates one table for each
        # software and so it is set up very different looping through the software
        for index, (_, json_obj) in enumerate(self.json_data.items()):
            table_letter = chr(index + 97)
            table_name = f'section_9_table_b16_5_2_16{table_letter}'
            software_name = json_obj['identifying_information']['software_name']
            table_caption = f'Table B16.5.2-16{table_letter}. June 28 Hourly Output - Case CE300 - {software_name}'
            data_table = []
            column_dict = {'Compressor (Wh)': ('compressor_Wh', 0),
                           'Condenser Fan (Wh)': ('condenser_fans_Wh', 0),
                           'Evaporator Total (Wh)': ('evaporator_total_Wh', 0),
                           'Evaporator Sensible (Wh)': ('evaporator_sensible_Wh', 0),
                           'Evaporator Latent (Wh)': ('evaporator_latent_Wh', 0),
                           'Zone Humidity Ratio (kg/kg)': ('zone_humidity_ratio_kg_kg', 4),
                           'COP2': ('cop2', 3),
                           'Outdoor Drybulb (C)': ('outdoor_drybulb_c', 2),
                           'Entering Drybulb (C)': ('entering_drybulb_c', 2),
                           'Entering Wetbulb (C)': ('entering_wetbulb_c', 2),
                           'Outdoor Humidity Ratio (kg/kg)': ('outdoor_humidity_ratio_kg_kg', 4),
                           }
            column_headings = ['Hour', ]
            column_headings.extend(column_dict.keys())
            for hour in range(1, 25):
                row = [str(hour), ]
                for json_key, _ in column_dict.values():
                    row.append(json_obj['june28_hourly'][str(hour)][json_key])
                data_table.append(row)
            formatted_table = [column_headings, ]
            for data_row in data_table:
                formatted_row = [data_row[0], ]
                for column_index, (_, digits) in enumerate(column_dict.values()):
                    formatting_string = '{:.' + str(digits) + 'f}'
                    formatted_row.append(formatting_string.format(data_row[column_index + 1]))
                formatted_table.append(formatted_row)
#             text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=2)
            self._make_markdown_from_table(table_name, table_caption, formatted_table, '')
        return

    def general_ce_b_table_delta(self, table_code, caption_end, json_dict, json_key, sig_digits):
        table_name = f'section_9_table_b16_5_2_{table_code}'
        table_caption = f'Table B16.5.2-{table_code}. Delta {caption_end}'
        delta_cases = [
            ('CE310', 'CE300', 'E310', 'E300'),
            ('CE320', 'CE300', 'E320', 'E300'),
            ('CE330', 'CE300', 'E330', 'E300'),
            ('CE330', 'CE320', 'E330', 'E320'),
            ('CE340', 'CE300', 'E340', 'E300'),
            ('CE330', 'CE340', 'E330', 'E340'),
            ('CE350', 'CE300', 'E350', 'E300'),
            ('CE360', 'CE300', 'E360', 'E300'),
            ('CE400', 'CE300', 'E400', 'E300'),
            ('CE410', 'CE300', 'E410', 'E300'),
            ('CE420', 'CE300', 'E420', 'E300'),
            ('CE430', 'CE300', 'E430', 'E300'),
            ('CE440', 'CE300', 'E440', 'E300'),
            ('CE500', 'CE300', 'E500', 'E300'),
            ('CE510', 'CE500', 'E510 May-Sep', 'E500 May-Sep'),
            ('CE525', 'CE520', 'E525', 'E520'),
            ('CE530', 'CE500', 'E530', 'E500'),
            ('CE545', 'CE540', 'E545', 'E540'),
        ]
        data_table = []
        footnotes = ['$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]', ]
        row_headings = [c[0] + '-' + c[1] for c in delta_cases]
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case_a, case_b, case_a_lookup, case_b_lookup in delta_cases:
            row = []
            for tst, json_obj in self.json_data.items():
                if json_dict != 'annual_sums_means':
                    case_a_lookup = case_a_lookup[:4]
                    case_b_lookup = case_b_lookup[:4]
                case_a_value = json_obj[json_dict][case_a_lookup][json_key]
                case_b_value = json_obj[json_dict][case_b_lookup][json_key]
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    row.append(float(case_a_value) - float(case_b_value))
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=sig_digits)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, footnotes)
        return

    def render_section_ce_b_table_b16_5_2_17a(self):
        self.general_ce_b_table_delta('17a', 'Annual Space Cooling Electricity Consumption - Total (kWh,e)',
                                      'annual_sums_means', 'cooling_energy_total_kWh', 0)

    def render_section_ce_b_table_b16_5_2_17b(self):
        self.general_ce_b_table_delta('17b', 'Annual Space Cooling Electricity Consumption - Compressors (kWh,e)',
                                      'annual_sums_means', 'cooling_energy_compressor_kWh', 0)

    def render_section_ce_b_table_b16_5_2_18a(self):
        self.general_ce_b_table_delta('18a', 'Annual Space Cooling Electricity Consumption - Supply Fan (kWh,e)',
                                      'annual_sums_means', 'indoor_fan_kWh', 0)

    def render_section_ce_b_table_b16_5_2_18b(self):
        self.general_ce_b_table_delta('18b', 'Annual Space Cooling Electricity Consumption - Condenser Fan (kWh,e)',
                                      'annual_sums_means', 'condenser_fan_kWh', 0)

    def render_section_ce_b_table_b16_5_2_19a(self):
        self.general_ce_b_table_delta('19a', 'Annual Cooling Sensible Coil Load (kWh,th)',
                                      'annual_sums_means', 'evaporator_load_sensible_kWh', 0)

    def render_section_ce_b_table_b16_5_2_19b(self):
        self.general_ce_b_table_delta('19b', 'Annual Cooling Latent Coil Load (kWh,th)',
                                      'annual_sums_means', 'evaporator_load_latent_kWh', 0)

    def render_section_ce_b_table_b16_5_2_20a(self):
        self.general_ce_b_table_delta('20a', 'Annual Mean - COP2',
                                      'annual_sums_means', 'cop2', 3)

    def render_section_ce_b_table_b16_5_2_20b(self):
        self.general_ce_b_table_delta('20b', 'Annual Mean - Indoor Dry Bulb Temperature (C)',
                                      'annual_sums_means', 'indoor_dry_bulb_c', 2)

    def render_section_ce_b_table_b16_5_2_21a(self):
        self.general_ce_b_table_delta('21a', 'Annual Mean - Zone Humidity Ratio (kg/kg)',
                                      'annual_sums_means', 'zone_humidity_ratio_kg_kg', 4)

    def render_section_ce_b_table_b16_5_2_21b(self):
        self.general_ce_b_table_delta('21b', 'Annual Mean - Zone Relative Humidity (%)',
                                      'annual_sums_means', 'zone_relative_humidity_perc', 2)

    def render_section_ce_b_table_b16_5_2_22(self):
        self.general_ce_b_table_delta('22', 'Hourly Integrated Maximum Total Consumption (Wh,e)',
                                      'annual_load_maxima', 'compressors_plus_fans_Wh', 0)

    def render_section_ce_b_table_b16_5_2_23a(self):
        self.general_ce_b_table_delta('23a', 'Hourly Integrated Maximum Total Coil Load (Wh,th)',
                                      'annual_load_maxima', 'evaporator_total_Wh', 0)

    def render_section_ce_b_table_b16_5_2_23b(self):
        self.general_ce_b_table_delta('23b', 'Hourly Integrated Maximum Sensible Coil Load (Wh,th)',
                                      'annual_load_maxima', 'evaporator_sensible_Wh', 0)

    def render_section_ce_b_table_b16_5_2_24(self):
        self.general_ce_b_table_delta('24', 'Hourly Integrated Maximum Latent Coil Load (Wh,th)',
                                      'annual_load_maxima', 'evaporator_latent_Wh', 0)

    def render_section_ce_b_table_b16_5_2_25a(self):
        self.general_ce_b_table_delta('25a', 'Hourly Integrated Maximum COP2',
                                      'annual_cop_zone', 'cop2_max_value', 3)

    def render_section_ce_b_table_b16_5_2_25b(self):
        self.general_ce_b_table_delta('25b', 'Hourly Integrated Minimum COP2',
                                      'annual_cop_zone', 'cop2_min_value', 3)

    def render_section_ce_b_table_b16_5_2_26a(self):
        self.general_ce_b_table_delta('26a', 'Hourly Integrated Maximum Indoor Dry Bulb Temperature (C)',
                                      'annual_cop_zone', 'indoor_db_max_c', 2)

    def render_section_ce_b_table_b16_5_2_26b(self):
        self.general_ce_b_table_delta('26b', 'Hourly Integrated Minimum Indoor Dry Bulb Temperature (C)',
                                      'annual_cop_zone', 'indoor_db_min_c', 2)

    def render_section_ce_b_table_b16_5_2_27a(self):
        self.general_ce_b_table_delta('27a', 'Hourly Integrated Maximum Zone Humidity Ratio (kg/kg)',
                                      'annual_cop_zone', 'indoor_hum_rat_max_kg_kg', 4)

    def render_section_ce_b_table_b16_5_2_27b(self):
        self.general_ce_b_table_delta('27b', 'Hourly Integrated Minimum Zone Humidity Ratio (kg/kg)',
                                      'annual_cop_zone', 'indoor_hum_rat_min_kg_kg', 4)

    def render_section_ce_b_table_b16_5_2_28a(self):
        self.general_ce_b_table_delta('28a', 'Hourly Integrated Maximum Zone Relative Humidity (%)',
                                      'annual_cop_zone', 'indoor_rel_hum_max_perc', 2)

    def render_section_ce_b_table_b16_5_2_28b(self):
        self.general_ce_b_table_delta('28b', 'Hourly Integrated Minimum Zone Relative Humidity (%)',
                                      'annual_cop_zone', 'indoor_rel_hum_min_perc', 2)

    def general_ce_b_table_29(self, table_letter, caption_end, json_key, sig_digits):
        """Generate tables that are like Table B16.4.2-16 but are comparative """
        table_name = f'section_9_table_b16_5_2_29{table_letter}'
        table_caption = f'Table B16.5.2-29{table_letter}. June 28 Hourly Output - Case CE300 - {caption_end}'
        data_table = []
        row_headings = [str(hr) for hr in range(1, 25)]
        column_headings = ['Hour']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for hour in range(1, 25):
            row = []
            for tst, json_obj in self.json_data.items():
                row.append(json_obj['june28_hourly'][str(hour)][json_key])
            data_table.append(row)
        text_table_with_stats = self._add_stats_to_table(row_headings, column_headings, data_table, digits=sig_digits)
        self._make_markdown_from_table(table_name, table_caption, text_table_with_stats, '')
        return

    def render_section_ce_b_table_b16_5_2_29a(self):
        self.general_ce_b_table_29('a', 'Compressor Energy Consumption (Wh)',
                                   'compressor_Wh', 0)

    def render_section_ce_b_table_b16_5_2_29b(self):
        self.general_ce_b_table_29('b', 'Condenser Fan Energy Consumption (Wh)',
                                   'condenser_fans_Wh', 0)

    def render_section_ce_b_table_b16_5_2_29c(self):
        self.general_ce_b_table_29('c', 'Total Evaporator Coil Load (Wh)',
                                   'evaporator_total_Wh', 0)

    def render_section_ce_b_table_b16_5_2_29d(self):
        self.general_ce_b_table_29('d', 'Sensible Evaporator Coil Load (Wh)',
                                   'evaporator_sensible_Wh', 0)

    def render_section_ce_b_table_b16_5_2_29e(self):
        self.general_ce_b_table_29('e', 'Latent Evaporator Coil Load (Wh)',
                                   'evaporator_latent_Wh', 0)

    def render_section_ce_b_table_b16_5_2_29f(self):
        self.general_ce_b_table_29('f', 'Zone Humidity Ratio (kg/kg)',
                                   'zone_humidity_ratio_kg_kg', 4)

    def render_section_ce_b_table_b16_5_2_29g(self):
        self.general_ce_b_table_29('g', 'COP2',
                                   'cop2', 3)

    def render_section_ce_b_table_b16_5_2_29h(self):
        self.general_ce_b_table_29('h', 'Outdoor Dry Bulb Temperature (C)',
                                   'outdoor_drybulb_c', 2)

    def render_section_ce_b_table_b16_5_2_29i(self):
        self.general_ce_b_table_29('i', 'Entering Dry Bulb Temperature (C)',
                                   'entering_drybulb_c', 2)

    def render_section_ce_b_table_b16_5_2_29j(self):
        self.general_ce_b_table_29('j', 'Entering Wet Bulb Temperature (C)',
                                   'entering_wetbulb_c', 2)

    def render_section_ce_b_table_b16_5_2_29k(self):
        self.general_ce_b_table_29('k', 'Outdoor Humidity Ratio (kg/kg)',
                                   'outdoor_humidity_ratio_kg_kg', 4)

    def general_ce_b_figure(self, chart_number, caption_end, yaxis, json_dict, json_key, key_suffix):
        chart_name = f'section_9_figure_b16_5_2_{chart_number:02d}'
        chart_caption = f'Figure B16.5.2-{chart_number}. HVAC BESTEST: CE300 - CE545 {caption_end}'
        data_table = []
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case in self.case_map_max.keys():
            row = []
            for tst, json_obj in self.json_data.items():
                case_json = json_obj[json_dict][case]
                row.append(case_json[f'{json_key}_{key_suffix}'])
            data_table.append(row)
        chart_row_headings = list(self.case_map_charts.values())
        chart_row_headings.remove('CE500 May-Sep')
        chart_row_headings.remove('CE510 May-Sep High PLR')
        self._create_plotly_bar(chart_name, data_table, chart_row_headings, column_headings, yaxis, chart_caption)
        return

    def render_section_ce_b_chart_b16_5_2_3(self):
        self.general_ce_b_figure(3, 'Peak Hour Total Electricity Consumption',
                                 'Electricity Consumption (Wh/h)', 'annual_load_maxima',
                                 'compressors_plus_fans', 'Wh')

    def render_section_ce_b_chart_b16_5_2_12(self):
        self.general_ce_b_figure(12, 'Peak Hour Total Coil Load',
                                 'Load (Wh/h thermal)', 'annual_load_maxima',
                                 'evaporator_total', 'Wh')

    def render_section_ce_b_chart_b16_5_2_16(self):
        self.general_ce_b_figure(16, 'Peak Hour Sensible Coil Load',
                                 'Load (Wh/h thermal)', 'annual_load_maxima',
                                 'evaporator_sensible', 'Wh')

    def render_section_ce_b_chart_b16_5_2_19(self):
        self.general_ce_b_figure(19, 'Peak Hour Latent Coil Load',
                                 'Load (Wh/h thermal)', 'annual_load_maxima',
                                 'evaporator_latent', 'Wh')

    def render_section_ce_b_chart_b16_5_2_23(self):
        self.general_ce_b_figure(23, 'Hourly Maximum COP2',
                                 'COP2', 'annual_cop_zone',
                                 'cop2_max', 'value')

    def render_section_ce_b_chart_b16_5_2_25(self):
        self.general_ce_b_figure(25, 'Hourly Minimum COP2',
                                 'COP2', 'annual_cop_zone',
                                 'cop2_min', 'value')

    def render_section_ce_b_chart_b16_5_2_29(self):
        self.general_ce_b_figure(29, 'Hourly Maximum Indoor Dry-Bulb Temperature',
                                 'Temperature (C)', 'annual_cop_zone',
                                 'indoor_db_max', 'c')

    def render_section_ce_b_chart_b16_5_2_31(self):
        self.general_ce_b_figure(31, 'Hourly Minimum Indoor Dry-Bulb Temperature',
                                 'Temperature (C)', 'annual_cop_zone',
                                 'indoor_db_min', 'c')

    def render_section_ce_b_chart_b16_5_2_34(self):
        self.general_ce_b_figure(34, 'Hourly Maximum Zone Humidity Ratio',
                                 'Humidity Ratio (kg/kg)', 'annual_cop_zone',
                                 'indoor_hum_rat_max', 'kg_kg')

    def render_section_ce_b_chart_b16_5_2_36(self):
        self.general_ce_b_figure(36, 'Hourly Minimum Zone Humidity Ratio',
                                 'Humidity Ratio (kg/kg)', 'annual_cop_zone',
                                 'indoor_hum_rat_min', 'kg_kg')

    def render_section_ce_b_chart_b16_5_2_39(self):
        self.general_ce_b_figure(39, 'Hourly Maximum Zone Relative Humidity',
                                 'Relative Humidity (%)', 'annual_cop_zone',
                                 'indoor_rel_hum_max', 'perc')

    def render_section_ce_b_chart_b16_5_2_41(self):
        self.general_ce_b_figure(41, 'Hourly Minimum Zone Relative Humidity',
                                 'Relative Humidity (%)', 'annual_cop_zone',
                                 'indoor_rel_hum_min', 'perc')

    def general_ce_b_figure_delta(self, chart_number, caption_end, yaxis, json_dict, json_key, divide=True):
        chart_name = f'section_9_figure_b16_5_2_{chart_number:02d}'
        chart_caption = f'Figure B16.5.2-{chart_number}. HVAC BESTEST: CE300 - CE545 <br>{caption_end}'
        delta_cases = [
            ('E310', 'E300', 'CE310-CE300, Latent Gains', '', 1.),
            ('E320', 'E300', 'CE320-CE300, Infiltration', '', 1.),
            ('E330', 'E300', 'CE330-CE300, 100% OA', '', 1.),
            ('E330', 'E320', 'CE330-CE320, OA-Infl', '', 1.),
            ('E340', 'E300', 'CE340-CE300, 50/50 OA/inf', '', 1.),
            ('E330', 'E340', 'CE330-CE340, OA-50/50', '', 1.),
            ('E350', 'E300', 'CE350-CE300, Tstat Set Up', '', 1.),
            ('E360', 'E300', '(CE360-CE300)/4, Overload', 'CE360-CE300, Overload', 4.),
            ('E400', 'E300', 'CE400-CE300, Ec. T Ctrl', '', 1.),
            ('E410', 'E300', 'CE410-CE300, Ec. No Compr.', '', 1.),
            ('E420', 'E300', 'CE420-CE300, Ec. ODB Lim.', '', 1.),
            ('E430', 'E300', 'CE430-CE300, Ec. Enth Ctrl', '', 1.),
            ('E440', 'E300', 'CE440-CE300, Ec. Enth Lim', '', 1.),
            ('E500', 'E300', '(CE500-CE300)/2, 0%OA', 'CE500-CE300, 0%OA', 2.),
            ('E510 May-Sep', 'E500 May-Sep', '(CE510-CE500)/4, PLR', 'CE510-CE500, PLR', 4.),
            ('E525', 'E520', 'CE525-CE520, EDB', '', 1.),
            ('E530', 'E500', 'CE530-CE500, Dry Coil', '', 1.),
            ('E545', 'E540', 'CE545-CE540, EDB (Dry)', '', 1.),
        ]
        data_table = []
        row_headings = []
        column_headings = ['Case']
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        for case_a_lookup, case_b_lookup, case_name, no_divide_case_name, divisor in delta_cases:
            row = []
            if not divide and no_divide_case_name:
                row_headings.append(no_divide_case_name)
            else:
                row_headings.append(case_name)
            for tst, json_obj in self.json_data.items():
                if json_dict != 'annual_sums_means':
                    case_a_lookup = case_a_lookup[:4]
                    case_b_lookup = case_b_lookup[:4]
                case_a_value = json_obj[json_dict][case_a_lookup][json_key]
                case_b_value = json_obj[json_dict][case_b_lookup][json_key]
                if math.isnan(case_a_value) or math.isnan(case_b_value):
                    row.append(math.nan)
                else:
                    if not divide:
                        divisor = 1.
                    row.append((float(case_a_value) - float(case_b_value)) / divisor)
            data_table.append(row)
        self._create_plotly_bar(chart_name, data_table, row_headings, column_headings, yaxis, chart_caption)
        return

    def render_section_ce_b_chart_b16_5_2_2(self):
        self.general_ce_b_figure_delta(2,
                                       'Annual Total Space Cooling Electricity Consumption Sensitivities',
                                       'Electricity Consumption (kWh)',
                                       'annual_sums_means',
                                       'cooling_energy_total_kWh')

    def render_section_ce_b_chart_b16_5_2_6(self):
        self.general_ce_b_figure_delta(6,
                                       'Annual Compressor Electricity Consumption Sensitivities',
                                       'Electricity Consumption (kWh)',
                                       'annual_sums_means',
                                       'cooling_energy_compressor_kWh')

    def render_section_ce_b_chart_b16_5_2_8(self):
        self.general_ce_b_figure_delta(8,
                                       'Annual Indoor (Supply) Fan Electricity Consumption Sensitivities',
                                       'Electricity Consumption (kWh)',
                                       'annual_sums_means',
                                       'indoor_fan_kWh',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_10(self):
        self.general_ce_b_figure_delta(10,
                                       'Annual Outdoor (Condenser) Fan Electricity Consumption Sensitivities',
                                       'Electricity Consumption (kWh)',
                                       'annual_sums_means',
                                       'condenser_fan_kWh')

    def render_section_ce_b_chart_b16_5_2_15(self):
        self.general_ce_b_figure_delta(15,
                                       'Annual Sensible Cooling Load Sensitivities',
                                       'Load (kWh thermal)',
                                       'annual_sums_means',
                                       'evaporator_load_sensible_kWh')

    def render_section_ce_b_chart_b16_5_2_18(self):
        self.general_ce_b_figure_delta(18,
                                       'Annual Latent Cooling Load Sensitivities',
                                       'Load (kWh thermal)',
                                       'annual_sums_means',
                                       'evaporator_load_latent_kWh')

    def render_section_ce_b_chart_b16_5_2_22(self):
        self.general_ce_b_figure_delta(22,
                                       'Annual Mean COP2 Sensitivities',
                                       'COP2',
                                       'annual_sums_means',
                                       'cop2',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_28(self):
        self.general_ce_b_figure_delta(28,
                                       'Annual Mean IDB Sensitivities',
                                       'Temperature (C)',
                                       'annual_sums_means',
                                       'indoor_dry_bulb_c',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_33(self):
        self.general_ce_b_figure_delta(33,
                                       'Annual Mean Humidity Ratio Sensitivities',
                                       'Humidity Ratio (kg/kg)',
                                       'annual_sums_means',
                                       'zone_humidity_ratio_kg_kg',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_38(self):
        self.general_ce_b_figure_delta(38,
                                       'Annual Mean Relative Humidity Sensitivities',
                                       'Relative Humidity (%)',
                                       'annual_sums_means',
                                       'zone_relative_humidity_perc',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_4(self):
        self.general_ce_b_figure_delta(4,
                                       'Hourly Maximum Total Space Cooling Consumption Sensitivities',
                                       'Electricity Consumption (Wh/h)',
                                       'annual_load_maxima',
                                       'compressors_plus_fans_Wh',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_13(self):
        self.general_ce_b_figure_delta(13,
                                       'Hourly Maximum Total Coil Load Sensitivities',
                                       'Load (Wh/h thermal)',
                                       'annual_load_maxima',
                                       'evaporator_total_Wh',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_20(self):
        self.general_ce_b_figure_delta(20,
                                       'Hourly Maximum Latent Coil Load Sensitivities',
                                       'Load (Wh/h thermal)',
                                       'annual_load_maxima',
                                       'evaporator_latent_Wh',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_24(self):
        self.general_ce_b_figure_delta(24,
                                       'Hourly Maximum COP2 Sensitivities',
                                       'COP2',
                                       'annual_cop_zone',
                                       'cop2_max_value',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_26(self):
        self.general_ce_b_figure_delta(26,
                                       'Hourly Minimum COP2 Sensitivities',
                                       'COP2',
                                       'annual_cop_zone',
                                       'cop2_min_value',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_30(self):
        self.general_ce_b_figure_delta(30,
                                       'Hourly Maximum IDB Sensitivities',
                                       'Temperature (C)',
                                       'annual_cop_zone',
                                       'indoor_db_max_c',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_35(self):
        self.general_ce_b_figure_delta(35,
                                       'Hourly Maximum Humidity Ratio Sensitivities',
                                       'Humidity Ratio (kg/kg)',
                                       'annual_cop_zone',
                                       'indoor_hum_rat_max_kg_kg',
                                       divide=False)

    def render_section_ce_b_chart_b16_5_2_40(self):
        self.general_ce_b_figure_delta(40,
                                       'Hourly Maximum Relative Humidity Sensitivities',
                                       'Relative Humidity (%)',
                                       'annual_cop_zone',
                                       'indoor_rel_hum_max_perc',
                                       divide=False)

    def general_ce_b_figure_24hr(self, chart_code, caption_end, yaxis, json_key):
        chart_name = f'section_9_figure_b16_5_2_{chart_code}'
        chart_caption = f'Figure B16.5.2-{chart_code}. HVAC BESTEST: CE300 June 28 Hourly {caption_end}'
        data_table = []
        column_headings = ['Hour', ]
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        row_headings = [str(x) for x in range(1, 25)]
        for hour in range(1, 25):
            row = []
            for tst, json_obj in self.json_data.items():
                hour_value = json_obj['june28_hourly'][str(hour)][json_key]
                row.append(hour_value)
            data_table.append(row)
        self._create_plotly_line(chart_name, data_table, row_headings, column_headings, yaxis, chart_caption)
        return

    def render_section_ce_b_chart_b16_5_2_48(self):
        self.general_ce_b_figure_24hr('48',
                                      'COP2',
                                      'COP2',
                                      'cop2')

    def render_section_ce_b_chart_b16_5_2_49(self):
        self.general_ce_b_figure_24hr('49',
                                      'Zone Humidity Ratio',
                                      'Humidity Ratio (kg/kg)',
                                      'zone_humidity_ratio_kg_kg')

    def render_section_ce_b_chart_b16_5_2_51(self):
        self.general_ce_b_figure_24hr('51',
                                      'Outdoor Dry-Bulb Temperature',
                                      'Temperature (C)',
                                      'outdoor_drybulb_c')

    def render_section_ce_b_chart_b16_5_2_52(self):
        self.general_ce_b_figure_24hr('52',
                                      'Outdoor Humidity Ratio',
                                      'Humidity Ratio (kg/kg)',
                                      'outdoor_humidity_ratio_kg_kg')

    def general_ce_b_figure_24hr_sum(self, chart_code, caption_end, yaxis, json_key_a, json_key_b):
        chart_name = f'section_9_figure_b16_5_2_{chart_code}'
        chart_caption = f'Figure B16.5.2-{chart_code}. HVAC BESTEST: CE300 June 28 Hourly {caption_end}'
        data_table = []
        column_headings = ['Hour', ]
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'])
        row_headings = [str(x) for x in range(1, 25)]
        for hour in range(1, 25):
            row = []
            for tst, json_obj in self.json_data.items():
                hour_value_a = json_obj['june28_hourly'][str(hour)][json_key_a]
                hour_value_b = json_obj['june28_hourly'][str(hour)][json_key_b]
                row.append(hour_value_a + hour_value_b)
            data_table.append(row)
        self._create_plotly_line(chart_name, data_table, row_headings, column_headings, yaxis, chart_caption)
        return

    def render_section_ce_b_chart_b16_5_2_46(self):
        self.general_ce_b_figure_24hr_sum('46',
                                          'Electricity Consumption (Compressor + OD Fan)',
                                          'Electricity Consumption (Wh/h)',
                                          'compressor_Wh', 'condenser_fans_Wh')

    def general_ce_b_figure_24hr_two_series(self, chart_code, caption_end, yaxis, name_a, name_b, json_key_a,
                                            json_key_b):
        chart_name = f'section_9_figure_b16_5_2_{chart_code}'
        chart_caption = f'Figure B16.5.2-{chart_code}. HVAC BESTEST: CE300 June 28 Hourly {caption_end}'
        data_table = []
        column_headings = ['Hour', ]
        row_headings = [str(x) for x in range(1, 25)]
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'] + ' ' + name_a)
        for _, json_obj in self.json_data.items():
            column_headings.append(json_obj['identifying_information']['software_name'] + ' ' + name_b)
        for hour in range(1, 25):
            row = []
            for tst, json_obj in self.json_data.items():
                hour_value = json_obj['june28_hourly'][str(hour)][json_key_a]
                row.append(hour_value)
            for tst, json_obj in self.json_data.items():
                hour_value = json_obj['june28_hourly'][str(hour)][json_key_b]
                row.append(hour_value)
            data_table.append(row)
        self._create_plotly_line(chart_name, data_table, row_headings, column_headings, yaxis, chart_caption)
        return

    def render_section_ce_b_chart_b16_5_2_47(self):
        self.general_ce_b_figure_24hr_two_series('47',
                                                 'Coil Loads',
                                                 'Load (Wh/h thermal)',
                                                 'Sensible',
                                                 'Latent',
                                                 'evaporator_sensible_Wh',
                                                 'evaporator_latent_Wh')

    def render_section_ce_b_chart_b16_5_2_50(self):
        self.general_ce_b_figure_24hr_two_series('50',
                                                 'Entering Dry- and Wet-Bulb',
                                                 'Temperature (C)',
                                                 'EDB',
                                                 'EWB',
                                                 'entering_drybulb_c',
                                                 'entering_wetbulb_c')
