import pathlib
import subprocess
import papermill as pm
from logger import Logger
from src.descriptors import VerifyInputFile

root_directory = pathlib.Path(__file__).parent.parent.resolve()
program_file_names = [
    'test-0.0.0-results5-2a.json',
]


class FileRenderer(Logger):
    """
    Render and create output files
    """

    file_name = VerifyInputFile()

    def __init__(self,
                 file_name,
                 logger_level="WARNING",
                 logger_name="console_only_logger"):
        super().__init__(logger_level=logger_level, logger_name=logger_name)
        self.file_name = file_name
        return

    def __repr__(self):
        rep = 'FileRenderer(file_path=' + str(self.file_name) + ')'
        return rep

    def run(self):
        notebook_name = '.'.join([self.file_name.stem, 'ipynb'])
        pm.execute_notebook(
            root_directory.joinpath('rendered', 'base', 'RESULTS5-2A.ipynb'),
            root_directory.joinpath('rendered', 'notebooks', notebook_name),
            parameters=dict(
                program_file=str(self.file_name),
                root_directory=str(root_directory)
            )
        )
        subprocess.run([
            'jupyter',
            'nbconvert',
            str(root_directory.joinpath('rendered', 'notebooks', notebook_name)),
            '--to',
            'html',
            '--output-dir',
            str(root_directory.joinpath('rendered', 'html')),
            '--no-input'
        ])
        return
