import pathlib
import subprocess
import papermill as pm

root_directory = pathlib.Path(__file__).parent.parent.resolve()

notebook_name = 'test_notebook.ipynb'
pm.execute_notebook(
    root_directory.joinpath('rendered', 'base', 'RESULTS5-2A.ipynb'),
    root_directory.joinpath('rendered', 'notebooks', notebook_name),
    parameters=dict(
        root_directory=str(root_directory),
        json_file_list=[
            'processed/RESULTS5-2A-EnergyPlus-9.0.1.json',
            'processed/RESULTS5-2A-CSE-0.861.1.json'],
        notebook_name=str(root_directory.joinpath('rendered', 'notebooks', notebook_name))
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

# subprocess.run([
#     'jupyter',
#     'nbconvert',
#     str(root_directory.joinpath('rendered', 'notebooks', notebook_name)),
#     '--to',
#     'pdf',
#     '--output-dir',
#     str(root_directory.joinpath('rendered', 'pdf')),
#     '--no-input'
# ])
