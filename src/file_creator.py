import pathlib
import subprocess
import papermill as pm

root_directory = pathlib.Path(__file__).parent.parent.resolve()
program_file_names = [
    'test-0.0.0-results5-2a.json',
]

for program_file_name in program_file_names:
    notebook_name = program_file_name.replace('.json', '.ipynb')
    pm.execute_notebook(
        root_directory.joinpath('rendered', 'base', 'RESULTS5-2A.ipynb'),
        root_directory.joinpath('rendered', 'notebooks', notebook_name),
        parameters=dict(
            program_file=program_file_name
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
