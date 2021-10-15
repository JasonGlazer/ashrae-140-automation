import pathlib
import subprocess
import papermill as pm

root_directory = pathlib.Path(__file__).parent.parent.resolve()

notebook_name = 'test_notebook.ipynb'
pm.execute_notebook(
    root_directory.joinpath('rendered', 'base', 'RESULTS5-2A.ipynb'),
    root_directory.joinpath('rendered', 'notebooks', notebook_name),
    parameters=dict(
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
