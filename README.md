# ashrae-140-automation
Automation of ASHRAE 140 Testing Verification

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/JasonGlazer/ashrae-140-automation/unit_tests.yml?branch=main)](https://github.com/JasonGlazer/ashrae-140-automation/actions)
[![Coverage Status](https://coveralls.io/repos/github/JasonGlazer/ashrae-140-automation/badge.svg?branch=main)](https://coveralls.io/github/JasonGlazer/ashrae-140-automation?branch=main)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/JasonGlazer/ashrae-140-automation/flake8.yml?branch=main)](https://github.com/JasonGlazer/ashrae-140-automation/actions)

### Processing Steps  
1. Read in tables from Std140_TF_Output.xlsx
2. Format as JSON objects into a 'processed' JSON file
3. Render graphics from processed file
4. Visualize graphics in Rmd files.


### Github Actions Workflow  
1. User submits file in 'input' folder via pull request.
2. Github Actions (GA) detects new files in 'input' folder.
3. GA runs verification workflow: schema, and sensible outputs. (Not yet implemented)
4. GA converts excel files to processed json file.
5. GA detects new files in 'processed' folder.
6. GA renders graphics based in presence of objects in processed file
7. Tests passed, PR ready for merge.

### Notes  
- If this repository is forked, then a personal access token must be [created](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) and saved as a [repository secret](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository) name 'PAT'.  Note, only `workflow` permissions need to be granted for this token.

### Detailed Step-by-step process  
1. A file path is created within the `input/` directory that has the following format: `/<software-name>/<version>/Std140_xx_output.xlsx`.  
    - This file contains at least the 'YourData' tab of the stock `Std140_xx_output.xlsx` file with the pertinent tables filled out.  
2. Github Actions makes a list of files to process by checking the created/modified files in the `input/` directory.
    - The command line call performed is `python src/main input/<software-name>/<version>/Std140_xx_output.xlsx`
3. For each created or modified file, the InputProcessor class picks up the file, performs some data validation via the DataCleanser class, and then creates a JSON file that is written to the `processed/` directory using the same file path as specified above.  This JSON file contains a structured object that should be consistent across all processed files.  
    - Future iterations of this program should include a schema validation step to ensure the data integrity.  
4. Github Actions makes a list of files to render by checking the created/modified files in the `processed/` directory.  
    - The command line call performed is `python src/main processed/<software-name>/<version>/std140_xx_output.json`
    - Individual graphics may be produced using the `rg` flag.  Example: `python src/main processed/<software-name>/<version>/std140_tf_output.json -rg section_7_table_b8_1`.  Multiple section* arguments will render multiple tables.
5. For each created or modified file, the GraphicsRenderer class walks attempts to generate all graphics for that section.  These graphs are stored as PNG files in the `rendered/images/<software-name>/<version>/images` directory using the same file path as specified above.  A markdown file will also be generated automatically under `rendered/images/<software-name>/<version>/` directory for a full rendering of the generated images.
