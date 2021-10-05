# ashrae-140-automation
Automation of ASHRAE 140 Testing Verification

## Todo

### Example Creation  
1. Read in tables from RESULTS-2A.xlsx for standard programs (make as a file in 'inputs' folder)
2. Format as JSON objects in 'processed' folder
3. Make a rendering in jupyter notebook (Fig. B8-9)
4. Export rendering to html and save in a 'static' folder.


### Short Term Workflow Using Github  
1. User submits file in 'input' folder via pull request.
2. Github Actions (GA) detects new files in 'input' folder.
3. GA converts any CSVs to json.
4. GA runs verification workflow: schema, and sensible outputs.
5. GA runs workflow to verify compliance for various tests, creates summaries for
   visualizations, adds them to json object, and writes file to 'processed' folder.
6. GA detects new files in 'processed' folder.
7. GA calls html module (e.g. nbconvert with Jupyter notebook) to render report
   and send html to 'static' folder.  Another Jupyter 'interactive' workbook
   can be created which allows users to change inputs in code for rendering, or
   have a more detailed diagnostic capability (if time allows).
8. GA updates index.html file to include report.
9. GA runs cleanup.
   a. Remove downstream files where the input file was deleted.
   b. Re-render any files that reference deleted files.
10. Tests passed, PR ready for merge.
