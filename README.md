# DHFeCorrLocal (D-D and D-HFe correlations)
This repository contains the code used to analyze correlations between heavy-flavour particles 
(D mesons and HF electrons) obtained from the ALICE experiment. The analysis is divided in the following steps:

## Filtering
The local analysis requires a small data sample (few GB if on local computer and ~100 GB for a local cluster). 
In order to select candidates, please use the task on [https://github.com/alisw/AliPhysics/blob/master/PWGHF/correlationHF/AliAnalysisTaskDHFeCorr.cxx][AliPhysics].
It is advisable to use the options "Derived data production" on the LEGO train system. This part does not depend 
in any code in this repository. Once the LEGO train has finished, you should download the output following the next steps.

## Downloading the output
In order to download the output, you will have to provide where the folders are saved. This can be obtained using:
1. Go to the train run and click on the "processing progress" of any "child". On the new page, it will show you the 
output folder of this 'child'. 
2. Edit the filter option for the output directory. This is the field between "Software versions" and "Job states".
Then remove all the content after the train rum number. You should change from something like 
"PWGHF/HFCJ_pp/561_20190909-1306_child_1$" to "PWGHF/HFCJ_pp/561".
3. Now copy all the content from the "Output directory" to a text file. You can use Excel to help you select only the 
correct columns.
4. Download the files using python dhfcorre/io/download_grid.py grid_username certificate_password location_txt_file destination_folder
Check download_grid.py --help to a description of each parameter.

## Convert from ROOT to parquet


## Select candidates

### Rectangular selections
### XGBoost

## Reduce data

## Perform correlation analysis

