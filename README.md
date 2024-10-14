# Project Title

Effect of short-term exposure to particulate matter on mortality rate by cardiovascular disease in Colombia: An observational ecological study with causal machine learning.

## Description

Code shared to reproduce the results of the paper "Effect of short-term exposure to particulate matter on mortality rate by cardiovascular disease in Colombia: An observational ecological study with causal machine learning".
The file data.csv is the dataset used for the results presented in the manuscript. 
The file DAG_test.R reproduces the results of the conditional independences and identification analysis.
The file Positivity_evaluation.R should be use to test the positivity assumption.
The file CVD_pm25.py allows to reproduce the results of the Average Treatment Effect (ATE), and Conditional Average Treatment Effect (CATE).  

## Author

Juan David Gutiérrez  

## libraries/modules

ggdag; dagitty; lavaan; CondIndTests; dplyr; GGally; tidyr; MKdescr; tidyverse 
import os; warnings; random; dowhy; econml; pandas; numpy; sklearn; zepid; matplotlib; scipy; plotnine




