# More info on individual run

- `clusterVariables`: 
    15 variables; manually pre-sorted for clustering; removed highly correlated variables; kept clinically interesting
    - `clusterVariables_allVars`: all 15 variables
    - `clusterVariables_boruta`: Boruta on pre-sorted cluster variables; between 5-7 variables



- `fullRegistry`: 
    32 variables, all variables from the dataset; 
    no removal of correlated variables --> therefore unfit for clustering! 
    Only removed variables with high missigness (>35%) & multiple variables of same nature (multiple scoring systems, three different ageing entries which are highly correlated)
    - `fullRegistry_allVars`: all 32 variables
    - `fullRegistry_boruta`:
