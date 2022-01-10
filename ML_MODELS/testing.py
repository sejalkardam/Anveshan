import hana_ml
from hana_ml import dataframe
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

# conn = dataframe.ConnectionContext('host', 'port', 'username', 'password')
data=pd.read_csv("test.csv")
data['date']=pd.to_datetime(data['date'], infer_datetime_format=True)
data=data.drop('date',1)
variables=data.columns  
matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
for col in matrix.columns:
    for row in matrix.index:
        test_result = grangercausalitytests(data[[row, col]], maxlag=20, verbose=False)            
        p_values = [round(test_result[i+1][0]['ssr_chi2test'][1],4) for i in range(20)]            
        min_p_value = np.min(p_values)
        matrix.loc[row, col] = min_p_value
matrix.columns = [var + '_x' for var in variables]
matrix.index = [var + '_y' for var in variables]
print(matrix)