import hana_ml 
from hana_ml import dataframe
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

def adfuller_test(series, sig=0.1, name=''):
    res = adfuller(series, autolag='AIC')    
    p_value = round(res[1], 3) 

    if p_value <= sig:
        print(f" {name} : P-Value = {p_value} => Stationary. ")
        return 1
    else:
        print(f" {name} : P-Value = {p_value} => Non-stationary.")
        return 0
main_df=pd.read_csv(os.getcwd()+"/Output/rice.csv")

for dist in main_df.Dist.unique():
    
    print(dist)
    dist_df=main_df[(main_df['Dist']==dist)]
    dist_df[['Year','Yield']].plot(x='Year',y='Yield')
    dist_df[['Year','Rain']].plot(x='Year',y='Rain')
    dist_df[['Year','N_TC']].plot(x='Year',y='N_TC')
    dist_df[['Year','P_TC']].plot(x='Year',y='P_TC')
    dist_df[['Year','K_TC']].plot(x='Year',y='K_TC')
    

    
    dist_df=dist_df[['Yield','Rain','N_TC','P_TC','K_TC']]
    #Granger Causality Test for 90%
    if dist==dist:
        try:
            variables=dist_df.columns  
            matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
            for col in matrix.columns:
                for row in matrix.index:
                    test_result = grangercausalitytests(dist_df[[row, col]], maxlag=7, verbose=False)            
                    p_values = [round(test_result[i+1][0]['ssr_chi2test'][1],4) for i in range(7)]            
                    min_p_value = np.min(p_values)
                    matrix.loc[row, col] = min_p_value
            matrix.columns = [var + '_x' for var in variables]
            matrix.index = [var + '_y' for var in variables]
            print(matrix)

            for name, column in dist_df.iteritems():
                stationary_count =  adfuller_test(column, name=column.name)

            diff = 0
            stationary_count = 0
            data_diff = dist_df
            cmd = 1
            while (stationary_count!=5 and cmd==1) :
                stationary_count = 0
                diff += 1
                print(str(diff)+" Difference")
                data_diff = data_diff.diff().dropna()
                for name, column in data_diff.iteritems():
                    stationary_count += adfuller_test(column, name=column.name)
                if (stationary_count==5):
                    break
                cmd = int(input("Go for next level differencing (1 for yes): "))
                

                            
        except:
            print("Data Insufficient for this district")
    n=input("Check for next district (press enter):")
    if(n=="break"):
        break




