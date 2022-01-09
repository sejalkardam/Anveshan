import hana_ml 
import matplotlib.pyplot as plt
import pandas as pd
import os


main_df=pd.read_csv(os.getcwd()+"/Output/rice.csv")
for dist in main_df.Dist.unique():
    print(dist)
    dist_df=main_df[(main_df['Dist']==dist)]
    dist_df[['Year','Yield']].plot(x='Year',y='Yield')
    dist_df[['Year','Rain']].plot(x='Year',y='Rain')
    dist_df[['Year','N_TC']].plot(x='Year',y='N_TC')
    dist_df[['Year','P_TC']].plot(x='Year',y='P_TC')
    dist_df[['Year','K_TC']].plot(x='Year',y='K_TC')
    

    plt.show()
    n=input()
    if(n=="break"):
        break

