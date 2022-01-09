import pandas as pd
import os

area_prod_df = pd.read_csv(os.getcwd()+"/dt_area_prod_a_web.csv")
fert_df = pd.read_csv(os.getcwd()+"/dt_fert_consumption_a_web.csv")
rainfall_df = pd.read_csv(os.getcwd()+"/dt_june_julyaug_rainfall_a_web.csv")
soil_df = pd.read_csv(os.getcwd()+"/dt_soil_type_a_web.csv")

print(area_prod_df.columns)
print(fert_df.columns)
print(rainfall_df.columns)

yeild_rainfall_fert = pd.DataFrame(columns=["Dist","Year","Yield","Rain","N_TC","P_TC","K_TC","NPK_TC"])

for index, row  in area_prod_df.iterrows():
    print(index)
    try:
        dist = row['DIST']
        year = row['YEAR']
        yeild = row['RICE_TQ']/row['RICE_TA']
        rain = rainfall_df[(rainfall_df['YEAR']==year) & (rainfall_df['DIST']==dist)].iloc[0]['ANNUAL']
        n_tc = fert_df[(fert_df['YEAR']==year) & (fert_df['DIST']==dist)].iloc[0]['N_TC'] / row['RICE_TA']
        p_tc = fert_df[(fert_df['YEAR']==year) & (fert_df['DIST']==dist)].iloc[0]['P_TC'] / row['RICE_TA']
        k_tc = fert_df[(fert_df['YEAR']==year) & (fert_df['DIST']==dist)].iloc[0]['K_TC'] / row['RICE_TA']
        npk_tc = fert_df[(fert_df['YEAR']==year) & (fert_df['DIST']==dist)].iloc[0]['NPK_TC'] / row['RICE_TA']
        yeild_rainfall_fert.loc[len(yeild_rainfall_fert.index)] = [dist,year,yeild,rain,n_tc,p_tc,k_tc,npk_tc]
    except:
        continue

print(yeild_rainfall_fert.head)
yeild_rainfall_fert.to_csv(os.getcwd()+"/Output/rice.csv")