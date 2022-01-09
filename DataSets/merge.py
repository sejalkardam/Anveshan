import pandas as pd
import os

area_prod_df = pd.read_csv(os.getcwd()+"/dt_area_prod_a_web.csv")
fert_df = pd.read_csv(os.getcwd()+"/dt_fert_consumption_a_web.csv")
rainfall_df = pd.read_csv(os.getcwd()+"/dt_normal_rainfall_a_web.csv")
soil_df = pd.read_csv(os.getcwd()+"/dt_soil_type_a_web.csv")

print(area_prod_df.columns)
print(fert_df.columns)
print(rainfall_df.columns)
print(soil_df.columns)

area_fert = pd.merge(area_prod_df, fert_df, on=['YEAR'])
print(area_fert.columns)
print(area_fert.head)

