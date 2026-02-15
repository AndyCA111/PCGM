import pandas as pd 
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression


def regress_out(X, Z):
    reg = LinearRegression().fit(X, Z)
    Z_residuals = Z - reg.predict(X)



### regress out sex (same as svol)only based on contrl
def test_original_csv_regressOutSvol2():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('800_lab_file.csv')

    # Not use this : Specify the columns you want to keep
    columns_to_keep = ['dx_raw',"Frontal_raw", "Insula_raw", "Parietal_raw", "svol_raw"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum

    # Select only those columns
    df_selected = df[columns_to_keep]
    df = df.dropna()


    # Fit the regression model including  svol  variables
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    reg = LinearRegression().fit(df_ctrl[["svol_raw"]], df_ctrl['Frontal_raw'])
    Z_residuals = df['Frontal_raw'] - reg.predict(df[["svol_raw"]])

    # Get the residuals
    df['residuals'] = Z_residuals
    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)
    
    # Fit the regression model including  svol  variables
    reg = LinearRegression().fit(df_ctrl[["svol_raw"]], df_ctrl['Frontal_Lobe_Sum'])
    Z_residuals = df['Frontal_Lobe_Sum'] - reg.predict(df[["svol_raw"]])
  

    # Get the residuals
    df['residuals'] =Z_residuals
    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']

    ### etoh['Frontal_Lobe_Sum'].dropna()
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Frontal_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")

    ##############-------------------------------#########################
# Fit the regression model including  svol  variables
    X = sm.add_constant(df[['svol_raw']])  # Include group, sex, age, and education
    model = sm.OLS(df['Insula_raw'], X).fit()

    # Get the residuals
    df['residuals'] = model.resid

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)
    
    # Fit the regression model including  svol  variables
    X = sm.add_constant(df[['svol_raw']])  # Include group, sex, age, and education
    model = sm.OLS(df['Insula_Sum'], X).fit()

    # Get the residuals
    df['residuals'] = model.resid

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']

    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Insula_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")


    ##############-------------------------------#########################
    reg = LinearRegression().fit(df_ctrl[["svol_raw"]], df_ctrl['Parietal_raw'])
    Z_residuals = df['Parietal_raw'] - reg.predict(df[["svol_raw"]])


    # Get the residuals
    df['residuals'] = Z_residuals

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)
    
    # Fit the regression model including  svol  variables
    reg = LinearRegression().fit(df_ctrl[["svol_raw"]], df_ctrl['Parietal_Lobe_Sum'])
    Z_residuals = df['Parietal_Lobe_Sum'] - reg.predict(df[["svol_raw"]])

    # Get the residuals
    df['residuals'] =Z_residuals
    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']

    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Parietal_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")
    ##############-------------------------------#########################

    return 

### regress out sex (same as svol)
def test_original_csv_regressOutSvol():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('/scratch/m000065/binxu/gcp/binxu/wepeng/codes/causal_MRI/new_6roi_synthseg.csv')
    # df = df[:200]
    # Not use this : Specify the columns you want to keep
    columns_to_keep = ['dx_raw',"Frontal_raw", "Insula_raw", "Parietal_raw", "svol_raw"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum

    # Select only those columns
    df_selected = df[columns_to_keep]
    df = df.dropna()


    # Fit the regression model including  svol  variables
    X = sm.add_constant(df[['svol_raw']])  # Include group, sex, age, and education
    model = sm.OLS(df['Frontal_raw'], X).fit()

    # Get the residuals
    df['residuals'] = model.resid

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)
    
    # Fit the regression model including  svol  variables
    X = sm.add_constant(df[['Total_Size']])  # Include group, sex, age, and education
    model = sm.OLS(df['Frontal_Lobe_Sum'], X).fit()

    # Get the residuals
    df['residuals'] = model.resid

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']

    ### etoh['Frontal_Lobe_Sum'].dropna()
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Frontal_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")

    ##############-------------------------------#########################
# Fit the regression model including  svol  variables



    X = sm.add_constant(df[['svol_raw']])  # Include group, sex, age, and education
    model = sm.OLS(df['Insula_raw'], X).fit()

    # Get the residuals
    df['residuals'] = model.resid

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)
    
    
    
    
    
    
    # Fit the regression model including  svol  variables
    X = sm.add_constant(df[['Total_Size']])  # Include group, sex, age, and education
    model = sm.OLS(df['Insula_Sum'], X).fit()

    # Get the residuals
    df['residuals'] = model.resid

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']

    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Insula_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")


    ##############-------------------------------#########################
    X = sm.add_constant(df[['svol_raw']])  # Include group, sex, age, and education
    model = sm.OLS(df['Parietal_raw'], X).fit()

    # Get the residuals
    df['residuals'] = model.resid

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)
    
    # Fit the regression model including  svol  variables
    X = sm.add_constant(df[['Total_Size']])  # Include group, sex, age, and education
    model = sm.OLS(df['Parietal_Lobe_Sum'], X).fit()

    # Get the residuals
    df['residuals'] = model.resid

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['residuals']
    values_etoh = df_etoh['residuals']

    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Parietal_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")
    ##############-------------------------------#########################

    return 

def test_original_csv():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('../800_lab_file.csv')

    # Not use this : Specify the columns you want to keep
    columns_to_keep = ['dx_raw',"Frontal_raw", "Insula_raw", "Parietal_raw"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum

    # Select only those columns
    df_selected = df[columns_to_keep]

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['dx_raw'] == 'ctrl']
    df_etoh = df[df['dx_raw'] == 'etoh']


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['Frontal_raw']
    values_etoh = df_etoh['Frontal_raw']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For SynthSeg
    values_ctrl = df_ctrl['Frontal_Lobe_Sum'].dropna()
    values_etoh = df_etoh['Frontal_Lobe_Sum'].dropna()
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Frontal_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")

    ##############-------------------------------#########################
    values_ctrl = df_ctrl['Insula_raw']
    values_etoh = df_etoh['Insula_raw']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For SynthSeg
    values_ctrl = df_ctrl['Insula_Sum'].dropna()
    values_etoh = df_etoh['Insula_Sum'].dropna()
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Insula_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")
    ##############-------------------------------#########################
    values_ctrl = df_ctrl['Parietal_raw']
    values_etoh = df_etoh['Parietal_raw']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For SynthSeg
    values_ctrl = df_ctrl['Parietal_Lobe_Sum'].dropna()
    values_etoh = df_etoh['Parietal_Lobe_Sum'].dropna()
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Parietal_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")
    ##############-------------------------------#########################

    return 

### change the counterfact and comepare two groups
def test_CF_two_csv():
    # Read the CSV file into a DataFrame
    # df = pd.read_csv('pa_data.csv')
    # df_cf = pd.read_csv('cf_pa_data.csv')

    df = pd.read_csv('pa_data_synthseg.csv')
    df_cf = pd.read_csv('cf_pa_data_synthseg.csv')

    # Not use this : Specify the columns you want to keep
    columns_to_keep = ['age',"diagnosis", "parietal", "frontal", "insula"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum

    # Select only those columns
    df_selected = df[columns_to_keep]

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['diagnosis'] == 0]
    df_etoh = df[df['diagnosis'] == 1]

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl_cf = df_cf[df_cf['diagnosis'] == 0]
    df_etoh_cf = df_cf[df_cf['diagnosis'] == 1]


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['frontal']
    values_etoh = df_etoh['frontal']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For do intervention
    values_ctrl_df = df_ctrl_cf['frontal']
    values_etoh_df = df_etoh_cf['frontal']
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl_df, values_etoh_df)

    t_stat, p_value2 = stats.ttest_ind(values_ctrl, values_ctrl_df)
    t_stat, p_value3 = stats.ttest_ind(values_ctrl, values_etoh_df)

    t_stat, p_value4 = stats.ttest_ind(values_etoh, values_etoh_df)
    t_stat, p_value5 = stats.ttest_ind(values_etoh, values_ctrl_df)

    print(f"Frontal_raw p-value: freesuer: {p_value}, cf_pgm: {p_value1}")
    print(f"Frontal Control p-value: ctrl_ctrl: {p_value2}, ctrl_etoh: {p_value3}")
    print(f"Frontal Drinker p-value: etoh_etoh: {p_value4}, etoh_ctrl: {p_value5}")
    print("##############-------------------------------#########################")
    ##############-------------------------------#########################
    values_ctrl = df_ctrl['insula']
    values_etoh = df_etoh['insula']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For intervention
    values_ctrl_df = df_ctrl_cf['insula']
    values_etoh_df = df_etoh_cf['insula']

    t_stat1, p_value1 = stats.ttest_ind(values_ctrl_df, values_etoh_df)

    t_stat, p_value2 = stats.ttest_ind(values_ctrl, values_ctrl_df)
    t_stat, p_value3 = stats.ttest_ind(values_ctrl, values_etoh_df)

    t_stat, p_value4 = stats.ttest_ind(values_etoh, values_etoh_df)
    t_stat, p_value5 = stats.ttest_ind(values_etoh, values_ctrl_df)

    print(f"Insula_raw p-value: freesuer: {p_value}, cf_pgm: {p_value1}")
    print(f"Insula Control p-value: ctrl_ctrl: {p_value2}, ctrl_etoh: {p_value3}")
    print(f"Insula Drinker p-value: etoh_etoh: {p_value4}, etoh_ctrl: {p_value5}")
    print("##############-------------------------------#########################")

    ##############-------------------------------#########################
    values_ctrl = df_ctrl['parietal']
    values_etoh = df_etoh['parietal']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For intervention
    values_ctrl_df = df_ctrl_cf['parietal']
    values_etoh_df = df_etoh_cf['parietal']
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl_df, values_etoh_df)

    t_stat, p_value2 = stats.ttest_ind(values_ctrl, values_ctrl_df)
    t_stat, p_value3 = stats.ttest_ind(values_ctrl, values_etoh_df)

    t_stat, p_value4 = stats.ttest_ind(values_etoh, values_etoh_df)
    t_stat, p_value5 = stats.ttest_ind(values_etoh, values_ctrl_df)

    print(f"Parietal_raw p-value: freesuer: {p_value}, cf_pgm: {p_value1}")
    print(f"Parietal Control p-value: ctrl_ctrl: {p_value2}, ctrl_etoh: {p_value3}")
    print(f"Parietal Drinker p-value: etoh_etoh: {p_value4}, etoh_ctrl: {p_value5}")
    print("##############-------------------------------#########################")
    
    ##############-------------------------------#########################

    return  
def test_cf_csv():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('/home/wepeng/codes/causal_MRI/sample_160_final_test.csv')
    # df = df[:200]
    # Not use this : Specify the columns you want to keep
    #columns_to_keep = ['dx_raw',"Frontal_raw", "Insula_raw", "Parietal_raw"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum
    columns_to_keep = ['diagnosis',"Parietal_Lobe_Sum", "Insula_Sum", "Frontal_Lobe_Sum"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum

    # Select only those columns
    df_selected = df[columns_to_keep]

    # Divide the DataFrame into two based on the 'dx_raw' column
    df_ctrl = df[df['diagnosis'] == 0]
    df_etoh = df[df['diagnosis'] == 1]


    ## compute the significance of each ROI between the two groups:
    # Extract the values for each group
    values_ctrl = df_ctrl['frontal']
    values_etoh = df_etoh['frontal']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For SynthSeg
    values_ctrl = df_ctrl['Frontal_Lobe_Sum']
    values_etoh = df_etoh['Frontal_Lobe_Sum']
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Frontal_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")

    ##############-------------------------------#########################
    values_ctrl = df_ctrl['insula']
    values_etoh = df_etoh['insula']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For SynthSeg
    values_ctrl = df_ctrl['Insula_Sum'].dropna()
    values_etoh = df_etoh['Insula_Sum'].dropna()
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Insula_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")
    ##############-------------------------------#########################
    values_ctrl = df_ctrl['parietal']
    values_etoh = df_etoh['parietal']
    t_stat, p_value = stats.ttest_ind(values_ctrl, values_etoh)

    ### For SynthSeg
    values_ctrl = df_ctrl['Parietal_Lobe_Sum'].dropna()
    values_etoh = df_etoh['Parietal_Lobe_Sum'].dropna()
    t_stat1, p_value1 = stats.ttest_ind(values_ctrl, values_etoh)
    print(f"Parietal_raw p-value: freesuer: {p_value}, SynthSeg: {p_value1}")
    ##############-------------------------------#########################

    return 
if __name__=="__main__":
    test_original_csv_regressOutSvol()
    # test_cf_csv()
    # test_CF_two_csv()
