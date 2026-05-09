import pandas as pd
import numpy as np

def feature_engineering(df):
    
    # ---- Context Features ----
    df['Is_Night'] = df['Light_conditions'].apply(
        lambda x: 1 if 'Dark' in str(x) else 0
    )
    
    df['Bad_Weather'] = df['Weather_conditions'].isin(
        ['Rain', 'Fog', 'Storm']
    ).astype(int)

    df['High_Risk_Scenario'] = df['Is_Night'] * df['Bad_Weather']
    
    # ---- Interaction Features ----
    df['Road_Weather'] = df['Road_surface_type'].astype(str) + "_" + df['Weather_conditions'].astype(str)
    df['Vehicle_Road'] = df['Type_of_vehicle'].astype(str) + "_" + df['Road_surface_type'].astype(str)
    
    # ---- NEW: Enhanced Interaction Features ----
    df['Time_Weather'] = df['Time'].astype(str) + "_" + df['Weather_conditions'].astype(str)
    df['Driver_Age_Experience'] = df['Age_band_of_driver'].astype(str) + "_" + df['Driving_experience'].astype(str)
    df['Junction_Road_Type'] = df['Types_of_Junction'].astype(str) + "_" + df['Road_surface_type'].astype(str)
    
    # ---- NEW: Derived Features ----
    # Risk scores based on collision and casualty patterns
    df['High_Casualty_Count'] = (df['Number_of_casualties'] > df['Number_of_casualties'].median()).astype(int)
    df['Multiple_Vehicles'] = (df['Number_of_vehicles_involved'] > 1).astype(int)
    
    # Driver risk factors
    df['Young_Driver'] = df['Age_band_of_driver'].isin(['18-30']).astype(int)
    df['Inexperienced_Driver'] = df['Driving_experience'].astype(str).str.contains('2-5|Below 2', na=False).astype(int)
    
    # Combined risk indicator
    df['Driver_Risk_Score'] = df['Young_Driver'] + df['Inexperienced_Driver'] + df['Bad_Weather'] + df['Is_Night']
    
    # ---- Rare Category Handling ----
    for col in ['Type_of_vehicle', 'Road_surface_type']:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        df[col] = df[col].replace(rare, 'Other')
    
    return df