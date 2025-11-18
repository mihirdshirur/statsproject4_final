import os
import pandas as pd
import numpy as np
import requests
import gzip
import pgeocode
from meteostat import Point, Hourly
import datetime
import pathlib
import holidays
import zoneinfo
import warnings

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def getWeatherFilePath(load_area):
    filepath = "data/weather/" + load_area + ".csv"
    return filepath

def getPjmFilePath(year):
    filepath = "data/pjm/" + "hrl_load_metered_" + str(year) + ".csv"
    return filepath

def getPjmFreshFilePath():
    return "data/pjm/hrl_load_metered_fresh.csv"

def getCompleteDfFilePath(load_area):
    filepath = "data/complete_dfs/" + load_area + ".csv"
    return filepath

def getModelFilePath(load_area):
    filepath = "models/" + load_area + "_ridge_model.pkl"
    return filepath

def makeDirectories(verbose=False):
    base = pathlib.Path("data")
    subdirs = ["complete_dfs", "pjm", "weather"]
    for sub in subdirs:
        path = base / sub
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created or already existed: {path}")
    path = pathlib.Path("models")
    path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Created or already existed: {path}")

def getLoadAreaToZips():
    # RTO is the *entire PJM footprint*, not a load zone. no meaningful ZIP.
    mymap = {
        # Atlantic City Electric (Southern New Jersey)
        'AECO': ['08401'],   # Atlantic City, NJ
    
        # American Electric Power - Appalachian Power (central and Southern West Virginia)
        'AEPAPT': ['25301'],   # Charleston, WV

        # American Electric Power - Indiana Michigan Power (northeast quadrant of indiana and southwest corner of michigan)
        'AEPIMP': ['46802'],   # Fort Wayne, IN

        # American Electric Power - Kentucky Power (eastern kentucky)
        'AEPKPT': ['41101'],   # Ashland, KY (Eastern Kentucky Power region)

        # American Electric Power - Ohio (central and southeast ohio)
        'AEPOPT': ['43215'],   # Columbus, OH

        # Allegheny Power (FirstEnergy West) serving MD/WV/PA panhandle
        'AP': ['21502'],     # Cumberland, MD

        # Baltimore Gas & Electric
        'BC': ['21201'],     # Baltimore, MD (city center)

        # Cleveland Electric Illuminating Company (FirstEnergy)
        'CE': ['44114'],     # Cleveland, OH (downtown)

        # Dayton Power & Light (AES Ohio)
        'DAY': ['45402'],    # Dayton, OH

        # Duke Energy Ohio/Kentucky load zone
        'DEOK': ['45202'],   # Cincinnati, OH

        # Dominion Virginia Power
        'DOM': ['23219'],    # Richmond, VA

        # Delmarva Power (Delaware & Eastern Shore MD)
        'DPLCO': ['19901'],  # Dover, DE

        # Duquesne Light
        'DUQ': ['15222'],    # Pittsburgh, PA

        # Easton Utilities (Maryland municipal)
        'EASTON': ['21601'], # Easton, MD

        # East Kentucky Power Cooperative
        'EKPC': ['40391'],   # Winchester, KY

        # Jersey Central Power & Light (FirstEnergy NJ)
        'JC': ['07728'],     # Freehold, NJ

        # Metropolitan Edison (FirstEnergy PA)
        'ME': ['19601'],     # Reading, PA

        # Ohio Edison (FirstEnergy OH)
        'OE': ['44308'],     # Akron, OH

        # Ohio Valley Electric Corporation
        'OVEC': ['45661'],   # Piketon, OH

        # Pennsylvania Power Company (FirstEnergy PA)
        'PAPWR': ['16101'],  # New Castle, PA

        # PECO (Philadelphia)
        'PE': ['19103'],     # Philadelphia, PA (Center City)

        # Potomac Electric Power Company (Washington DC + Montgomery Co MD)
        'PEPCO': ['20001'],  # Washington, DC

        # Potomac Edison (FirstEnergy MD/WV)
        'PLCO': ['21740'],   # Hagerstown, MD

        # Pennsylvania Electric Company (Penelec - FirstEnergy Northwest/Central PA)
        'PN': ['16601'],     # Altoona, PA

        # Public Service Electric & Gas (PSE&G NJ)
        'PS': ['07102'],     # Newark, NJ

        # Rockland Electric (Northern NJ / small NY portion)
        'RECO': ['07450'],   # Ridgewood, NJ

        # Southern Maryland Electric Cooperative
        'SMECO': ['20650'],  # Leonardtown, MD

        # UGI Electric (NE Pennsylvania, Luzerne County)
        'UGI': ['18702'],    # Wilkes-Barre, PA

        # VMEU = Virginia Municipal Electric Utility (multiple small cities)
        'VMEU': ['22801']    # Harrisonburg, VA (representative muni)
    }
    return mymap

# Returns the id of the load area (1-indexed)
def getLoadAreaId(load_area):
    load_areas = getLoadAreaToZips().keys()
    load_areas = sorted(load_areas)
    return load_areas.index(load_area)+1


def checkInconsistencies():
    load_area_to_zips = getLoadAreaToZips()
    for year in range(2016,2026):
        df = pd.read_csv(getPjmFilePath(year))
        #print(df['zone'].unique())
        #print(df['load_area'].unique())
        #print(len(df['load_area'].unique()))
        set1 = set(df['load_area'].unique())
        set2 = set(load_area_to_zips.keys())
        set2.add("RTO")
        # Note that years 2016 and 2017 do not match. Thus, we will start with 2018 inclusive
        if len(set1.symmetric_difference(set2)) > 0:
            print(str(year) + " did not match! The difference is: " + str(set1.symmetric_difference(set2)))
    
    print("Finished checking inconsistencies!")

def getYears():
    return range(2018,2026)

# the flag recent_only_no_cache is used when constructing the test-time df
def getWeatherDf(load_area, recent_only_no_cache=False, force_refresh=False, verbose=False):
    # Create a Nominatim geocoder instance for the desired country (For the USA, use 'us')
    nomi = pgeocode.Nominatim('us')
    years = getYears()
    # First check if we have the information cached
    file_path = getWeatherFilePath(load_area)
    if os.path.exists(file_path) and not force_refresh and not recent_only_no_cache:
        temp = pd.read_csv(file_path)
        temp['time'] = pd.to_datetime(temp['time'])
        temp = temp.set_index('time')
        if temp.index.min().year == years[0] and temp.index.max().year == years[-1]:
            if verbose:
                print("Cached weather file for " + load_area + " was found")
            return temp

    # Query the postal code
    load_area_to_zips = getLoadAreaToZips()
    zip_code = load_area_to_zips[load_area][0]
    location_data = nomi.query_postal_code(zip_code)
    latitude = float(location_data.latitude)
    longitude = float(location_data.longitude)

    location = Point(latitude, longitude)
    start = datetime.datetime(years[0], 1, 1)
    if recent_only_no_cache:
        # hardcoded but whatever. start in nov 1 to allow sufficient lagged features to accumulate
        start = datetime.datetime(2025, 11, 1)
    end   = datetime.datetime(years[-1], 12, 31)

    data = Hourly(location, start, end, timezone='UTC').fetch()
    # df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    if recent_only_no_cache:
        #print("Returning recent only no cache:")
        #print(data)
        return data

    data.to_csv(getWeatherFilePath(load_area))
    if verbose:
        print("Added new cached weather file for " + load_area + " using zip code:" + str(zip_code))
    return data

def getAllWeatherDfs(verbose=False):
    load_area_to_zips = getLoadAreaToZips()
    for key, value in load_area_to_zips.items():
        load_area = key
        getWeatherDf(load_area, verbose=verbose)


def getUSHolidays():
    years = getYears()
    us_holidays = holidays.US(years=years)
    # Add black friday
    for year in years:
        thanksgiving = [day for day, name in us_holidays.items() if name == "Thanksgiving Day" and day.year == year][0]
        us_holidays[thanksgiving + datetime.timedelta(days=1)] = "Black Friday"
        us_holidays[thanksgiving - datetime.timedelta(days=1)] = "Thanksgiving Eve"

    return us_holidays

def isHoliday(datetime_obj, holidays_map):
    return datetime_obj.date() in holidays_map

def isThanksgiving(datetime_obj, holidays_map):
    return holidays_map.get(datetime_obj.date()) == "Thanksgiving Day"
    
def isThanksgivingEve(datetime_obj, holidays_map):
    return holidays_map.get(datetime_obj.date()) == "Thanksgiving Eve"
    
def isBlackFriday(datetime_obj, holidays_map):
    return holidays_map.get(datetime_obj.date()) == "Black Friday"
    
def isWeekend(datetime_obj):
    weekno = datetime.datetime.today().weekday()
    if weekno < 5:
        return False
    return True

# Load PJM data
def getEnergyDf(load_area, force_refresh=False, verbose=False):
    years = getYears()
    load_area_to_zips = getLoadAreaToZips()
    if load_area not in load_area_to_zips.keys():
        if verbose:
            print("Error: load_area(" + load_area + ") not found!")
        return -1

    ret_df = None
    for year in years:
        pjm_df = pd.read_csv(getPjmFilePath(year))
        pjm_df = pjm_df[pjm_df['load_area'] == load_area]
        
        if ret_df is None:
            ret_df = pjm_df
        else:
            ret_df = pd.concat([ret_df, pjm_df])

    # Note: 2025 is hardcoded, but whatever
    if 2025 in years:
        # first check if it is cached
        file_path = getPjmFreshFilePath()
        if not os.path.exists(file_path) or force_refresh:
            url = "https://raw.githubusercontent.com/SurtaiHan/stats604fa25proj4/refs/heads/main/data/pjm/hrl_load_metered_fresh.csv"
            out_path = pathlib.Path(getPjmFreshFilePath())
            response = requests.get(url)
            response.raise_for_status()  # raises if download failed
            out_path.write_bytes(response.content)
            if verbose:
                print(f"Downloaded fresh pjm csv to: {out_path}")
        else:
            if verbose:
                print("Cached pjm fresh file was found")

        if verbose:
            print("Loading fresh data in addition to historical data")
        pjm_df = pd.read_csv(getPjmFreshFilePath())
        pjm_df = pjm_df[pjm_df['load_area'] == load_area]
        if ret_df is None:
            ret_df = pjm_df
        else:
            ret_df = pd.concat([ret_df, pjm_df])

    ret_df['datetime_beginning_utc'] = pd.to_datetime(ret_df['datetime_beginning_utc'], utc=True)
    ret_df = ret_df.set_index('datetime_beginning_utc')
    ret_df = ret_df.sort_index()   # Always a good idea after setting index

    return ret_df

# Last step is to augment the pjm data with:
# 1) weather at that current time  (we just use the ground truth weather)
# 2) previous day's weather
# 3) isHoliday, isThanksgiving, isThanksgivingEve, isBlackFriday, isWeekend
# 4) lagged loads (72 hrs ago, and older)

# The resulting dataframe could be viewed as X,y tuples
# in the sense that everything but 

# weather variables available:
# temp	dwpt	rhum	prcp	snow	wdir	wspd	wpgt	pres	tsun	coco
def add_features(energy_df, weather_df, dropna=True, outer=False):
    """
    df must contain at least:
    - mw (float)
    - temp, dwpt, rhum, wspd, tsun, etc. (weather columns)
    - index is hourly timestamps (pd.DatetimeIndex)
    """
    df = energy_df.copy()

    if outer:
        df = df.join(weather_df[['temp']], how='outer')
    else:
        df = df.join(weather_df[['temp']], how='left')
    
    # --- Calendar Features ---
    df['hour'] = df.index.tz_convert('America/New_York').hour
    df['dayofweek'] = df.index.tz_convert('America/New_York').dayofweek
    df['month'] = df.index.tz_convert('America/New_York').month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    # Holiday features (assuming you already defined isHoliday / isThanksgiving):
    us_holidays = getUSHolidays()
    df['is_holiday'] = df.index.tz_convert('America/New_York').to_series().apply(isHoliday, args=(us_holidays,))
    df['is_thanksgiving'] = df.index.tz_convert('America/New_York').to_series().apply(isThanksgiving, args=(us_holidays,))
    df['is_thanksgiving_eve'] = df.index.tz_convert('America/New_York').to_series().apply(isThanksgivingEve, args=(us_holidays,))
    df['is_black_friday'] = df.index.tz_convert('America/New_York').to_series().apply(isBlackFriday, args=(us_holidays,))
    df["is_daylight_savings"] = df.index.tz_convert("America/New_York").to_series().apply(lambda t: t.dst() != pd.Timedelta(0))

    # --- Degree Days ---
    df['HDD'] = (18 - df['temp']).clip(lower=0)
    df['CDD'] = (df['temp'] - 18).clip(lower=0)

    # --- Rolling Weather Averages ---
    df['temp_rolling_24'] = df['temp'].rolling('24h', min_periods=1).mean()
    df['temp_rolling_48'] = df['temp'].rolling('48h', min_periods=1).mean()
    df['temp_rolling_72'] = df['temp'].rolling('72h', min_periods=1).mean()
    # df['dwpt_rolling_24'] = df['dwpt'].rolling(24, min_periods=1).mean()

    # --- Lagged Weather Features ---
    # 24 48 72 96 120 144 168 192 216 240
    df['temp_lag_24'] = df['temp'].shift(freq='24h')
    df['temp_lag_48'] = df['temp'].shift(freq='48h')
    df['temp_lag_72'] = df['temp'].shift(freq='72h')
    df['temp_lag_96'] = df['temp'].shift(freq='96h')
    df['temp_lag_120'] = df['temp'].shift(freq='120h')
    df['temp_lag_144'] = df['temp'].shift(freq='144h')
    df['temp_lag_168'] = df['temp'].shift(freq='168h')
    df['temp_lag_192'] = df['temp'].shift(freq='192h')
    df['temp_lag_216'] = df['temp'].shift(freq='216h')
    df['temp_lag_240'] = df['temp'].shift(freq='240h')

    # --- Lagged Load Features (Key Predictors) ---
    # 72 96 120 144 168 192 216 240
    df['mw_lag_72'] = df['mw'].shift(freq='72h')
    df['mw_lag_96'] = df['mw'].shift(freq='96h')
    df['mw_lag_120'] = df['mw'].shift(freq='120h')
    df['mw_lag_144'] = df['mw'].shift(freq='144h')
    df['mw_lag_168'] = df['mw'].shift(freq='168h')
    df['mw_lag_192'] = df['mw'].shift(freq='192h')
    df['mw_lag_216'] = df['mw'].shift(freq='216h')
    df['mw_lag_240'] = df['mw'].shift(freq='240h')

    #print("Shape before dropna:", df.shape)
    # Drop rows where lag features are missing
    if dropna:
        df = df.dropna()
    #print("Shape after dropna:", df.shape)

    return df

def getCompleteDf(load_area, verbose=False):
    # First check if we have the information cached
    file_path = getCompleteDfFilePath(load_area)
    if os.path.exists(file_path):
        temp = pd.read_csv(file_path)
        temp['datetime_beginning_utc'] = pd.to_datetime(temp['datetime_beginning_utc'], utc=True)
        temp = temp.set_index('datetime_beginning_utc')
        temp = temp.sort_index()
        if verbose:
            print("Cached complete file for " + load_area + " was found")
        return temp

    energy_df = getEnergyDf(load_area, verbose=verbose)
    weather_df = getWeatherDf(load_area, verbose=verbose)
    df_augmented = add_features(energy_df, weather_df)
    # load_area_to_complete_df[load_area] = df_augmented
    df_augmented.to_csv(getCompleteDfFilePath(load_area))
    if verbose:
        print("Added new cached complete file for " + load_area)
    return df_augmented

def getAllCompleteDfs(verbose=False):
    load_area_2_complete_df = {}
    load_area_to_zips = getLoadAreaToZips()
    for load_area in load_area_to_zips.keys():
        load_area_2_complete_df[load_area] = getCompleteDf(load_area, verbose)
    return load_area_2_complete_df

# This function gets the EPT midnight and expresses it as UTC
# calling it with num_days = 0 gives midnight today
# num_days = 1 gives midnight tomorrow, etc
def getMidnight(num_days):
    time_ept = datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York")) + datetime.timedelta(days=num_days)
    midnight_ept = time_ept.replace(hour=0, minute=0, second=0, microsecond=0)
    midnight_utc = midnight_ept.astimezone(zoneinfo.ZoneInfo("UTC"))
    return midnight_utc

def getPredictionFeatures(load_area):
    energy_df = getEnergyDf(load_area, force_refresh=True)
    weather_df = getWeatherDf(load_area, recent_only_no_cache=True, force_refresh=True)
    prediction_df = add_features(energy_df, weather_df, dropna=False, outer=True)

    start = getMidnight(1)
    end = getMidnight(2) - datetime.timedelta(hours=1)
    prediction_df = prediction_df.loc[start:end]
    prediction_df["load_area"] = load_area
    prediction_df["datetime_beginning_ept"] = prediction_df.index.tz_convert('America/New_York')
    prediction_df['mw_lag_216'] = prediction_df['mw_lag_216'].fillna(prediction_df['mw_lag_240'])
    prediction_df['mw_lag_192'] = prediction_df['mw_lag_192'].fillna(prediction_df['mw_lag_216'])
    prediction_df['mw_lag_168'] = prediction_df['mw_lag_168'].fillna(prediction_df['mw_lag_192'])
    prediction_df['mw_lag_144'] = prediction_df['mw_lag_144'].fillna(prediction_df['mw_lag_168'])
    prediction_df['mw_lag_120'] = prediction_df['mw_lag_120'].fillna(prediction_df['mw_lag_144'])
    prediction_df['mw_lag_96'] = prediction_df['mw_lag_96'].fillna(prediction_df['mw_lag_120'])
    prediction_df['mw_lag_72'] = prediction_df['mw_lag_72'].fillna(prediction_df['mw_lag_96'])

    prediction_df['temp_rolling_48'] = prediction_df['temp_rolling_48'].fillna(prediction_df['temp_rolling_72'])
    prediction_df['temp_rolling_24'] = prediction_df['temp_rolling_24'].fillna(prediction_df['temp_rolling_48'])

    prediction_df['temp_lag_216'] = prediction_df['temp_lag_216'].fillna(prediction_df['temp_lag_240'])
    prediction_df['temp_lag_192'] = prediction_df['temp_lag_192'].fillna(prediction_df['temp_lag_216'])
    prediction_df['temp_lag_168'] = prediction_df['temp_lag_168'].fillna(prediction_df['temp_lag_192'])
    prediction_df['temp_lag_144'] = prediction_df['temp_lag_144'].fillna(prediction_df['temp_lag_168'])
    prediction_df['temp_lag_120'] = prediction_df['temp_lag_120'].fillna(prediction_df['temp_lag_144'])
    prediction_df['temp_lag_96'] = prediction_df['temp_lag_96'].fillna(prediction_df['temp_lag_120'])
    prediction_df['temp_lag_72'] = prediction_df['temp_lag_72'].fillna(prediction_df['temp_lag_96'])

    #print("generated prediction_df:")
    #print(prediction_df[["datetime_beginning_ept", "temp", "temp_lag_72","temp_lag_240"]])

    return prediction_df





def get_feature_names_from_ct(preprocessor, cat_cols, num_cols):
    """
    Given a ColumnTransformer with:
      ("num", ..., num_cols)
      ("cat", OneHotEncoder, cat_cols)
    return the full list of feature names in the order seen by the model.
    """
    num_feature_names = list(num_cols)

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = []
    for col_name, cats in zip(cat_cols, cat_encoder.categories_):
        for cat in cats:
            cat_feature_names.append(f"{col_name}={cat}")

    return num_feature_names + cat_feature_names



def save_ridge_weights_pickle(ridge_pipeline, cat_cols, num_cols, prefix):
    """
    ridge_pipeline: Pipeline([('prep', ...), ('ridge', Ridge(...))])
    Saves:
      - {prefix}_ridge_coefs.csv
      - {prefix}_ridge_intercept.txt
      - {prefix}_ridge_model.pkl     <-- pickle instead of joblib
    """
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    prep = ridge_pipeline.named_steps["prep"]
    ridge = ridge_pipeline.named_steps["ridge"]

    feature_names = get_feature_names_from_ct(prep, cat_cols, num_cols)
    coefs = ridge.coef_
    intercept = ridge.intercept_

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df.to_csv(f"{prefix}_ridge_coefs.csv", index=False)

    with open(f"{prefix}_ridge_intercept.txt", "w") as f:
        f.write(str(intercept))

    # Save full pipeline as pickle
    with open(f"{prefix}_ridge_model.pkl", "wb") as f:
        pickle.dump(ridge_pipeline, f)

def split_time_series_with_gaps_df(
    df,
    time_col="datetime_beginning_utc",
    covid_exclude=("2020-03-15 00:00:00", "2020-06-30 23:59:59"),  # set to None to keep all rows
    train_frac=0.70,
    val_frac=0.15,
    gap_hours=168,
):
    """
    Train | gap | Val | gap | Test split that AUTO-matches your time format.

    - If your column is tz-naive (dtype datetime64[ns]), we keep everything naive
      and create naive cutoff timestamps.
    - If your column is tz-aware, we convert to UTC and use UTC-aware cutoffs.

    Returns: train_df, val_df, test_df, summary_dict
    """

    d = df.copy()

    # --- Ensure datetime type; do not force timezone yet ---
    if not pd.api.types.is_datetime64_any_dtype(d[time_col]):
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")

    # Detect tz awareness
    is_tz_aware = pd.api.types.is_datetime64tz_dtype(d[time_col])

    # Normalize to one consistent timeline
    if is_tz_aware:
        # Keep tz-aware; convert everything to UTC for clean math
        d[time_col] = d[time_col].dt.tz_convert("UTC")
        # Helper to build UTC-aware cutoffs
        def _ts(s): return pd.Timestamp(s, tz="UTC")
    else:
        # Keep naive timeline (no tz)
        d[time_col] = d[time_col].dt.tz_localize(None)
        # Helper to build naive cutoffs
        def _ts(s): return pd.Timestamp(s)

    d = d.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    # --- Exclude peak-COVID (optional) ---
    if covid_exclude is not None:
        covid_start = _ts(covid_exclude[0])
        covid_end   = _ts(covid_exclude[1])
        mask_covid = (d[time_col] >= covid_start) & (d[time_col] <= covid_end)
        d = d.loc[~mask_covid].copy()

    # --- Compute boundaries with gaps ---
    gap = pd.Timedelta(hours=gap_hours)
    tmin, tmax = d[time_col].min(), d[time_col].max()
    total_span = tmax - tmin
    eff_span = total_span - 2 * gap
    if eff_span <= pd.Timedelta(0):
        raise ValueError("Time span too short for the requested gaps. Increase range or reduce gap_hours.")

    train_end = tmin + eff_span * train_frac
    val_end   = train_end + gap + eff_span * val_frac

    mask_train = d[time_col] <= train_end
    mask_val   = (d[time_col] > train_end + gap) & (d[time_col] <= val_end)
    mask_test  = d[time_col] > val_end + gap

    train_df = d.loc[mask_train].copy()
    val_df   = d.loc[mask_val].copy()
    test_df  = d.loc[mask_test].copy()

    summary = {
        "time_col": time_col,
        "dtype": str(d[time_col].dtype),
        "time_min": tmin.isoformat(),
        "train_last": train_end.isoformat(),
        "val_last": val_end.isoformat(),
        "time_max": tmax.isoformat(),
        "gap_hours": gap_hours,
        "rows_train": len(train_df),
        "rows_val": len(val_df),
        "rows_test": len(test_df),
        "covid_excluded": covid_exclude is not None,
        "covid_window": None if covid_exclude is None else (str(covid_start), str(covid_end)),
    }
    return train_df, val_df, test_df, summary


def trainRegion(region_name, force_refresh=False):
    csv_path = getCompleteDfFilePath(region_name)
    model_path = getModelFilePath(region_name)
    if os.path.exists(model_path) and not force_refresh:
        print("Existing model for " + region_name + " found. Skipping...")
        return
    
    print(f"\n=== Processing {region_name} from {csv_path} ===")
    df = pd.read_csv(csv_path)
    time_col = "datetime_beginning_ept"  # adjust if your main time col is different

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    
    train_df, val_df, test_df, info = split_time_series_with_gaps_df(
     df,
     time_col="datetime_beginning_ept",
     covid_exclude=("2020-03-15 00:00:00", "2020-06-30 23:59:59"),  # or None
     train_frac=0.70,
     val_frac=0.15,
     gap_hours=168
    )


    # ---- Config (match your CSV) ----
    TIME_COL = "datetime_beginning_ept"
    Y_COL    = "mw"
    TEMP_COL = "temp"

    # Optional extras the CSV already had (only used if present)
    OPTIONAL_NUMERIC = ["CDD", "HDD", "temp_rolling_24"]  # add more if you want
    OPTIONAL_CATS    = [
        "is_holiday", "is_weekend",
        "blackFriday", "thanksgiving", "thanksgiving_eve",
        "is_black_friday", "is_thanksgiving", "is_thanksgiving_eve"
    ]

    # ---- Helper: ensure calendar cols exist without overwriting yours ----
    def ensure_calendar(d):
        d = d.copy()
        if "hour" not in d.columns:
            d["hour"] = pd.to_datetime(d[TIME_COL]).dt.hour
        # prefer existing 'dayofweek' if present
        if "dow" not in d.columns:
            if "dayofweek" in d.columns:
                d["dow"] = d["dayofweek"].astype(int)
            else:
                d["dow"] = pd.to_datetime(d[TIME_COL]).dt.dayofweek
        if "month" not in d.columns:
            d["month"] = pd.to_datetime(d[TIME_COL]).dt.month
        # cast known boolean-ish flags to bool if present
        for c in OPTIONAL_CATS:
            if c in d.columns:
                d[c] = d[c].astype(bool)
        return d

    train_df = ensure_calendar(train_df)
    val_df   = ensure_calendar(val_df)
    test_df  = ensure_calendar(test_df)

    # ---- Detect 24h-spaced lag columns
    #      - mw:   72,96,120,144,168
    #      - temp: 24,48,72,96,120,144,168
    # ------------------------------------------
    MW_LAG_HOURS   = tuple(range(72, 169, 24))  # (72, 96, 120, 144, 168)
    TEMP_LAG_HOURS = tuple(range(24, 169, 24))  # (24, 48, 72, 96, 120, 144, 168)

    def detect_lags(frame, base_name, allowed_hours):
        pat = re.compile(rf"^{re.escape(base_name)}_lag_(\d+)$")
        cols = []
        for c in frame.columns:
            m = pat.match(c)
            if m:
                h = int(m.group(1))
                if h in allowed_hours:
                    cols.append((h, c))
        return [c for _, c in sorted(cols)]

    # mw lags: only 72h and beyond
    mw_lags   = detect_lags(train_df, Y_COL,   allowed_hours=MW_LAG_HOURS)
    # temp lags: from 24h onward (24,48,...,168)
    temp_lags = detect_lags(train_df, TEMP_COL, allowed_hours=TEMP_LAG_HOURS)

    lag_block_cols = mw_lags + temp_lags

    if not lag_block_cols:
        raise ValueError(
            "No suitable lag columns found. "
            "Expected names like mw_lag_72, mw_lag_96, ..., and/or "
            "temp_lag_24, temp_lag_48, ..., temp_lag_168."
        )

    # ---- HARD DROP: remove any rows with NaNs in lag columns + core vars ----
    drop_cols = lag_block_cols + [Y_COL, TEMP_COL]

    train_df = train_df.dropna(subset=drop_cols).copy()
    val_df   = val_df.dropna(subset=drop_cols).copy()
    test_df  = test_df.dropna(subset=drop_cols).copy()

    print("\nAfter dropping NaNs from lags + core variables:")
    print(f"  TRAIN rows: {train_df.shape[0]}")
    print(f"  VAL rows:   {val_df.shape[0]}")
    print(f"  TEST rows:  {test_df.shape[0]}")

    # ---- Stage A: Elastic Net selection on lag blocks (TRAIN only) ----
    sel_scaler = StandardScaler()
    Xtrain_lags = sel_scaler.fit_transform(train_df[lag_block_cols].values)
    ytrain      = train_df[Y_COL].values

    enet = ElasticNetCV(
        l1_ratio=[0.2, 0.4, 0.6],   # small, stable grid
        alphas=None,                # search full path
        cv=3,
        max_iter=5000,
        random_state=0
    ).fit(Xtrain_lags, ytrain)

    coef = enet.coef_
    selected_idx = np.where(coef != 0)[0]
    selected_lag_cols = [lag_block_cols[i] for i in selected_idx]

    # Fallback if ENet is very ridge-y and selects nothing
    if len(selected_lag_cols) == 0:
        corr = train_df[lag_block_cols].corrwith(train_df[Y_COL]).abs().sort_values(ascending=False)
        selected_lag_cols = list(corr.head(2).index)

    print("Selected lag features:")
    for c in selected_lag_cols:
        print("  •", c)

    # Categorical set (only those that are actually present)
    base_cat = ["hour", "dow", "month"]
    present_flags = [c for c in OPTIONAL_CATS if c in train_df.columns]
    cat_cols = base_cat + present_flags

    # Numeric set = target-hour weather + optional numeric extras (if present) + selected lags
    num_base = [col for col in [TEMP_COL] + OPTIONAL_NUMERIC if col in train_df.columns]
    num_cols = num_base + selected_lag_cols

    # In case any optional numeric columns have NaNs, clean again on the exact feature set
    needed_cols = cat_cols + num_cols + [Y_COL]
    train_df_B = train_df.dropna(subset=needed_cols).copy()
    val_df_B   = val_df.dropna(subset=needed_cols).copy()
    test_df_B  = test_df.dropna(subset=needed_cols).copy()

    print("\nAfter final NaN drop on all used features:")
    print(f"  TRAIN (B) rows: {train_df_B.shape[0]}")
    print(f"  VAL   (B) rows: {val_df_B.shape[0]}")
    print(f"  TEST  (B) rows: {test_df_B.shape[0]}")

    # Build transformer & model
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    model = Pipeline([
        ("prep", pre),
        ("ridge", Ridge(alpha=10.0))
    ])

    # ---- Fit on TRAIN, evaluate on VAL ----
    Xtr = train_df_B[cat_cols + num_cols]
    ytr = train_df_B[Y_COL].values
    Xva = val_df_B[cat_cols + num_cols]
    yva = val_df_B[Y_COL].values

    model.fit(Xtr, ytr)
    pred_val = model.predict(Xva)
    rmse_val = np.sqrt(mean_squared_error(yva, pred_val))
    print(f"\n[VAL] RMSE (post-Lasso Ridge): {rmse_val:0.3f}")

    # Baselines (only if the cols exist, and drop NaNs rowwise)
    def baseline_rmse(df_part, hours):
        col = f"{Y_COL}_lag_{hours}"
        if col in df_part.columns:
            tmp = df_part[[Y_COL, col]].dropna()
            if tmp.empty:
                return None
            return np.sqrt(mean_squared_error(tmp[Y_COL], tmp[col]))
        return None

    for h in (72, 168):
        r = baseline_rmse(val_df_B, h)
        if r is not None:
            print(f"[VAL] Naive-{h} RMSE: {r:0.3f}")

    # ---- Refit on TRAIN+VAL, evaluate on TEST ----
    trainval_df = pd.concat([train_df_B, val_df_B], axis=0).sort_values(TIME_COL)
    Xtva = trainval_df[cat_cols + num_cols]
    ytva = trainval_df[Y_COL].values
    Xte  = test_df_B[cat_cols + num_cols]
    yte  = test_df_B[Y_COL].values

    model.fit(Xtva, ytva)
    pred_test = model.predict(Xte)
    rmse_test = np.sqrt(mean_squared_error(yte, pred_test))
    print(f"\n[TEST] RMSE (post-Lasso Ridge): {rmse_test:0.3f}")

    for h in (72, 168):
        r = baseline_rmse(test_df_B, h)
        if r is not None:
            print(f"[TEST] Naive-{h} RMSE: {r:0.3f}")

    print("\nCategorical features used:", cat_cols)
    print("Numeric (unpenalized + selected lags) used:", num_cols)
   
    # 2. Save Ridge "weights" for this region
    prefix = f"models/{region_name}"

    save_ridge_weights_pickle(model, cat_cols, num_cols, prefix)

    print(f"Saved Ridge weights for {region_name} under prefix '{prefix}_*'")





def trainAll(force_refresh=False):
    load_areas = sorted(getLoadAreaToZips().keys())
    for load_area in load_areas:
        trainRegion(load_area, force_refresh)





TIME_COL      = "datetime_beginning_ept"
LOAD_AREA_COL = "load_area"
MODELS_DIR    = "models"


def load_ridge_model_for_zone(zone_name, base_dir=MODELS_DIR):
    path = os.path.join(base_dir, f"{zone_name}_ridge_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ridge model not found at: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_feature_cols_from_pipeline(pipeline):
    prep = pipeline.named_steps["prep"]
    num_cols = list(prep.transformers_[0][2])
    cat_cols = list(prep.transformers_[1][2])
    return num_cols, cat_cols


def ensure_calendar_like_training(df):
    """
    Make test_df look like the training data w.r.t calendar cols:
      - ensure TIME_COL is datetime
      - create hour if missing
      - create dow if missing (prefer 'dayofweek' if present)
      - create month if missing
    """
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL])

    if "hour" not in d.columns:
        d["hour"] = d[TIME_COL].dt.hour

    if "dow" not in d.columns:
        if "dayofweek" in d.columns:
            d["dow"] = d["dayofweek"].astype(int)
        else:
            d["dow"] = d[TIME_COL].dt.dayofweek

    if "month" not in d.columns:
        d["month"] = d[TIME_COL].dt.month

    return d


def impute_mw_lags(df, used_num_cols):
    """
    For any columns named mw_lag_<number> that are actually used in num_cols,
    forward-fill missing values from the last available value.
    """
    d = df.copy()
    lag_cols = [c for c in used_num_cols if re.match(r"^mw_lag_\d+$", c)]
    for col in lag_cols:
        if col in d.columns:
            d[col] = d[col].ffill().bfill()
    return d


def impute_other_features(df, num_cols, cat_cols):
    """
    Impute remaining NaNs in used numeric and categorical feature columns,
    without dropping any rows.
    """
    d = df.copy()

    # Numeric features: simple strategy = forward-fill, then back-fill, then median
    for col in num_cols:
        if col not in d.columns:
            continue
        if d[col].isna().any():
            # ffill/bfill first (time-structure-friendly)
            d[col] = d[col].ffill().bfill()
            # If still NaNs (all values were NaN), fill with overall median or 0
            if d[col].isna().any():
                median_val = d[col].median()
                if np.isnan(median_val):
                    median_val = 0.0
                d[col] = d[col].fillna(median_val)

    # Categorical features: fill NaNs with a placeholder
    for col in cat_cols:
        if col not in d.columns:
            continue
        if d[col].isna().any():
            d[col] = d[col].fillna("missing")

    return d


def build_single_zone_line(current_date_str, test_df, output_txt_path, verbose=False):
    """
    current_date_str : "YYYY-MM-DD" (submission date)
    test_df          : 24-row DataFrame for ONE zone and ONE target date.
    output_txt_path  : where to write: "YYYY-MM-DD", L_00..L_23, PH, PD

    Steps:
      - infer zone from load_area
      - load correct Ridge model
      - ensure calendar cols (hour, dow, month)
      - drop columns not used by the model
      - impute mw_lag_* used by the model
      - impute remaining NaNs in used num/cat features
      - predict 24 loads, compute peak hour + peak-day flag
    """

    # 1) Calendar fix
    df = ensure_calendar_like_training(test_df)

    # 2) Zone + target date checks
    zones = df[LOAD_AREA_COL].unique()
    if len(zones) != 1:
        raise ValueError(f"Expected exactly 1 zone in {LOAD_AREA_COL}, got {len(zones)}: {zones}")
    zone_name = zones[0]
    if verbose:
        print(f"Detected zone: {zone_name}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    target_dates = df[TIME_COL].dt.date.unique()
    if len(target_dates) != 1:
        raise ValueError(
            f"Expected exactly 1 target date in test_df, got {len(target_dates)}: {target_dates}"
        )
    target_date = target_dates[0]
    if verbose:
        print(f"Target date for this zone: {target_date}")

    # 3) Load model & get feature columns
    ridge_model = load_ridge_model_for_zone(zone_name)
    num_cols, cat_cols = get_feature_cols_from_pipeline(ridge_model)

    # 4) Drop columns that are not used in prediction (keep only needed + minimal meta)
    keep_cols = set([TIME_COL, LOAD_AREA_COL, "hour", "dow", "month"]) | set(num_cols) | set(cat_cols)
    df = df[[c for c in df.columns if c in keep_cols]].copy()

    # 5) Impute mw_lag_* that are actually used by the model
    df = impute_mw_lags(df, num_cols)

    # 6) Impute any remaining NaNs in the used features (num + cat)
    df = impute_other_features(df, num_cols, cat_cols)

    # 7) Build design matrix and predict
    X = df[cat_cols + num_cols]
    preds = ridge_model.predict(X)

    # Sort by hour so L_00..L_23 are in order
    tmp = pd.DataFrame({
        "hour": df["hour"].values,
        "pred": preds
    }).sort_values("hour")

    preds_sorted = tmp["pred"].values
    loads_rounded = np.rint(preds_sorted).astype(int)

    # 8) Task 2: peak hour (0..23)
    peak_hour = int(np.argmax(preds_sorted))

    # 9) Task 3: very simple baseline for peak day
    if (target_date.month == 11) and (target_date.day in (24, 25, 26)):
        peak_day_flag = 1
    else:
        peak_day_flag = 0

    # 10) Assemble and write line
    values = [current_date_str] + loads_rounded.tolist() + [peak_hour, peak_day_flag]
    line = ",".join(str(v) for v in values)

    if verbose:
        with open(output_txt_path, "w") as f:
            f.write(line + "\n")
        print(f"Wrote single-zone line for {zone_name} to {output_txt_path}")
        print("Line content:")
        print(line)
    return (loads_rounded.tolist(), peak_hour, peak_day_flag)


def predictLoadArea(load_area):
    # ------------------ EXAMPLE USAGE ------------------ #
    test_df = getPredictionFeatures(load_area)
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    date_str = tomorrow.strftime("%Y-%m-%d")
    output_path = load_area + "_" + date_str + ".txt"
    return build_single_zone_line(date_str, test_df, output_path)

def generateHeader(n_load_areas):
    # --- 1. Generate L_i_j Columns
    L_columns = []
    for i in range(1, n_load_areas+1):
        # j runs from 0 to 23 (00 to 23)
        # The format code {:02d} ensures the number is zero-padded to two digits (e.g., 5 becomes 05)
        L_columns.extend([f"L{i}_{j:02d}" for j in range(24)])

    # --- 2. Generate PH_i Columns
    PH_columns = [f"PH_{i}" for i in range(1, n_load_areas+1)]

    # --- 3. Generate PD_i Columns
    PD_columns = [f"PD_{i}" for i in range(1, n_load_areas+1)]

    # --- 4. Combine all columns into a single list ---
    # The order is L_columns, then PH_columns, then PD_columns
    header_list = L_columns + PH_columns + PD_columns

    # --- 5. Join the list into a single comma-separated string ---
    csv_header_string = "date," + ", ".join(header_list)

    return csv_header_string

def predictAll():
    load_areas = sorted(getLoadAreaToZips().keys())
    hourly_preds = []
    peak_hours = []
    peak_day_flags = []
    with warnings.catch_warnings():
        # 1. Temporarily set the filter to ignore ALL warnings (or specific ones)
        warnings.filterwarnings("ignore")
        for load_area in load_areas:
            hourly_pred, peak_hour, peak_day_flag = predictLoadArea(load_area)
            hourly_preds = hourly_preds + hourly_pred
            peak_hours.append(peak_hour)
            peak_day_flags.append(peak_day_flag)

    header_str = generateHeader(len(load_areas))
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    date_str = tomorrow.strftime("%Y-%m-%d")
    final_str = date_str + ","

    final_str += ",".join(str(v) for v in hourly_preds)
    final_str += ","
    final_str += ",".join(str(v) for v in peak_hours)
    final_str += ","
    final_str += ",".join(str(v) for v in peak_day_flags)

    
    print(final_str)
