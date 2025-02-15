import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import cfgrib
import numpy as np
import pandas as pd
import pytz
import xarray as xr
from dateutil.relativedelta import relativedelta

# Mapping from GRIB parameter codes to human-readable field names
AROME_FACTORS = {
    "1": "mean_sea_level_pressure",
    "6": "geopotential",
    "11": "temperature",
    "17": "dew_point_temperature",
    "20": "visibility",
    "33": "u_component_of_wind",
    "34": "v_component_of_wind",
    "52": "relative_humidity",
    "61": "total_precipitation",
    "65": "water_equivalent_of_accumulated_snow_depth",
    "66": "snow_cover",
    "67": "boundary_layer_height",
    "71": "cloud_cover",
    "73": "low_cloud_cover",
    "74": "medium_cloud_cover",
    "75": "high_cloud_cover",
    "81": "land_cover",
    "111": "net_short_wave_radiation",
    "112": "net_long_wave_radiation",
    "117": "global_radiation",
    "122": "sensible_heat_flux",
    "132": "latent_heat_flux",
    "162": "u_component_max_squall",
    "163": "v_component_max_squall",
    "181": "_rain_water",
    "184": "_snow_water",
    "186": "cloud_base",
    "201": "_graupel",
    "SD Snow depth m": "snow_depth",
    "T Temperature K": "temperature - Extra copy",
}


def _build_lat_lon_grid(
    grib_message: cfgrib.Message,
) -> Tuple[List[float], List[float]]:
    """
    Build latitude and longitude grids from a GRIB message.

    Args:
        grib_message: A cfgrib.Message object from which to extract grid parameters.

    Returns:
        A tuple containing the list of latitudes and the list of longitudes.
    """
    min_lat = float(grib_message["latitudeOfFirstGridPointInDegrees"])
    max_lat = float(grib_message["latitudeOfLastGridPointInDegrees"])
    min_lon = float(grib_message["longitudeOfFirstGridPointInDegrees"])
    max_lon = float(grib_message["longitudeOfLastGridPointInDegrees"])

    step_lat = int(grib_message["jDirectionIncrement"])
    step_lon = int(grib_message["iDirectionIncrement"])

    # Multiply and then back-scale to include the last grid point properly
    latitudes = [
        x / 1_000
        for x in range(int(min_lat * 1_000), int(max_lat * 1_000 + step_lat), step_lat)
    ]
    longitudes = [
        y / 1_000
        for y in range(int(min_lon * 1_000), int(max_lon * 1_000 + step_lon), step_lon)
    ]
    return latitudes, longitudes


def _process_grib_message(
    grib_message: cfgrib.Message, prediction_moment: datetime, predicted_hour: int
) -> Tuple[str, xr.Dataset]:
    """
    Process a single GRIB message into an xarray.Dataset with an appropriate field name.

    Args:
        grib_message: A cfgrib.Message containing the data.
        prediction_moment: Datetime representing when the data was predicted.
        predicted_hour: Offset (in hours) for the prediction.

    Returns:
        A tuple with the field name and the corresponding xarray.Dataset.
    """
    # Format the level type from CamelCase to lower-case with underscores
    level_type = "_".join(
        re.findall("[A-Z][^A-Z]*", grib_message["typeOfLevel"])
    ).lower()
    level = grib_message["level"]

    # Determine field name based on parameter name and AROME_FACTORS mapping
    param = grib_message["parameterName"]
    if param in AROME_FACTORS:
        field_name = AROME_FACTORS[param]
        # If the factor name starts with an underscore, prepend the step type
        if field_name.startswith("_"):
            field_name = f"{grib_message['stepType']}{field_name}"
    else:
        field_name = f"unknown_code_{param}"

    # Add level information to the field name
    if level == 0 and level_type == "above_ground":
        field_name = f"surface_{field_name}"
    else:
        field_name = f"{level}m_{level_type}_{field_name}"
    field_name = field_name.strip()

    # Build spatial grid and reshape data values accordingly
    lats, lons = _build_lat_lon_grid(grib_message)
    field_values = np.reshape(grib_message["values"], len(lats) * len(lons))

    data_dict = {field_name: (["time", "coord"], [field_values])}
    pred_time = np.datetime64(
        (prediction_moment + relativedelta(hours=predicted_hour))
    ).astype("datetime64[ns]")

    # Create coordinates for the dataset
    coords = {
        "time_of_prediction": [prediction_moment],
        "time": [pred_time],
        "coord": xr.DataArray(
            pd.MultiIndex.from_product([lats, lons], names=["lat", "lon"]), dims="coord"
        ),
    }

    dataset = xr.Dataset(data_vars=data_dict, coords=coords)
    dataset.time.encoding["units"] = "hours since 2018-01-01"
    return field_name, dataset


def _get_times_from_filename(filename: str) -> Tuple[datetime, int]:
    """This function extracts the timeframe a file represents from its name and translates that into the datetime
     that the prediction was made, and the exact hour it represents of that prediction.

    Args:
        filename:   The filename to parse

    Returns:
        A datetime and an int value indicating the prediction datetime and the predicted hour respectively

    """
    components = filename.split("_")
    prediction_str = components[2]
    predicted_str = components[3]

    prediction_moment = pytz.timezone("Europe/Amsterdam").localize(
        datetime.strptime(prediction_str, "%Y%m%d%H%M")
    )
    prediction_moment = prediction_moment.astimezone(pytz.utc).replace(tzinfo=None)
    predicted_hour = int(predicted_str[:3])

    print(
        f"File [{filename}] was parsed to prediction moment and hour: [{prediction_moment}],[{predicted_hour}]"
    )
    return prediction_moment, predicted_hour


def convert_grib_file_to_dataset(grib_file: Path) -> xr.Dataset:
    """
    Convert a single GRIB file into an xarray.Dataset by processing each message.

    Args:
        grib_file: Path to the GRIB file.

    Returns:
        An xarray.Dataset with all valid fields merged.
    """

    prediction_moment, predicted_hour = _get_times_from_filename(grib_file.stem)
    grib_filestream = cfgrib.FileStream(str(grib_file))
    combined_dataset = xr.Dataset()

    for key, message in grib_filestream.items():
        # Process only non-rotated grid data
        if message["gridType"] == "regular_ll":
            field_name, msg_dataset = _process_grib_message(
                message, prediction_moment, predicted_hour
            )
            if not combined_dataset:
                combined_dataset = msg_dataset
            else:
                combined_dataset[field_name] = msg_dataset[field_name]

    combined_dataset = combined_dataset.unstack("coord")
    combined_dataset.time.encoding["units"] = "hours since 2018-01-01"
    return combined_dataset


def netcdf_to_dataframe(
    ds: xr.Dataset,
    locations_filter: List[int] | None = None,
) -> pd.DataFrame:
    """
    Convert an xarray.Dataset (from GRIB conversion) into a pandas DataFrame and add a unique 'location' column.

    Args:
        ds: xarray.Dataset to convert.
        locations_filter: Optional list of location indices to filter the DataFrame

    Returns:
        A pandas DataFrame with columns for coordinates and a computed unique location index.
    """
    df = ds.to_dataframe().reset_index()

    # Compute unique location based on lat/lon indices
    lat_to_index = {lat: idx for idx, lat in enumerate(ds["lat"].values)}
    lon_to_index = {lon: idx for idx, lon in enumerate(ds["lon"].values)}
    lat_index = df["lat"].map(lat_to_index)
    lon_index = df["lon"].map(lon_to_index)
    n_lon = len(ds["lon"].values)
    df["location_idx"] = lat_index * n_lon + lon_index
    df = df.drop(columns=["lat", "lon"])

    # Filter locations if necessary
    df = (
        df[df["location_idx"].isin(locations_filter)]
        if locations_filter is not None
        else df
    )

    return df


def grib_files_to_dataframe(
    grib_files: List[Path],
    locations_filter: List[int] | None = None,
) -> pd.DataFrame:
    """
    Process a list of GRIB files and write the concatenated data to a single Parquet file.

    Args:
        grib_files: List of Path objects pointing to GRIB files.
    """
    if not grib_files:
        raise ValueError("No GRIB files provided for processing.")

    dataframes = []
    for file in grib_files:
        ds = convert_grib_file_to_dataset(file)
        df = netcdf_to_dataframe(ds, locations_filter=locations_filter)
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


# Example usage:
if __name__ == "__main__":
    locations_filter_df = pd.read_csv("selected_locations.csv")
    locations_filter = locations_filter_df["location_idx"].values

    # Single file processing example:
    files = sorted(Path("../data/HA43_P1_2020100100").glob("*_GB"))
    df = grib_files_to_dataframe(files, locations_filter=locations_filter)
    df.to_parquet("single_file.parquet")
