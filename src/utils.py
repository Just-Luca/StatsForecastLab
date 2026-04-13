
from collections.abc import Sequence
from pathlib import Path
from src import constants as cnsts 
from src import grid_search_parameters as gsp
from typing import Tuple

import numpy as np
import os
import pandas as pd

os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

# @TODO: i can move this function into the class directly, and then add a my_lab.py file containing all the pipeline (def main()...)

# def create_folder_structure(self):
#     if self.normalization:
#         cnsts.CSV_FOLDER_NORM.mkdir(parents=True, exist_ok=True)
#       return cnsts.CSV_FOLDER_NORM
#     else:
#         cnsts.CSV_FOLDER.mkdir(parents=True, exist_ok=True)
#       return cnsts.CSV_FOLDER

def create_folder_structure():
    cnsts.CSV_FOLDER_NORM.mkdir(parents=True, exist_ok=True)


def create_output_folder_structure(
        horizon: int, 
        transformation: str, 
        n_windows: int
    ) -> Tuple[Path, Path, Path, Path]:

    horizon_folder = cnsts.CSV_FOLDER_NORM / f"horizon={horizon}"
    horizon_folder.mkdir(parents=True, exist_ok=True)

    base_folder = horizon_folder / f"SF_container_{transformation}"
    base_folder.mkdir(parents=True, exist_ok=True)

    inner_folder = base_folder / f"stats_training"
    inner_folder.mkdir(parents=True, exist_ok=True)

    results_folder = inner_folder / f"stats_training_results_(nw={n_windows})"
    results_folder.mkdir(parents=True, exist_ok=True)

    fit_results_path = inner_folder / f"test_forecast_df.csv"
    cv_results_path = results_folder / f"cv_df_(nw={n_windows}).csv"
    
    fore_metric_results_path = inner_folder / f"test_fore_metric_bm_df.csv"
    cross_metric_results_path = results_folder / f"cross_metric_bm_df_(nw={n_windows}).csv"

    return fit_results_path, cv_results_path, fore_metric_results_path, cross_metric_results_path


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    # @TODO: check if is needed
    norm_df = df.copy()

    norm_df["y"] = norm_df.groupby("unique_id")["y"].transform(lambda x: x / x.max())
    return norm_df


def remove_outliers_iqr(
        df: pd.DataFrame, 
        column: str = "y", 
        m: float = 0.25, 
        n: float = 0.75
    ) -> pd.DataFrame:

    """
    Removes outliers from a pandas DataFrame using the IQR method and replaces
    the missing values with the mean of the non-outlier values.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column (str): The column name where outliers need to be removed.

    Returns:
    pd.DataFrame: DataFrame with outliers removed and replaced with the mean.
    """
    # Calculate Q_m (m-th percentile) and Q_n (n-th percentile)
    Q_m = df[column].quantile(m)
    Q_n = df[column].quantile(n)

    IQR = Q_n - Q_m

    # Define outlier bounds
    lower_bound = Q_m - 1.5 * IQR
    upper_bound = Q_n + 1.5 * IQR

    # Create a mask for outliers
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

    # Replace outliers with NaN
    df.loc[outlier_mask, column] = np.nan

    # Fill the NaN values (outliers) with the mean of the remaining non-outlier values
    df[column] = df[column].fillna(df[column].mean())

    return df


def apply_transformation_to_dataframe(
        data: pd.DataFrame, 
        transf: str="identity"
    ) -> pd.DataFrame:

    cols = ["unique_id", "ds"]
    df_outl = remove_outliers_iqr(data)

    match transf:
        case "identity":
            data_frame = data
        case "log":
            data_frame = pd.DataFrame(
                np.log(data.set_index(cols))
            ).reset_index(cols)
        case "root3":
            data_frame = pd.DataFrame(
                data.set_index(cols)
                .pow(1 / 3)
                .reset_index(cols)
            )
        case "outl+log":
            data_frame = pd.DataFrame(
                np.log(df_outl.set_index(cols))
            ).reset_index(cols)
        case "outl+root3":
            data_frame = pd.DataFrame(
                df_outl.set_index(cols)
                .pow(1 / 3)
                .reset_index(cols)
            )
        case "outl":
            data_frame = df_outl
        case _:
            raise NotImplementedError(f"Transformation not supported, got {transf}")

    return pd.DataFrame(data_frame)


def apply_inverse_transformation_to_dataframe(
        # @TODO: check if "data: pd.DataFrame," raises some errors
        data, 
        transf: str="identity"
    ) -> pd.DataFrame:
    
    cols = ["unique_id", "ds", "cutoff"]
    if "cutoff" not in data.columns:
        cols.pop(-1)

    match transf:
        case "identity":
            data_frame = data
        case "root3":
            data_frame = data.set_index(cols).pow(3).reset_index(cols)
        case "log":
            data_frame = pd.DataFrame(np.exp(data.set_index(cols))).reset_index(cols)
        case "outl+log":
            data_frame = pd.DataFrame(np.exp(data.set_index(cols))).reset_index(cols)
        case "outl+root3":
            data_frame = data.set_index(cols).pow(3).reset_index(cols)
        case "outl":
            data_frame = data
        case _:
            raise NotImplementedError(
                f"Inverse transformation not supported, got {transf}"
            )

    return pd.DataFrame(data_frame)


def _load_data():
    df_pivot = pd.read_csv(cnsts.DATA_PATH).assign(
        date=lambda x: pd.to_datetime(x["date"])
    )

    rename_dict = {
        col: col.split("_")[1].lower()
        for col in df_pivot.columns
        if col.startswith("volatility")
    }
    rename_dict["date"] = "ds"

    df_pivot = df_pivot.rename(columns=rename_dict)

    return df_pivot


def process_data(
        transformation: str = "identity",
        normalization: bool = False,
        test: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Loads and processes data and returns the correct dataframe ready for training
    """

    df = _load_data()

    df = (
        df
        .melt(id_vars="ds", value_name="y", var_name="unique_id")
        .sort_values(["unique_id", "ds"])
        .iloc[:, [1, 0, 2]]
    )

    df = normalize_data(df) if normalization else df

    df_actual = apply_transformation_to_dataframe(df, transformation)

    if test:
        split_date = df["ds"].max() - pd.DateOffset(months=cnsts.TEST_SIZE)

        _df_train = df[df['ds'] < split_date].reset_index(drop=True)

        df_train = apply_transformation_to_dataframe(_df_train, transformation)
        df_test = df[df['ds'] >= split_date].reset_index(drop=True)

        return df_actual, df_train, df_test
    
    else:
        return df_actual, df_actual, df_actual


def get_id():
    return process_data()[0]["unique_id"].unique().tolist()
    

def id_control(inserted_id):

    if isinstance(inserted_id, str):
        ids = [inserted_id]
    elif isinstance(inserted_id, Sequence) and all(isinstance(item, str) for item in inserted_id):
        ids = inserted_id
    else:
        raise ValueError('Please insert a valid id format. It has to be a string or a list of strings.')
    
    def is_present(country_list, country_input):
        if isinstance(country_input, list):
            return all(nome in country_list for nome in country_input)
        
    unique_ids = get_id()

    if is_present(unique_ids, ids):
        pass
    else:
        raise TypeError(
            f'The desired data are not available, got {inserted_id}.'
                + "\n Here a list of possible unique_ids: \n"
                + f"{unique_ids}"
            )
    
    return ids

# n_windows * horizon = quanto la cross validation va nel passato (3 * 72 = 216 ore, cioè 9 giorni)
def get_n_windows(
        data_frame: pd.DataFrame,
        horizon: int
    ):

    # return int((data_frame.groupby('unique_id').size().min() - gsp.LB_DAYS) / horizon - .1)
    return 3

def accuracy_metrics(
        df: pd.DataFrame, 
        metrics=None,
        normalization: bool=False
    ) -> pd.DataFrame:
    
    # Define metric functions
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def norm_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred)) / np.max(y_true)
    
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

    # Default metrics
    all_metrics = {"mae": mae, "norm_mae": norm_mae, "rmse": rmse, "mape": mape, "smape": smape}

    if metrics is None:
        metrics = list(all_metrics.keys())
    else:
        if isinstance(metrics, str):
            metrics = [metrics]  
    
    # Validate metrics
    metrics = [metric.lower() for metric in metrics]
    invalid_metrics = set(metrics) - set(all_metrics.keys())
    if invalid_metrics:
        raise ValueError(f"Invalid metrics specified: {', '.join(invalid_metrics)}. Valid options are: {', '.join(all_metrics.keys())}.")

    # Filter to selected metrics
    metric_functions = {metric: all_metrics[metric] for metric in metrics}

    if "y" not in df.columns:
        df["y"] = process_data(normalization=normalization)[2]["y"]

    # Identify model columns
    non_model_columns = {"unique_id", "ds", "cutoff", "y"}
    model_columns = [col for col in df.columns if col not in non_model_columns]
    
    # Initialize list to store results
    results = []
    
    # Group by unique_id
    for unique_id, group in df.groupby("unique_id"):
        y_true = group["y"].values
        
        # Calculate metrics for each model
        metrics_results = []
        for model in model_columns:
            y_pred = group[model].values
            metrics_results.append({
                "model": model,
                **{metric: func(y_true, y_pred) for metric, func in metric_functions.items()}
            })
        
        # Find the best model for each metric
        for metric_name in metrics:
            best_model = min(metrics_results, key=lambda x: x[metric_name])
            results.append({
                "metric": metric_name,
                "unique_id": unique_id,
                "metric_value": best_model[metric_name],
                "best_model": best_model["model"]
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df.sort_values(["metric", "metric_value"]).reset_index(drop=True)


def calculate_mean_metrics(df_path: Path, metric: str = "mae"):

    df = pd.read_csv(df_path)

    match metric:
        case "mae":
            return df.query('metric == "mae"').mean(numeric_only=True).iloc[0]
        case "norm_mae":
            return df.query('metric == "norm_mae"').mean(numeric_only=True).iloc[0]
        case "mape":
            return round(df.query('metric == "mape"').mean(numeric_only=True).iloc[0], 2)
        case "smape":
            return round(df.query('metric == "smape"').mean(numeric_only=True).iloc[0], 2)
        case "rmse":
            return df.query('metric == "rmse"').mean(numeric_only=True).iloc[0]
        case _:
            raise ValueError(f"Invalid metric specified: {metric}. Choose from {cnsts.valid_metrics}.") 
    

def main():
    os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

    create_folder_structure()


if __name__ == "__main__":
    main()
