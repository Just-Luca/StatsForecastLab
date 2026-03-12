
from datetime import timedelta
from matplotlib.ticker import MaxNLocator
from statsforecast import StatsForecast
from pathlib import Path
from polars import Unknown
from sklearn.metrics import mean_pinball_loss
from src import constants as cnsts 
from src import grid_search_parameters as gsp
from src import utils
from statsforecast.models import SeasonalNaive
from typing import Tuple

import itertools
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd

os.environ.setdefault("NIXTLA_ID_AS_COL", "1")


class HupStatsMacrocast:

    def __init__(
            self,
            horizons: list = gsp.HORIZONS,
            transformations: list = gsp.TRANSFORMATIONS,
            models: list = gsp.MODELS,
            normalization: bool = False
        ) -> None:

        self.horizons = horizons
        self.transformations = transformations
        self.models = models
        self.normalization = normalization


    def predict(self):

        for horizon, transformation, model in itertools.product(
                self.horizons, self.transformations, self.models
            ):
            
            df_train, fit_results_path, _, sf, model_name = self._train_loop(
                horizon,
                transformation,
                model
            )

            if fit_results_path.is_file():
                df_old_fore = pd.read_csv(fit_results_path).assign(
                    ds=lambda x: pd.to_datetime(x["ds"])
                )
            else:
                df_old_fore = pd.DataFrame()
            
            if model_name not in df_old_fore.columns:

                sf.fit(df=df_train)

                df_fore = sf.predict(h=horizon)

                df_fore_new = utils.apply_inverse_transformation_to_dataframe(
                    df_fore, transformation
                ).assign(ds=lambda x: pd.to_datetime(x["ds"]))

                if not df_old_fore.empty:
                    _dffore = pd.merge(
                        df_old_fore, df_fore_new, on=["ds", "unique_id"], how="inner"
                    )
                else:
                    _dffore = df_fore_new

                _dffore.to_csv(fit_results_path, index=False, encoding="utf-8")
            else:
                pass


    def cross_validation(self):

        for horizon, transformation, model in itertools.product(
                self.horizons, self.transformations, self.models
            ):
            # print("check 1")
            df_train, _, cv_results_path, sf, model_name = self._train_loop(
                horizon,
                transformation,
                model
            )
            # print("check 2")

            n_windows = utils.get_n_windows(
                data_frame=df_train,
                horizon=horizon
            )
            # print("check 3")

            if cv_results_path.is_file():
                df_old_cv = (
                    pd.read_csv(cv_results_path)
                    .assign(ds=lambda x: pd.to_datetime(x["ds"]))
                    .assign(cutoff=lambda x: pd.to_datetime(x["cutoff"]))
                )
            else:
                df_old_cv = pd.DataFrame()
            # print("check 4")

            if model_name not in df_old_cv.columns:
                df_cross = sf.cross_validation(
                    df=df_train, 
                    h=horizon,
                    n_windows=n_windows, 
                    step_size=horizon
                )
                # print("check 5")
                
                df_cross_new = (
                    utils.apply_inverse_transformation_to_dataframe(df_cross, transformation)
                    .assign(ds=lambda x: pd.to_datetime(x["ds"]))
                    .assign(cutoff=lambda x: pd.to_datetime(x["cutoff"]))
                )
                # print("check 6")
                
                if not df_old_cv.empty:

                    _dfcv = (
                        pd.merge(df_old_cv, df_cross_new, on=["ds", "unique_id", "cutoff"], how="inner", suffixes=('_old', '_new'))
                        .drop_duplicates(subset=['ds', 'unique_id', 'cutoff'])
                        .rename(columns={"y_new": "y"})
                        .drop(columns=['y_old'])
                    )
                    # print("check 7")

                else:
                    _dfcv = df_cross_new
                # print("check 8")
                
                _dfcv.to_csv(cv_results_path, index=False, encoding="utf-8")
            else:
                pass

    
    def best_results_plots(
            self,
            horizon: int = gsp.HORIZONS[0],
            metric: str = "mae",
            result: str ='forecast', 
            eval_horizon: int = cnsts.EVAL_HORIZONS[0], 
            countries=None,
            actual: bool=False
        ):
        
        if countries is None:
            countries = utils.get_id()
        else:
            pass

        if result == 'forecast' or result == 'crossval':

            df_plot = self.best_results_prediction_dataframe(
                horizon=horizon,
                result=result,
                metric=metric,
                countries=countries
            )

        else:
            pass
        
        # crossvalidation (cv) is needed to decide which model is the best. 
        # but, if there is only one model, then cv is kinda useless.
        # NB: this function does not work if cv is not performed. @TODO We need to adjust this!
        df_crossval = self.best_results_prediction_dataframe(
            horizon=horizon,
            result='crossval',
            metric=metric,
            countries=countries
        )

        match result:
            case 'forecast':
                return self._plot_forecast(
                    df_forecast=df_plot, 
                    df_crossval=df_crossval,
                    eval_horizon=eval_horizon, 
                    actual=actual
                )        
            
            case 'crossval':
                return self._plot_cross_validation(
                    df_crossval=df_plot, 
                    eval_horizon=eval_horizon, 
                    actual=actual
                )

            case 'metrics':
                return self._metrics_plot(
                    countries=countries,
                    horizon=horizon,
                )
            case _:
                raise ValueError(f"Invalid result specified, got {result}.")
            

    def best_model_metric_evaluation(
            self,
            result: str = "forecast"
        ):
        
        for horizon, transformation in itertools.product(
            self.horizons, self.transformations
            ):

            _, df_train, _ = utils.process_data(
                transformation=transformation,
                normalization=self.normalization
            )        

            n_windows = utils.get_n_windows(
                data_frame=df_train,
                horizon=horizon
            )
            
            fit_results_path, cv_results_path, fore_metric_results_path, cross_metric_results_path = (
                utils.create_output_folder_structure(horizon, transformation, n_windows)
            )

            if result == "forecast":
                metric_results_path = fore_metric_results_path
                df_forecast = pd.read_csv(fit_results_path)
                if Path.exists(metric_results_path):   
                    continue

                evaluation_df = utils.accuracy_metrics(
                    df_forecast,
                    normalization=self.normalization
                )

            elif result == "crossval":
                metric_results_path = cross_metric_results_path
                df_crossval = pd.read_csv(cv_results_path)
                if Path.exists(metric_results_path):   
                        continue

                evaluation_df = utils.accuracy_metrics(
                    df_crossval,
                    normalization=self.normalization
                )
            else:
                raise ValueError(f"Invalid result specified, got {result}.")

            new_rows = []

            for _, row in evaluation_df.iterrows():

                new_row = {
                    'metric': row['metric'],
                    'unique_id': row['unique_id'],
                    'metric_value': row['metric_value'], # select the value from the column whose name matches that of best_model
                    'best_model': row['best_model']
                }
                new_rows.append(new_row)

            metric_best_model_df = pd.DataFrame(new_rows)
            metric_best_model_df.to_csv(metric_results_path, index=False, encoding='utf-8')


    def best_results_metric_dataframe(
            self,
            horizon: int = gsp.HORIZONS[0],
            metric: str = "mae",
            result: str = "forecast"
        ):

        self.best_model_metric_evaluation(result=result)
        
        metric_results_df = pd.DataFrame()

        for transformation in self.transformations:

            _, df_train, _ = utils.process_data(
                transformation=transformation,
                normalization=self.normalization
            )
        
            n_windows = utils.get_n_windows(
                data_frame=df_train,
                horizon=horizon
            )
            
            _, _, fore_metric_results_path, cross_metric_results_path = (
                utils.create_output_folder_structure(horizon, transformation, n_windows)
            )

            if result == "forecast":
                metric_file_path = fore_metric_results_path

            elif result == "crossval":
                metric_file_path = cross_metric_results_path
            else:
                raise ValueError(f"Invalid result specified, got {result}.")
            
            metric_df = pd.read_csv(metric_file_path)   
            metric_df['transformation'] = transformation

            if metric in cnsts.valid_metrics:
                filtered_df = metric_df.query(f'metric == "{metric}"')
                metric_results_df = pd.concat([metric_results_df, filtered_df], ignore_index=True)
            else:
                raise ValueError(f"Invalid metric specified: {metric}. Choose from {cnsts.valid_metrics}.")

        min_mape_idx = metric_results_df.groupby('unique_id')['metric_value'].idxmin()
        return metric_results_df.loc[min_mape_idx].reset_index(drop=True)


    def best_results_summary(
            self,
            metric: str = "mape",
            result: str = "forecast"
        ):

        self.best_model_metric_evaluation(result=result)

        results_df = pd.DataFrame()

        for horizon, transformation in itertools.product(
            self.horizons, self.transformations
            ):

            _, df_train, _ = utils.process_data(
                transformation=transformation,
                normalization=self.normalization
            )
        
            n_windows = utils.get_n_windows(
                data_frame=df_train,
                horizon=horizon
            )
            
            _, _, fore_metric_results_path, cross_metric_results_path = (
                utils.create_output_folder_structure(horizon, transformation, n_windows)
            )

            if result == "forecast":
                metric_results_path = fore_metric_results_path

            elif result == "crossval":
                metric_results_path = cross_metric_results_path
            else:
                raise ValueError(f"Invalid result specified, got {result}.")

            m_metric = utils.calculate_mean_metrics(metric_results_path, metric)

            metric_column_name = f'mean_{metric}'

            data = {
                'transformation': transformation,
                'horizon (days)': horizon,
                metric_column_name: [m_metric]
            }
            
            new_df = pd.DataFrame(data)
            results_df = pd.concat([results_df, new_df], ignore_index=True)

        # from results_df, for each horizon saves only the best value of the metrics       
        min_mape_idx = results_df.groupby('horizon (days)')[metric_column_name].idxmin()
        return results_df.loc[min_mape_idx].sort_values('horizon (days)', ascending=False).reset_index(drop=True)


    # it returns the best results of the training, whether they are crossvalidation or forecast results. one can choose which product and client display and which metric use in order to establish the "best" results
    def best_results_prediction_dataframe(
            self,
            horizon: int = gsp.HORIZONS[0],
            metric: str = "mape",
            result: str ="forecast", 
            countries=None
        ):

        if countries is None:
            countries = utils.get_id()
        else:
            pass

        ids = utils.id_control(countries)

        metric_best_results_df = self.best_results_metric_dataframe(
            horizon=horizon,
            metric=metric,
            result=result
        )

        best_results = metric_best_results_df.set_index('unique_id').drop(columns=['metric', 'metric_value']).to_dict('index')
        
        if result == 'forecast':

            forecast_results_df = pd.DataFrame()

            for transformation in self.transformations:

                _, df_train, _ = utils.process_data(
                    transformation=transformation,
                    normalization=self.normalization
                )
            
                n_windows = utils.get_n_windows(
                    data_frame=df_train,
                    horizon=horizon
                )

                file_path, _, _, _ = (
                    utils.create_output_folder_structure(horizon, transformation, n_windows)
                )

                forecast_df = pd.read_csv(file_path)   
                forecast_df['transformation'] = transformation

                forecast_results_df = pd.concat([forecast_results_df, forecast_df], ignore_index=True)

            filtered_data = []

            for unique_id in best_results.keys():
                filtered_rows = forecast_results_df[
                    (forecast_results_df['unique_id'] == unique_id)
                    & (forecast_results_df['transformation'] == best_results[unique_id]['transformation'])
                ][['ds', best_results[unique_id]['best_model']]].copy()

                filtered_rows.rename(columns={best_results[unique_id]['best_model']: 'forecast'}, inplace=True)
                filtered_rows['unique_id'] = unique_id
                filtered_data.append(filtered_rows)

            filtered_df = pd.concat(filtered_data, ignore_index=True)
            filtered_df = filtered_df.iloc[:, [2, 0, 1]]

        elif result == 'crossval':

            crossval_results_df = pd.DataFrame()

            for transformation in self.transformations:

                _, df_train, _ = utils.process_data(
                    transformation=transformation,
                    normalization=self.normalization
                )
            
                n_windows = utils.get_n_windows(
                    data_frame=df_train,
                    horizon=horizon
                )
                
                _, file_path, _, _ = (
                    utils.create_output_folder_structure(horizon, transformation, n_windows)
                )

                crossval_df = pd.read_csv(file_path)   
                crossval_df['transformation'] = transformation

                crossval_results_df = pd.concat([crossval_results_df, crossval_df], ignore_index=True)

            filtered_data = []

            for unique_id in best_results.keys():
                filtered_rows = crossval_results_df[
                    (crossval_results_df['unique_id'] == unique_id)
                    & (crossval_results_df['transformation'] == best_results[unique_id]['transformation'])
                    ][['ds', 'cutoff', 'y', best_results[unique_id]['best_model']]].copy()

                filtered_rows.rename(columns={best_results[unique_id]['best_model']: 'prediction'}, inplace=True)
                filtered_rows['unique_id'] = unique_id
                filtered_data.append(filtered_rows)

            filtered_df = pd.concat(filtered_data, ignore_index=True)
            filtered_df = filtered_df.rename(columns={'y': 'volatility'})
            filtered_df = filtered_df.iloc[:, [4, 0, 1, 2, 3]]

        return filtered_df[filtered_df['unique_id'].isin(ids)].reset_index(drop=True)


    def _train_loop(
            self,
            horizon,
            transformation,
            model
        ) -> Tuple[pd.DataFrame, Path, Path, StatsForecast, Unknown]:

        _, df_train, _ = utils.process_data(
            transformation=transformation,
            normalization=self.normalization
        )

        # cross validation
        n_windows = utils.get_n_windows(
            data_frame=df_train,
            horizon=horizon
        )

        fit_results_path, cv_results_path, _, _ = utils.create_output_folder_structure(
            horizon, transformation, n_windows
        )

        sf = StatsForecast(
            models=[model],
            n_jobs=-1,
            freq="B",
            fallback_model=SeasonalNaive(season_length=1, alias=f"SeasNaive_sl1_{model.alias}"),
            verbose=True
        )

        model_name = model.alias

        train_info = [
            horizon,
            transformation,
            model_name
        ]

        print(f"Currently running: {train_info}")

        return df_train, fit_results_path, cv_results_path, sf, model_name    


    def _start_date_forecast(
            self,
            df: pd.DataFrame, 
            eval_horizon: int=cnsts.EVAL_HORIZONS[0], 
            date_column: str='ds'
        ):

        df[date_column] = pd.to_datetime(df[date_column])
        days_offset = eval_horizon

        return df[date_column].iloc[0] - timedelta(days=days_offset)


    def _end_date(self, df: pd.DataFrame, date_column: str='ds'):

        df[date_column] = pd.to_datetime(df[date_column])

        return df[date_column].max()


    def _start_date_crossval(
            self,
            df: pd.DataFrame, 
            eval_horizon: int=cnsts.EVAL_HORIZONS[0], 
            date_column: str='ds'
        ):

        df[date_column] = pd.to_datetime(df[date_column])
        days_offset = eval_horizon*15

        return df[date_column].max() - timedelta(days=days_offset)


    def _apply_plot_style(self, style_dict):
        plt.style.use(style_dict['style'])
        plt.rcParams.update(style_dict['rcParams'])

    # @TODO questo metodo funziona solo se ho fatto sia prediction che crossvalidation, e non va bene. è così per i prediction intervals, che in produzione non posso calcolare solo con il dataframe di forecast
    def _plot_forecast(
            self,
            df_forecast: pd.DataFrame, 
            df_crossval: pd.DataFrame,
            eval_horizon: int=cnsts.EVAL_HORIZONS[0],
            actual: bool=False
        ):

        self._apply_plot_style(cnsts.DARK_STYLE)

        df_actual, df_train, _ = utils.process_data(
            normalization=self.normalization
        )

        df_actual = df_actual.assign(ds=lambda x: pd.to_datetime(x["ds"]))
        df_train = df_train.assign(ds=lambda x: pd.to_datetime(x["ds"]))
        df_forecast = df_forecast.assign(ds=lambda x: pd.to_datetime(x["ds"]))

        data_list = []

        for unique_id in df_forecast["unique_id"].unique():

            min_date = df_crossval.query(f"unique_id == '{unique_id}'")["ds"].min()
            max_date = df_crossval.query(f"unique_id == '{unique_id}'")["ds"].max()
            min_date_actual = df_actual.query(f"unique_id == '{unique_id}'")["ds"].min()
            # min_date_actual = pd.to_datetime("2024-01-01")

            data_list.append(
                {
                    "title": f"{unique_id}", 
                    "actual_x": df_actual[(df_actual["unique_id"] == unique_id) & (df_actual["ds"] >= min_date_actual)]['ds'],
                    "actual_y": df_actual[(df_actual["unique_id"] == unique_id) & (df_actual["ds"] >= min_date_actual)]['y'],
                    "forecast_x": df_forecast.query(f"unique_id == '{unique_id}'")['ds'],
                    "forecast_y": df_forecast.query(f"unique_id == '{unique_id}'")['forecast'],
                    "crossval_y": df_crossval.query(f"unique_id == '{unique_id}'")['prediction'],
                    "actual_res_y": df_train[(df_train["unique_id"] == unique_id) & (df_train["ds"] >= min_date) & (df_train["ds"] <= max_date)]['y']
                },
            )

        n_plots = len(data_list)
        
        if n_plots == 1:
            n_cols = 1
        else:
            n_cols = 2

        n_rows = math.ceil(n_plots / n_cols)  

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols, 7 * n_rows), squeeze=False)
        fig.suptitle("Forecast Volatility Plot", fontsize=55)

        # Flatten the axes array for easier iteration
        axes = axes.flatten() 

        for idx, data in enumerate(data_list):

            low_limit = mean_pinball_loss(
                y_true=data['actual_res_y'],
                y_pred=data['crossval_y'],
                alpha=cnsts.LOW_QUANTILE
            )

            up_limit = mean_pinball_loss(
                y_true=data['actual_res_y'],
                y_pred=data['crossval_y'],
                alpha=cnsts.UP_QUANTILE
            )

            ax = axes[idx]
            
            if actual:
                ax.plot(data['actual_x'], data['actual_y'], label="actual", color='#1f77b4')
                ax.plot(data['forecast_x'], data['forecast_y'], label="forecast", color='#e8320e')
                ax.set_xlim(self._start_date_forecast(df_forecast, eval_horizon=eval_horizon), self._end_date(df_forecast))
                ax.fill_between(
                    data['forecast_x'], 
                    data['forecast_y']-low_limit, 
                    data['forecast_y']+up_limit, 
                    color='#e8320e', 
                    alpha=.25, 
                    label = '95% interval'
                )

            else:
                ax.plot(data['forecast_x'], data['forecast_y'], label="forecast", color='#1f77b4')
                ax.fill_between(
                    data['forecast_x'], 
                    data['forecast_y']-low_limit, 
                    data['forecast_y']+up_limit, 
                    color='#1f77b4', 
                    alpha=.25, 
                    label = '95% interval'
                )

            ax.legend()
            ax.set_title(data['title'])
            ax.tick_params(axis='x', rotation=45)
            ax.yaxis.set_major_formatter(ticker.EngFormatter())
            # ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

        # Hide any extra subplots if there are unused spaces in the grid
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # type: ignore # Adjust layout to make room for the main title

        plt.show()


    def _plot_cross_validation(
            self,
            df_crossval: pd.DataFrame, 
            eval_horizon: int=cnsts.EVAL_HORIZONS[0],
            actual: bool=False
        ):

        self._apply_plot_style(cnsts.DARK_STYLE)

        _, df_train, _ = utils.process_data(
            normalization=self.normalization
        )

        df_train = df_train.assign(ds=lambda x: pd.to_datetime(x["ds"]))
        df_crossval = df_crossval.assign(ds=lambda x: pd.to_datetime(x["ds"]))

        data_list = []

        for unique_id in df_crossval["unique_id"].unique():

            min_date = df_crossval.query(f"unique_id == '{unique_id}'")["ds"].min()
            max_date = df_crossval.query(f"unique_id == '{unique_id}'")["ds"].max()

            data_list.append(
                {
                    "title": f"{unique_id}", 
                    "actual_x": df_train.query(f"unique_id == '{unique_id}'")['ds'],
                    "actual_y": df_train.query(f"unique_id == '{unique_id}'")['y'],
                    "crossval_x": df_crossval.query(f"unique_id == '{unique_id}'")['ds'],
                    "crossval_y": df_crossval.query(f"unique_id == '{unique_id}'")['prediction'],
                    "actual_res_y": df_train[(df_train["unique_id"] == unique_id) & (df_train["ds"] >= min_date) & (df_train["ds"] <= max_date)]['y']
                },
            )

        n_plots = len(data_list)
        
        if n_plots == 1:
            n_cols = 1
        else:
            n_cols = 2

        n_rows = math.ceil(n_plots / n_cols)  

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols, 7 * n_rows), squeeze=False)
        fig.suptitle("CrossValidation Volatility Plot", fontsize=55)

        # Flatten the axes array for easier iteration
        axes = axes.flatten() 

        for idx, data in enumerate(data_list):

            low_limit = mean_pinball_loss(
                y_true=data['actual_res_y'],
                y_pred=data['crossval_y'],
                alpha=cnsts.LOW_QUANTILE
            )

            up_limit = mean_pinball_loss(
                y_true=data['actual_res_y'],
                y_pred=data['crossval_y'],
                alpha=cnsts.UP_QUANTILE
            )

            ax = axes[idx]
            
            if actual:
                ax.plot(data['actual_x'], data['actual_y'], label="actual", color='#1f77b4')
                ax.plot(data['crossval_x'], data['crossval_y'], label="crossvalidation", color='#e8320e')
                ax.fill_between(
                    data['crossval_x'], 
                    data['crossval_y']-low_limit, 
                    data['crossval_y']+up_limit, 
                    color='#e8320e', 
                    alpha=.25, 
                    label = '95% interval'
                )
                
            else:
                ax.plot(data['crossval_x'], data['crossval_y'], label="forecast", color='#1f77b4')
                ax.fill_between(
                    data['crossval_x'], 
                    data['crossval_y']-low_limit, 
                    data['crossval_y']+up_limit, 
                    color='#1f77b4', 
                    alpha=.25, 
                    label = '95% interval'
                )

            ax.set_xlim(self._start_date_crossval(df_crossval, eval_horizon=eval_horizon), self._end_date(df_crossval))
            ax.legend()
            ax.set_title(data['title'])
            # ax.tick_params(axis='x', rotation=45)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.yaxis.set_major_formatter(ticker.EngFormatter())
            ax.xaxis.set_major_locator(MaxNLocator(nbins=9))

        # Hide any extra subplots if there are unused spaces in the grid
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # type: ignore # Adjust layout to make room for the main title

        plt.show()
    

    def _metrics_plot(
            self,
            countries=None,
            horizon: int = gsp.HORIZONS[0],
        ):

        if countries is None:
            countries = utils.get_id()
        else:
            pass   

        ids = utils.id_control(countries)

        plt.style.use('default')  
        plt.rcdefaults()

        selected_metrics = ['mae', 'rmse']

        _, df_train, _ = utils.process_data(
            normalization=self.normalization
        )

        # create one new dataframe for each metrc
        for metric in selected_metrics:
            
            df_best_results = self.best_results_metric_dataframe(
                horizon=horizon,
                metric=metric,
            )

            df_best_results = df_best_results[df_best_results['unique_id'].isin(ids)]

            df_metric = (
                df_best_results
                .merge(
                    df_train
                    .assign(max_y=lambda x: x.groupby('unique_id')["y"].transform('max'))
                    .assign(mean_norm_y=lambda x: x["y"] / x["max_y"])
                    .groupby('unique_id')
                    .mean(numeric_only=True)
                    .reset_index()
                    .rename(columns={'y' : "mean_y"})
                )
                .assign(norm_metric_value=lambda x: x["metric_value"]/x["max_y"])
            )

            plt.figure(figsize=(10, 6))

            # plot for every model
            for best_model in df_metric.sort_values('best_model')['best_model'].unique():

                subset = df_metric[df_metric['best_model'] == best_model]
                plt.bar(
                    subset['unique_id'], 
                    subset['mean_norm_y'], 
                    yerr=subset['norm_metric_value'], 
                    color=cnsts.COLORS[best_model], 
                    label=best_model,
                    capsize=5
                )

            plt.ylim((0,1))
            plt.xlabel('countries')
            plt.ylabel(f'mean normalized volatility')
            plt.title(f'metric plot: "{metric}" - horizon = {horizon}', fontsize=15)
            plt.xticks(rotation=45) 
            plt.legend(title='best model', loc='upper left')
            plt.grid(axis='y', alpha=0.25)
            plt.show()


        plt.figure(figsize=(10, 6))

        df_best_results_mape = self.best_results_metric_dataframe(
            horizon=horizon,
            metric="smape",
        )

        df_best_results_mape = df_best_results_mape[df_best_results_mape['unique_id'].isin(ids)]

        for best_model in df_best_results_mape.sort_values('best_model')['best_model'].unique():

            subset = df_best_results_mape[df_best_results_mape['best_model'] == best_model]
            plt.bar(
                subset.query('metric == "smape"')['unique_id'], 
                subset.query('metric == "smape"')['metric_value'], 
                color=cnsts.COLORS[best_model], 
                label=best_model
            )

        plt.ylim((0,110))
        plt.xlabel('countries')
        plt.title(f'metric plot: "smape" - horizon = {horizon}', fontsize=15)
        plt.legend(title='best model', loc='upper left')
        plt.grid(axis='y', alpha=0.25)
        plt.xticks(rotation=45) 
        plt.show()
