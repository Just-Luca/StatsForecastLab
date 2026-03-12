
# from src.hup_neuralmacrocast import process_data
from statsforecast.models import (
    
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive,
    Naive,
    ARIMA,
    AutoETS,
    AutoARIMA,
    GARCH
)

HORIZONS = [
    72,
    48,
    24,
]

# @TODO: lb_days and horizons should be calculated automatically
LB_DAYS = 1* 12 * 20 * 24   # (years * months * days * hours)
# LB_DAYS = 2 * 12 * 20   # (years * months * days)

TRANSFORMATIONS = [
    # "log", 
    # "root3", 
    # "outl+log", 
    # "outl+root3", 
    # "outl", 
    "identity"
]

MODELS = [
    # HoltWinters(), # lui non runna, fullbacka sul modello di fullback
    # Croston(),
    # SeasonalNaive(season_length=7, alias="SeasNaive_sl7"),
    # HistoricAverage(),
    # DOT(season_length=1 , alias = 'DOT_sl1'),
    # DOT(season_length=7 , alias = 'DOT_sl7'),
    Naive(),
    ARIMA((3,1,5), alias = 'ARIMA'),
    ARIMA(season_length = 24, alias = 'ARIMA_sl24'),
    AutoARIMA(seasonal=False),
    AutoARIMA(season_length = 24, alias = 'AutoARIMA_sl24'),
    AutoETS(alias = 'AutoETS'),
    AutoETS(season_length = 24, alias = 'AutoETS_sl24'),
    # GARCH(1,1),
    # GARCH(2,1),
    # GARCH(2,2)
]

# n_series = 16

# HIDDEN_SIZE = 25
# RANDOM_SEED = 42
# BATCH_SIZE = 4

# def get_model_config(lb_days, horizon):
    
#     return{
#         "input_size": lb_days,
#         "batch_size": BATCH_SIZE,
#         "h": horizon,
#         # "loss": DistributionLoss(distribution="StudentT", level=[80, 90]),
#         "loss": MAE(),
#         "learning_rate": 1e-3,
#         # "val_check_steps": 10,
#         "scaler_type": "robust",
#         "random_seed": RANDOM_SEED,
#         "deterministic": "warn",
#         "start_padding_enabled": True,
#     }