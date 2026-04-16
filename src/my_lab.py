
from src.statsforecastlab import StatsForecastLab
from src import constants as cnsts 
from tqdm import TqdmExperimentalWarning

import os
import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def main():

    os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

    sflab = StatsForecastLab(
        freq="h",
        normalization=True,
        # test=True
    )

    sflab.create_folder_structure()
    sflab.cross_validation()
    sflab.predict()

    print(sflab.best_results_summary())
    

if __name__ == "__main__":
    main()