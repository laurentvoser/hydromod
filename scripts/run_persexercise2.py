import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scripts.persex2_snow_model as persex2_snow_model
import scripts.persex2_sensitivity as persex2_sensitivity

if __name__ == "__main__":
    persex2_snow_model.run()
    persex2_sensitivity.run()
