import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ex2a_linear_storage
import ex2b_cascade
import ex2c_capacity_et

if __name__ == "__main__":
    ex2a_linear_storage.run()
    ex2b_cascade.run()
    ex2c_capacity_et.run()
