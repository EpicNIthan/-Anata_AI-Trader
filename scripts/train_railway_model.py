from __future__ import annotations

import sys

from train_best_model import main


if __name__ == "__main__":
    if "--model-types" not in sys.argv:
        sys.argv.extend(["--model-types", "sklearn_hist_gradient_boosting,random_forest"])
    main()
