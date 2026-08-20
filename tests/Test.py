import sys
sys.path.append("source")   # adjust if your folder is named "src"

import yaml
from preprocess import load_and_preprocess

def test_preprocess_output():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    X, y = load_and_preprocess(config)

    assert X.isnull().sum().sum() == 0          # no missing values
    assert set(y.unique()) == {0, 1}             # target is binary
    assert X.shape[0] == y.shape[0]              # rows match