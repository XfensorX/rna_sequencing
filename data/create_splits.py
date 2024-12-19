import os
import random
import sys

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append("../")

from data.original import pertdata as pt  # noqa: E402

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1


##### Outputs -> pandas, numpy, pytorch, csv |||| .pd, .np, .pt, .csv


def get_pandas_data(norman_dataset):
    X = pd.DataFrame(
        norman.adata.X.toarray(),
        columns=norman.adata.var["gene_name"],
        index=norman.adata.obs.index,
    )

    conditions = pt.generate_fixed_perturbation_labels(norman.adata.obs["condition"])
    single_conditions = conditions.str.split("+")

    mlb = MultiLabelBinarizer()
    y = pd.DataFrame(
        mlb.fit_transform(single_conditions),
        columns=pd.Series(mlb.classes_, name="gene_perturbation"),
        index=conditions.index,
    ).drop("ctrl", axis=1)

    return X, y


def random_split(seed: int, **datas):
    N = len(list(datas.values())[0])

    indices = list(range(N))

    random.seed(seed)
    random.shuffle(indices)

    val_size = int(VALIDATION_SIZE * N)
    test_size = int(TEST_SIZE * N)

    splitted_datas = {}

    for name, data in datas.items():
        assert len(data) == N

        validation = data.iloc[indices[:val_size]]
        test = data.iloc[indices[val_size : val_size + test_size]]
        training = data.iloc[indices[val_size + test_size :]]

        splitted_datas[name] = {"train": training, "val": validation, "test": test}

    return splitted_datas


if __name__ == "__main__":
    print("Creating Data Splits")

    norman = pt.PertData.from_repo(name="norman", save_dir="original")
    X, y = get_pandas_data(norman)
    splitted_datas = random_split(seed=42, X=X, y=y)
    for name, datasets in splitted_datas.items():
        for stage_name, data in datasets.items():
            filepath = f"./splits/{stage_name}/{name}_pandas.pck"
            directory = os.path.dirname(filepath)
            os.makedirs(directory, exist_ok=True)
            data.to_pickle(filepath)
