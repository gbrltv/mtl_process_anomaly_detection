import os
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from meta_feature_extraction import sort_files


def read_data(path: str, file: str):
    """
    Reads file containing the encodings for a given event log

    Parameters
    -----------------------
    path: str,
        File path
    log: str,
        File name
    Returns
    -----------------------
    The encoding vectors and their corresponding labels
    """
    df = pd.read_csv(f"{path}/{file}")
    y = list(df["label"])
    df.drop(["case", "time", "label"], axis=1, inplace=True)
    vectors = df.to_numpy()

    return vectors, y


def compute_metrics(y_true, y_pred, binary=True):
    """
    Computes classification metrics

    Parameters
    -----------------------
    y_true,
        List of true instance labels
    y_pred,
        List of predicted instance labels
    binary,
        Controls the computation of binary only metrics
    Returns
    -----------------------
    Classification metrics
    """
    accuracy = accuracy_score(y_true, y_pred)

    if binary:
        f1 = f1_score(y_true, y_pred, pos_label="normal")
    else:
        f1 = f1_score(y_true, y_pred, average="macro")

    return [accuracy, f1]


encodings = ["alignment", "doc2vec", "node2vec"]
files = sort_files(os.listdir("encoding_results/alignment"))

out = []
for file in tqdm(files, total=len(files), desc="Calculating classification metrics"):
    for encoding in encodings:
        path = f"encoding_results/{encoding}"
        X, y = read_data(path, file)

        metrics_list = []
        for step in range(30):
            start_time = time.time()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=step
            )

            rf = RandomForestClassifier(random_state=step, n_jobs=-1).fit(
                X_train, y_train
            )
            y_pred = rf.predict(X_test)

            if "scenario" in file and "all" not in file:
                metrics_ = compute_metrics(y_test, y_pred)
            else:
                metrics_ = compute_metrics(y_test, y_pred, False)

            end_time = time.time() - start_time

            metrics_.extend([end_time])
            metrics_list.append(metrics_)

        metrics_array = np.array(metrics_list)
        metrics_array = list(np.mean(metrics_array, axis=0))
        file_encoding = [file.split(".csv")[0], encoding]
        file_encoding.extend(metrics_array)

        out.append(file_encoding)


columns = ["log", "encoding", "accuracy", "f1", "time"]
pd.DataFrame(out, columns=columns).to_csv("classification.csv", index=False)
