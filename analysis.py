import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import autumn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def plot_rank(df_):
    metrics = ["f1_rank", "accuracy_rank"]

    final_rank = df_.groupby("encoding")[metrics].mean()
    final_rank.columns = ["F1", "Acc"]
    final_rank.index.name = "Encoding"

    plt.figure()
    sns.heatmap(final_rank, annot=True, cmap="YlGnBu")
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/ranking.pdf")
    plt.close()


def plot_importances(dic):
    plt.figure()
    sns.barplot(x=list(dic.keys())[:9], y=list(dic.values())[:9], color="r")
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/bottom10.pdf")
    plt.close()

    plt.figure()
    # plt.xticks(rotation=90)
    sns.barplot(x=list(dic.values())[-9:], y=list(dic.keys())[-9:], color="g")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/top10.pdf")
    plt.close()


def plot_classification(df):
    sns.set_theme({"legend.frameon": True}, style="whitegrid", palette="pastel")
    plt.figure(figsize=(4, 3))
    sns.violinplot(
        x="Method", y="Performance", hue="Metric", data=df, palette="Reds"
    )
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/classification_perf.pdf")
    plt.close()


def plot_enc_anom(df_enc_anom):
    df_enc_anom.drop(["log", "accuracy", "time"], axis=1, inplace=True)
    df_enc_anom.columns = ["Anomaly", "Encoding", "F-score"]

    sns.set_theme(
        {
            "legend.frameon": True,
            "legend.title_fontsize": "small",
            "legend.fontsize": "x-small",
            "legend.borderpad": 0.2,
            "legend.labelspacing": 0.2,
            "patch.linewidth": 1.5,
        },
        style="whitegrid",
        palette="pastel",
    )
    plt.figure(figsize=(5, 3))
    sns.boxplot(
        x="Anomaly",
        y="F-score",
        hue="Encoding",
        data=df_enc_anom,
        palette=["m", "g", "orange"],
    )

    plt.tight_layout()
    # sns.despine()
    # plt.show()
    plt.savefig("plots/enc_vs_anom.pdf")
    plt.close()


def plot_positions(df):
    sns.set_theme(style="whitegrid", palette="pastel")
    metrics = ["f1_rank"]
    encodings = ["alignment", "doc2vec", "node2vec"]

    for metric in metrics:
        fig, axs = plt.subplots(ncols=3, figsize=(8, 3))
        fig.subplots_adjust(wspace=0.7)
        for id, encoding in enumerate(encodings):
            sns.histplot(
                df[df.encoding == encoding].loc[:, metric],
                legend=encoding,
                color=autumn(id / 2),
                ax=axs[id],
                binwidth=0.3,
            )
            axs[id].set(ylim=(0, 90), ylabel="Frequency", xlabel="Position")
        fig.legend(encodings)
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/positions.pdf")
    plt.close()


def plot_positions2(df):
    metrics = ["f1_rank"]
    encodings = ["alignment", "doc2vec", "node2vec"]

    out = []
    for encoding in encodings:
        for pos in range(1, 4):
            out.append(
                [
                    encoding,
                    pos,
                    len(df[(df["encoding"] == encoding) & (df["f1_rank"] == pos)]),
                ]
            )
    df_plot = pd.DataFrame(out, columns=["Encoding", "Position", "Frequency"])

    sns.set_theme(
        {
            "legend.frameon": True,
            "legend.title_fontsize": "small",
            "legend.fontsize": "x-small",
            "legend.borderpad": 0.2,
            "legend.labelspacing": 0.2,
            "patch.linewidth": 1.5,
        },
        style="whitegrid",
        palette="pastel",
    )
    plt.figure(figsize=(4, 3))

    ax = sns.barplot(
        x="Position",
        y="Frequency",
        hue="Encoding",
        data=df_plot,
        # palette=["m", "g", "orange"],
        palette="Blues"
    )

    c_ = (0.2980392156862745, 0.2980392156862745, 0.2980392156862745, 1.0)
    [patch.set_edgecolor(c_) for patch in ax.patches]

    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/positions2.pdf")
    plt.close()


def add_anomaly_info(df):
    df.insert(1, "anomaly", "all")
    df.loc[df["log"].str.contains("attribute"), "anomaly"] = "attribute"
    df.loc[df["log"].str.contains("early"), "anomaly"] = "early"
    df.loc[df["log"].str.contains("insert"), "anomaly"] = "insert"
    df.loc[df["log"].str.contains("late"), "anomaly"] = "late"
    df.loc[df["log"].str.contains("rework"), "anomaly"] = "rework"
    df.loc[df["log"].str.contains("skip"), "anomaly"] = "skip"

    return df

random.seed(542)

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("classification.csv")
df = add_anomaly_info(df)

plot_enc_anom(df.copy())

df["f1_rank"] = df.groupby("log")["f1"].rank(method="min", ascending=False)
df["accuracy_rank"] = df.groupby("log")["accuracy"].rank(method="min", ascending=False)

plot_rank(df)
plot_positions(df)
plot_positions2(df)

df_stats = pd.read_csv("log_meta_features.csv")
df_stats = df_stats.iloc[:, :81] # we do not use entropy meta-features in this work


best = []
for log in df_stats["log"]:
    encs = list(df[(df["log"] == log) & (df["f1_rank"] == 1)].encoding)
    if len(encs) == 1:
        best.append(encs[0])
    else:
        best.append("-")

df_stats.insert(1, "encoding", best)
df_stats = df_stats[df_stats["encoding"] != "-"].copy()
y = df_stats["encoding"]
df_stats.drop(["log", "encoding"], axis=1, inplace=True)

opts = ["alignment", "doc2vec", "node2vec"]
importances = []
accs, f1s = [], []
maj_accs, maj_f1s = [], []
rand_accs, rand_f1s = [], []
for step in range(30):
    X_train, X_test, y_train, y_test = train_test_split(
        df_stats, y, test_size=0.2, random_state=step
    )

    rf = RandomForestClassifier(random_state=step, n_jobs=-1).fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Meta-model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    accs.append(accuracy)
    f1s.append(f1)

    # Majority
    majority_pred = ["node2vec"] * len(y_pred)
    maj_accs.append(accuracy_score(y_test, majority_pred))
    maj_f1s.append(f1_score(y_test, majority_pred, average="macro"))

    # Random
    random_pred = [random.choice(opts) for x in range(len(y_pred))]
    rand_accs.append(accuracy_score(y_test, random_pred))
    rand_f1s.append(f1_score(y_test, random_pred, average="macro"))

    importances.append(rf.feature_importances_)

feature_importances = dict(zip(df_stats.columns, np.mean(importances, axis=0)))
feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1])}
plot_importances(feature_importances)

print("Meta-model")
print("Acc:", np.mean(accs))
print("F1:", np.mean(f1s))

print("Majority")
print("Acc:", np.mean(maj_accs))
print("F1:", np.mean(maj_f1s))

print("Random")
print("Acc:", np.mean(rand_accs))
print("F1:", np.mean(rand_f1s))


# plot
length = len(accs)

method = ["Meta-model"] * length * 2
method.extend(["Majority"] * length * 2)
method.extend(["Random"] * length * 2)

metric = ["Accuracy"] * length
metric.extend(["F-score"] * length)
metric *= 3

perf = accs.copy()
perf.extend(f1s)
perf.extend(maj_accs)
perf.extend(maj_f1s)
perf.extend(rand_accs)
perf.extend(rand_f1s)

df_perf = pd.DataFrame(
    zip(method, metric, perf), columns=["Method", "Metric", "Performance"]
)
plot_classification(df_perf)
