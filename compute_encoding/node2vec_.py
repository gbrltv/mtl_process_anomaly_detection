import os
import time
import pandas as pd
from tqdm import tqdm
import networkx as nx
from karateclub.node_embedding.neighbourhood import Node2Vec
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces
from utils import convert_traces_mapping
from utils import create_graph
from utils import trace_feature_vector_from_nodes
import warnings
warnings.filterwarnings('ignore')


def save_results(vector, dimension, ids, time, y, path):
    out_df = pd.DataFrame(vector, columns=[f'feature_{i}' for i in range(dimension)])
    out_df['case'] = ids
    out_df['time'] = time
    out_df['label'] = y
    out_df.to_csv(path, index=False)


dimension = 32
path = './event_logs'
save_path = './encoding_results/node2vec'
os.makedirs(save_path, exist_ok=True)

for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # create graph
    log = read_log(path, file)
    graph = create_graph(log)
    mapping = dict(zip(graph.nodes(), [i for i in range(len(graph.nodes()))]))
    graph = nx.relabel_nodes(graph, mapping)

    # read event log, import case id and labels and transform activities names
    ids, traces_raw, y = retrieve_traces(log)
    traces = convert_traces_mapping(traces_raw, mapping)

    start_time = time.time()
    # generate model
    model = Node2Vec(dimensions=dimension)
    model.fit(graph)
    training_time = time.time() - start_time

    # calculating the average and max feature vector for each trace
    start_time = time.time()
    node_average, _ = trace_feature_vector_from_nodes(model.get_embedding(), traces, dimension)
    node_time = training_time + (time.time() - start_time)

    # saving
    fname = file.split(".xes")[0]
    save_results(node_average, dimension, ids, node_time, y, f'{save_path}/{fname}.csv')
