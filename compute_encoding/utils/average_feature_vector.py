import numpy as np


def average_feature_vector_doc2vec(model, traces):
    """
    Retrieves the document feature vector for doc2vec

    Parameters
    -----------------------
    model,
        Text-based model containing the computed encodings
    traces: List,
        List of traces treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """
    vectors = []
    for trace in traces:
        vectors.append(model.infer_vector(trace))

    return vectors


def trace_feature_vector_from_nodes(embeddings, traces, dimension):
    """
    Computes average feature vector for each trace

    Parameters
    -----------------------
    embeddings,
        Text-based model containing the computed encodings
    traces: List,
        List of traces treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """
    vectors_average, vectors_max = [], []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(embeddings[token])
            except KeyError:
                pass
        if len(trace_vector) == 0:
            trace_vector.append(np.zeros(dimension))
        vectors_average.append(np.array(trace_vector).mean(axis=0))
        vectors_max.append(np.array(trace_vector).max(axis=0))

    return vectors_average, vectors_max
