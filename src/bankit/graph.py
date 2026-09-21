"""Distance matrices and the municipality adjacency graph used by the GNN."""

import numpy as np
import pandas as pd
import torch

from .data import create_final


def filter_distances(distance_matrix, threshold):
    filtered_matrix = np.where(distance_matrix < threshold, distance_matrix, 0)
    np.fill_diagonal(filtered_matrix, 0)
    return filtered_matrix


def inverse_distance(distance_matrix, power=1):
    mask = distance_matrix != 0
    result = np.zeros_like(distance_matrix, dtype=float)
    result[mask] = 1 / (distance_matrix[mask] ** power)
    return result


def top_distances(distance_matrix, top_k):
    filtered_matrix = np.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        row = distance_matrix[i, :]
        if np.count_nonzero(row) > top_k:
            top_indices = np.argsort(row)[:top_k+1]
            filtered_matrix[i, top_indices] = row[top_indices]
        else:
            filtered_matrix[i, :] = row
    return np.maximum(filtered_matrix, filtered_matrix.T)


def haversine_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return r * c


def _match_istat_to_bdap(gi_comuni_df, final):
    """For each `codice_istat`, the last-occurring BDAP index in `final` with
    a matching `Codice_ISTAT_Comune` (mirrors `final[...].index[-1]`), plus a
    print for any ISTAT code with more than one match.
    """
    last_bdap_by_istat = {}
    matches_by_istat = {}
    for bdap, codice in final['Codice_ISTAT_Comune'].items():
        last_bdap_by_istat[codice] = bdap
        matches_by_istat.setdefault(codice, []).append(bdap)

    for codice_istat in gi_comuni_df['codice_istat']:
        matches = matches_by_istat.get(codice_istat, [])
        if len(matches) > 1:
            print(f"Multiple matches found for ISTAT code {codice_istat}: {matches}")

    return gi_comuni_df['codice_istat'].map(last_bdap_by_istat)


def prepare_graph(comuni, distance=30):

    final = create_final(comuni)

    gi_comuni_df = pd.read_csv('gi_comuni.csv', sep=';', decimal=',', encoding='latin_1', low_memory=False)
    gi_comuni_df['lat'] = gi_comuni_df['lat'].astype(float)
    gi_comuni_df['lon'] = gi_comuni_df['lon'].astype(float)

    gi_comuni_df['BDAP'] = _match_istat_to_bdap(gi_comuni_df, final)
    gi_comuni_df = gi_comuni_df.dropna(subset=['BDAP'])
    gi_comuni_df = gi_comuni_df.set_index('BDAP')
    gi_comuni_df.sort_index(inplace=True)
    gi_comuni_df.index = gi_comuni_df.index.astype(int)

    lat = gi_comuni_df['lat'].to_numpy()
    lon = gi_comuni_df['lon'].to_numpy()
    distance_matrix = haversine_distance(lat[:, np.newaxis], lon[:, np.newaxis], lat[np.newaxis, :], lon[np.newaxis, :])

    graph = inverse_distance(filter_distances(distance_matrix, distance))

    adj = torch.tensor(graph)

    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    edge_weight = graph[edge_index[0], edge_index[1]]
    edge_weight = torch.tensor(edge_weight).unsqueeze(-1).float()

    del distance_matrix
    del graph

    return adj, edge_weight
