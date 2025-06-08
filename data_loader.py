# data_loader.py
import pandas as pd  
import torch  
from torch_geometric.data import Data  
from torch_geometric.utils import degree  


###########################################################################################################

# Function to load data from a file and return it as a DataFrame
def load_data(file_path):
    triplets_df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])  # Read a tab-separated file with no header and name columns
    return triplets_df  # Return the DataFrame containing triplets (head, relation, tail)

###########################################################################################################

def convert_to_indices(df, entity_to_idx, relation_to_idx):
    df['head'] = df['head'].map(entity_to_idx)  # Map head entities to their corresponding indices
    df['tail'] = df['tail'].map(entity_to_idx)  # Map tail entities to their corresponding indices
    df['relation'] = df['relation'].map(relation_to_idx)  # Map relations to their corresponding indices
    df = df.dropna()  # Drop rows with NaNs (in case mapping fails)
    edge_index = torch.tensor(df[['head', 'tail']].values.T, dtype=torch.long)  # Create a tensor for edge indices (head and tail pairs)
    edge_attr = torch.tensor(df['relation'].values, dtype=torch.long)  # Create a tensor for edge attributes (relations)
    return edge_index, edge_attr  # Return the edge index tensor and edge attribute tensor

###########################################################################################################

# Function to filter and retain only the top-k highest-degree neighbors for each node
def filter_top_k_neighbors(edge_index, k=10):
    row, col = edge_index  # Split the edge_index tensor into source (row) and destination (col) nodes
    num_nodes = max(row.max().item(), col.max().item()) + 1  # Calculate the total number of nodes in the graph
    deg = degree(col, num_nodes=num_nodes)  # Compute the degree of each node

    filtered_edges = []  # Initialize a list to store the filtered edges
    for node in range(num_nodes):  # Iterate over each node in the graph
        neighbors = col[row == node]  # Find all neighbors of the current node
        if len(neighbors) <= k:  # If the number of neighbors is less than or equal to k, keep all
            filtered_edges.extend([(node, nbr.item()) for nbr in neighbors])  # Add all neighbors as edges
        else:  # If the number of neighbors is greater than k
            neighbor_degrees = deg[neighbors]  # Get the degree of each neighbor
            top_neighbors = neighbors[torch.topk(neighbor_degrees, k).indices]  # Select the top-k neighbors by degree
            filtered_edges.extend([(node, nbr.item()) for nbr in top_neighbors])  # Add the top-k neighbors as edges

    return torch.tensor(filtered_edges, dtype=torch.long).T  # Return the filtered edge list as a tensor (transposed)

###########################################################################################################

