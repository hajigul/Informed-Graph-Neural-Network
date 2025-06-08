import torch  
import os  
import pandas as pd 
from torch_geometric.data import Data  # Data class for handling graph data in PyTorch Geometric
from torch_geometric.loader import DataLoader  # DataLoader class for loading graph data in batches
from data_loader import load_data, convert_to_indices, filter_top_k_neighbors  
from model import GraphSAGE  # Custom GNN model class
from train import train  # Training function
from utils import set_seed  # Utility function for reproducibility




###########################################################################################################

# Set the path for the dataset directory
#dataset_dir = 'D:/informed-GNN/data/YAGO3-10/'  # Directory containing the dataset files
#dataset_dir = 'D:/informed-GNN/data/nell_v1/' 
# Other dataset paths (commented out)
#dataset_dir = 'D:/informed-GNN/data/codex-l/'
#dataset_dir = 'D:/informed-GNN/data/codex-m/'
#dataset_dir = 'D:/informed-GNN/data/codex-s/'
#dataset_dir = 'D:/informed-GNN/data/conceptnet-100k/'
dataset_dir = 'D:/Informed-GNN-text-entity/wn18rr/'


###########################################################################################################



# Define the paths for the train, validation, and test files
train_file_path = os.path.join(dataset_dir, 'train.txt')  # Path to the training data
valid_file_path = os.path.join(dataset_dir, 'valid.txt')  # Path to the validation data
test_file_path = os.path.join(dataset_dir, 'test.txt')  # Path to the test data

# Define the directory and file paths for results
results_dir = "D:/informed-GNN/results"  # Directory for saving results
os.makedirs(results_dir, exist_ok=True)  # Create the directory if it does not exist
loss_file_path = os.path.join(results_dir, "train_val_loss.txt")  # Path for saving training and validation loss
test_results_file_path = os.path.join(results_dir, "test_results.txt")  # Path for saving test results
checkpoint_path = os.path.join(results_dir, "checkpoint.pth")  # Path for saving model checkpoints

###########################################################################################################

# Load and prepare data
train_df = load_data(train_file_path)  # Load training data as a DataFrame
valid_df = load_data(valid_file_path)  # Load validation data as a DataFrame
test_df = load_data(test_file_path)  # Load test data as a DataFrame

# Extract unique entities and relations from the data
entities = pd.concat([train_df['head'], train_df['tail'], valid_df['head'], valid_df['tail'], test_df['head'], test_df['tail']]).unique()
entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}  # Create a mapping from entity names to indices
relations = pd.concat([train_df['relation'], valid_df['relation'], test_df['relation']]).unique()
relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}  # Create a mapping from relation names to indices

# Convert dataframes to indexed graph data and filter top-k neighbors
train_edge_index, train_edge_attr = convert_to_indices(train_df, entity_to_idx, relation_to_idx)  # Convert training data
train_edge_index = filter_top_k_neighbors(train_edge_index, k=20)  # Filter top-k neighbors for training data

valid_edge_index, valid_edge_attr = convert_to_indices(valid_df, entity_to_idx, relation_to_idx)  # Convert validation data
#valid_edge_index = filter_top_k_neighbors(valid_edge_index, k=20)  # Filter top-k neighbors for validation data
test_edge_index, test_edge_attr = convert_to_indices(test_df, entity_to_idx, relation_to_idx)  # Convert test data
#test_edge_index = filter_top_k_neighbors(test_edge_index, k=20)  # Filter top-k neighbors for test data


###########################################################################################################

# Get the total number of unique nodes (entities) and relations
num_nodes, num_relations = len(entities), len(relations)
# Create node features as an identity matrix (one-hot encoding)
x = torch.eye(num_nodes, dtype=torch.float)  # Each node is represented as a one-hot vector

# Create Data objects for training, validation, and test datasets
train_data = Data(x=x, edge_index=train_edge_index, edge_attr=train_edge_attr)  # Training data object
valid_data = Data(x=x, edge_index=valid_edge_index, edge_attr=valid_edge_attr)  # Validation data object
test_data = Data(x=x, edge_index=test_edge_index, edge_attr=test_edge_attr)  # Test data object

# Create DataLoaders for batching and loading data
train_loader = DataLoader([train_data], batch_size=16, shuffle=True)  # DataLoader for training data
valid_loader = DataLoader([valid_data], batch_size=16, shuffle=False)  # DataLoader for validation data
test_loader = DataLoader([test_data], batch_size=16, shuffle=False)  # DataLoader for test data
set_seed()

# Initialize the GNN model
Informed_GNN = GraphSAGE(num_nodes, num_relations)

# Train the model with specified parameters
train(Informed_GNN, train_loader, valid_loader, test_loader, epochs=50, lr=0.005,
      checkpoint_path=checkpoint_path, loss_file_path=loss_file_path, test_results_file_path=test_results_file_path)


###########################################################################################################

