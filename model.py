# model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


###########################################################################################################


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=16):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_nodes, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_relations)
        self.dropout = torch.nn.Dropout(0.5)
        """
        :param x: Tensor of node features (num_nodes, in_channels)
        :param edge_index: Tensor of shape (2, num_edges), edge list
        :return: Final node embeddings (num_nodes, out_channels)
        """
    def forward(self, data):
       
        x, edge_index = data.x, data.edge_index  # Extract node features (x) and edge connections (edge_index) from the data object
        x = F.relu(self.conv1(x, edge_index))    # Pass the node features through the first GCN layer and apply ReLU activation
        x = self.dropout(x)                      # Apply dropout for regularization
        x = F.relu(self.conv2(x, edge_index))    # Pass the result through the second GCN layer with ReLU activation
       
        x = self.dropout(x)                      # Apply dropout again
        x = self.conv3(x, edge_index)            # Pass the result through the third GCN layer without activation
 
        tail_scores = self.fc(x)                 # Apply the fully connected layer to produce scores for tail entity prediction
        return tail_scores                       # Return the scores for tail entities

'''
class Informed_Graph_Neural_Network(torch.nn.Module):
    
    def __init__(self, num_nodes, num_relations, hidden_dim=16):    # Constructor for initializing the model's layers
        super(Informed_Graph_Neural_Network, self).__init__()       # Call the parent class constructor
        self.conv1 = GCNConv(num_nodes, hidden_dim)                 # First GCN (Graph Convolutional Network) layer: transforms input features to hidden_dim size
        self.conv2 = GCNConv(hidden_dim, hidden_dim)                # Second GCN layer: keeps the feature size as hidden_dim
        self.conv3 = GCNConv(hidden_dim, hidden_dim)                # Third GCN layer: keeps the feature size as hidden_dim
        self.fc = torch.nn.Linear(hidden_dim, num_relations)        # Fully connected layer to produce scores for each relation or entity, output size is num_relations
        self.dropout = torch.nn.Dropout(0.5)                        # Dropout layer to help prevent overfitting during training

    
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index  # Extract node features (x) and edge connections (edge_index) from the data object
        x = F.relu(self.conv1(x, edge_index))    # Pass the node features through the first GCN layer and apply ReLU activation
        x = self.dropout(x)                      # Apply dropout for regularization
        x = F.relu(self.conv2(x, edge_index))    # Pass the result through the second GCN layer with ReLU activation
        
        x = self.dropout(x)                      # Apply dropout again
        x = self.conv3(x, edge_index)            # Pass the result through the third GCN layer without activation

        tail_scores = self.fc(x)                 # Apply the fully connected layer to produce scores for tail entity prediction
        return tail_scores                       # Return the scores for tail entities

###########################################################################################################
'''