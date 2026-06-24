import matplotlib.pyplot as plt
import networkx as nx

def generate_flowchart(output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    G = nx.DiGraph()
    
    # Define nodes
    nodes = {
        'Datasets': 'LIMUC & TMC\nDatasets',
        'Preprocessing': 'Image\nPreprocessing',
        'Feature_Extraction': 'Feature Extraction\n(GLCM & DWT)',
        'Super_Agent': 'Super Agent\n(Fusion & Analysis)',
        'Medical_Report': 'Medical Report\nVisualization'
    }
    
    # Add nodes
    for node_id, node_label in nodes.items():
        G.add_node(node_id, label=node_label)
        
    # Define edges
    edges = [
        ('Datasets', 'Preprocessing'),
        ('Preprocessing', 'Feature_Extraction'),
        ('Feature_Extraction', 'Super_Agent'),
        ('Super_Agent', 'Medical_Report')
    ]
    G.add_edges_from(edges)
    
    # Set node positions
    pos = {
        'Datasets': (0, 0),
        'Preprocessing': (1, 0),
        'Feature_Extraction': (2, 0),
        'Super_Agent': (3, 0),
        'Medical_Report': (4, 0)
    }
    
    # Draw parameters
    node_size = 5000
    node_color = '#e0f7fa'
    edge_color = '#006064'
    font_size = 11
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_shape='s', 
                           node_color=node_color, edgecolors=edge_color, linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, 
                           arrows=True, arrowsize=20, width=2,
                           connectionstyle="arc3,rad=0.0")
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=font_size, font_weight='bold')
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    
    plt.title('Texture Analysis Architecture Pipeline', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Flowchart saved to {output_path}")

if __name__ == "__main__":
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'architecture_flowchart.png')
    generate_flowchart(output_file)
