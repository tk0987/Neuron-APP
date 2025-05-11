import networkx as nx
import matplotlib.pyplot as plt

def visualize_neuron_network(neurons):
    G = nx.DiGraph()  # Directed graph for neuron connections
    
    # Add neurons as nodes
    for i, neuron in enumerate(neurons):
        G.add_node(f"Neuron {i+1}", label=f"Neuron {i+1}")

    # Add edges based on autolinking and recurrence connections
    for i, neuron in enumerate(neurons):
        for j, target_neuron in enumerate(neurons):
            if i != j:  # Prevent self-linking
                mapping = {
                    "dopamine": (0, 0),
                    "opioid": (0, 0),
                    "acetylcholine": (0, 0),
                    "serotonin": (0, 0),
                    "glutamate": (0, 0)
                }

                # Autolink connections
                neuron.autolinker(target_neuron, mapping)
                
                # If recurrent, connect back to self (optional)
                if neuron.recurrent:
                    G.add_edge(f"Neuron {i+1}", f"Neuron {i+1}", color="red", style="dashed")

                # Basic neurotransmitter connection
                for nt in neuron.neurotransmitters:
                    if getattr(neuron, f"{nt}_o").any() > 0:  # Check for active output
                        G.add_edge(f"Neuron {i+1}", f"Neuron {j+1}", label=nt)

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Positioning of nodes
    labels = nx.get_node_attributes(G, "label")
    edge_labels = nx.get_edge_attributes(G, "label")
    edge_colors = [G[u][v].get('color', 'black') for u, v in G.edges]
    edge_styles = [G[u][v].get('style', 'solid') for u, v in G.edges]

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold", edge_color=edge_colors, style=edge_styles)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Neuron Network Visualization")
    plt.show()

# ==========================
#      Example Usage
# ==========================
neurons = [
    Neuron(acetylcholine_i=1, dopamine_o=2, acetylcholine_o=2, opioid_o=1, norepinephrine_o=1, callback_val=1.1),
    Neuron(opioid_i=1, dopamine_i=1, acetylcholine_i=1, dopamine_o=1, opioid_o=1, callback_val=1.1),
    Neuron(norepinephrine_i=1, dopamine_i=1, dopamine_o=1, acetylcholine_i=1, acetylcholine_o=1, opioid_o=1, callback_val=1.1, recurrent=True),
    Neuron(opioid_i=2, dopamine_i=2, acetylcholine_i=1, dopamine_o=1, callback_val=1.1)
]

visualize_neuron_network(neurons)
