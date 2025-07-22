import networkx as nx
import matplotlib.pyplot as plt

def visualize_neuron_network(layer_dicts):
    G = nx.DiGraph()

    layout_labels = ["Input Layer", "Hidden Layer", "Output Layer"]
    node_ids = {}
    total_index = 0

    # Add neurons from each layout as nodes, labeled by layer
    for layer_idx, layout in enumerate(layer_dicts):
        for key in layout:
            node_label = f"{key} ({layout_labels[layer_idx]})"
            G.add_node(key, label=node_label, layer=layout_labels[layer_idx])
            node_ids[key] = layout[key]
            total_index += 1

    # Add edges from connect_neurons history — simulate actual wiring
    for source_dict in layer_dicts:
        for source_key, source_neuron in source_dict.items():
            for target_dict in layer_dicts:
                for target_key, target_neuron in target_dict.items():
                    for nt in source_neuron.neurotransmitters:
                        src_array = getattr(source_neuron, f"{nt}_o", None)
                        tgt_array = getattr(target_neuron, f"{nt}_i", None)
                        if src_array is not None and tgt_array is not None:
                            if src_array.size > 0 and tgt_array.size > 0:
                                # Visual connection exists
                                G.add_edge(source_key, target_key, label=nt)

    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(14, 10))
    node_labels = nx.get_node_attributes(G, "label")
    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw(G, pos, with_labels=True, labels=node_labels,
            node_size=2000, node_color="lightblue", edge_color="gray",
            font_size=10, font_weight="bold", arrowsize=15)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_color="red", font_size=9)
    plt.title("🧠 Neuron Network Architecture (Biologically Inspired)", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()






layer_dicts = [first_layout, second_layout, third_layout] # layers are dicts
visualize_neuron_network(layer_dicts)
