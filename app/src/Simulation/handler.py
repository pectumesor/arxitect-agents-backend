from .constants import BUFFER_SIZE

"""
    This file contains all the necessary methods to extract all the relevant data from the 
    hospital layouts .json files
"""
def find_boundaries(graph_data):
    # Calculate graph center and offset positions to center in the box
    node_positions = [(node["value"]["x"], node["value"]["y"]) for node in graph_data["nodes"]]
    # Extract x and y coordinates
    x_coords = [pos[0] for pos in node_positions]
    y_coords = [pos[1] for pos in node_positions]
    # Compute min and max for both axes
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    return node_positions,min_x, min_y, max_x, max_y

def  calculate_render_box_size(min_x, min_y, max_x, max_y):
    width = max_x - min_x
    height = max_y - min_y

    return (width, height), (min_x, min_y, max_x, max_y)

def compute_box_and_boundaries(graph_data):
    node_positions,min_x, min_y, max_x, max_y = find_boundaries(graph_data)
    box_size, bounding_box = calculate_render_box_size(min_x, min_y, max_x, max_y)
    return graph_data,node_positions,box_size, bounding_box

