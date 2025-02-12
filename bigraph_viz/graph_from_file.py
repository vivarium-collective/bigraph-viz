import os
from graphviz import Source


def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, relative_path)


def generate_graph_from_file(file_path, output_file=None):
    # Read the content of the file
    with open(file_path, 'r') as file:
        graph_data = file.read()

    # Create a Source object from the Graphviz data
    graph = Source(graph_data)

    # Render the graph to a file (default format is PDF, so we explicitly specify PNG)
    graph.render(filename=output_file, format='png', cleanup=True)


def test_graph_data():
    file_path = get_absolute_path('../data/gut_microbiome')
    output_file = 'out/gut_microbiome'
    generate_graph_from_file(file_path=file_path, output_file=output_file)


if __name__ == '__main__':
    test_graph_data()
