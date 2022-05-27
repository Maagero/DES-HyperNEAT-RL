import copy
from matplotlib.pyplot import connect
import numpy as np
from torch import true_divide
from hyperneat import get_cppn_bias, query_cppn, query_cppn_bias
from visualize_cppn import draw_es

import activations
import feed_forward
import recurrent


class DESNetwork:

    def __init__(self,input_substrates, output_substrates, genome, params, config):
        self.input_substrates = genome.input_substrates
        self.output_substrates = genome.output_substrates
        #Add the cords to the substrates
        if config.genome_config.num_inputs == 4:
            self.cppn_bias = False
        else:
            self.cppn_bias = True

        self.ann_bias = config.layout_config.enable_cppn_bias

        self.input_coordinates = []
        self.input_coordinates_by_substrate = {}
        for i, substrate in enumerate(self.input_substrates.values()):
            substrate.coordinates = input_substrates[i]
            self.input_coordinates += substrate.coordinates
            self.input_coordinates_by_substrate[substrate.key] = substrate.coordinates
        
        self.output_coordinates = []
        self.output_coordinates_by_substrate = {}
        for i, substrate in enumerate(self.output_substrates.values()):
            substrate.coordinates = output_substrates[i]
            self.output_coordinates += substrate.coordinates
            self.output_coordinates_by_substrate[substrate.key] = substrate.coordinates



        self.substrates = genome.substrates
        self.paths = genome.paths

        # DES ANN bias
        self.enable_des_bias = config.layout_config.enable_des_bias
        if self.enable_des_bias:
            self.bias_cppn = feed_forward.FeedForwardNetwork.create(genome.bias_cppn, config)

        #DES LEO 
        self.enable_leo = config.layout_config.enable_leo

        self.initial_depth = params["initial_depth"]
        self.max_depth = params["max_depth"]
        self.variance_threshold = params["variance_threshold"]
        self.band_threshold = params["band_threshold"]
        self.iteration_level = params["iteration_level"]
        self.division_threshold = params["division_threshold"]
        self.max_weight = params["max_weight"]
        self.connections = set()
        self.config = config
        # Number of layers in the network.
        self.activations = 2 ** params["max_depth"]*len(list(genome.substrates.keys())) + 1 #TODO check this out maybe multiply by num of substrates
        activation_functions = activations.ActivationFunctionSet()
        self.activation = activation_functions.get(params["activation"])

    def create_phenotype_network(self, filename=None):

        input_nodes = list(range(len(self.input_coordinates)))
        output_nodes = list(range(len(input_nodes), len(
            input_nodes)+len(self.output_coordinates)))
        hidden_idx = len(self.input_coordinates)+len(self.output_coordinates)


        #ADD input and output with substrate to coords:
        coords_in = set()
        for sub_key, coordinates in self.input_coordinates_by_substrate.items():
            for coord in coordinates:
                coords_in.add((sub_key, (coord)))
        
        coords_out = set()
        for sub_key, coordinates in self.output_coordinates_by_substrate.items():
            for coord in coordinates:
                coords_out.add((sub_key, (coord)))


        coordinates, indices, draw_connections, node_evals = [], [], [], []
        nodes = {}

        coordinates.extend(coords_in)
        coordinates.extend(coords_out)
        indices.extend(input_nodes)
        indices.extend(output_nodes)

        #Unpack coordinates
        coords_to_id = dict(zip(coordinates, indices))
        
        hidden_nodes, connections = self.des_hyperneat()
        # Map hidden coordinates to their IDs.
        for substrate_key, node_set in hidden_nodes.items():
            for node in node_set:
                coords_to_id[substrate_key, node] = hidden_idx
                hidden_idx += 1
        # For every coordinate:
        # Check the connections and create a node with corresponding connections if appropriate.
        for key, idx in coords_to_id.items():
            substrate_key, (x,y) = key
            for element_key, cons in connections.items():
                if type(element_key) == tuple and element_key[1] != substrate_key:
                    continue
                elif element_key == substrate_key:
                    continue
                for c in cons:
                    if c.x2 == x and c.y2 == y:
                        draw_connections.append(c)
                        inc_node = element_key
                        if type(element_key) == tuple:
                            inc_node = element_key[0]
                        if idx in nodes:
                            initial, bias = nodes[idx]
                            initial.append((coords_to_id[inc_node, (c.x1, c.y1)], c.weight))
                            nodes[idx] = (initial, bias)
                        else:
                            nodes[idx] = ([(coords_to_id[inc_node, (c.x1, c.y1)], c.weight)], c.bias)
        # Combine the indices with the connections/links;
        # forming node_evals used by the RecurrentNetwork.

        #With CPPN layout bias


        for idx, links_bias in nodes.items():
            node_evals.append((idx, self.activation, sum, links_bias[1], 1.0, links_bias[0]))


        # Visualize the network?
        if filename is not None:
            print(node_evals)
            #draw_es(coords_to_id, draw_connections, filename)
        
        # This is actually a feedforward network.
        return recurrent.RecurrentNetwork(input_nodes, output_nodes, node_evals)

    @staticmethod
    def get_weights(p):
        """
        Recursively collect all weights for a given QuadPoint.
        """
        temp = []

        def loop(pp):
            if pp is not None and all(child is not None for child in pp.cs):
                for i in range(0, 4):
                    loop(pp.cs[i])
            else:
                if pp is not None:
                    temp.append(pp.w)
        loop(p)
        return temp

    def variance(self, p):
        """
        Find the variance of a given QuadPoint.
        """
        if not p:
            return 0.0
        return np.var(self.get_weights(p))

    def division_initialization(self, coord, outgoing, cppn, depth):
        """
        Initialize the quadtree by dividing it in appropriate quads.
        """
        root = QuadPoint(0.0, 0.0, 1.0, 1)
        q = [root]

        cppn = feed_forward.FeedForwardNetwork.create(cppn, self.config)

        while q:
            p = q.pop(0)

            p.cs[0] = QuadPoint(p.x - p.width/2.0, p.y -
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[1] = QuadPoint(p.x - p.width/2.0, p.y +
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[2] = QuadPoint(p.x + p.width/2.0, p.y +
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[3] = QuadPoint(p.x + p.width/2.0, p.y -
                                p.width/2.0, p.width/2.0, p.lvl + 1)

            for c in p.cs:
                c.w = query_cppn(coord, (c.x, c.y), outgoing,
                                 cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo)
                if self.ann_bias:
                    c.b = query_cppn_bias((0.0, 0.0), (c.x, c.y), outgoing,
                                    cppn, bias = self.cppn_bias)
                elif self.enable_des_bias:
                    c.b = get_cppn_bias(coord, self.bias_cppn, bias=self.cppn_bias)
                else:
                    c.b = 0
            if (p.lvl < self.initial_depth) or ( p.lvl < self.max_depth and self.variance(p)
                                                > self.division_threshold):
                for child in p.cs:
                    q.append(child)
        return root


    def pruning_extraction(self, coord, p, outgoing, cppn):
        """
        Determines which connections to express - high variance = more connetions.
        """
        if type(cppn) != feed_forward.FeedForwardNetwork:
            cppn = feed_forward.FeedForwardNetwork.create(cppn, self.config)
        for c in p.cs:
            d_left, d_right, d_top, d_bottom = None, None, None, None

            if self.variance(c) > self.variance_threshold:
                self.pruning_extraction(coord, c, outgoing, cppn)
            else:
                d_left = abs(c.w - query_cppn(coord, (c.x - p.width,
                                                      c.y), outgoing, cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo))
                d_right = abs(c.w - query_cppn(coord, (c.x + p.width,
                                                       c.y), outgoing, cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo))
                d_top = abs(c.w - query_cppn(coord, (c.x, c.y - p.width),
                                             outgoing, cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo))
                d_bottom = abs(c.w - query_cppn(coord, (c.x, c.y +
                                                        p.width), outgoing, cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo))

                con = None
                if max(min(d_top, d_bottom), min(d_left, d_right)) > self.band_threshold:

                    if outgoing:
                        con = Connection(coord[0], coord[1], c.x, c.y, c.w, c.b)
                    else:
                        con = Connection(c.x, c.y, coord[0], coord[1], c.w, c.b)
                if con is not None:
                    # Nodes will only connect upwards.
                    # If connections to same layer is wanted, change to con.y1 <= con.y2.
                    if not c.w == 0.0 and not (con.x1 == con.x2 and con.y1 == con.y2):
                        self.connections.add(con)

    def des_hyperneat(self):
        # Hidden nodes are (substrate_key, (nodes)) same with unexplored
        hidden_nodes, unexplored_hidden_nodes = {}, {}
        connections = {}
        #Sort the substrates
        self.substrates = self.sort_substrates()
        #Start the search from the input substrates
        for key, substrate in self.input_substrates.items():
            inputs = substrate.coordinates
            for path in self.paths.values():
                if path.key[0] == key:
                    connections[path.key] = set()
                    for x, y in inputs:  # Explore from inputs.
                        root = self.division_initialization((x, y), True, path.cppn, 1)
                        self.pruning_extraction((x, y), root, True, path.cppn)
                        connections[path.key] = connections[path.key].union(self.connections)
                        if path.key[1] not in hidden_nodes:
                            hidden_nodes[path.key[1]] = set()
                        for c in connections[path.key]:
                            hidden_nodes[path.key[1]].add((c.x2, c.y2))
                        self.connections = set()

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)
        
        #Search for the hidden substrates
        for substrate in self.substrates.values():
            connections[substrate.key] = set()
            #First do a local search in this substrate
            for _ in range(substrate.depth):  # Explore from hidden.
                if substrate.key in unexplored_hidden_nodes:
                    inputs = copy.deepcopy(unexplored_hidden_nodes[substrate.key])
                    for x, y in inputs:
                        root = self.division_initialization((x, y), True, substrate.cppn, substrate.depth)
                        self.pruning_extraction((x, y), root, True, substrate.cppn)
                        connections[substrate.key] = connections[substrate.key].union(self.connections)
                        for c in connections[substrate.key]:
                            hidden_nodes[substrate.key].add((c.x2, c.y2))
                        self.connections = set()
                        
                    unexplored_hidden_nodes[substrate.key] = hidden_nodes[substrate.key] - unexplored_hidden_nodes[substrate.key]
            if substrate.key in hidden_nodes:
                inputs = copy.deepcopy(hidden_nodes[substrate.key])
                # Then search to other substrates from the current
                for path in self.paths.values():
                    if path.key[0] == substrate.key:
                        connections[path.key] = set()
                        for x, y in inputs:  # Explore from inputs.
                            root = self.division_initialization((x, y), True, path.cppn, 1)
                            self.pruning_extraction((x, y), root, True, path.cppn)
                            connections[path.key] = connections[path.key].union(self.connections)
                            if path.key[1] not in hidden_nodes:
                                hidden_nodes[path.key[1]] = set()
                            for c in connections[path.key]:
                                hidden_nodes[path.key[1]].add((c.x2, c.y2))
                            self.connections = set()
                            
                #If hidden nodes in new substrates are found they will be added to be searched for later
                unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)
        #Seartch from the outputs
        for substrate in self.output_substrates.values():
            for path in self.paths.values():
                if substrate.key == path.key[1]:
                    connections[path.key[0]] = set()
            inputs = substrate.coordinates
            for path in self.paths.values():
                if path.key[1] == substrate.key:
                    connections[path.key] = set()
                    for x, y in inputs:
                        root = self.division_initialization((x, y), False, path.cppn, 1)
                        self.pruning_extraction((x, y), root, False, path.cppn)
                        connections[path.key] = connections[path.key].union(self.connections)
                        self.connections = set()
        return self.clean_net(connections)


    def sort_substrates(self):
        '''
        Sorting substrates w
        '''
        sorted_substrate = {}
        no_in_path_substrates = {}

        edge_paths = []
        
        #Remove the paths from inputsubs
        for path in self.paths.keys():
            if path[0] not in self.input_substrates.keys():
                edge_paths.append(path)
        #Find substrates without incoming paths
        for substrate in self.substrates.values():
            in_path = False
            for path_key in edge_paths:
                if substrate.key == path_key[1]:
                    in_path = True
                    break
            if not in_path:
                no_in_path_substrates[substrate.key] = substrate
        while no_in_path_substrates:
            substrate = no_in_path_substrates.pop(list(no_in_path_substrates.keys())[0])
            sorted_substrate[substrate.key] = substrate

            for path in copy.deepcopy(edge_paths):
                if path[0]==substrate.key:
                    edge_paths.remove(path)
                    if path[1] not in self.output_substrates.keys():
                        new_sub = self.substrates[path[1]]
                        in_path = False
                        for path_key in edge_paths:
                            if new_sub.key == path_key[1]:
                                in_path = True
                                break
                        if not in_path:
                            no_in_path_substrates[new_sub.key] = new_sub

        return sorted_substrate


    def merge_dict(self, dict1, dict2):
        new_dict = copy.deepcopy(dict1)
        for key, value in dict2.items():
            if key in new_dict:
                new_dict[key] = new_dict[key].union(value)
            else:
                new_dict[key] = value
        return new_dict

    def clean_net(self, connections):
        """
        Clean a net for dangling connections:
        Intersects paths from input nodes with paths to output.
        """
        connected_to_inputs = {}
        for sub_key, coordinates in self.input_coordinates_by_substrate.items():
            connected_to_inputs[sub_key] = set()
            for path in self.paths.values():
                if path.key[0] == sub_key:
                    for coord in coordinates:
                        connected_to_inputs[sub_key].add(coord)
        
        connected_to_outputs = {}
        for sub_key, coordinates in self.output_coordinates_by_substrate.items():
            connected_to_outputs[sub_key] = set()
            for path in self.paths.values():
                if path.key[1] == sub_key:
                    for coord in coordinates:
                        connected_to_outputs[sub_key].add(coord)

        true_connections = {}
        initial_input_connections = copy.deepcopy(connections)
        initial_output_connections = copy.deepcopy(connections)

        add_happened = True
        while add_happened:  # The path from inputs.
            add_happened = False
            for element_key in connections.keys():
                temp_input_connections = copy.deepcopy(initial_input_connections[element_key])
                for c in temp_input_connections:
                    if type(element_key) == tuple:
                        if element_key[0] in connected_to_inputs and (c.x1, c.y1) in connected_to_inputs[element_key[0]]:
                            if element_key[1] not in connected_to_inputs:
                                connected_to_inputs[element_key[1]] = set()
                            connected_to_inputs[element_key[1]].add((c.x2, c.y2))
                            initial_input_connections[element_key].remove(c)
                            add_happened = True
                    else:
                        if element_key in connected_to_inputs and (c.x1, c.y1) in connected_to_inputs[element_key]:
                            connected_to_inputs[element_key].add((c.x2, c.y2))
                            initial_input_connections[element_key].remove(c)
                            add_happened = True

        add_happened = True
        while add_happened:  # The path to outputs.
            add_happened = False
            for element_key in connections.keys():
                temp_output_connections = copy.deepcopy(initial_output_connections[element_key])
                for c in temp_output_connections:
                    if type(element_key) == tuple:
                        if element_key[1] in connected_to_outputs and (c.x2, c.y2) in connected_to_outputs[element_key[1]]:
                            if element_key[0] not in connected_to_outputs:
                                connected_to_outputs[element_key[0]] = set()
                            connected_to_outputs[element_key[0]].add((c.x1, c.y1))
                            initial_output_connections[element_key].remove(c)
                            add_happened = True
                    else:
                        if element_key in connected_to_outputs and (c.x2, c.y2) in connected_to_outputs[element_key]:
                            connected_to_outputs[element_key].add((c.x1, c.y1))
                            initial_output_connections[element_key].remove(c)
                            add_happened = True

        #True node is intersection of the connected to input and connected to output
        true_nodes = {}
        for element_key in connected_to_inputs.keys():
            if element_key in connected_to_outputs:
                true_nodes[element_key] = connected_to_inputs[element_key].intersection(connected_to_outputs[element_key])

        for element_key in connections.keys():
            for c in connections[element_key]:
                # Only include connection if both source and target node resides in the real path from input to output
                if type(element_key)==tuple:
                    if element_key[0] in true_nodes and element_key[1] in true_nodes:
                        if (c.x1, c.y1) in true_nodes[element_key[0]] and (c.x2, c.y2) in true_nodes[element_key[1]]:
                            if element_key not in true_connections:
                                true_connections[element_key] = set()
                            true_connections[element_key].add(c)
                else:
                    if element_key in true_nodes and (c.x1, c.y1) in true_nodes[element_key] and (c.x2, c.y2) in true_nodes[element_key]:
                        if element_key not in true_connections:
                            true_connections[element_key] = set()
                        true_connections[element_key].add(c)



        #Remove input and output nodes from the hidden node list
        for sub_key, coordinates in self.input_coordinates_by_substrate.items():
            connected_to_inputs[sub_key] = set()
            for path in self.paths.values():
                if path.key[0] == sub_key:
                    for coord in coordinates:
                        if sub_key in true_nodes and coord in true_nodes[sub_key]:
                            true_nodes[sub_key].remove(coord)
        
        for sub_key, coordinates in self.output_coordinates_by_substrate.items():
            connected_to_outputs[sub_key] = set()
            for path in self.paths.values():
                if path.key[1] == sub_key:
                    for coord in coordinates:
                        if sub_key in true_nodes and coord in true_nodes[sub_key]:
                            true_nodes[sub_key].remove(coord)
        return true_nodes, true_connections


class QuadPoint:
    """
    Class representing an area in the quadtree.
    Defined by a center coordinate and the distance to the edges of the area.
    """

    def __init__(self, x, y, width, lvl):
        self.x = x
        self.y = y
        self.w = 0.0
        self.width = width
        self.cs = [None] * 4
        self.lvl = lvl


class Connection:
    """
    Class representing a connection from one point to another with a certain weight.
    """

    def __init__(self, x1, y1, x2, y2, weight, bias = 0.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight
        self.bias = bias

    # Below is needed for use in set.
    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.x1, self.y1, self.x2, self.y2) == (other.x1, other.y1, other.x2, other.y2)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2, self.weight))

    def __str__(self) -> str:
        return str(self.x1) + ', ' + str(self.y1) + ' to ' + str(self.x2) + ', ' + str(self.y2)

def find_pattern(cppn, coord, res=60, max_weight=5.0):
    """
    From a given point, query the cppn for weights to all other points.
    This can be visualized as a connectivity pattern.
    """
    im = np.zeros((res, res))

    for x2 in range(res):
        for y2 in range(res):

            x2_scaled = -1.0 + (x2/float(res))*2.0
            y2_scaled = -1.0 + (y2/float(res))*2.0

            i = [coord[0], coord[1], x2_scaled, y2_scaled, 1.0]
            n = cppn.activate(i)[0]

            im[x2][y2] = n * max_weight

    return im