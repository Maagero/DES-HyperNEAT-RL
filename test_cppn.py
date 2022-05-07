"""
Visualizes a CPPN - remember to edit path in visualize.py, sorry.
"""

import pickle
from es_hyperneat import find_pattern
from visualize_cppn import draw_pattern

path_to_cppn = "es_hyperneat_xor_small_cppn.pkl"

# For now, path_to_cppn should match path in visualize.py, sorry.
with open(path_to_cppn, 'rb') as cppn_input:
    CPPN = pickle.load(cppn_input)
    pattern = find_pattern(CPPN, (0.0, -1.0))
    draw_pattern(pattern)
