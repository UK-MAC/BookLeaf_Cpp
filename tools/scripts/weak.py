#!/usr/bin/python3

import sys
from math import sqrt



yaml1 = """TIME:
    time_start: 0.0
    time_end: 20.0
    dt_initial: 1.0e-3
    dt_max: 1.0e-1
    dt_min: 1.0e-8
    dt_g: 1.02

HYDRO:
    cfl_sf: 0.5
    cvisc1: 0.5
    cvisc2: 0.75

EOS:
    - { type: ideal gas, params: [1.4] }

MESH:
    - type: LIN1"""

yaml2 = """      sides:"""

yaml3 = """
ALE:
    zeul: true

INDICATORS:
    regions:
        - { type: background, name: air }
        - { type: shape, name: helium, value: 0 }
    materials:
        - { type: background, name: shocktube }

INITIAL_CONDITIONS:
    thermodynamics:
        - { type: region, value: 0, density: 1.0, energy: 2.5 }
        - { type: region, value: 1, density: 0.125, energy: 2.0 }

SHAPES:"""



# nprocs
N = int(sys.argv[1])

# Mesh size
elperpe = (15000.0 * 3750.0) / (32.0 * 4.0)
meshprod = elperpe * (4.0 * N)
meshh = sqrt(meshprod / 4.0)
meshw = 4.0 * meshh
meshh = int(meshh + 0.5)
meshw = int(meshw + 0.5)

# Domain size
domx = 200. / (15000.0 / meshw)
domy = 50. / (3750.0 / meshh)

# Rect corners
x1 = domx / 2.0
y1 = -10.0
x2 = domx * 1.5
y2 = domy + 10.0



# Print deck
print(yaml1)

print("      dims: [{0}, {1}]".format(meshw, meshh))

print(yaml2)

tmp = "          - - {{ type: LINE, bc: SLIPY, pos: [0.0, 0.0, {0}, 0.0] }}"
print(tmp.format(domx))
tmp = "          - - {{ type: LINE, bc: SLIPX, pos: [{0}, 0.0, {0}, {1}] }}"
print(tmp.format(domx, domy))
tmp = "          - - {{ type: LINE, bc: SLIPY, pos: [{0}, {1}, 0.0, {1}] }}"
print(tmp.format(domx, domy))
tmp = "          - - {{ type: LINE, bc: SLIPX, pos: [0.0, {0}, 0.0, 0.0] }}"
print(tmp.format(domy))

print(yaml3)

tmp = "    - {{ type: rectangle, params: [{0}, {1}, {2}, {3}] }}"
print(tmp.format(x1, y1, x2, y2))
