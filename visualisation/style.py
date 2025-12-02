# Set global styles for visualisations, change rcParams (runtime configuration). 

import matplotlib as mpl 

def set_default_style():
    mpl.rcParams["figure.figsize"] = (8,5)
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.grid"] = True

    