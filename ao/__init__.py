from .abs_ao import AbsAO
from .nasnet import NASNET
from .randomsearch import RandomSearch

AO = ["reinforce", "randomsearch"]


def ao_selector(ao_name):
    fn = None
    
    if ao_name == "reinforce":
        fn = nasnet.NASNET
    elif ao_name == "random":
        fn = randomsearch.RandomSearch
    else:
        print("Error AO name not found")
        
    return fn