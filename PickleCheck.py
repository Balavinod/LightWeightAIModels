class MyClass:  # ✅ top-level
    pass


import pickle
import pandas as pd
import numpy as np
import os

with open("scaler.pkl", "rb") as f:
    obj = pickle.load(f)


