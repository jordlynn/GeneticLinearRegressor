import matplotlib
import numpy
import pandas
import random
import sklearn
import matplotlib.pyplot as plt

TARGET_NAME = "target_kazutsugi"

class numeraiData():

    def __init__(self):
        print("Loading data...")
        self.df = pandas.read_csv("data/numerai_training_data.csv").set_index("id")
        self.td = pandas.read_csv("data/numerai_tournament_data.csv").set_index("id")

        self.TrainingDataFeatureNames = [f for f in self.df.columns if f.startswith("feature")]
        self.TestDataFeatureNames = [f for f in self.td.columns if f.startswith("feature")]

        self.target_name = TARGET_NAME
        print(f"Loaded {len(self.TrainingDataFeatureNames)} features from training data")
        print(f"Loaded {len(self.TestDataFeatureNames)} features from test data")

        self.feature_groups = {
            g: [c for c in self.df if c.startswith(f"feature_{g}")]
            for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]
        }

