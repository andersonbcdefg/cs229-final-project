# File: util.py
# Description: Utilities for CS229 final project.
import pandas as pd
import numpy as np

def load_dataset(file_path, nrows=None):
	head = pd.read_csv(file_path, nrows=1)
	cols = head.columns
	cols_to_read = [c for c in cols if c not in ["Unnamed: 0", "content", "account_category"]]
	df = pd.read_csv(file_path, usecols=cols_to_read, nrows=nrows).dropna()
	y = df["troll"].to_numpy()
	X = df.drop(["troll"], axis=1).to_numpy()
	print(f"Loaded X with shape {X.shape}, y with shape {y.shape}.")
	return X, y


def create_learning_curve(model_type, model_params, 
		X_train, y_train, X_val, y_val, train_sizes):
	results = {
		"n": [],
		"model": [],
		"train": [],
		"val": []
	}
	for n in train_sizes:
		print(f"Training model with {n} training examples...")
		model = model_type(**model_params)
		model.fit(X_train[:n, :], y_train[:n])
		train_preds = model.predict(X_train[:n, :])
		train_accuracy = np.mean(train_preds == y_train[:n])
		val_preds = model.predict(X_val)
		val_accuracy = np.mean(val_preds == y_val)
		results["n"].append(n)
		results["model"].append(model)
		results["train"].append(train_accuracy)
		results["val"].append(val_accuracy)
	return results