#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    train_data = pd.read_csv("train_data.csv")
    val_data = pd.read_csv("validation_data.csv")
    print("Data loaded successfully.")
    return train_data, val_data

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    DAG_edges = [
        ('Start_Stop_ID', 'Distance'),
        ('Start_Stop_ID', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('Start_Stop_ID', 'Fare_Category'),
        ('Start_Stop_ID', 'End_Stop_ID'),
        ('Distance', 'Zones_Crossed'),
        ('Distance', 'Fare_Category'),
        ('Distance', 'Route_Type'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Zones_Crossed', 'Route_Type'),
        ('Fare_Category', 'Route_Type'),
        ('Distance', 'End_Stop_ID'),
        ('Zones_Crossed', 'End_Stop_ID'),
        ('Route_Type', 'End_Stop_ID'),
        ('Fare_Category', 'End_Stop_ID')
    ]
    
    model = bn.make_DAG(DAG_edges)
    
    model = bn.parameter_learning.fit(model, df)
    print("Initial Bayesian Network created and fitted successfully.")
    
    bn.plot(model, params_static={"layout": "spring", "title": "Initial Bayesian Network"})
    
    return model

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    DAG_edges = [
        ('Start_Stop_ID', 'Distance'),
        ('Start_Stop_ID', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('Start_Stop_ID', 'Fare_Category'),
        ('Start_Stop_ID', 'End_Stop_ID'),
        ('Distance', 'Zones_Crossed'),
        ('Distance', 'Fare_Category'),
        ('Distance', 'Route_Type'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Zones_Crossed', 'Route_Type'),
        ('Fare_Category', 'Route_Type'),
        ('Distance', 'End_Stop_ID'),
        ('Zones_Crossed', 'End_Stop_ID'),
        ('Route_Type', 'End_Stop_ID'),
        ('Fare_Category', 'End_Stop_ID')
    ]
    
    model = bn.make_DAG(DAG_edges)
    
    model = bn.parameter_learning.fit(model, df)
    
    initial_score = bn.structure_scores(model, df)
    print(f"Initial model structure score (BIC): {initial_score['bic']}")
    
    best_score = initial_score['bic']
    best_model = model

    for edge in DAG_edges:
        pruned_edges = [e for e in DAG_edges if e != edge]
        
        pruned_model = bn.make_DAG(pruned_edges)
        pruned_model = bn.parameter_learning.fit(pruned_model, df)
        
        pruned_score = bn.structure_scores(pruned_model, df)
        print(f"Score after pruning {edge}: {pruned_score['bic']}")

        if pruned_score['bic'] > best_score:
            best_score = pruned_score['bic']
            best_model = pruned_model
            print(f"Improved model found by removing {edge} with score {best_score}")
    
    print("Pruned Bayesian Network created and fitted successfully.")
    
    bn.plot(best_model, params_static={"layout": "spring", "title": "Pruned Bayesian Network"})

    output_filename = "pruned_network_plot.png"
    bn.save(best_model, params_static={"filename": output_filename})
    print(f"Pruned Bayesian Network plot saved as {output_filename}")
    
    return best_model

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    DAG_edges = [
        ('Distance', 'Zones_Crossed'),
        ('Distance', 'Fare_Category'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Route_Type', 'Fare_Category'),
        ('Route_Type', 'Zones_Crossed'),
        ('Route_Type', 'Distance')
    ]
    
    initial_model = bn.make_DAG(DAG_edges)
    initial_model = bn.parameter_learning.fit(initial_model, df)
    
    optimized_model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')

    optimized_score = bn.structure_scores(optimized_model, df)
    print(f"Optimized model structure score (BIC): {optimized_score['bic']}")

    print("Optimized Bayesian Network created and fitted successfully.")
    return optimized_model

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model successfully saved to {fname}.")

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df)
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df)
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df)
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

