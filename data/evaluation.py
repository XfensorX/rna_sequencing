import os
import sklearn as sk
from sklearn import metrics
import numpy as np
import torch
import neptune

# Ggf. zu ändern:
name = "Luan" #Name
model_bezeichnung = "Random Forest, n_estimators=4" #Verwendetes Modell und Params, z.B.: FFN, act_func = relu, batch_size = 16, usw...
y_pred = y_pred 
y_test = y_test #Shape: (9120, 105)
#Ab hier keine Veränderung im Code mehr notwendig

run = neptune.init_project(
    project="JPL/rna-sequencing",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YWM2N2QyNC0yMTFhLTRlNDEtOWZmZi0wNGVhZDViMmY1YmYifQ=="  # Dein Neptune API-Token
)

def compute_metrics(y_pred, y_test):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='micro')
    recall = metrics.recall_score(y_test, y_pred, average='micro')
    auc = metrics.roc_auc_score(y_test, y_pred, average='macro', multi_class='ovr')

    return accuracy, precision, recall, auc

def convert_inputs(y_p):
    if isinstance(y_p, torch.Tensor):
        # Tensor auf CPU schieben
        if y_p.is_cuda:
            y_p = y_p.cpu()
        y_p_array = y_p.numpy()
        return y_p_array
    else:
        return y_p

y_pred_array = convert_inputs(y_pred)
accuracy, precision, recall, auc = compute_metrics(y_pred_array, y_test)

run[f"{name}/{model_bezeichnung}/metrics/accuracy"] = accuracy
run[f"{name}/{model_bezeichnung}/metrics/precision"] = precision
run[f"{name}/{model_bezeichnung}/metrics/recall"] = recall
run[f"{name}/{model_bezeichnung}/metrics/auc"] = auc

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"AUC: {auc}")

run.stop()
