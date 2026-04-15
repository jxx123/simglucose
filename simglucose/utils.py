import importlib.resources
import pandas as pd
import os

# Compatibility helper for Python < 3.9
def _get_resource_path(package, resource):
    if hasattr(importlib.resources, 'files'):
        return str(importlib.resources.files(package) / resource)
    else:
        # Fallback for Python 3.8
        with importlib.resources.path(package, '__init__.py') as p:
            return str(p.parent / resource)

CONTROL_QUEST = _get_resource_path("simglucose", "params/Quest.csv")
PATIENT_PARA_FILE = _get_resource_path("simglucose", "params/vpatient_params.csv")


def fetch_patient_params(patient_name: str):
    all_params = pd.read_csv(PATIENT_PARA_FILE)
    patient_params = lookup_patient_meta_data(all_params, patient_name)
    return patient_params


def fetch_patient_quest(patient_name: str):
    all_quests = pd.read_csv(CONTROL_QUEST)
    quest = lookup_patient_meta_data(all_quests, patient_name)
    return quest


def lookup_patient_meta_data(df: pd.DataFrame, patient_name: str) -> dict:
    idx = df['Name'] == patient_name
    params = {}
    if idx.any():
        params = df[idx].iloc[0].to_dict()
    return params
