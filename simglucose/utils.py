import pkg_resources
import pandas as pd

CONTROL_QUEST = pkg_resources.resource_filename('simglucose',
                                                'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


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
