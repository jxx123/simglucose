import pandas as pd


class DataWarehouse:
    def __init__(self):
        pass

    def put(self, idx, records):
        raise NotImplementedError

    def getAll(self):
        raise NotImplementedError


class InMemoryDataWarehouse(DataWarehouse):
    def __init__(self, column_names, index_name):
        # NOTE: need to define the schema first, so the data is consistent all
        # the time
        self._data = pd.DataFrame(columns=column_names)
        self._data.index.name = index_name

    def put(self, idx, **records):
        self._data.loc[idx, list(records.keys())] = list(records.values())
        # if idx in self._data.keys():
        #     self._data[idx].update(records)
        # else:
        #     self._data[idx] = records

    def getAll(self):
        return self._data
        # return pd.DataFrame.from_dict(self._data, orient='index').sort_index()

    def get(self, key):
        return self._data[key]
