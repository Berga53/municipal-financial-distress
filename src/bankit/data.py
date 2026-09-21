"""Data-assembly helpers for the classification pipeline and the map notebook.

`create_data` (used by `classification.ipynb`) and `create_data_map` (used by
`map.ipynb`) build superficially similar sliding-window panels but take
different inputs and are kept as separate functions rather than merged.
"""

import numpy as np
import pandas as pd


def create_data(comuni, pop, spese=None, entrate=None, zone=None, anticipazioni=None, indicatori=None, input=0, target=0, periodo=1):

    years = np.arange(2009, 2016) if periodo == 1 else np.arange(2016, 2024)
    if periodo == 1:
        year_ranges = [np.arange(year, year + input) for year in np.arange(years[0], years[-1] - input + 2)]
    else:
        year_ranges = [np.arange(year, year + input) for year in np.arange(years[0], years[-1] - input - target + 3)]

    data = []
    y = []

    for year_range in year_ranges:
        data_temp = []
        for year in year_range:

            temp = pd.DataFrame(index=comuni.index)

            for elem in [spese, entrate, anticipazioni, indicatori]:
                if elem is not None:
                    temp2 = elem[f'{year}'].loc[temp.index.intersection(elem[f'{year}'].index)]
                    temp = pd.concat([temp, temp2], axis=1)

            if spese is not None:
                if pop is not None:
                    col = pd.Series(np.log(pop[f'{year}'].copy()), index=pop.index)
                    col = col.loc[temp.index.intersection(col.index)]
                    temp = pd.concat([temp, col], axis=1)
                if zone is not None:
                    zone2 = zone.loc[temp.index.intersection(zone.index)]
                    temp = pd.concat([temp, zone2], axis=1)

            data_temp.append(temp)

        temp = np.stack([elem.values for elem in data_temp])
        temp = np.swapaxes(temp, 0, 1)
        data.append(temp)

        temp_y = sum([comuni[f'Target {year_range[-1] + i + 1}'].values for i in range(target)])
        y.append((temp_y > 0).astype(int))

    X = np.concatenate(data)
    y = np.concatenate(y)

    return X, y


def create_data_map(dataframes, comuni, pop, category, features, lag, pred, mode, add=None, zone=None):
    years = sorted(set(int(col.split('_')[1]) for col in dataframes.keys()))
    year_ranges = [np.arange(year, year + lag) for year in np.arange(years[0], years[-1] - lag + 2)]

    data = []
    y = []

    for year_range in year_ranges:
        data_temp = []
        for year in year_range:
            pop_temp = pop[f'pop_{year}'].values[:, np.newaxis]
            if mode == 'pop':
                temp = pd.concat([dataframes[f'{feature}_{year}_{category}'] for feature in features], axis=1) / pop_temp
            elif mode == 'abs':
                temp = pd.concat([dataframes[f'{feature}_{year}_{category}']
                                  .div(dataframes[f'{feature}_{year}_{category}'].sum(axis=1), axis=0).fillna(0)
                                  for feature in features], axis=1)
            if add is not None:
                for elem in add:
                    if elem == 'pop':
                        col = np.log(pop[f'pop_{year}'].copy())
                        temp = pd.concat([temp, col], axis=1)
                    if elem == 'zone':
                        for zone_col in zone.columns:
                            col = zone[zone_col].copy()
                            temp = pd.concat([temp, col], axis=1)
            data_temp.append(temp)

        temp = np.stack([elem.values for elem in data_temp])
        temp = np.swapaxes(temp, 0, 1)
        data.append(temp)
        temp_y = sum([comuni[f'Target {year_range[-1] + i + 1}'].values for i in range(pred)])
        y.append((temp_y > 0).astype(int))

    X = np.concatenate(data)
    y = np.concatenate(y)
    print(X.shape)

    return X, y


def create_final(comuni):
    columns = ['Denominazione', 'Dizione_Provincia', 'Dizione_Regione', 'Dizione_zona', 'Codice_ISTAT_Comune']

    final = pd.read_csv('data//Anagrafe_comuni.csv', sep=';', encoding='latin_1', low_memory=False).set_index('Id_Ente')

    final['Data_Istituzione'] = pd.to_datetime(final['Data_Istituzione'], format='%Y-%m-%d')
    final = final[final['Data_Istituzione'].dt.year < 2008]

    final = final[final['Data_Cessazione'].isna()]

    final = final[final['Codice_Tipologia_DLGS_118_2011'] == 'ELCOMU'][columns]

    final = final.rename(columns={'Denominazione': 'Comune', 'Dizione_Provincia': 'Provincia', 'Dizione_Regione': 'Regione', 'Dizione_zona': 'Zona'})
    final.index.names = ['BDAP']

    final.loc[602642930517270801, "Codice_ISTAT_Comune"] = 21014

    final = final.loc[comuni.index.intersection(final.index)]

    return final
