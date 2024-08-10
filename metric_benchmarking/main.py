import glob as glob
import pandas as pd
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    files = glob.glob('vdz_stats/*.csv')
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)

    # Combine all DataFrames into one
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    idx_orbs = list(combined_dataframe.columns).index('log_fci_dim')
    print(list(combined_dataframe.columns))
    sorted_dataframe = combined_dataframe.apply(
        lambda x: x.sort_values().values if x.name != combined_dataframe.columns[idx_orbs] else x)

    plt.figure(figsize=(10, 6))

    for column in sorted_dataframe.columns:
        if 'df' in str(column):
            plt.plot(sorted_dataframe.index, sorted_dataframe[column], label=column)

    plt.title('Sorted Data Plot')
    plt.xlabel(r'Chemical Systems (sorted by $\log_{10} \text{FCI Dim.})$')
    plt.ylabel('Metric values')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.show()

    df1 = pd.read_csv('shci_ref_es.csv')
    df2 = pd.read_csv('ccsdt_ref_es.csv')

    # Perform an inner merge on 'system' and 'basis'
    merged_df = pd.merge(df1, df2, on=['system', 'basis'], suffixes=('_1', '_2'))

    # Calculate the absolute difference in CCSDT energies
    merged_df['error'] = abs(merged_df['SHCI'] - merged_df['CCSDT'])

    # Determine if the error is less than 0.0016 Hartree
    merged_df['below_threshold'] = merged_df['error'] < 0.0016

    below_threshold_results = merged_df['below_threshold']

    print(below_threshold_results)
