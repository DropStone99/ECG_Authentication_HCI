import wfdb
import os
import pandas as pd

# Set the path to the PTB Diagnostic ECG Database files
data_path = 'Data'

# Get a list of all the files and folders in the directory
file_list = os.listdir(data_path)

# Use a list comprehension to filter out only the folders with the name "person"
person_folders = [f for f in file_list if os.path.isdir(os.path.join(data_path, f)) and 'Person_' in f]

# Create an ExcelWriter object to write the data to a single file
writer = pd.ExcelWriter('Data.xlsx', engine='xlsxwriter')

# Loop over the person folders
for folder_name in person_folders:
    # Create an empty DataFrame to store the ECG data for this person
    df = pd.DataFrame()

    # Loop over the files in the folder
    for i in range(1, 21):
        # Set the record name
        record_name = f'{folder_name}/rec_{i}'

        if not os.path.exists(os.path.join(data_path, record_name) + ".hea"):
            continue

        # Read the ECG record
        record = wfdb.rdrecord(os.path.join(data_path, record_name))

        # Read the signal data
        signal, _ = wfdb.rdsamp(os.path.join(data_path, record_name))

        # Add the ECG data to the DataFrame
        df[f'ECG_{i}'] = signal[:, 0]

    # Add the DataFrame as a sheet in the Excel file
    df.to_excel(writer, sheet_name=folder_name, index=False)

# Save the Excel file
writer.close()
