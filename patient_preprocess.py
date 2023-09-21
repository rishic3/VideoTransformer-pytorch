import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Load the patient_id.CSV
patient_df = pd.read_csv('patient_data.csv')
print("Loaded patient_id.CSV with {} records.".format(len(patient_df)))

# Extract unique patient IDs from patient_df and convert them to strings
unique_patient_ids = patient_df['record_id'].astype(str).unique()

print(f"Unique Patient IDs: {unique_patient_ids}")

# Split the unique patient IDs into train, valid, and test (60-20-20)
train_ids, temp_ids = train_test_split(unique_patient_ids, test_size=0.4, random_state=42)
valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Define a function to extract the patient ID from the video filename
def extract_and_match_patient_id(video_path):
    match = re.search(r"(\d+(-\d+)?)", video_path)
    if not match:
        return None
    
    extracted_id = match.group(1)
    # If the extracted ID contains a hyphen, try to match with the entire string first
    if '-' in extracted_id:
        if extracted_id in unique_patient_ids:
            return extracted_id
        # If not found, then match using only the first part of the split string
        return str(int(extracted_id.split('-')[0]))
        
    return str(int(extracted_id))  # Convert to integer to remove leading zeros and then back to string

# Define a function to read and append rows to the appropriate dataset
def process_file(file_path, train_data, valid_data, test_data):
    df = pd.read_csv(file_path)
    print(f"Processing {file_path} with {len(df)} records.")
    
    for _, row in df.iterrows():
        patient_id = extract_and_match_patient_id(row['video_path'])
        if not patient_id:
            print(f"Could not extract patient ID from {row['video_path']}")
            continue
            
        if patient_id in train_ids:
            train_data = train_data.append(row, ignore_index=True)
        elif patient_id in valid_ids:
            valid_data = valid_data.append(row, ignore_index=True)
        elif patient_id in test_ids:
            test_data = test_data.append(row, ignore_index=True)
        else:
            print(f"Patient ID {patient_id} from {row['video_path']} is not in unique patient ids list.")
    return train_data, valid_data, test_data

# Create empty dataframes to store stratified data
train_data = pd.DataFrame(columns=['video_path', 'start', 'end', 'label', 'caption'])
valid_data = pd.DataFrame(columns=['video_path', 'start', 'end', 'label', 'caption'])
test_data = pd.DataFrame(columns=['video_path', 'start', 'end', 'label', 'caption'])

# Process each CSV file
files = ['train.csv', 'valid.csv', 'test.csv']
for file in files:
    train_data, valid_data, test_data = process_file(file, train_data, valid_data, test_data)

print(f"Train data length: {len(train_data)}")
print(f"Valid data length: {len(valid_data)}")
print(f"Test data length: {len(test_data)}")

# Save the new stratified datasets
train_data.to_csv('stratified_train.csv', index=False)
valid_data.to_csv('stratified_valid.csv', index=False)
test_data.to_csv('stratified_test.csv', index=False)
