import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Define a function to extract and normalize the patient ID from the video filename
def extract_and_normalize_patient_id(video_path):
    match = re.search(r"(\d+(-\d+)?)", video_path)
    if not match:
        return None
    
    extracted_id = match.group(1)
    # If the extracted ID contains a hyphen, normalize using only the first part of the split string
    if '-' in extracted_id:
        return str(int(extracted_id.split('-')[0]))
        
    return str(int(extracted_id))  # Convert to integer to remove leading zeros and then back to string

# Create a set to store unique patient IDs
unique_patient_ids_set = set()

# Extract unique patient IDs from each CSV file
files = ['train.csv', 'valid.csv', 'test.csv']
for file in files:
    df = pd.read_csv(file)
    print(f"Processing {file} with {len(df)} records.")
    
    for _, row in df.iterrows():
        patient_id = extract_and_normalize_patient_id(row['video_path'])
        if patient_id:
            unique_patient_ids_set.add(patient_id)

# Convert the set to a list
unique_patient_ids = list(unique_patient_ids_set)

# Split the unique patient IDs into train, valid, and test (60-20-20)
train_ids, temp_ids = train_test_split(unique_patient_ids, test_size=0.4, random_state=42)
valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Define a function to read and append rows to the appropriate dataset
def process_file(file_path, train_data, valid_data, test_data):
    df = pd.read_csv(file_path)
    print(f"Processing {file_path} with {len(df)} records.")
    
    for _, row in df.iterrows():
        patient_id = extract_and_normalize_patient_id(row['video_path'])
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
            print(f"Unexpected error: Patient ID {patient_id} from {row['video_path']} is not in unique patient ids list.")
    return train_data, valid_data, test_data

# Create empty dataframes to store stratified data
train_data = pd.DataFrame(columns=['video_path', 'start', 'end', 'label', 'caption'])
valid_data = pd.DataFrame(columns=['video_path', 'start', 'end', 'label', 'caption'])
test_data = pd.DataFrame(columns=['video_path', 'start', 'end', 'label', 'caption'])

# Process each CSV file again to stratify data based on extracted unique patient IDs
for file in files:
    train_data, valid_data, test_data = process_file(file, train_data, valid_data, test_data)

print(f"Train data length: {len(train_data)}")
print(f"Valid data length: {len(valid_data)}")
print(f"Test data length: {len(test_data)}")

# Save the new stratified datasets
train_data.to_csv('stratified_train.csv', index=False)
valid_data.to_csv('stratified_valid.csv', index=False)
test_data.to_csv('stratified_test.csv', index=False)
