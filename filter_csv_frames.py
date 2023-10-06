import csv

min_frames = 16
input_files = ['fragment_train.csv', 'fragment_valid.csv', 'fragment_test.csv']
output_files = [x.replace('.csv', '_min16.csv') for x in input_files]

for input_file, output_file in zip(input_files, output_files):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        writer.writeheader() 
        
        for row in reader:
            end_value = int(row['end'])
            if end_value >= 16:
                writer.writerow(row)
