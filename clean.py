import numpy as np # Replaced pandas
import matplotlib.pyplot as plt
import os
import csv # Standard library for robust CSV reading


# 1. SETUP THE FILE PATH
# Get the absolute path of the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))


# Construct the full path to 'data/legend.csv'
# This ensures it works even if you run the script from a different terminal location
csv_file_path = os.path.join(script_dir, 'facial_expressions', 'data', 'legend.csv')


# Check if file exists
if not os.path.exists(csv_file_path):
   print(f"Error: Could not find file at {csv_file_path}")
   print("Make sure 'visualize_emotions.py' is in the 'facial_expressions' folder next to the 'data' folder.")
   exit()


# 2. LOAD THE DATA (Using Numpy + csv module)
print(f"Loading data from: {csv_file_path}")

try:
   with open(csv_file_path, 'r', encoding='utf-8') as f:
       reader = csv.reader(f)
       data_list = list(reader)
  
   if not data_list:
       print("Error: The CSV file is empty.")
       exit()


   # Convert list of lists to a 2D numpy array
   # We use object type temporarily to handle potential jagged arrays safely,
   # though valid CSVs should be rectangular.
   dataset = np.array(data_list, dtype=object)


except Exception as e:
   print(f"Error reading CSV: {e}")
   exit()


# Separate headers (first row) from the actual data (rest of the rows)
headers = dataset[0]
data = dataset[1:]


# 3. INSPECT DATA
print("\n--- Columns in your dataset ---")
print(headers)
print("-------------------------------")


# AUTOMATICALLY FIND THE EMOTION COLUMN
possible_names = ['emotion', 'Emotion', 'label', 'Label', 'expression', 'class']
target_index = -1


for name in possible_names:
   # np.where returns the indices where the condition is true
   matches = np.where(headers == name)[0]
   if len(matches) > 0:
       target_index = matches[0]
       break


if target_index != -1:
   print(f"Found emotion column: '{headers[target_index]}' at index {target_index}")
else:
   print("\nWARNING: Could not automatically identify the emotion column.")
   print("Using the last column by default.")
   target_index = -1


# 4. PREPARE THE DATA FOR PLOTTING
# Extract the column with the emotions (all rows, target column index)
# We treat them as strings
raw_emotions = data[:, target_index].astype(str)


# FIX: Normalize to lowercase to merge 'HAPPINESS' and 'happiness'
emotion_column = np.char.lower(raw_emotions)


# Use numpy to get unique values and their counts
unique_emotions, counts = np.unique(emotion_column, return_counts=True)


# Sort by count (descending) to match Pandas value_counts() behavior
sort_indices = np.argsort(counts)[::-1]
unique_emotions = unique_emotions[sort_indices]
counts = counts[sort_indices]


print("\n--- Emotion Counts (Normalized) ---")
# Print loop to mimic the pandas output
for emotion, count in zip(unique_emotions, counts):
   print(f"{emotion}: {count}")


# 5. CREATE THE HISTOGRAM
plt.figure(figsize=(10, 6))


bars = plt.bar(unique_emotions, counts, color='skyblue', edgecolor='black')


plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Distribution of Facial Expressions in Dataset', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)


# Add text labels
for bar in bars:
   height = bar.get_height()
   plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom')


# 6. SHOW AND SAVE
plt.tight_layout()
output_path = os.path.join(script_dir, 'emotion_histogram.png')
plt.savefig(output_path)
print(f"\nSuccess! Chart saved as '{output_path}'.")
plt.show()
