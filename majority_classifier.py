import os
import numpy as np
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, 'facial_expressions', 'data', 'legend.csv')

if not os.path.exists(csv_file_path):
    print(f"Error: Could not find file at {csv_file_path}")
    exit()

try:
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    if not data_list:
        print("Error: CSV file is empty.")
        exit()

    dataset = np.array(data_list, dtype=object)

except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

headers = dataset[0]
data = dataset[1:]

possible_names = ['emotion', 'Emotion', 'label', 'Label', 'expression', 'class']
target_index = -1

for name in possible_names:
    matches = np.where(headers == name)[0]
    if len(matches) > 0:
        target_index = matches[0]
        break

if target_index == -1:
    print("WARNING: Emotion column not found. Using last column by default.")
    target_index = -1

raw_emotions = data[:, target_index].astype(str)
emotion_column = np.char.lower(raw_emotions)

unique_emotions, counts = np.unique(emotion_column, return_counts=True)

# find majority class
max_index = np.argmax(counts)
majority_emotion = unique_emotions[max_index]
majority_count = counts[max_index]
total_samples = len(emotion_column)

baseline_accuracy = majority_count / total_samples

# print the accuracy of the model
print("Majority Class Baseline")
print(f"Most common emotion: {majority_emotion}")
print(f"Count: {majority_count} / {total_samples}")
print(f"Baseline Accuracy: {baseline_accuracy * 100:.2f}%")
