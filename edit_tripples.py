import csv

csv_file_path = 'conditional_triples.csv'

with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

def cl(num):
    if num > 500:
        num = 500
    return int(num / 20)

# append to the csv header row
data[0].append('Hours')
data[0].append('Class')

# append to all other rows
for i in range(1, len(data)):
    data[i].append(int(data[i][1]) / 3600)
    data[i].append(cl(int(data[i][2])))

# write to a new csv file
with open("init_training.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)