import csv

# ---- CONFIG ----
input_file = "elonmusk1.csv"       # your original CSV file
output_file = "tweettimes.csv"  # new file to create
# ----------------

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    writer.writerow(["date","time","char_count","is_RT"])
    char_count = 0
    is_rt = 0
    jan1 = False
    for row in reader:
        if len(row) >= 1 and (row[-1][-3:] == 'EST' or row[-1][-3:] == 'EDT'): 
            dt = row[-1].strip('"').split(", ")
            if dt[0] == "Jan 1":
                jan1 = True
            if jan1:
                dt[0] += " 2025"
            else:
                dt[0] += " 2024"
            if len(row[0]) == 19 and row[0].isnumeric():
                text = "".join(row[1:-1])
                dt.append(len(text))
                if text[:4] == "RT @":
                    is_rt = 1
            else:
                dt.append(char_count + len("".join(row[:-1])))
            dt.append(is_rt)
            char_count = 0
            is_rt = 0
            writer.writerow(dt)  
        elif len(row) > 0:
            if len(row[0]) == 19 and row[0].isnumeric() and len(row) > 1:
                text = "".join(row[1:])
                char_count += len(text)
                if text[:4] == "RT @":
                    is_rt = 1
            else:
                char_count += len("".join(row))
