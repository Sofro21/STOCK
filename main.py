import preproccessing
import newLSTM
import newNN
import os
import csv
import pandas as pd
import gc

if not os.path.exists("results_folder"):
    os.makedirs("results_folder")

folder_path = "snp500_data"
files = os.listdir(folder_path)
files.sort(reverse=True)

csv_path = "results.csv"
header = ["Company", "Percentage Change"]

# Writing the header to the CSV file
# with open(csv_path, "w", newline="") as csv_file:
#   csv_writer = csv.writer(csv_file)
#    csv_writer.writerow(header)

for file in files:
    print("-------------------------------------")
    print("BEGINNING " + file)
    file_path = folder_path + "/" + file
    try:
        preproccessing.main(file_path)
        newLSTM.main1(file_path)
        newNN.main2(file_path)
        df = pd.read_csv(file_path, index_col=False)
        Close = df["Close"][len(df["Close"]) - 60 :]
        Close = list(Close[::-1])
        score = 0
        for i in range(1, len(Close)):
            if Close[i - 1] < Close[i]:
                score += 1
            else:
                score -= 1
        new_row = [file_path, score]
        print("Designating Scores On:" + file_path)
        with open(csv_path, "a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(new_row)

        out = Close = df["Close"][len(df["Close"]) - 120 :]
        out.to_csv(f"prices/{file}")

        # Explicitly collect and dump memory
        gc.collect()
        gc.garbage.clear()

        os.remove(file_path)

    except Exception as e:
        if e == KeyboardInterrupt:
            exit()
        else:
            with open("BITCHES.txt", "a", encoding="utf-8") as f:
                f.writelines(file_path + ": " + "\n" + str(e) + "\n")
