#pip install import_notebook
import argparse
import os
import json
import pandas as pd
import tensorflow as tf
#from import_notebook import Notebook

#notebook = Notebook.load(os.path('data_pipeline.ipynb'))
#create_model_input = notebook.create_input_data # speicher es so ab wie beim Trainieren des Modells
#from mistralus import model, predict # Model soll von Huggingface geladen werden, es werden Funktionen bereitgestellt

def generate_output(input_dir, output_dir):
    #model = load_model() # load model from Huggingface
    baseline_model = tf.keras.models.load_model("my_model.h5")

    columns = ['paragraph1', 'paragraph2']

    for file_name in os.listdir(input_dir):
        counter = 1 # TO DO: CHANGE THAT --> there were some files that were not working
        if file_name.endswith(".txt"):
            data = []
            with open(os.path.join(input_dir, file_name), "r") as f:
                paragraphs = [str.strip(line) for line in f.readlines()]
                for j in range(len(paragraphs)-1):
                    para1 = paragraphs[j]
                    para2 = paragraphs[j+1]
                    data.append([para1,para2])
                df = pd.DatFrame(data, columns=columns) # this is enough as input for baseline model
                # Get predictions using the baseline model
                predictions_base = baseline_model.predict(df)
                #prediction = [0,1,1,1]# get predictions from model, should be a list, otherwise transform to list
                output_file_name = f"solution-problem-{counter}.json"
                counter += 1
                with open(os.path.join(output_dir, output_file_name), "w") as g:
                    json.dump({"changes": predictions_base}, g)


def main():
    parser = argparse.ArgumentParser(description='PAN24 Style Change Detection Task: Creating Output Files')
    parser.add_argument("-i", "--input", type=str, help="path to the dir holding the input data", required=True)
    parser.add_argument("-o", "--output", type=str, help="path to the dir to write predicted labels to", required=True)
    args = parser.parse_args()

    generate_output(args.input, args.output)

if __name__ == "__main__":
    main()