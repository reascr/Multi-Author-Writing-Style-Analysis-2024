!pip install import_notebook
import argparse
import os
import json
import pandas as pd
from import_notebook import Notebook

notebook = Notebook.load(os.path('data_pipeline.ipynb'))
create_model_input = notebook.create_input_data # speicher es so ab wie beim Trainieren des Modells
#from mistralus import model, predict # Model soll von Huggingface geladen werden, es werden Funktionen bereitgestellt
# für jede Inputfile soll das dann ausgegeben werden
# kann ich die data pipeline irgendwie importieren um unnötige Schreibarbeit zu umgehen

def generate_output(input_dir, output_dir):
    #model = load_model() # load model from Huggingface
    for file_name in os.listdir(input_dir):
        counter = 1 # geht besser!
        if file_name.endswith(".txt"):
            with open(os.path.join(input_dir, file_name), "r") as f:
                paragraphs = f.read()
                # was für einen Input muss ich dem Model geben?
                prediction = [0,1,1,1]# get predictions from model, should be a list, otherwise transform to list
                output_file_name = f"solution-problem-{counter}.json"
                counter += 1
                with open(os.path.join(output_dir, output_file_name), "w") as g:
                    json.dump({"changes": prediction}, g)


def main():
    parser = argparse.ArgumentParser(description='PAN24 Style Change Detection Task: Creating Output Files')
    parser.add_argument("-i", "--input", type=str, help="path to the dir holding the input data", required=True)
    parser.add_argument("-o", "--output", type=str, help="path to the dir to write predicted labels to", required=True)
    args = parser.parse_args()

    generate_output(args.input, args.output)

if __name__ == "__main__":
    main()