import argparse
import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data_baseline(df):
    '''Get BOW representations of the data the model should make predictions for.'''
    df_train = pd.read_csv('/Users/rea/Documents/Multi-Author-Writing-Style-Analysis-2024/create_prediction_files/training_data_for_CountVectorizer/df_train.csv') # path to data that was used to fit CountVectorizer of the baseline models

    vectorizer_par1 = CountVectorizer(max_features=5000)
    vectorizer_par2 = CountVectorizer(max_features=5000)

    X_train_par1 = vectorizer_par1.fit_transform(df_train['paragraph1']).toarray()
    X_train_par2 = vectorizer_par2.fit_transform(df_train['paragraph2']).toarray()
    
    # get bow representation of paragraph1
    X_par1 = vectorizer_par1.transform(df['paragraph1']).toarray()
    # get bow representation of paragraph2
    X_par2 = vectorizer_par2.transform(df['paragraph2']).toarray()
    
    X = np.concatenate((X_par1, X_par2), axis=1)

    return X

def generate_output(input_dir, output_dir, model_path, model_type):
    model_type = "baseline" # mistral
    if model_type == "baseline":
        model = tf.keras.models.load_model(model_path)
    #else:
       #model =
    columns = ['paragraph1', 'paragraph2']

    if model_type == "baseline":
        for i, file_name in enumerate(os.listdir(input_dir)):
            if file_name.endswith(".txt"):
                data = []
                with open(os.path.join(input_dir, file_name), "r") as f:
                    paragraphs = [str.strip(line) for line in f.readlines()]
                    for j in range(len(paragraphs)-1):
                        para1 = paragraphs[j]
                        para2 = paragraphs[j+1]
                        data.append([para1,para2])
                    df = pd.DataFrame(data, columns=columns)
                    X = preprocess_data_baseline(df)
                    print(X.shape)
                    predictions = model.predict(X) # get predictions using the baseline model
                    print(predictions)
                    predictions = [0,1,1,1] # dummy predictions
                    output_file_name = f"solution-problem-{i}.json" 

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    with open(os.path.join(output_dir, output_file_name), "w") as g:
                        json.dump({"changes": predictions}, g) # save predictions as a json file

def main():
    print("GENERATING PREDICTIONS") # example for terminal: python3 /Users/rea/Documents/Multi-Author-Writing-Style-Analysis-2024/create_prediction_files/Predictions.py -i /Users/rea/Documents/Multi-Author-Writing-Style-Analysis-2024/create_prediction_files/test_files/easy -o /Users/rea/Documents/Multi-Author-Writing-Style-Analysis-2024/create_prediction_files/predictions -m /Users/rea/Documents/Multi-Author-Writing-Style-Analysis-2024/baseline/Models/baseline_model_emb_org_a.h5 -t baseline
    parser = argparse.ArgumentParser(description='PAN24 Style Change Detection Task: Creating Output Files')
    parser.add_argument("-i", "--input", type=str, help="path to the dir holding the input data", required=True)
    parser.add_argument("-o", "--output", type=str, help="path to the dir to write predicted labels to", required=True)
    parser.add_argument("-m", "--model_path", type=str, help="path to the model that makes predictions", required=True)
    parser.add_argument("-t", "--model_type", type=str, help="type 'baseline' for a baseline model and 'mistral' for mistral models", required=True)
    args = parser.parse_args()

    generate_output(args.input, args.output, args.model_path, args.model_type) # generate predictions and save them locally


if __name__ == "__main__":
    main()