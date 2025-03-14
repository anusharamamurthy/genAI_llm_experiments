# This file contains code to run few shot learning on mistral 7B, on both validation and test set(CS dataset) only. The code will write output into csv files.

from langchain_community.llms import Ollama
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from typing import List, Dict, Optional
from IPython.display import display
from io import StringIO
from zipfile import PyZipFile
from evaluate import load

import getpass
import sys
import os
import langchain
import pandas as pd
import json
from tqdm import tqdm

import csv


def read_data_from_json_file(filename) ->list:
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    return data

def get_predictions(model, train_cs_df:pd.DataFrame, model_input:pd.DataFrame) -> pd.DataFrame:

    training_examples = train_cs_df[['utterance','plan']].rename(columns={'utterance':'input','plan':'output'}).to_dict(orient='records')
    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt_with_train_set = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=training_examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", " You are an expert semantic parser. \
          The input is English-language dialogue, featuring natural conversations about tasks involving calendars, weather, places, and people. Parse the input to provide an executable dataflow program featuring API calls,\
          function composition, and complex constraints built from strings, numbers, dates and times. Do not interpolate or make up any data."),
        few_shot_prompt_with_train_set,
        ("human", "{input}"),
        ]
    )
    
    results = []

    for row in tqdm(model_input.to_dict(orient='records')):
        chain = final_prompt | model
        output = chain.invoke({"input": row['utterance']})        
        row['prediction'] = dict(output)['content']
        results.append(row)
        pd.DataFrame(results).to_csv("output.csv")

    
    return pd.DataFrame(results)

def compute_exact_match_score(model_name:str,dataset:str, results_df:pd.DataFrame):
    exact_match = load("exact_match")
    scores = exact_match.compute(predictions=results_df['prediction'].values, references=results_df['plan'].values)
    
    print(f"Score on {dataset} \n {scores}")

def compute_bleu_score(model_name:str,dataset:str, results_df:pd.DataFrame):
    bleu = load("bleu")
    scores = bleu.compute(predictions=results_df['prediction'].values, references=results_df['plan'].values)
    
    print(f"Score on {dataset} \n {scores}")

def compute_rouge_score(model_name:str,dataset:str, results_df:pd.DataFrame):
    
    rouge = load("rouge")
    scores = rouge.compute(predictions=results_df['prediction'].values, references=results_df['plan'].values)
    
    print(f"Score on {dataset} \n {scores}")
    
def get_train_set_CS(trainFile):
    temp = pd.json_normalize(read_data_from_json_file(trainFile))
    temp['tags'] = temp['tags'].apply(str)
    train_cs_df  = temp[temp['tags'].str.contains('events.create') & (temp['tags'].str.contains('events_with_orgchart'))]
    return train_cs_df

def get_test_set_CS(testFile):
    temp = pd.json_normalize(read_data_from_json_file(testFile))
    print(temp.columns)
    prediction = ["" for x in range(len(temp))]
    
    temp = temp.assign(prediction=prediction)
    temp['tags'] = temp['tags'].apply(str)
    
    test_cs_df = temp[temp['tags'].str.contains('events.create') & (temp['tags'].str.contains('events_with_orgchart'))]
    return test_cs_df

def compute_average_score(filename:str,column_name:str)-> float:
    results_df = pd.read_csv(filename)
    return results_df[column_name].mean()

def main():
    # get train, val, test data, filtered down to only rows that contain CS skills
    # aka "events_with_orgchart" and "event.create"

    # windows system
    trainFile = r'data\smcalflow_cs\calflow.orgchart.event_create\source_domain_with_target_num128\train.jsonl'
    validFile = r'data\smcalflow_cs\calflow.orgchart.event_create\source_domain_with_target_num128\valid.jsonl'
    testFile = r'data\smcalflow_cs\calflow.orgchart.event_create\source_domain_with_target_num128\test.jsonl'
    testFile_L = r'/Documents/input/sample_for_large_model.jsonl'

    train_df = get_train_set_CS(trainFile)
    validation_df = get_test_set_CS(validFile)
    test_df = get_test_set_CS(testFile_L)
   
    if sys.argv[1] == "s":
        # few shot learning on mistralAI 7B
        model = Ollama(model="mistral")
        model_name = "mistral7B"
    
    elif sys.argv[1] == "l":
        # few shot learning on mistralAI Large model through API
        
        # os.environ["MISTRAL_API_KEY"] = getpass.getpass()
        # model = ChatMistralAI(model="mistral-large-latest")
        # model_name = "mistral-large"
        # test_df = get_test_set_CS(testFile_L)

        OPENAI_API_KEY = getpass.getpass()
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # base_url="...",
            # organization="...",
            # other params...
        )
        model_name = "mistral-large"

        # if local model is availanle (needs 69GB space)
        # model = Ollama(model="mistral-large")
        # model_name = "mistral-large"

    else:
        print("invalid value for argument, please type `s` for running mistral7B from local, `l` to run mistral-large model using API Key")
        exit(1)

    
    # validation set
    results_df = get_predictions(model, train_df,validation_df)
    results_df.to_csv(f"{model_name}_predictions_for_val_set.csv")
    
    # # compute metrics
    compute_bleu_score(model_name,"validationSet", results_df)
    compute_exact_match_score(model_name,"validationSet", results_df)
    compute_rouge_score(model_name,"validationSet", results_df)

    # test set
    
    results_df = get_predictions(model, train_df,test_df)
    results_df.to_csv(f"{model_name}_predictions_for_test_set.csv")
    
    # compute metrics
    compute_bleu_score(model_name,"testSet", results_df)
    compute_exact_match_score(model_name,"testSet", results_df)
    compute_rouge_score(model_name,"testSet", results_df)



if __name__ == "__main__":
    main()
