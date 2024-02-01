import pandas as pd
from itertools import permutations
import sys
import os
from tqdm import tqdm

from torch import nn
import torch

from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


# Function to extract float values from tensors
def extract_float_values(tensor_list):
    float_values = [tensor_item.item() for tensor_item in tensor_list]
    return float_values

def process_files(txt_path, model):
    base_dir = os.path.abspath(os.path.dirname(txt_path))
    lines_list = []

    try:
        with open(txt_path, 'r') as file:
            lines_list = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return


    for lines in lines_list:
        book_name = lines
        fileIn = os.path.join(base_dir, f'{book_name}_Paraphrases_Oversampling.xlsx')
        fileOut = os.path.join(base_dir, f'{book_name}_Paraphrases_Oversampling.xlsx')
        data = pd.read_excel(fileIn)

        parts = book_name.split('_-_')
        book_name = parts[0].replace('_', ' ')
        author_name = parts[1].replace('_', ' ')
        print(f"Starting book - {book_name} by {author_name}")

        if model == "ChatGPT":
            softmax = nn.Softmax(dim=0)
            mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            A_probabilities = []
            B_probabilities = []
            C_probabilities = []
            D_probabilities = []
            Max_Label = []
           
            for i in tqdm(range(len(data))):
                gpt_extra_prompt = f"""Question: Which of the following passages is verbatim from the \"{book_name}\" book by {author_name}?\nOptions:\n"""
                prompt = gpt_extra_prompt + 'A. ' + data.iloc[i][0] + '\n' + 'B. ' + data.iloc[i][1] + '\n' + 'C. ' + data.iloc[i][2] + '\n' + 'D. ' + data.iloc[i][3] + '\n' + 'Answer: '
                response = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt= prompt,
                    max_tokens=1,
                    temperature=0,
                    seed = 2319,
                    logprobs = 4,
                    logit_bias={32:+100, 33:+100, 34:+100, 35:+100}#Increase probabilities of tokens A,B,C,D equally, such that model answers one of those.
                )
                dict_probs = response.choices[0].logprobs.top_logprobs[0]
                logits = torch.tensor([dict_probs["A"], dict_probs["B"], dict_probs["C"], dict_probs["D"]], dtype=torch.float32)
                probabilities = softmax(logits)
                # probabilities = [torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1)]
                A_probabilities.append(probabilities[0])
                B_probabilities.append(probabilities[1])
                C_probabilities.append(probabilities[2])
                D_probabilities.append(probabilities[3])
                Max_Label.append(mapping.get(torch.argmax(probabilities).item(), 'Unknown'))
                # Max_Label.append('A')

            # Extracting float values from the lists of tensors
            float_list1 = extract_float_values(A_probabilities)
            float_list2 = extract_float_values(B_probabilities)
            float_list3 = extract_float_values(C_probabilities)
            float_list4 = extract_float_values(D_probabilities)
            data["A_Probability"] = float_list1
            data["B_Probability"] = float_list2
            data["C_Probability"] = float_list3
            data["D_Probability"] = float_list4
            data["Max_Label_NoDebias"] = Max_Label 
            data.to_excel(fileOut, index=False)
            print(f"Completed book - {book_name}!")





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python name_of_file.py <path_to_txt_file> <model_name>")
        sys.exit(1)

    txt_path = sys.argv[1]
    model = sys.argv[2]

    if model == "ChatGPT":
        api_key="Add your OpenAI key here"
        client = OpenAI(api_key=api_key)
    elif model == "Claude":
        claude_api_key="Add your OpenAI key here"
        anthropic = Anthropic(api_key = claude_api_key)
    else:
        print("Available models are: ChatGPT, Claude, LLaMA-2, Mistral, Mixtral")



    process_files(txt_path, model)
