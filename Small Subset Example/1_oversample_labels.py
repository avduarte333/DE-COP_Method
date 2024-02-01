import pandas as pd
from itertools import permutations
import sys
import os

def generate_permutations():
    variables = [0, 1, 2, 3]
    permuted_variables = list(permutations(variables))
    results = []

    for perm in permuted_variables:
        fifth_variable_position = perm.index(0)
        fifth_variable = fifth_variable_position
        result_row = list(perm) + [fifth_variable]
        results.append(result_row)

    columns = variables + ['Answer']
    base_permutations_df = pd.DataFrame(results, columns=columns)
    new_column_names = ['Example_A', 'Example_B', 'Example_C', 'Example_D', 'Answer']
    base_permutations_df.columns = new_column_names
    return base_permutations_df

def process_files(txt_path, base_permutations_df):
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
        fileIn = os.path.join(base_dir, f'{book_name}.xlsx')
        fileOut = os.path.join(base_dir, f'{book_name}_Paraphrases_Oversampling.xlsx')
        data = pd.read_excel(fileIn)

        multiplication_factor = len(data)
        full_base_permutations = pd.concat([base_permutations_df] * multiplication_factor, ignore_index=True)

        new_df_aux = pd.DataFrame(index=range(len(base_permutations_df)), columns=base_permutations_df.columns[:-1])

        new_df = pd.DataFrame(index=range(0), columns=base_permutations_df.columns[:-1])
        mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        for j in range(len(data)):
            for i in range(len(base_permutations_df)):
                new_df_aux.at[i, 'Example_A'] = data.iloc[j, full_base_permutations.at[i, 'Example_A']]
                new_df_aux.at[i, 'Example_B'] = data.iloc[j, full_base_permutations.at[i, 'Example_B']]
                new_df_aux.at[i, 'Example_C'] = data.iloc[j, full_base_permutations.at[i, 'Example_C']]
                new_df_aux.at[i, 'Example_D'] = data.iloc[j, full_base_permutations.at[i, 'Example_D']]
                new_df_aux.at[i, 'True Answer'] = mapping[full_base_permutations.at[i, 'Answer']]
            new_df = pd.concat([new_df, new_df_aux], ignore_index=True)
        new_df.to_excel(fileOut, index=False)





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python name_of_file.py <path_to_txt_file>")
        sys.exit(1)

    txt_path = sys.argv[1]
    base_permutations_df = generate_permutations()
    process_files(txt_path, base_permutations_df)
