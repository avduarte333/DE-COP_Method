import pandas as pd
from itertools import permutations

def generate_permutations(document_df):
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


    multiplication_factor = len(document_df)
    full_base_permutations = pd.concat([base_permutations_df] * multiplication_factor, ignore_index=True)

    new_df_aux = pd.DataFrame(index=range(len(base_permutations_df)), columns=base_permutations_df.columns[:-1])

    new_df = pd.DataFrame(index=range(0), columns=base_permutations_df.columns[:-1])
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    for j in range(len(document_df)):
        for i in range(len(base_permutations_df)):
            new_df_aux.at[i, 'Example_A'] = document_df.iloc[j, full_base_permutations.at[i, 'Example_A']]
            new_df_aux.at[i, 'Example_B'] = document_df.iloc[j, full_base_permutations.at[i, 'Example_B']]
            new_df_aux.at[i, 'Example_C'] = document_df.iloc[j, full_base_permutations.at[i, 'Example_C']]
            new_df_aux.at[i, 'Example_D'] = document_df.iloc[j, full_base_permutations.at[i, 'Example_D']]
            new_df_aux.at[i, 'True Answer'] = mapping[full_base_permutations.at[i, 'Answer']]
        new_df = pd.concat([new_df, new_df_aux], ignore_index=True)

    new_df['ID'] = document_df.at[0, 'ID']
    new_df['Label'] = document_df.at[0, 'Label']
    return new_df