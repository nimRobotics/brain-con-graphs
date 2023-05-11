import pandas as pd

# read the csv file into a pandas dataframe
df = pd.read_csv('./input/graph_results.csv')

# define a function to map the condition values to the new columns
def map_fatigue(x):
    if x == 'NHALR' or x == 'NHAHR':
        return 0
    elif x == 'FHALR' or x == 'FHAHR':
        return 1
    else:
        return None
    
def map_reliability(x):
    if x == 'NHALR' or x == 'FHALR':
        return 0
    elif x == 'NHAHR' or x == 'FHAHR':
        return 1
    else:
        return None

# create the new columns based on the condition column
df['fatigue'] = df['condition'].apply(lambda x: map_fatigue(x))
df['reliability'] = df['condition'].apply(lambda x: map_reliability(x))

# save the updated dataframe to a new csv file
df.to_csv('./input/graph_results_updated.csv', index=False)
