import pandas as pd

def load_data(file_path, text_col, label_col):
    print(f"Loading data from {file_path}...")
    dataset = pd.read_csv(file_path)
    
    if text_col not in dataset.columns or label_col not in dataset.columns:
        raise ValueError(f"CRITICAL: Columns '{text_col}' or '{label_col}' not found in dataset!")

    print(f"Dataset shape : {dataset.shape}\n")
    print(f"Dataset head : \n{dataset.head()}\n")
    print(f"Dataset output counts:\n{dataset[label_col].value_counts()}\n")

    # Flexible generalization to binary integer 1 and 0 mapping
    if not pd.api.types.is_numeric_dtype(dataset[label_col]):
        dataset[label_col] = dataset[label_col].astype(str).str.strip().str.lower()
        pos_mapping = ['positive', 'pos', 'yes', '1', 'true']
        neg_mapping = ['negative', 'neg', 'no', '0', 'false']
        
        def map_sentiment(val):
            if val in pos_mapping: return 1
            if val in neg_mapping: return 0
            raise ValueError(f"Unrecognized label variation found: '{val}'")
            
        dataset[label_col] = dataset[label_col].apply(map_sentiment)

    dataset[label_col] = dataset[label_col].astype(int)
    print(f"Dataset head after numeric binary encoding :\n{dataset.head(10)}\n")
    
    return dataset
