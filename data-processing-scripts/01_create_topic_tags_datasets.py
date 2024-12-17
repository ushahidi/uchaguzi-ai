import pandas as pd
import os
from ast import literal_eval
from utils import topic_split, tag_split


if __name__ == "__main__":
    
    # read processed dataset
    print('Reading dataset from parquet file...')
    processed_df = pd.read_parquet('./data/uchaguzi-2022_processed.parquet')
    processed_df['tags'] = processed_df['tags'].apply(lambda x: literal_eval(x))
    
    # split dataframe into 8:1:1 train/val/test for topic classification task
    print('Creating train/val/test splits for topic classification task...')
    local_path_to_dir = './data/topic_classification'
    if not os.path.exists(local_path_to_dir):
        os.makedirs(local_path_to_dir)
        
    topic_train_df, topic_val_df, topic_test_df = topic_split(processed_df)
    topic_train_df[['id']].to_parquet(f'{local_path_to_dir}/train.parquet', index=False)
    topic_val_df[['id']].to_parquet(f'{local_path_to_dir}/validation.parquet', index=False)
    topic_test_df[['id']].to_parquet(f'{local_path_to_dir}/test.parquet', index=False)    
    
    # split dataframe into 8:1:1 train/val/test for each tag classification task
    print('Creating train/val/test splits for the tag classification tasks...')
    
    final_tag_tasks_df = processed_df[processed_df['tags'].apply(lambda x: len(x))>0].copy()
    tag_tasks = ['Counting and Results', 'Opinions', 'Polling Station Administration', 'Positive Events', 'Security Issues', 'Voting Issues']
    for task in tag_tasks:
        local_path_to_dir = f'./data/tag_classification/{task.lower().replace(" ", "_")}'
        if not os.path.exists(local_path_to_dir):
            os.makedirs(local_path_to_dir)
        
        tag_train_df, tag_val_df, tag_test_df = tag_split(final_tag_tasks_df[final_tag_tasks_df['topic']==task].copy())
        tag_train_df[['id']].to_parquet(f'{local_path_to_dir}/train.parquet', index=False)
        tag_val_df[['id']].to_parquet(f'{local_path_to_dir}/validation.parquet', index=False)
        tag_test_df[['id']].to_parquet(f'{local_path_to_dir}/test.parquet', index=False)