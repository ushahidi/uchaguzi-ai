import pandas as pd
import os
from ast import literal_eval
from utils import has_at_least_one_tag_in_common
from sklearn.metrics import cohen_kappa_score
import nltk
from nltk.metrics.distance import jaccard_distance


if __name__ == "__main__":
    
    # read processed dataset
    processed_df = pd.read_parquet('./data/uchaguzi-2022_processed.parquet')
    processed_df['tags'] = processed_df['tags'].apply(lambda x: literal_eval(x))
    
    # generate expert-annotation dataset (500 samples)
    print('Generating the expert-annotation dataset (500 samples)...')
    
    tag_tasks = ['Counting and Results', 'Opinions', 'Polling Station Administration', 'Positive Events', 'Security Issues', 'Voting Issues']
    max_num_to_sample = 100
    annotation_dfs = []

    for task in tag_tasks:
        local_path_to_dir = f'./data/tag_classification/{task.lower().replace(" ", "_")}'
        tag_test_df = pd.read_parquet(f'{local_path_to_dir}/test.parquet')
        task_annotation_df = tag_test_df.sample(n=min([max_num_to_sample,len(tag_test_df)]), random_state=0)
        annotation_dfs.append(task_annotation_df)
    
    remaining_tag_tasks = ['Media Reports', 'Political Rallies', 'Staffing Issues', 'Irrelevant Report']
    num_to_sample = 42    
    
    for task in remaining_tag_tasks:
        task_annotation_df = processed_df[processed_df['topic']==task].sample(n=num_to_sample, random_state=0)
        annotation_dfs.append(task_annotation_df[['id']])
    
    annotation_dataset_df = pd.concat(annotation_dfs)
    annotation_dataset_df = annotation_dataset_df.sample(frac=1, random_state=0)

    # merge with annotations
    print('Merging with annotations...')
    
    annotations_df = pd.read_parquet('./data/annotated_dataset.parquet')
    annotations_df['annotated_tags'] = annotations_df['annotated_tags'].apply(lambda x: literal_eval(x))
    
    merged_df = annotation_dataset_df.merge(annotations_df, on='id')
    merged_df = merged_df.merge(processed_df[['id','topic','tags']], on='id')
    merged_df['tags'] = merged_df['tags'].apply(lambda x: x if x!=[] else ['No Nature of Incident'])
    
    # evaluate inter-annotator reliability
    print('Evaluating inter-annotator reliability...')
    
    # agreement on topic annotation (0.504, i.e., 50.4%)
    print(f"Percent agreement on topic annotation: {round(sum(merged_df['topic']==merged_df['annotated_topic'])/len(merged_df),3)}")
    
    # cohen's kappa for topic annotation (0.425)
    print(f"Cohen's kappa on topic annotation: {round(cohen_kappa_score(merged_df['topic'], merged_df['annotated_topic']),3)}")
    
    # agreement on tag annotation (at least one tag in common) (0.71, i.e., 71.0%)
    tmp_merged_df = merged_df[merged_df['topic']==merged_df['annotated_topic']].copy()
    tmp_merged_df['has_at_least_one_tag_in_common'] = tmp_merged_df.apply(has_at_least_one_tag_in_common, axis=1)
    print(f"Percent agreement on tag annotation: {round(sum(tmp_merged_df['has_at_least_one_tag_in_common'])/len(tmp_merged_df),3)}")
    
    # cohen's kappa for tag annotation (0.624)
    data = []
    for idx in range(len(tmp_merged_df)):
        gt_tuple = ('gt',tmp_merged_df.iloc[idx]['id'],frozenset(tmp_merged_df.iloc[idx]['tags']))
        if tmp_merged_df.iloc[idx]['annotated_tags'] is None:
            ann_tuple = ('ann',tmp_merged_df.iloc[idx]['id'],frozenset([None]))
        else:
            ann_tuple = ('ann',tmp_merged_df.iloc[idx]['id'],frozenset(tmp_merged_df.iloc[idx]['annotated_tags']))
        data.append(gt_tuple)
        data.append(ann_tuple)
    
    jaccard_task = nltk.AnnotationTask(distance=jaccard_distance)
    jaccard_task.load_array(data)
    print(f"Cohen's kappa for tag annotation: {round(jaccard_task.kappa(),3)}")
    
    # update the topic and tag classification train/val/test splits
    print('Updating the topic and tag classification train/val/test splits...')
    
    # remove expert-annotated samples from topic train/val splits
    annotated_sample_ids = list(merged_df['id'])

    local_path_to_dir = './data/topic_classification'
    topic_train_df = pd.read_parquet(f'{local_path_to_dir}/train.parquet')
    topic_val_df = pd.read_parquet(f'{local_path_to_dir}/validation.parquet')
    
    updated_topic_train_df = topic_train_df[topic_train_df['id'].apply(lambda x: x not in annotated_sample_ids)].copy()
    updated_topic_val_df = topic_val_df[topic_val_df['id'].apply(lambda x: x not in annotated_sample_ids)].copy()
    
    updated_topic_train_df[['id']].to_parquet(f'{local_path_to_dir}/train.parquet', index=False)
    updated_topic_val_df[['id']].to_parquet(f'{local_path_to_dir}/validation.parquet', index=False)
    
    # replace topic and tag test splits with expert-annotated dataset
    
    merged_df[['id']].to_parquet(f'{local_path_to_dir}/test.parquet', index=False)
    
    for task in tag_tasks:
        local_path_to_dir = f'./data/tag_classification/{task.lower().replace(" ", "_")}'
        expert_tag_test_df = merged_df[merged_df['annotated_topic']==task].copy()
        expert_tag_test_df[['id']].to_parquet(f'{local_path_to_dir}/test.parquet', index=False)