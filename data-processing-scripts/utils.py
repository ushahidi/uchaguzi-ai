import pandas as pd
from ast import literal_eval


def topic_split(df):
    """Splits a dataframe into train:val:test 8:1:1"""
    df = df.sample(frac=1,random_state=0)
    size = len(df)
    train_df = df[:int(.8*size)]
    val_df = df[int(.8*size):int(.9*size)]
    test_df = df[int(.9*size):]

    return train_df, val_df, test_df


def tag_split(df):
    """Splits a dataframe into train:val:test 8:1:1 and omits labels with less than 20 observations"""
    df = df.sample(frac=1, random_state=42)
    size = len(df)
    train_df = df[:int(.8*size)].copy()
    val_df = df[int(.8*size):int(.9*size)].copy()
    test_df = df[int(.9*size):].copy()

    label_counts = df['tags'].explode().value_counts()
    all_labels = label_counts[label_counts >= 20].index.tolist()

    train_df['tags'] = train_df['tags'].apply(lambda x: [y for y in x if y in all_labels])
    val_df['tags'] = val_df['tags'].apply(lambda x: [y for y in x if y in all_labels])
    test_df['tags'] = test_df['tags'].apply(lambda x: [y for y in x if y in all_labels])

    return train_df, val_df, test_df


def has_at_least_one_tag_in_common(row):
    if row['annotated_tags'] is not None:
        if sorted(row['tags'])==sorted(row['annotated_tags']):
            return True
        elif list(set(row['tags']) & set(row['annotated_tags']))!=[]:
            return True
        else:
            return False
    else:
        return False