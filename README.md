This repo hosts the dataset site and code related to the paper:
> Uchaguzi-2022: A Dataset of Citizen Reports on the 2022 Kenyan Election

Getting Started
---

We recommend using [Anaconda](https://www.anaconda.com/download) or a similar dependency manager.
To get started, install the `uchaguzi-ai` package and its dependencies.
```{bash}
conda create -n uchaguzi-ai python=3.10
conda activate uchaguzi-ai
pip install -e .
```

Data-Processing Scripts
---

This section describes the scripts used to generate and evaluate the expert-annotated dataset (Section 2.4 of the paper) and to generate the datasets for the topic and tag classification tasks (Section 3.1 of the paper).

After filling out the [data access form](https://docs.google.com/forms/d/e/1FAIpQLSdVtM1p3UPW3bV7v_wt2TOxSZ2pMCKqsAq6NkrThTHGV1WNOA/viewform) and receiving the dataset, `data.zip` should be unzipped and placed within this repo.

The first script is called `01_create_topic_tags_datasets.py` and can be run as follows:

```{bash}
python data-processing-scripts/01_create_topic_tags_datasets.py
```

Firstly, this script loads the processed dataset (`data/uchaguzi-2022_processed.parquet`). In this dataset, we removed the text for posts from sources other than sms:
- For posts from X (formerly Twitter), we provide the ID
- For posts from Facebook, TikTok, or YouTube, we provide the original link.

The dataset (14,169 posts) contains the following fields:
- `id`: unique ID of the post
- `link`: original link for posts from Facebook, TikTok, or YouTube
- `tweet_id`: ID for posts from X (formerly Twitter)
- `text`: text of the post (empty for posts from X, Facebook, TikTok, or YouTube)
- `topic`: assigned topic
- `title`: assigned title
- `tags`: assigned tags (as a list) 
- `coordinates`: assigned location coordinates in *(latitude, longitude)* format
- `location`: assigned location name.

Secondly, the script generates the train, validation, and test splits for the *topic classification* task using an 80-10-10% random split. The generated files are: `train.parquet`, `validation.parquet`, and `test.parquet` in the `data/topic_classification` folder. These files only contain the `id` field since the remaining fields can be obtained by joining with the full dataset.

Finally, the script generates the train, validation, and test splits for each *tag classification* task using an 80-10-10% random split. As discussed in the paper, for each task we omit labels with less than 20 observations (see `tag_split` method in `utils.py`). The generated files follow the format above and reside in the `data/tag_classification` folder (and sub-folders).

The second script is called `02_create_annotation_dataset.py` and can be run as follows:

```{bash}
python data-processing-scripts/02_create_annotation_dataset.py
```

This script performs the following steps.

First, it generates the dataset consisting of 500 samples to be annotated by an expert using the sampling strategy described in the paper (Section 2.4).

Second, it merges the dataset with the expert annotations (the `annotated_dataset.parquet` file in `data`) and evaluates the inter-annotator reliability, reproducing the results presented in the paper (percent agreement and Cohen's kappa).

Finally, it generates the updated train, validation, and test splits for the *topic classification* and *tag classification* tasks by removing the annotated samples from the train and validation splits (to avoid data leakage during training and evaluation) and by replacing the test sets with the expert-annotated dataset (used as ground truth as described in the paper).