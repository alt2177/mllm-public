r"""°°°
# Data Visualization

This notebook is meant for data visualization and (potential) preprocessing before using the data for model training. Our data is stored on XetHub, and we will use the Python SDK they provide to access our data without the need for local copies. 

First, installation of `pyxet` and other dependencies:
°°°"""
# |%%--%%| <ISV3Q4HYZQ|ESapP0plJi>

%pip install pyxet
%pip install polars
%pip install pandas
%pip install numpy
%pip install pyarrow

#|%%--%%| <ESapP0plJi|WthiWdIKYx>

%pip install datasets

# |%%--%%| <WthiWdIKYx|qIursMh7Hg>

import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas
import numpy as np
import pyarrow
import matplotlib
import matplotlib.pyplot as plt
# List available font styles
print(plt.rcParams['font.family'])

# Set to a font that supports the character, e.g., 'Arial Unicode MS'
plt.rcParams['font.family'] = 'Arial Unicode MS'

# fix matplotlib error
# matplotlib.rcParams.update(
#     {
#         'text.usetex': False,
#         'font.family': 'stixgeneral',
#         'mathtext.fontset': 'stix',
#     }
# )

fs = pyxet.XetFS()

# |%%--%%| <qIursMh7Hg|4jPSkLc6p9>
r"""°°°
# Drug Review Dataset (UCI)
°°°"""
# |%%--%%| <4jPSkLc6p9|46L995bhcr>

# get the drug training and testing data
df_drugs_train = pl.read_csv('xet://drug_data/drugsComTrain_raw.tsv', separator = '\t')
df_drugs_test = pl.read_csv('xet://drug_data/drugsComTest_raw.tsv', separator = '\t')
print(df_drugs_train.shape)
print(df_drugs_test.shape)
df_drugs_train.head(10)

# |%%--%%| <46L995bhcr|wgFoMZLF7L>

df_drugs_train["review"][0]

#|%%--%%| <wgFoMZLF7L|dAqe1ERVIx>

print(df_drugs_train.select("rating"))

# create scatter plot
plt.figure(figsize = (10, 6))

# Remove top and 
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# labels
plt.hist(df_drugs_train.select("rating"))
plt.title("Predicting `MedHouseval` on `MedInc`")
plt.xlabel('MedInc')
plt.ylabel('MedHouseVal')
plt.legend()
plt.savefig("images/rating_distribution.png")
plt.show()
# |%%--%%| <dAqe1ERVIx|Svziulo8Sh>
r"""°°°
## Data Description / Visualization
°°°"""
# |%%--%%| <Svziulo8Sh|9MapHZqZD1>
r"""°°°
### Training Data
°°°"""
# |%%--%%| <9MapHZqZD1|Sf419kjb0F>

# df_drugs_train.dtypes  #= [Int64, String, String, String, Float64, String, Int64]
df_drugs_train.describe()
df_drugs_train.select(pl.all().is_null().sum())

#df_drugs_train.select(['rating']).unique()
# for cond in df_drugs_train.select(['condition']).unique()['condition']:
#     print(cond)
# for drug in df_drugs_train.select(['drugName']).unique()['drugName']:
#     print(drug)
# for rev in df_drugs_train.select(['review']).unique()['review'][0:20]:
#     print(rev)

review_word_count = df_drugs_train.select(['review']).map_rows(lambda t: len(t[0].split(" ")))
#plt.hist(review_word_count)
review_word_count.describe()
plt.hist(df_drugs_train['rating'], bins = 10)
plt.title("Distribution of Rating")
plt.savefig("rating_dist")
plt.hist(df_drugs_train['usefulCount'], bins = 100)
#df_drugs_train.select(['usefulCount']).unique()


#|%%--%%| <Sf419kjb0F|snB1DM1M8z>

# FOR INTERIM REPORT
from datasets import load_dataset

dataset = load_dataset("billsum")
#|%%--%%| <snB1DM1M8z|7NI6RKV0z7>

import re

billsum_train = dataset["train"]

# Example function to remove non-ASCII characters
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Apply this function to your dataset if it's text data
# Assuming `feature_data` is a list of strings
cleaned_text = [remove_non_ascii(text) for text in billsum_train["text"]]

#|%%--%%| <7NI6RKV0z7|cPIbIpftsX>



# Remove top and 
plt.figure(figsize = (10, 6))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.hist(cleaned_text, bins = 30)
plt.title(f'Distribution of Text Feature')
plt.xlabel("text")
plt.ylabel('Frequency')
# plt.savefig("images/billsum_text_dist.png")
plt.show()

# |%%--%%| <cPIbIpftsX|668K8p14C1>

print(np.quantile(review_word_count, 0.95))
print(np.quantile(review_word_count, 0.05)) # 14 words

for rev in df_drugs_train.filter(review_word_count['map'] < 5)['review']:
   print(rev.strip('\n'))


#    print()
#review_word_count
#df_drugs_train.filter(review_word_count['map'] > 1000)['review']
#df_drugs_train.filter(review_word_count['map'] < 5)['review']



# |%%--%%| <668K8p14C1|ZXGIsF9SJS>
r"""°°°
There are 5 reviews over 1000 words and two of them look identical. There might be more duplicates in the review column.
°°°"""
# |%%--%%| <ZXGIsF9SJS|3mha2jEL5s>

print("garbage")

# |%%--%%| <3mha2jEL5s|FreBAOWV0l>
r"""°°°
Getting the basic descriptive stats, data types, etc.
°°°"""
# |%%--%%| <FreBAOWV0l|rvNIjVVQD2>

# Get descripive stats
df_drugs_train.describe()

# |%%--%%| <rvNIjVVQD2|lI1CDAjSdq>

# get column names and their dtypes
df_drugs_train.schema

# |%%--%%| <lI1CDAjSdq|LhTqSZoKTo>

# check if all the rows are unique 
print(df_drugs_train.is_unique().all())

# |%%--%%| <LhTqSZoKTo|ncdP6b5ea8>

# check for nulls
df_drugs_train.null_count()

# |%%--%%| <ncdP6b5ea8|xhwI5K3ug6>
r"""°°°
Since there are over 800 `null` values in `condition`, let's take a closer look. This column represents the medical conditions of respondents, so we cannot just drop those. Instead, we fill with `not_reported`. This ensures no null values.
°°°"""
# |%%--%%| <xhwI5K3ug6|NfSU56oLQV>

# fill missing values with "not_reported"
df_filled = df_drugs_train.with_columns(
    pl.col("condition").fill_null(pl.lit("not_reported")),
)

# check if we no longer have null values
print(df_filled.null_count())

# |%%--%%| <NfSU56oLQV|TlA6M2SNHD>
r"""°°°
### Testing Data
°°°"""
# |%%--%%| <TlA6M2SNHD|oqIlkjmkac>



# |%%--%%| <oqIlkjmkac|M6zNKeGhsf>



# |%%--%%| <M6zNKeGhsf|usVwDwg3iN>
r"""°°°
# XSUM Dataset
°°°"""
# |%%--%%| <usVwDwg3iN|ZoYWry144X>


# df_xsum = pl.from_pandas(pd.read_parquet('xet://xsum/predictions.parquet'))
# print(df_xsum.shape)
# df_xsum.head()

# |%%--%%| <ZoYWry144X|YA2gz70CKy>



# |%%--%%| <YA2gz70CKy|OQUCsUjILc>
r"""°°°
# CNN/Dailymail Dataset
°°°"""
# |%%--%%| <OQUCsUjILc|rK67ZRraL8>

# files_cnn = fs.ls('xet://cnn_dailymail/1.0.0')
# df_dir = pl.from_dicts(files_cnn)
# print("==================== Current Working Dir ls =====================")
# print(df_dir.head())
# # print(df_dir.select(["name"]).head(1).item())
# df_cnn = pl.from_pandas(pd.read_parquet('xet://{}'.format(df_dir.select(["name"]).head(1).item())))
# # df_cnn = pl.from_pandas(pd.read_parquet('xet://cnn_dailymail/1.0.0/test-00000-of-00001.parquet'))
# print(df_cnn.shape)
# df_cnn.head()

# |%%--%%| <rK67ZRraL8|hh7Mst4TfO>


