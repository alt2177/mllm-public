r"""°°°
# Data Visualization

This notebook is meant for data visualization and (potential) preprocessing before using the data for model training. Our data is stored on XetHub, and we will use the Python SDK they provide to access our data without the need for local copies. 

First, installation of `pyxet` and other dependencies:
°°°"""
# |%%--%%| <sdycpZXZrM|nMxrpLiwao>

%pip install pyxet
%pip install polars
%pip install pandas
%pip install numpy
%pip install pyarrow

#|%%--%%| <nMxrpLiwao|7BYmUih6BP>
r"""°°°
Now, let's install all the relevant libraries:
°°°"""
# |%%--%%| <7BYmUih6BP|pfw6CUAGvu>

import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas
import numpy as np
import pyarrow
import matplotlib.pyplot as plt

fs = pyxet.XetFS()

# |%%--%%| <pfw6CUAGvu|nFFgVYu23E>
r"""°°°
## Data Loading 

Now, let's access our data and load it into `pd.DataFrame` objects,
°°°"""
# |%%--%%| <nFFgVYu23E|XDe5SYSB0k>

# get the drug training and testing data
df_drugs_train = pl.read_csv('xet://drug_data/drugsComTrain_raw.tsv', separator = '\t')
df_drugs_test = pl.read_csv('xet://drug_data/drugsComTest_raw.tsv', separator = '\t')
df_drugs_train.head(11)

#|%%--%%| <XDe5SYSB0k|iSvjEduknc>

df_drugs_train.select(pl.col[165907]

#|%%--%%| <iSvjEduknc|mxG2mMa32O>

# get the shape of the training and testing sets
print(df_drugs_train.shape)
print(df_drugs_test.shape)

# |%%--%%| <mxG2mMa32O|7D4EMr5SJM>

# preview one of the reviews
df_drugs_train["review"][0]

# |%%--%%| <7D4EMr5SJM|Gk50aixwSm>
r"""°°°
## Data Description / Visualization

We now begin EDA, first looking at some simple summary statistics, checking for null values, and getting a feel for the data.
°°°"""
#|%%--%%| <Gk50aixwSm|2yb9jP9hP0>

# basic descriptions
df_drugs_train.describe()
# |%%--%%| <2yb9jP9hP0|FSBP8jnmPJ>

# check for null values
df_drugs_train.select(pl.all().is_null().sum())
#|%%--%%| <FSBP8jnmPJ|pbGCONbb9F>
r"""°°°
Now, we look at the unique values we have in our columns
°°°"""
#|%%--%%| <pbGCONbb9F|5Dl54EdTfK>

# check our ratings and see what unique values we have
df_drugs_train.select(['rating']).unique()
#|%%--%%| <5Dl54EdTfK|C5cPMqN2mq>

# check unique conditions
df_drugs_train.select(['condition']).unique()['condition']

#|%%--%%| <C5cPMqN2mq|qoJEnEq3Zi>

# check unique drug names 
df_drugs_train.select(['drugName']).unique()['drugName']
#|%%--%%| <qoJEnEq3Zi|t2c45vrRJz>

# look at the first 20 reviews
df_drugs_train.select(['review']).unique()['review'][0:20]
#|%%--%%| <t2c45vrRJz|j9uuzqiR0A>
r"""°°°
We now turn our attention to the reviews themselves. Some basic visualization:
°°°"""
#|%%--%%| <j9uuzqiR0A|HWc657s24h>

review_word_count = df_drugs_train.select(['review']).map_rows(lambda t: len(t[0].split(" ")))
review_word_count.describe()
#|%%--%%| <HWc657s24h|N9hsnW9cp9>

# create figure 
plt.figure(figsize = (10, 6))

# Remove top and right borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plotting
plt.hist(df_drugs_train['rating'], bins = 10, color = "cornflowerblue", label = "Rating")
plt.axvline(df_drugs_train['rating'].mean(), color = "gold", linestyle = "dashed", linewidth = 2, label = "Average Rating")
plt.title("Distribution of Drug Ratings")
plt.savefig("rating_dist")
plt.xlabel('Score (from 1 to 10)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("images/drug_ratings.png")
plt.show()

#|%%--%%| <N9hsnW9cp9|X6s3rvd4gD>

# create figure 
plt.figure(figsize = (10, 6))

# Remove top and right borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plotting useful count (the number of users who found a particular review helpful)
plt.hist(df_drugs_train['usefulCount'], range = [0, 200], bins = 100, color = "violet")
plt.xlabel('Number of users who found the review helpful')
plt.ylabel('Frequency')
plt.title("Distribution of Useful Ratings")
plt.savefig("images/drug_usefulness.png")
plt.show()

# |%%--%%| <X6s3rvd4gD|fjfsqaLMgZ>

print(np.quantile(review_word_count, 0.95))
print(np.quantile(review_word_count, 0.05)) # 14 words
#|%%--%%| <fjfsqaLMgZ|ihHMAnQ3Pc>
plt.figure(figsize = (10, 6))

# Remove top and right borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.title("Distribution of Review Word Counts")
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xlim(14, 200)
plt.hist(review_word_count, bins = 200, color = "lightskyblue")
<<<<<<< HEAD
<<<<<<< HEAD
plt.axvline(review_word_count.mean().row(0)[0], color = "gold", linestyle = "dashed", linewidth = 2, label = "Average Word Count")
=======
plt.axvline(review_word_count.mean(), color = "gold", linestyle = "dashed", linewidth = 2, label = "Average Word Count")
>>>>>>> a63225b39 (updating notebooks for presentation figures)
=======
plt.axvline(review_word_count.mean().row(0)[0], color = "gold", linestyle = "dashed", linewidth = 2, label = "Average Word Count")
>>>>>>> c1dc9724a (created new figuers in notebooks)
plt.savefig("images/review_word_count.png")
plt.show()

#|%%--%%| <ihHMAnQ3Pc|CDPsEMelgx>

plt.figure(figsize = (10, 6))

# Remove top and right borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


review_char_count = df_drugs_train.select(['review']).map_rows(lambda t: len(t[0]))
print(list(review_char_count))
plt.hist(review_word_count, bins = 200, color = "lightskyblue", label = "Review Word Count")
plt.axvline(review_word_count.mean().row(0)[0], color = "gold", linestyle = "dashed", linewidth = 2, label = "Average Word Count")
plt.hist(review_char_count, bins = 500, alpha = 0.5, color = "violet", label = "Review Character Count")
plt.axvline(review_char_count.mean().row(0)[0], color = "purple", linestyle = "dashed", linewidth = 2, label = "Average Character Count")
plt.xlim(14, 900)
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.legend()
<<<<<<< HEAD
<<<<<<< HEAD
plt.title("Distribution of Length of Drug Reviews")
plt.savefig("images/count_dist.png")
=======
plt.savefig("images/count_dist.png")
plt.title("Distribution of Length of Drug Reviews")
>>>>>>> c1dc9724a (created new figuers in notebooks)
=======
plt.title("Distribution of Length of Drug Reviews")
plt.savefig("images/count_dist.png")
>>>>>>> 2b81c5dbd (fixed image title)
plt.show()
#for rev in df_drugs_train.filter(review_word_count['map'] < 5)['review']:
#   print(rev.strip('\n'))


#    print()
#review_word_count
#df_drugs_train.filter(review_word_count['map'] > 1000)['review']
#df_drugs_train.filter(review_word_count['map'] < 5)['review']



# |%%--%%| <CDPsEMelgx|GWBY7VfJKo>
r"""°°°
There are 5 reviews over 1000 words and two of them look identical. There might be more duplicates in the review column.
°°°"""
# |%%--%%| <GWBY7VfJKo|MicaUlg1rK>


# |%%--%%| <MicaUlg1rK|oBD1Dolf6c>
r"""°°°
Getting the basic descriptive stats, data types, etc.
°°°"""
# |%%--%%| <oBD1Dolf6c|lXWP6Ypvta>

# Get descripive stats
df_drugs_train.describe()

# |%%--%%| <lXWP6Ypvta|JpGtxbn2QX>

# get column names and their dtypes
df_drugs_train.schema

# |%%--%%| <JpGtxbn2QX|v4l3KuCL1t>

# check if all the rows are unique 
print(df_drugs_train.is_unique().all())

# |%%--%%| <v4l3KuCL1t|gtNzKvKgCv>

# check for nulls
df_drugs_train.null_count()

# |%%--%%| <gtNzKvKgCv|6jlsonLhbC>
r"""°°°
Since there are over 800 `null` values in `condition`, let's take a closer look. This column represents the medical conditions of respondents, so we cannot just drop those. Instead, we fill with `not_reported`. This ensures no null values.
°°°"""
# |%%--%%| <6jlsonLhbC|rDXvdNTgrT>

# fill missing values with "not_reported"
df_filled = df_drugs_train.with_columns(
    pl.col("condition").fill_null(pl.lit("not_reported")),
)

# check if we no longer have null values
print(df_filled.null_count())

# |%%--%%| <rDXvdNTgrT|2eqByAP9BG>
r"""°°°
°°°"""
# |%%--%%| <2eqByAP9BG|AZDKulL54M>



# |%%--%%| <AZDKulL54M|eq7QM3PAp6>



# |%%--%%| <eq7QM3PAp6|KY2m8XKprf>
r"""°°°
# XSUM Dataset
°°°"""
# |%%--%%| <KY2m8XKprf|QoAjM81eKc>

df_xsum = pl.from_pandas(pd.read_parquet('xet://xsum/predictions.parquet'))
print(df_xsum.shape)
df_xsum.head()

# |%%--%%| <QoAjM81eKc|VP6E0UOBhb>



# |%%--%%| <VP6E0UOBhb|uVCqSdLwbK>
r"""°°°
# CNN/Dailymail Dataset
°°°"""
# |%%--%%| <uVCqSdLwbK|mnxXyXMNXV>

files_cnn = fs.ls('xet://cnn_dailymail/1.0.0')
df_dir = pl.from_dicts(files_cnn)
print("==================== Current Working Dir ls =====================")
print(df_dir.head())
# print(df_dir.select(["name"]).head(1).item())
df_cnn = pl.from_pandas(pd.read_parquet('xet://{}'.format(df_dir.select(["name"]).head(1).item())))
# df_cnn = pl.from_pandas(pd.read_parquet('xet://cnn_dailymail/1.0.0/test-00000-of-00001.parquet'))
print(df_cnn.shape)
df_cnn.head()

# |%%--%%| <mnxXyXMNXV|Nv3yr1vYoS>


