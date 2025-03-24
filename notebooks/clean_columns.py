r"""°°°
# Cleaning `condition` feature

Notebook used for cleaning and visualizing the `condition` column in the Drug Review dataset.

First, installation of `pyxet` and other dependencies:
°°°"""
# |%%--%%| <diUQuwgpou|N6xHq8Vfgn>

%pip install pyxet
%pip install polars
%pip install pandas
%pip install numpy
%pip install pyarrow

# |%%--%%| <N6xHq8Vfgn|8xbt8E955f>

import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas
import numpy as np
import pyarrow
import matplotlib.pyplot as plt

fs = pyxet.XetFS()

# |%%--%%| <8xbt8E955f|5nuJSLA6BX>
r"""°°°
# Drug Review Dataset (UCI)
°°°"""
# |%%--%%| <5nuJSLA6BX|K7859xiLFZ>

# get the drug training and testing data
df_drugs_train = pl.read_csv('xet://drug_data/drugsComTrain_raw.tsv', separator = '\t')
df_drugs_test = pl.read_csv('xet://drug_data/drugsComTest_raw.tsv', separator = '\t')
print(df_drugs_train.head())

# |%%--%%| <K7859xiLFZ|k1yBOtm0bB>

# get the shape of our data
print("Training Data shape: {}".format(df_drugs_train.shape))
print("Testing Data shape: {}".format(df_drugs_test.shape))

# |%%--%%| <k1yBOtm0bB|8VRIzRNq6P>
r"""°°°
## `condition` column
°°°"""
# |%%--%%| <8VRIzRNq6P|U4GTRIc7tf>
r"""°°°
### Training Data
°°°"""
# |%%--%%| <U4GTRIc7tf|wjx34txpEn>

df_drugs_train["condition"].describe()

# |%%--%%| <wjx34txpEn|lVUucYD6qv>

# fill missing values with "not_reported"
df_filled = df_drugs_train.with_columns(
    pl.col("condition").fill_null(pl.lit("not_reported")),
)

# check if we no longer have null values
print(df_filled.null_count())

# |%%--%%| <lVUucYD6qv|O9ZFtotMkz>

# set the number of rows we want to see
pl.Config.set_tbl_rows(100)

# get only the condition column
print(df_filled.select("condition").unique())

# |%%--%%| <O9ZFtotMkz|ibTsQadMzG>
r"""°°°
There are many entries with html tags like `</span>` inside, which should not be showing up in our data.
°°°"""
# |%%--%%| <ibTsQadMzG|7H2tQJbaBz>

# removing html tags
df_condition = df_filled.filter(~pl.col("condition").str.contains("</span>"))


# |%%--%%| <7H2tQJbaBz|XesbJrEdVV>



# |%%--%%| <XesbJrEdVV|qcZhzioRic>

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
plt.hist(df_drugs_train['usefulCount'], bins = 100)
#df_drugs_train.select(['usefulCount']).unique()


# |%%--%%| <qcZhzioRic|2ZCjld2fWS>

print("garbage")

# |%%--%%| <2ZCjld2fWS|pXAWsCfcSG>
r"""°°°
Getting the basic descriptive stats, data types, etc.
°°°"""
# |%%--%%| <pXAWsCfcSG|nS5DroNUfj>

# Get descripive stats
df_drugs_train.describe()

# |%%--%%| <nS5DroNUfj|OlV6xQTntx>

# get column names and their dtypes
df_drugs_train.schema

# |%%--%%| <OlV6xQTntx|cSOfEKT7rL>

# check if all the rows are unique 
print(df_drugs_train.is_unique().all())

# |%%--%%| <cSOfEKT7rL|ZMVUgXcWNy>

# check for nulls
df_drugs_train.null_count()

# |%%--%%| <ZMVUgXcWNy|JMfpV6ugQZ>
r"""°°°
Since there are over 800 `null` values in `condition`, let's take a closer look. This column represents the medical conditions of respondents, so we cannot just drop those. Instead, we fill with `not_reported`. This ensures no null values.
°°°"""
# |%%--%%| <JMfpV6ugQZ|j7TNXSjNVt>

# fill missing values with "not_reported"
df_filled = df_drugs_train.with_columns(
    pl.col("condition").fill_null(pl.lit("not_reported")),
)

# check if we no longer have null values
print(df_filled.null_count())

# |%%--%%| <j7TNXSjNVt|G8DaFCgU8F>
r"""°°°
### Testing Data
°°°"""
# |%%--%%| <G8DaFCgU8F|xFf4mu8omd>



# |%%--%%| <xFf4mu8omd|lbG08gwvDy>


