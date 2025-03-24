r"""°°°
# Investigating High Test Loss with Drug Dataset 

°°°"""
# |%%--%%| <IeS9KzIxpW|Gz5EGgfL1C>

%pip install pyxet
%pip install polars
%pip install pandas
%pip install numpy
%pip install pyarrow

# |%%--%%| <Gz5EGgfL1C|zaXx0yw20g>

import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas
import numpy as np
import pyarrow
import matplotlib.pyplot as plt
import os

fs = pyxet.XetFS()

# |%%--%%| <zaXx0yw20g|3WzLxPYYUB>

# get to correct directory
os.chdir("..")

print(os.getcwd())

#|%%--%%| <3WzLxPYYUB|Fh6sWFkqI7>

# load all probabilities

# ties merge
ties_test = pl.read_csv("test_loss_exp/test_probabilities.csv")
ties_val = pl.read_csv("test_loss_exp/validation_probabilities.csv")

# dare linear merge
dare_lin_test = pl.read_csv("test_loss_exp/dare_linear_test_probabilities.csv")
dare_lin_val = pl.read_csv("test_loss_exp/dare_linear_validation_probabilities.csv")

# GPT2-XL 
gpt2_xl_test = pl.read_csv("test_loss_exp/gpt2_xl_test_probabilities.csv")
gpt2_xl_val = pl.read_csv("test_loss_exp/gpt2_xl_validation_probabilities.csv")

# base fine tuned gpt2
base_test = pl.read_csv("test_loss_exp/base_ft_test_probabilities.csv")
base_val = pl.read_csv("test_loss_exp/base_ft_validation_probabilities.csv")

# second base model
base_4_test = pl.read_csv("test_loss_exp/base_ft_4test_probabilities.csv")
base_4_val = pl.read_csv("test_loss_exp/base_ft_4validation_probabilities.csv")


#|%%--%%| <Fh6sWFkqI7|eLq6t46PoP>

# create dataframes with only the probabilities
ties_test_probs = ties_test.select(pl.exclude("true_label"))
ties_val_probs = ties_val.select(pl.exclude("true_label"))

dare_lin_test_probs = dare_lin_test.select(pl.exclude("true_label"))
dare_lin_val_probs = dare_lin_val.select(pl.exclude("true_label"))

gpt2_xl_test_probs = gpt2_xl_test.select(pl.exclude("true_label"))
gpt2_xl_val_probs = gpt2_xl_val.select(pl.exclude("true_label"))


base_test_probs = base_test.select(pl.exclude("true_label"))
base_val_probs = base_val.select(pl.exclude("true_label"))


base_4_test_probs = base_4_test.select(pl.exclude("true_label"))
base_4_val_probs = base_4_val.select(pl.exclude("true_label"))

#|%%--%%| <eLq6t46PoP|jphG4SNWO1>

# get distribution of max values
ties_test_maxes = ties_test_probs.with_columns(max = pl.max_horizontal(ties_test_probs.columns))
ties_val_maxes = ties_val_probs.with_columns(max = pl.max_horizontal(ties_val_probs.columns))


dare_lin_test_maxes = dare_lin_test_probs.with_columns(max = pl.max_horizontal(dare_lin_test_probs.columns))
dare_lin_val_maxes = dare_lin_val_probs.with_columns(max = pl.max_horizontal(dare_lin_val_probs.columns))

gpt2_xl_test_maxes = gpt2_xl_test_probs.with_columns(max = pl.max_horizontal(gpt2_xl_test_probs.columns))
gpt2_xl_val_maxes = gpt2_xl_val_probs.with_columns(max = pl.max_horizontal(gpt2_xl_val_probs.columns))


base_test_maxes = base_test_probs.with_columns(max = pl.max_horizontal(base_test_probs.columns))
base_val_maxes = base_val_probs.with_columns(max = pl.max_horizontal(base_val_probs.columns))

base_4_test_maxes = base_4_test_probs.with_columns(max = pl.max_horizontal(base_4_test_probs.columns))
base_4_val_maxes = base_4_val_probs.with_columns(max = pl.max_horizontal(base_4_val_probs.columns))

#|%%--%%| <jphG4SNWO1|GWfy9fyqU3>

base_4_test_maxes.head()

#|%%--%%| <GWfy9fyqU3|vvCOfdMBjs>

# Plotting the distribution
# VAL 
plt.figure(figsize = (10, 6), dpi = 100)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.hist(ties_val_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "TIES merge", color='dodgerblue')
plt.hist(dare_lin_val_maxes.select("max"), bins = 20, density = True, alpha = 0.3, label = "DARE Linear merge", color='springgreen')
plt.hist(gpt2_xl_val_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "GPT2-XL", color='violet')
#plt.hist(ties_val_maxes.select("max"), bins = 30, alpha = 0.75, label = "gpt2-ties-merge (validation)", color='violet')
plt.title('Distribution of Maximum Values per Row (Validation)')
plt.xlabel('Max Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.savefig("notebooks/images/val_max_distributions.png")
plt.show()

#|%%--%%| <vvCOfdMBjs|UFAEAd68hQ>


# Plotting the distribution
# BASE MODELS
plt.figure(figsize = (10, 6), dpi = 100)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.hist(base_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "Fine-Tune 2", color='blue')
plt.hist(base_4_test_maxes.select("max"), bins = 20, density = True, alpha = 0.3, label = "Fine-Tune 4", color='orange')
plt.hist(gpt2_xl_val_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "GPT2-XL", color='violet')
plt.hist(ties_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "TIES Merge", color='dodgerblue')
plt.hist(dare_lin_test_maxes.select("max"), bins = 20, density = True, alpha = 0.3, label = "DARE Linear Merge", color='lightskyblue')
#plt.hist(ties_val_maxes.select("max"), bins = 30, alpha = 0.75, label = "gpt2-ties-merge (validation)", color='violet')
plt.title('Distribution of Base Maximum Values per Row (Testing)')
plt.xlabel('Max Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.savefig("notebooks/images/base_max_distributions.png")
plt.show()

#|%%--%%| <UFAEAd68hQ|IIwOj6odrR>


# Plotting the distribution
# TEST 
plt.figure(figsize = (10, 6), dpi = 100)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.hist(ties_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "TIES Merge", color='dodgerblue')
plt.hist(dare_lin_test_maxes.select("max"), bins = 20, density = True, alpha = 0.3, label = "DARE Linear Merge", color='lightskyblue')
plt.hist(gpt2_xl_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "GPT2-XL", color='violet')
plt.hist(ties_test_maxes.select("max"), bins = 30, alpha = 0.75, label = "gpt2-ties-merge (validation)", color='violet')
plt.title('Distribution of Maximum Values per Row (Testing)')
plt.xlabel('Max Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.savefig("notebooks/images/test_max_distributions.png")
plt.show()

#|%%--%%| <IIwOj6odrR|SbuiSHsCrv>

# counts, bins, _ = plt.hist(ties_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "TIES merge", color='dodgerblue')
# counts = (ties_test_maxes.select("max") / sum(ties_test_maxes.select("max"))) * 100
# 
# weights = np.ones_like(ties_test_maxes.select("max")) / len(ties_test_maxes.select("max")) * 100
# ties_test_maxes.shape
# print(weights.shape)
# plt.hist(, bins = bins, weights = weights, alpha = 0.5)
# # plt.hist(ties_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "TIES merge", color='dodgerblue')


#|%%--%%| <SbuiSHsCrv|klpftFZAqZ>



#|%%--%%| <klpftFZAqZ|2ZVXGw5sHA>
r"""°°°
# START LOOKING HERE
°°°"""
#|%%--%%| <2ZVXGw5sHA|OVrFUGCDdX>

true_labels = ties_test.select("true_label")
true_labels = true_labels.to_numpy()

#|%%--%%| <OVrFUGCDdX|LYQPCdwt1j>
print(true_labels[5][0])

inlabel_ties = []
for i, row in enumerate(ties_test.rows()):
    inlabel_ties.append(row[true_labels[i][0]])
   # row.select(str(i)) 

print(inlabel_ties)

#|%%--%%| <LYQPCdwt1j|ypshuD3Ocy>

print(inlabel_ties[:10])

#|%%--%%| <ypshuD3Ocy|vWPz8sDGtQ>

first_ten = ties_test.select("true_label")[:10].to_numpy()
first_ten[0][0]

ties_test.rows()[1][first_ten[1][0]]
ties_test.head(2)
for i, rows in enumerate(ties_test.rows()):
    print(row[])


#|%%--%%| <vWPz8sDGtQ|t0kM7yAz5G>

true_labels = base_test.select("true_label").to_numpy()
inlabel_base_2 = []
for i, row in enumerate(base_test.rows()):
    inlabel_base_2.append(row[true_labels[i][0]])
#|%%--%%| <t0kM7yAz5G|X272jg2h0G>

true_labels = dare_lin_test.select("true_label").to_numpy()
inlabel_dare_lin = []
for i, row in enumerate(dare_lin_test.rows()):
    inlabel_dare_lin.append(row[true_labels[i][0]])

#|%%--%%| <X272jg2h0G|ix8r2rWPfm>



#|%%--%%| <ix8r2rWPfm|528FBLOXZD>

# FIXED 
plt.figure(figsize = (10, 6), dpi = 100)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.hist(inlabel_ties, bins = 50, alpha = 0.3, label = "TIES Merge", color='dodgerblue')
# plt.hist(inlabel_base_2, bins = 20, alpha = 0.3, label = "Fine-Tune 2", color='blue')
# plt.hist(inlabel_dare_lin, bins = 20, alpha = 0.5, label = "DARE Linear Merge", color='springgreen')
plt.title('Distribution of Values Conditioned on True Label (Testing)')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
# plt.savefig("notebooks/images/inlabel_ties.png")
plt.show()

#|%%--%%| <528FBLOXZD|C3v4ZYezvu>
r"""°°°
## Plot where we started zooming in hella and adding hella bins
°°°"""
#|%%--%%| <C3v4ZYezvu|mVn6FMzKXR>

# plotting on different plots
plt.figure(figsize = (10, 6), dpi = 100)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(inlabel_base_2, alpha = 0.3, label = "Fine-Tune 2", color='blue')
plt.hist(inlabel_ties, alpha = 0.3, label = "TIES Merge", color='green')
# plt.hist(inlabel_dare_lin, bins = 20, alpha = 0.5, label = "DARE Linear Merge", color='springgreen')
plt.xlim([0.0, 0.00025])
plt.title('Distribution of Values Conditioned on True Label (Testing)')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.savefig("notebooks/images/zoom.png")
plt.show()

#|%%--%%| <mVn6FMzKXR|U3WfEp1aGO>


# plotting on different plots
plt.figure(figsize = (10, 6), dpi = 100)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.hist(inlabel_base_2, bins = 20, alpha = 0.3, label = "Fine-Tune 2", color='blue')
plt.hist(inlabel_dare_lin, bins = 20, alpha = 0.5, label = "DARE Linear Merge", color='springgreen')
plt.title('Distribution of Values Conditioned on True Label (Testing)')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.savefig("notebooks/images/inlabel_ties.png")
plt.show()

#|%%--%%| <U3WfEp1aGO|p2l2p1l1Zm>



#|%%--%%| <p2l2p1l1Zm|vBlmdeKn7v>

ties_test.columns

#|%%--%%| <vBlmdeKn7v|B4HRe6dplS>

# create dataframes with only the probabilities
ties_test_probs = get_true_probabilities(ties_test, 10)
print(ties_test_probs.columns)
ties_val_probs = ties_val.select(pl.exclude("true_label"))

dare_lin_test_probs = dare_lin_test.select(pl.exclude("true_label"))
dare_lin_val_probs = dare_lin_val.select(pl.exclude("true_label"))

gpt2_xl_test_probs = gpt2_xl_test.select(pl.exclude("true_label"))
gpt2_xl_val_probs = gpt2_xl_val.select(pl.exclude("true_label"))


base_test_probs = base_test.select(pl.exclude("true_label"))
base_val_probs = base_val.select(pl.exclude("true_label"))


base_4_test_probs = base_4_test.select(pl.exclude("true_label"))
base_4_val_probs = base_4_val.select(pl.exclude("true_label"))


#|%%--%%| <B4HRe6dplS|ztH4Feg8in>









