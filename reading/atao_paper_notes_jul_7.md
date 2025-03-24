# ZIPIT

[https://arxiv.org/pdf/2305.03053](paper)

Notes:

- Existing literature permutes one model into the space of the other models, then averages them
- Zipit generalizes for two arbitrary models with the same architecture
- Achieve merging by concatenating one layer from both models to get one long concatenated feature
  then define a merge matrix $$M_i \in \mathbb{R}^{n_i \times 2n_i}$$ such that
  the new feature is gnerated with $M$.
- $M$ averages any matched features. Every row vector in $M$ is zeros except at the two indices
  where there is a match
- Use M to zip two layers, zero and average matches, then use U to unzip and feed forward
- Demonstrates effective results by evaluating the merged model on each individual task
  the original models are designed for.

Questions:

- Still not entirely clear on how they do the matching to generate matrix $M$
- How does their model compare to each individual fine-tuned model?
- So when they "propagate" $M$ and $U$, are they just keeping them in memory and ignoring any
  layers that don't have weights?

# CDEvalSumm

[https://arxiv.org/pdf/2010.05139](paper)

Notes:

- results of evaluation are defined as $\textbf{r} = U \in \mathbb{R}^{N \times N}$. Each entry in $U$
  represents the scalar result $r = \mathrm{eval}(D, S, m)$ when summarize $S$ is trained on dataset $D_i$
  and tested on $D_j$.
- So stiffness as a metric is just how well some system performs in all cross-datasets, and stableness
  is the variance of a system between in-dataset and cross-dataset
- Could be an additional resource to learn about summarization models [https://aclanthology.org/D18-1208v2.pdf](Kedzie, McKeown, Daume III)

Questions:

- Are the summarizers being trained on datasets? I thought that these summarizers like $Trans_{non}$
  are using pre-trained transformers. So do they mean fine-tuning when they say training? I feel like training
  all of these summarizers would take forever.
- What exactly is a "system"? Is it just the summarizer being used? And a summarizer is just a deep learning
  model trained on summarization right?
- Even if stiffness and stableness are not unanimous, do they have some predictable relationship?

# OOD

[https://aclanthology.org/2024.lrec-main.720.pdf](paper)

Notes:

- _I feel like this paper doesn't relate too much to our work since it's about OOD detection in AI safety
  as opposed to evalauting models on OOD data_
- So OOD just means inputs into an LLM that are not appropriate based on what the LLM was originally
  trained on
- Far-OOD is already easily detected by LLMs. The issue is near-OOD, which gets improved with fine-tuning.

Questions:
