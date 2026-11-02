# Toxic Prompt Classification with `BAAI/bge-base-en-v1.5`

**Notebook:** `classification_v3_bge_cv.ipynb` · **Owner:** Riya · **Result:** test macro F1 **0.868** (target was 0.80)

## What this notebook does

Our system first **detects** whether a prompt is harmful. This notebook handles the next step, **classification**: given a harmful prompt, which of 10 categories does it belong to?

| | | |
|---|---|---|
| Disinformation | Economic harm | Expert advice |
| Fraud/Deception | Government decision-making | Harassment/Discrimination |
| Malware/Hacking | Physical harm | Privacy |
| Sexual/Adult content | | |

The notebook takes a pretrained language model, trains it to recognize these 10 categories, picks the best of four training methods, and reports how well the final model does on prompts it has never seen.

## Data

- **Dataset:** [JailbreakBench `JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors): 100 harmful + 100 benign prompts, 20 per category. Benign prompts are included because they share each category's topic, which gives the model more examples to learn from.
- **Split:** 160 training prompts / 40 test prompts (4 per category), using `TEST_SIZE = 0.2`, `SEED = 42` and stratified sampling.
- **The test set is the same as in every teammate's classification notebook**, so the scores can be compared directly. The 40 test prompts are never used for training or for any decision. They are scored once, at the very end.

## Results

Scores on the 40 test prompts:

| Model / method | Test macro F1 | Test accuracy |
|---|---|---|
| `unitary/unbiased-toxic-roberta`, standard fine-tuning (assigned model; best grid-search run) | ~0.72 | n/a |
| **`BAAI/bge-base-en-v1.5`, ensemble D (this notebook)** | **0.868** | **0.875** |

- **Accuracy on harmful prompts only:** 0.95 (19 of 20). These are the prompts the detection stage actually passes to this step.
- **Per category:** 6 categories are perfect or nearly perfect (F1 ≥ 0.86). The weakest are **Fraud/Deception** and **Privacy** (2 of 4 correct each). Their errors went to Government decision-making and Malware/Hacking.
- **Macro F1** averages F1 across the 10 categories equally, so a model can't score well by getting only the common categories right.

## How it works

### The pretrained model

`bge-base-en-v1.5` (BAAI, 110M parameters, BERT architecture) is a **sentence-embedding** model. It turns a sentence into 768 numbers so that sentences with similar meaning get similar numbers, even without shared words. "Write a keylogger" and "build software that records keystrokes" land close together. We start from this pretrained knowledge and only teach it our 10 categories. This is called *transfer learning*.

### Four ways of training it

| | Approach | What it does |
|---|---|---|
| **A** | Frozen embeddings + logistic regression | bge is not trained at all. Its embeddings feed a small, simple classifier. |
| **B** | Contrastive fine-tuning + logistic regression | bge is lightly trained to pull same-category prompts closer together. A small classifier sits on top. This method is designed for small datasets. |
| **C** | Standard fine-tuning, 3-seed average | A classification layer is added and the whole model is trained, using small learning rates for lower layers, a higher rate for the new layer, and label smoothing. Three copies trained with different random seeds are averaged. |
| **D** | Ensemble of B + C | Averages B's and C's predicted probabilities (50/50). B and C make different mistakes, so averaging cancels some of them out. |

### Choosing the winner without touching the test set

1. **5-fold cross-validation on the 160 training prompts.** The training prompts are split into 5 groups of 32. Each approach is trained 5 times, each time on 4 groups (128 prompts), and predicts the 5th group (32 prompts). Every training prompt gets predicted once by a model that never trained on it. Cross-validation also chooses each approach's settings: B's and C's number of training epochs and D's mix.
2. **Pick the approach with the best cross-validated F1:**

   | Approach | Cross-validated F1 (160 training prompts) |
   |---|---|
   | A | 0.866 |
   | B (3 epochs) | 0.887 |
   | C (8 epochs) | 0.880 |
   | **D (50% B + 50% C)** | **0.893** |

3. **Retrain the winner on all 160 training prompts.**
4. **Score the 40 test prompts once.** Result: F1 0.868, close to the cross-validated 0.893, so it isn't a lucky test split.

An earlier version picked the winner using a 24-prompt validation set. At that size one prompt changes F1 by about 4 points, so the choice was mostly luck. Cross-validation scores each approach on all 160 prompts instead.

## Why it beat `unbiased-toxic-roberta`

**1. The pretrained model already does a task close to ours.** `unbiased-toxic-roberta` was pretrained to judge *how toxic* text is (insults, threats, identity attacks). Our task is to sort prompts by *topic*, and toxicity says little about whether a prompt is about malware or fraud. `bge` was pretrained to group sentences by *meaning*, which is close to topic classification. Approach A shows this: bge with **no training at all** reached 0.866 cross-validated F1, above what full fine-tuning of the toxicity model achieved.

**2. The data is tiny, and the methods were chosen for that.** There are only 16 training prompts per category. Standard fine-tuning (the method in the roberta notebook) has to learn a new classification layer from scratch with so few examples, and results swing a lot between runs. Approaches B and D are designed for small datasets, and averaging several models smooths out the swings.

**3. Fairer settings selection.** Cross-validation picked the training length: 3 epochs for B and 8 for C. Training longer made both worse. The roberta grid search ran 10–20 epochs.

**About the comparison:** the roberta ~0.72 is the best of a grid search that was **scored on the test set**. Picking the best of many test scores inflates it, so the true gap is, if anything, larger than shown.

## How to run

1. Open the notebook in Google Colab and select `Runtime → Change runtime type → T4 GPU`.
2. Run all cells. It takes about **40–60 minutes**, most of it approach C's cross-validation.
3. **"Test F1 to report"** is printed in Step 8.

Settings (model name, epochs, learning rates, seeds) are all in Step 2. Keep `SEED`, `TEST_SIZE` and `USE_BENIGN` unchanged so the test set stays the same as the team's.

**Expected warnings (harmless):** `num_labels=10 is incompatible with id2label of length 1`, and `classifier.weight / classifier.bias MISSING`. bge has no classification layer, so the notebook adds a new one and trains it.

## Outputs

| File / folder | Contents |
|---|---|
| `classification_results.csv` | One results row in the team's shared format: model, approach, accuracy, macro precision/recall/F1, harmful-slice accuracy |
| `classification_bge_v3_cv.csv` | Cross-validation scores for approaches A–D |
| `bge_contrastive_final/` | Approach B's trained encoder |
| `bge_finetuned_final/` | Approach C's fine-tuned model (seed 42) |

Files are lost when the Colab session ends unless Google Drive is mounted or they are downloaded.

## Trying your own prompts

Step 10 shows predictions for the prompts listed in `CUSTOM_PROMPTS` (Step 2). For interactive testing after a full run, add the two "Step 12" cells: one loads the final ensemble, then `classify(["your prompt", ...])` prints the top 3 categories with confidence scores. They must run in the same Colab session as the full notebook.

## Limitations

- **Small test set.** 40 prompts means each mistake costs about 2.5 accuracy points, so scores would likely shift by a few points on a different split.
- **One dataset.** All prompts come from JailbreakBench and share its writing style. The score shows the model handles new JailbreakBench-style prompts. It does not show how it handles real user prompts phrased differently.
- **Not the assigned model.** The assigned model for this task is `unitary/unbiased-toxic-roberta` (~0.72 F1). `bge-base-en-v1.5` is reported as a stronger alternative.
- **Slower to run.** The final ensemble runs 4 models per prompt (one B, three C), so it is slower and uses more memory than a single model.
