# Research plan

Only decisions we have agreed on go in this file.

**Status (2026-10-14):** goal, ground rules, research question and first specifications agreed.
Next up: your predictions for Parts A and B (Rule 6), then the remaining specifications, then
the steps and the plan.

## Goal
*Agreed 2026-10-14*

Learn to implement machine learning by doing it, learn how Jupyter notebooks work, and use this
project as a stepping stone into research like the work done at the UW Allen School. By the end,
I can explain what is happening underneath every part of it.

**Context:** my starting point is basic neural networks. I have about 10 hours of my own time,
spread over a few days. The computer can also run overnight when that helps.

## Ground rules
*Agreed 2026-10-14. "You" is the researcher; "the assistant" is the AI assistant.*

**Learning while doing**
1. **Split the code by what's worth learning.** You write the core pieces, the 2–3 small
   functions that *are* the experiment. The assistant writes routine setup code (data download,
   plotting), and you read it until you can say what it does.
2. **Open the black box once.** For the project's central idea, you write a tiny version by hand
   once and check that it matches the library's version. For everything else you use the
   library, but you can say in one sentence what it does.
3. **Tiny first.** New code runs on a few images before it runs on the full dataset.

**Notebooks**

4. **One numbered notebook per step.** Each starts with the question it answers and ends with a
   few lines in your own words: what happened, what you learned, what still confuses you. Code
   that's used in more than one notebook goes into a small `.py` file.
5. **A notebook only counts as done once "Restart & Run All" works.**

**Research habits**

6. **Predict before you run.** Write one or two sentences before each experiment.
7. **Keep the test set locked away.** Choices are made on a separate validation split, never on
   the test set.
8. **Compare against a baseline, and repeat the main result** (for example with 3 random seeds)
   before believing it.
9. **End with a one-page write-up** covering the question, the method, one or two key figures,
   the result and its limitations.

**Working together**

10. **You set the pace.** The assistant proposes one step at a time and waits for your OK.
11. **Timebox each session.** Every session ends with something that runs and something you
    understand. If a step takes too long, we shrink the experiment, not the explanation.

## Research question
*Agreed 2026-10-14. Builds on Hooker et al. 2019 (*What Do Compressed Deep Neural Networks Forget?*).
This replaces the earlier idea of artificial image degradations.*

**In one sentence:** when pruned, do ViTs, like CNNs, lose accuracy disproportionately on the
images that humans find hard?

- **Main question:** Does pruning hurt a CNN and a ViT differently on images that humans find
  ambiguous?
- **Part A (reproduce):** In a CNN (ResNet-18), does pruning reduce accuracy *more* on
  human-ambiguous CIFAR-10 images than on clear ones? This is a conceptual replication: Hooker et
  al. found their hard images through disagreement between pruned and unpruned models, and we
  use human judgements instead.
- **Part B (extend):** Does the same happen in a ViT (DeiT-Tiny), and is its gap between ambiguous
  and clear images larger or smaller than the CNN's?
- **Bonus (exploratory, only if time allows):** Are the images that pruning hurts most the same
  ones that humans find ambiguous?

*Left to the specifications:* how much is pruned and how; where the line between ambiguous and
clear is drawn; whether "more" is measured in percentage points or relative to starting accuracy;
whether the comparison is ambiguous vs clear or ambiguous vs all images.

## Specifications

### Data
- **Decided 2026-10-14:** CIFAR-10. "Unclear" images are defined by human disagreement, using CIFAR-10H (about 50 human labels per CIFAR-10 test image).

### Models
- **Decided 2026-10-14:** ResNet-18 (CNN) vs DeiT-Tiny (ViT), both fine-tuned from ImageNet-1k
  pretrained weights.

### Tools
- **Decided 2026-10-14:** notebooks run in VS Code (Jupyter extension), with an AI assistant in the sidebar used under Rule 1.
- **Decided 2026-10-14:** the project lives in this repo (`image-classification`). The Fashion-MNIST warm-up moved to `~/Developer/fashion-mnist`.

## Steps

## Plan

## Next project
The week after this project: read *Ensembling Sparse Autoencoders* (Gadgil, Lin, Lee; ICML 2026;
arXiv 2505.16077) and plan an extension to vision models, which the paper names as future work.
