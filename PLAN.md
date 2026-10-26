# Research plan

Only decisions we have agreed on go in this file.

**Status (2026-10-25): project complete.** You asked the assistant to "ignore previous limits/learning
goals" and finish the whole project, so the assistant did the remaining steps alone (decisions in
*Decided by the assistant, 2026-10-25* below): notebooks `01_vit.ipynb`, `02_cnn.ipynb` and
`03_test.ipynb` all run top to bottom; results are in `results/`; the 4-page paper is
`paper/paper.pdf`. Next up: read the paper and notebooks, put your name in `paper/paper.typ`,
rewrite each notebook's "What happened" in your own words, then the next project.

## Goal
*Agreed 2026-10-14*

Learn to implement machine learning by doing it, learn how Jupyter notebooks work, and use this
project as a stepping stone into research like the work done at the UW Allen School. By the end,
I can explain what is happening underneath every part of it.

*Added 2026-10-15, in my own words:*
- Have practice doing/making a "real" research paper.
- Learn how to use Jupyter notebooks and become more proficient using PyTorch.
- Have a tangible result I can show people.
- Figure out what next steps of research I can pursue/take.
- Be able to explain every decision made in the research paper.

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
   *Amended 2026-10-15:* for time, each condition runs once (one seed). The write-up lists this
   as a limitation.
9. **End with a one-page write-up** covering the question, the method, one or two key figures,
   the result and its limitations.
   *Amended 2026-10-15:* instead, a **4-page workshop-style paper** (plus references) with the
   standard sections: abstract; introduction and related work; method; results (one or two
   figures, each prediction checked); limitations and next steps. Why: one page has no room for the
   method, which is where every decision gets explained; a full 8+ page paper would need more seeds,
   datasets and ablations than this project has. Workshops at ML conferences accept papers of this
   length for preliminary results like these.

**Working together**

10. **You set the pace.** The assistant proposes one step at a time and waits for your OK.
11. **Timebox each session.** Every session ends with something that runs and something you
    understand. If a step takes too long, we shrink the experiment, not the explanation.

## Research question
*Agreed 2026-10-14. Builds on Hooker et al. 2019 (*What Do Compressed Deep Neural Networks Forget?*).
This replaces the earlier idea of artificial image degradations.*

**In one sentence:** when pruned, do ViTs, like CNNs, lose accuracy disproportionately on the
images that humans find hard?

**Contribution** *(added 2026-10-15)***:** Hooker et al. defined hard images by pruning itself
(images whose prediction changes when the model is pruned) and asked humans only afterwards. We
choose the hard images from human judgements (CIFAR-10H), before and independently of any model,
and then test whether pruning hurts them more, in a CNN and in a ViT. *Limitation to keep in mind:*
human-ambiguous images are also harder for the unpruned model, so its starting accuracy on them is
already lower.

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

## Experiment description
*Written by me 2026-10-15, corrected where necessary. This is the text used in the final paper.*

> The experiment compares two machine learning model types, ViTs and CNNs, and measures whether
> the distribution of their failure rates is the same on harder images as on easier images.
>
> The experiment does this by quickly fitting available vision models to the data, and then
> progressively pruning the models with two different pruning methods, magnitude pruning and random
> pruning. Afterwards, we will measure whether degradation, in percentage points of accuracy, is the
> same on standard and hard images, and compare the two models.
>
> The null hypothesis is that models degrade uniformly: standard and hard images lose the same
> number of percentage points of accuracy. The alternative hypothesis is that both types of models
> degrade more on harder images after pruning.
>
> Moreover, we are also looking to see if one type of architecture has more uniform degradation
> than the other.

## Predictions
*Written 2026-10-15, before any experiment (Rule 6). In my own words:*

> I believe that CNN's would have double the loss on hard problems, while ViT's 1.5 times the
> loss. But I also believe that ViT's would have an even lower relative effect when we use random
> pruning, and random pruning will be worse per percentage of weights active but at the same time
> the relative difference from hard and not as hard problems won't be as big for both.

*Measurable version, agreed 2026-10-15.* "Ratio" means (accuracy drop on ambiguous images) ÷
(accuracy drop on clear images), with drops in **percentage points**, compared at a matched overall
accuracy drop.

*Confirmed 2026-10-15:* drops are measured in percentage points, not relative to starting accuracy.
- Why: ambiguous images start with lower accuracy. Dividing a drop by starting accuracy would make
  their drop look bigger and tilt the result toward the predictions, and dividing by starting error
  would tilt it the other way. Percentage points divide by neither: they count how many images out
  of 100 each group lost.
- Known weakness: accuracy can't fall below chance (10%), so at the heaviest pruning levels
  ambiguous images have less room to fall. This shows up on the plot.

| # | Prediction | Measured as |
|---|---|---|
| 1 | CNN, magnitude pruning: hard images lose 2× as much | ResNet-18 ratio ≈ 2 |
| 2 | ViT, magnitude pruning: 1.5× | DeiT-Tiny ratio ≈ 1.5 |
| 3 | Random pruning is worse per weight kept | at the same sparsity, both models lose more overall accuracy with random than with magnitude pruning |
| 4 | Random pruning narrows the gap for both, and the ViT's stays smaller | with random pruning both ratios are closer to 1 than with magnitude pruning, and the ViT's ratio is still below the CNN's |

*Checked 2026-10-25 (notebook 03), after the fine-tune at the matched overall drop D\* = 12.1
points:* ResNet-18 ratio 1.67 [1.38, 1.97] (#1 not supported), DeiT-Tiny 1.90 [1.61, 2.21] (#2 not
supported: larger, not smaller), random pruning worse at every sparsity (#3 supported), random
ratios 1.57 and 1.54 vs magnitude 1.67 and 1.90 (#4 supported by the pre-set rule, but the ViT–CNN
part is a tie). Uniform degradation is rejected for both models and both methods.

## Specifications

### Data
- **Decided 2026-10-14:** CIFAR-10. "Unclear" images are defined by human disagreement, using CIFAR-10H (about 50 human labels per CIFAR-10 test image).
- **Decided 2026-10-15:** an image's *human agreement* is the share of CIFAR-10H annotators who
  chose its CIFAR-10 label (using the authors' cleaned `cifar10h-counts.npy`, which leaves out 16
  inattentive annotators). **Ambiguous** = every test image with agreement ≤ 86.8%: 1,011 images,
  the bottom ~10%, with ties included rather than broken at random. **Clear** = the other 8,989.
  - Why: the agreement distribution has no natural break (44% of images have 100% agreement; 12%
    are below 90%, 6% below 80%, 1.2% below 50%), so the cut is a convention fixed before any
    model is run. The bottom 10% already contains every strongly ambiguous image, gives a stable
    accuracy figure, and avoids a group made mostly of mislabeled images the unpruned model already
    gets wrong. Including mildly ambiguous images can only shrink the gap, so the choice is
    conservative.
- **Decided 2026-10-15:** validation split (Rule 7): 5,000 of the 50,000 training images, 500 per
  class, drawn with a fixed seed and saved as an index file so every run uses the same split. All
  models train on the other 45,000. Validation is used to choose the base-model recipe and to check
  that each pruned model recovers after its fine-tune. The test set and CIFAR-10H are loaded only
  for the final evaluation.
  - Why: CIFAR-10H covers only the test set, so validation can't be split into ambiguous and clear;
    it is for training choices only, and the ambiguous-vs-clear gap is measured once, on test.
    Retraining on all 50,000 afterwards was skipped: one more run and a harder-to-follow setup
    for about 1% more data.

### Models
- **Decided 2026-10-14:** ResNet-18 (CNN) vs DeiT-Tiny (ViT), both fine-tuned from ImageNet-1k
  pretrained weights.

### Pruning
- **Decided 2026-10-15:** the main method is global unstructured magnitude pruning (the smallest-magnitude
  weights across the whole network are set to zero), applied to both models. It is the same family
  as Hooker et al. and means the same thing for conv and linear layers.
- **Decided 2026-10-15:** random pruning (the same fraction of weights, chosen at random) is the control.
- **Decided 2026-10-15:** the CNN and the ViT are compared at **matched overall accuracy loss**, not at matched
  sparsity. For each model we sweep sparsity and plot the ambiguous-vs-clear gap against the overall
  accuracy drop. This keeps "the method suits one architecture better" from biasing the comparison.
- **Decided 2026-10-15:** in both models only conv and linear weights are pruned. Biases,
  normalization layers (BatchNorm, LayerNorm), the first layer (ResNet's first conv, the ViT's patch
  embedding) and the last layer (the 10-class output) are left alone, as is common practice.
- *Optional, only if time allows:* repeat with a structured method (pruning whole heads or channels).

### Fine-tuning after pruning
- **Decided 2026-10-15:** each pruned model is fine-tuned for 1 epoch on the training set, with the
  pruned weights held at zero. It is tested twice: right after pruning (the damage pruning does on
  its own) and after the fine-tune (what it still gets wrong after recovering). The after-fine-tune
  test is the main result, matching Hooker et al.
- **Decided 2026-10-15:** before any pruning, each pretrained model gets a new 10-class output layer
  and is fine-tuned on CIFAR-10 for a few epochs. This base model is the starting point for every
  pruning run.
- **Decided 2026-10-15:** the unpruned (0%) model also gets the same 1 extra epoch, so the extra
  training is not mistaken for an effect of pruning.

### Training recipe
- **Decided 2026-10-15:** one recipe for both models, so the only difference between them is the
  architecture. The optimizer and weight decay shape the spread of weight sizes, which is exactly
  what magnitude pruning acts on.
  - **Images:** 32×32 upscaled to 224×224, the size both models were pretrained at (at 32×32 the
    ViT would see only 4 patches). Normalized with ImageNet mean and std.
  - **Augmentation:** random crop (4-pixel padding) and horizontal flip at 32×32, before the
    upscale.
  - **Optimizer:** AdamW, peak learning rate 1e-4, weight decay 0.05, batch size 128.
  - **Schedule:** short linear warm-up, then cosine decay to 0.
  - **Base model:** 3 epochs on the 45,000 training images.
  - **Fine-tune after pruning:** 1 epoch, same optimizer and peak learning rate, with the warm-up
    and cosine decay fitted into that epoch.
- **Decided 2026-10-15:** no hyperparameter search. Validation accuracy is checked after every
  base-model epoch. Only if a model ends below ~94%, or is still clearly improving at epoch 3, is
  one thing changed (learning rate or epochs), judged on validation only.
- **Decided 2026-10-15:** if the first timing run (Rule 3) shows the runs won't fit, the image size
  drops to 128×128 for both models.

### Sparsity levels and runs
- **Decided 2026-10-15:** 0, 30, 50, 70, 90, 95% of weights pruned.
- **Decided 2026-10-15:** one seed per condition (see the amendment to Rule 8).
- That makes 2 base-model runs plus 22 short runs (2 models × 2 methods × 5 pruned levels, plus
  the 2 unpruned models), estimated at 2–4 hours.

### Tools
- **Decided 2026-10-14:** notebooks run in VS Code (Jupyter extension), with an AI assistant in the sidebar used under Rule 1.
- **Decided 2026-10-14:** the project lives in this repo (`image-classification`). The Fashion-MNIST warm-up moved to `~/Developer/fashion-mnist`.

## Steps
*Decided 2026-10-15. Steps after these are decided when we get there.*

1. **ViT, together.** Load pretrained DeiT-Tiny, train it into the base model, check it on the
   validation split (Rule 7: the test set stays locked), and prune it.
2. **Decide what to do with the CNN.**
3. **The rest:** The assistant does it and you double-check it.

## Decided by the assistant, 2026-10-25
*At your request ("Continue and complete the whole project"); not agreed step by step like the rest.*

- **The CNN (Step 2):** same recipe, sparsities and notebook structure as the ViT (`02_cnn.ipynb`),
  so the architecture is the only difference. ResNet-18 weights come from timm's `resnet18.tv_in1k`,
  torchvision's original ImageNet checkpoint, because torchvision's own download failed on an SSL
  certificate error. Skipped layers: `conv1` and `fc`.
- **Shared code:** the prune-and-fine-tune loop and its plot moved to `sweep.py`, model creation to
  `models.py`, test-set and CIFAR-10H loading to `data.py`, so notebooks 01 and 02 run the same code.
- **Base models:** both cleared the 94% bar with less than half a point gained in epoch 3
  (DeiT-Tiny 96.5%, ResNet-18 95.0% validation), so the recipe was not changed.
- **Analysis rules** (fixed in `03_test.ipynb` before it first ran on the real models): ratios
  only where the clear-image drop is ≥ 1 point; 95% intervals from a bootstrap over test images
  (2,000 resamples, within each group, the same for every model); D\* = the largest overall drop all
  four model–method curves reach, with linear interpolation between sparsity levels; verdicts:
  #1–2 supported if the predicted ratio lies in the interval, #3 if every random-minus-magnitude
  difference is above 0, #4 on point estimates.
- **Added after seeing the results** (labelled as such in notebook 03 and the paper): the
  chance-floor lines in Figure 2, and the bonus analysis repeated at matched overall drop, because at
  95% DeiT-Tiny has collapsed.
- **Robustness checks** (written with the analysis rules): a class-matched gap and the loss rate
  among images the reference gets right.
- **Paper:** written in Typst (no LaTeX on this Mac); `paper/make_table.py` builds Table 1 from
  `results/drops.csv`. Build command at the top of `paper/paper.typ`.

## Plan

## Next project
The week after this project: read *Ensembling Sparse Autoencoders* (Gadgil, Lin, Lee; ICML 2026;
arXiv 2505.16077) and plan an extension to vision models, which the paper names as future work.
