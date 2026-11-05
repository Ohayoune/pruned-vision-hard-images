// Workshop-style paper (PLAN.md, Rule 9 as amended): 4 pages plus references.
// Build (from the repo root): .venv/bin/python paper/make_table.py && .venv/bin/python -c "import typst; typst.compile('paper/paper.typ', output='paper/paper.pdf', root='.')"
// Numbers come from results/summary.json and results/drops.csv (notebook 03).

#let author = "Eli Ohayon"
#let affiliation = "Independent research project"

#set document(title: "Do Pruned Vision Transformers Also Forget What Humans Find Hard?", author: author)
#set page(paper: "us-letter", margin: (x: 1in, y: 0.95in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, spacing: 0.9em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.2em, below: 0.7em, text(size: 11pt, weight: "bold", it))
#show heading.where(level: 2): it => block(above: 1em, below: 0.6em, text(size: 10pt, weight: "bold", it))
#show figure.caption: set text(size: 8.5pt)
#show figure: set block(above: 1em, below: 1em)
#set table(stroke: none, inset: (x: 4pt, y: 2.2pt))
#show table: set text(size: 8.5pt)
#show figure.where(kind: table): set figure.caption(position: top)
#show bibliography: set text(size: 8.5pt)

#align(center)[
  #text(size: 15pt, weight: "bold")[Do Pruned Vision Transformers Also Forget \ What Humans Find Hard?]
  #v(0.6em)
  #text(size: 10.5pt)[#author] \
  #text(size: 9pt, style: "italic")[#affiliation]
  #v(0.8em)
]

#block(inset: (x: 0.4in))[
  #align(center, text(weight: "bold")[Abstract])
  #set text(size: 9pt)
  #set par(justify: true)
  Pruning removes most of a network's weights at little cost in average accuracy, but earlier work
  found that the cost falls mostly on hard, often ambiguous images. That work identified its hard
  images through pruning itself and studied only CNNs. We fix the hard set in advance from human
  labels: the 1,011 CIFAR-10 test images with the lowest agreement among CIFAR-10H annotators. We
  prune ResNet-18 and DeiT-Tiny with global magnitude pruning and a random-pruning control,
  fine-tune each pruned model for one epoch, and compare the accuracy lost on ambiguous and clear
  images at a matched overall loss. At an overall drop of 12 points, ambiguous images lose 1.7×
  (ResNet-18, 95% CI 1.4--2.0) and 1.9× (DeiT-Tiny, 1.6--2.2) as many points as clear images. We
  find no detectable difference between the architectures, although DeiT-Tiny reaches that loss at
  70% sparsity and ResNet-18 only at 95%. The gap shrinks but survives when ambiguous images are
  compared with clear images of the same class. It is larger still among images the unpruned
  model classified correctly. Where it can be checked, it appears during the recovery fine-tune
  rather than at pruning. Near
  chance accuracy it reverses, because ambiguous images have less room to fall.
]

= Introduction

Pruning removes most of a network's weights at a small cost in top-line accuracy
@han2015learning @gale2019state @blalock2020state. Hooker et al. @hooker2019forget showed that the
small average cost is concentrated: the images whose predictions change under pruning
(_pruning-identified exemplars_) are dominated by atypical, low-quality, ambiguous or mislabeled
images. But they found these hard images through pruning itself, and they studied only CNNs.
Vision transformers (ViTs) @dosovitskiy2021vit @touvron2021deit process images differently, with
global attention from the first layer and weaker spatial priors @raghu2021vision, and they are
increasingly pruned for deployment @chen2021chasing. Whether they forget the same images is unknown.

We ask: *when pruned, do ViTs, like CNNs, lose accuracy disproportionately on the images that
humans find ambiguous?* We fix the hard set from CIFAR-10H human labels @peterson2019human before
any model is trained. We prune ResNet-18 @he2016resnet and DeiT-Tiny @touvron2021deit with global
magnitude pruning and a random control, and compare the drops on ambiguous and clear images at
matched overall accuracy loss, against predictions written before any experiment. We contribute
(i) a conceptual replication of Hooker et al.'s question with a model-independent, human-defined
hard set; (ii) a CNN--ViT comparison at matched accuracy loss; and (iii) checks that separate
ambiguity from class mix and starting accuracy. Both architectures lose disproportionately on
ambiguous images, by similar factors. Where it can be measured, the disproportion appears during
the recovery fine-tune rather than at pruning.

*Related work.* Global magnitude pruning @han2015learning is a strong baseline @gale2019state
@blalock2020state, and random pruning is the standard control @liu2019rethinking
@frankle2019lottery. Hooker et al. @hooker2019forget @hooker2020bias showed that pruning's errors
concentrate on particular images and groups. Example difficulty is usually measured from models
@baldock2021difficulty. CIFAR-10H @peterson2019human measures it from people. ViT pruning studies
@chen2021chasing report accuracy and efficiency, not which images are lost.

= Method

*Data and the human-defined hard set.* CIFAR-10 @krizhevsky2009cifar has 50,000 training and
10,000 test images in 10 classes. We hold out 5,000 training images (500 per class, fixed seed) as
a validation split, used only to check training, and train on the remaining 45,000. CIFAR-10H
@peterson2019human gives 47--63 human labels per test image (median 51; we use the authors'
cleaned counts). An image's _human agreement_ is the share of annotators who chose its CIFAR-10
label. We call an image *ambiguous* if its agreement is at most 86.8%: the bottom ≈10% of the test
set, with ties kept, giving 1,011 ambiguous and 8,989 *clear* images (@fig-data). The
distribution has no natural break (44% of images have 100% agreement), so the cut is a convention,
fixed before any model was run. The test set and CIFAR-10H were opened once, for the final
evaluation.

#figure(
  image("../results/figures/fig1_data.pdf", width: 100%),
  caption: [Human agreement on the CIFAR-10 test set (left, log scale; 1,011 ambiguous and 8,989 clear images; dashed line: the 86.8%
  cut) and random examples from each group (right), titled with the CIFAR-10 label and its share of
  human votes and, for ambiguous images, the most popular other class.],
) <fig-data>

*Models and training.* ResNet-18 (11.2M parameters, torchvision's ImageNet weights) and DeiT-Tiny
(5.5M parameters, ImageNet-1k weights) are loaded through timm @wightman2019timm, each with a new
10-class output layer. Both use one recipe, so the architecture is the only difference. This
matters because the optimizer and weight decay shape the weight magnitudes that magnitude pruning
acts on. Images get a random crop (4-pixel padding) and a horizontal flip at 32×32, are upscaled
bilinearly to the 224×224 pretraining resolution, and are normalized with ImageNet statistics. We
fine-tune for 3 epochs with AdamW @loshchilov2019adamw (peak learning rate $10^(-4)$, weight decay
0.05 except on biases and normalization layers, batch size 128, 5% linear warm-up, then cosine
decay to 0), with no hyperparameter search.

*Pruning.* Global unstructured magnitude pruning zeroes the weights with the smallest $|w|$ across
all prunable layers at once. Random pruning zeroes as many weights, chosen uniformly at random. We
prune convolutional and linear weights except those of the first layer (ResNet's `conv1`, DeiT's
patch embedding) and the output layer, leaving biases and normalization layers intact: 11.16M
weights in 19 layers of ResNet-18, and 5.31M in 48 layers of DeiT-Tiny (`qkv`, `proj`, `fc1` and
`fc2` in 12 blocks). Each base model is pruned to 30, 50, 70, 90 and 95% sparsity and evaluated
twice: right after pruning, and after one epoch of fine-tuning with the same recipe and the pruned
weights held at zero. The unpruned model gets the same extra epoch and is the reference for the
fine-tuned models. In all: 2 base models and 22 one-epoch runs, one seed each.

*Measures and statistics.* We measure the drop $Delta$ in accuracy, in percentage points, on
ambiguous, clear and all test images relative to the reference. Percentage points avoid dividing by
starting accuracy, which would inflate the drop of ambiguous images (dividing by starting error
would shrink it). We report the gap $Delta_"amb" - Delta_"clear"$ and the ratio
$Delta_"amb" slash Delta_"clear"$. Uniform degradation, the null hypothesis, means gap 0 and ratio 1.
Because a method may suit one architecture better, models are compared at a *matched overall
drop* $D^*$: the largest overall drop all four model--method curves reach, with each curve
interpolated
linearly between sparsity levels. Four predictions, written before any experiment, are listed in
@tab-pred. A predicted ratio counts as supported if it lies inside the 95% interval. Intervals come
from a paired bootstrap over test images @efron1993bootstrap (2,000 resamples, stratified by group)
and capture test-set sampling noise only, not training noise.

= Results

*Base models.* Both models clear the 94% validation bar with the recipe unchanged (DeiT-Tiny
96.5%, ResNet-18 95.0%, each gaining under half a point in the last epoch). On the test set the
references reach 96.4% and 95.0%. Ambiguous images are much harder for both: 82.9% vs 97.9% on
clear images for DeiT-Tiny, and 78.5% vs 96.8% for ResNet-18.

*Robustness to pruning.* ResNet-18 is far more compressible (@tab-main). After the fine-tune it
stays within 0.5 points of its reference up to 70% sparsity and keeps 82.8% at 95%, while DeiT-Tiny
falls to 85.1% at 70% and 32.0% at 95%. Random pruning is worse than magnitude pruning at every
sparsity in both models (P3 supported). Right after pruning, before the fine-tune, most models have
collapsed to chance (10%), often predicting a single class.

#figure(
  include "table1.typ",
  caption: [Test accuracy (%) after the 1-epoch fine-tune, and the gap $Delta_"amb" - Delta_"clear"$
  (points) against the unpruned reference. Bold: 95% interval excludes 0. The references score
  78.5% / 96.8% (ResNet-18) and 82.9% / 97.9% (DeiT-Tiny) on ambiguous / clear images.],
) <tab-main>

*Main result.* @fig-gap plots the gap against the overall drop. After the fine-tune, all four
curves rise together: while overall drops are small (up to about 20 points), ambiguous images lose
more than clear ones in both architectures and with both methods. The matched overall drop is
$D^* = 12.1$ points, set by ResNet-18 with magnitude pruning at 95% sparsity. DeiT-Tiny reaches it
at about 70%. At $D^*$, ResNet-18 loses 19.0 [15.9, 22.1] points on ambiguous images and 11.4
[10.6, 12.1] on clear ones, a ratio of 1.67 [1.38, 1.97]. DeiT-Tiny loses 21.1 [18.1, 24.2] and
11.1 [10.4, 11.8], a ratio of 1.90 [1.61, 2.21]. The null hypothesis of uniform degradation is
rejected for all four model--method curves (gaps of 6.2--10.0 points, every interval above 0). The
architectures do not differ detectably: ResNet-18’s ratio minus DeiT-Tiny's is −0.22 [−0.62, 0.14].
As the models approach chance the curves turn down and the gap becomes negative, following the
chance floor (thin lines in @fig-gap). Ambiguous images start lower and have less room to fall, as
anticipated in our plan.

#figure(
  image("../results/figures/fig2_gap.pdf", width: 100%),
  caption: [Gap between the drops on ambiguous and clear images, against the overall drop. Left:
  after the 1-epoch fine-tune (main result). Right: right after pruning. Each curve starts at the
  unpruned reference $(0, 0)$ and passes through 30, 50, 70, 90 and 95% sparsity. Bars are 95%
  bootstrap intervals. Black: no gap (the null hypothesis). Gray: the gaps implied by the predicted
  ratios. Vertical line: $D^*$. Thin coloured lines (added after seeing the results): the gap if
  both groups lost the same share of their accuracy above chance.],
) <fig-gap>

*The disproportion appears during recovery.* Right after pruning, only two settings have not yet
collapsed (ResNet-18 also keeps BatchNorm statistics from the unpruned network). In both, the
damage is nearly uniform. ResNet-18 at 70% (magnitude) loses 11.9 points
on ambiguous and 10.2 on clear images, a gap of 1.7 [−1.3, 4.9]. DeiT-Tiny at 50% loses 30.1 and
28.2, a gap of 1.9 [−1.6, 5.3]. After the fine-tune the same models are left with drops of 1.7 vs
0.4 points and 5.4 vs 1.3 points. The fine-tune recovers clear images almost fully and ambiguous
ones much less. This observation is exploratory and rests on two points.

*Predictions* (@tab-pred). The disproportion is smaller than predicted for ResNet-18 and larger,
not smaller, for DeiT-Tiny.

#figure(
  table(
    columns: (auto, 1.25fr, 2fr, 1fr),
    align: (left, left, left, left),
    table.hline(),
    table.header([], [Prediction], [Result (after fine-tune, at $D^*$)], [Verdict]),
    table.hline(stroke: 0.5pt),
    [P1], [ResNet-18 ratio ≈ 2], [1.67 [1.38, 1.97]], [not supported],
    [P2], [DeiT-Tiny ratio ≈ 1.5], [1.90 [1.61, 2.21]], [not supported],
    [P3], [random loses more, same sparsity], [+1.9 to +48.9 points, all intervals > 0], [supported],
    [P4], [random: ratios nearer 1; ViT < CNN], [1.57 vs 1.67; 1.54 vs 1.90; ViT − CNN −0.03 [−0.32, 0.25]],
      [supported (ViT--CNN tie)],
    table.hline(),
  ),
  caption: [The four predictions, written before the experiments (ratios: $Delta_"amb" slash Delta_"clear"$
  under magnitude pruning for P1--P2; random vs magnitude for P4), with 95% intervals.],
) <tab-pred>

*Robustness.* Ambiguous images cluster in a few classes: deer and cats make up 42% of them.
Comparing ambiguous images with clear images of the same class shrinks the gap at $D^*$ from 7.6
to 5.2 points (ResNet-18) and from 10.2 to 8.0 (DeiT-Tiny). Class mix explains part of the gap, but
not most of it. Drops are net (images lost minus images gained), so we also measure the _loss
rate_: the share of initially correct images that the pruned model gets wrong. By this measure
ambiguous images are lost 2.3× (ResNet-18, 29.9% vs 12.8%) and 2.7× (DeiT-Tiny, 29.8% vs 11.2%) as
often as clear ones, and 5--7× as often at light sparsity. So the effect is not an artifact of
ambiguous images starting with more errors.

*Bonus: are the images pruning hurts the ones humans find ambiguous?* Partly. At matched overall
drop (ResNet-18 at 95%, DeiT-Tiny at 70%), 18% and 20% of the images that pruning flips from right
to wrong are ambiguous, against 8% and 9% of all initially correct images (2.1× and 2.3×). Human
disagreement separates flipped from unflipped images only weakly (AUROC 0.62 and 0.64). The two
architectures flip many of the same images: 498 shared, against 141 expected if they were
independent. Only 19% of those shared images are ambiguous, so most of what both architectures
forget is clear to humans.

= Limitations and next steps

Each condition was run once, and our intervals cover test-set noise only, so the CNN--ViT
comparison in particular could change with more seeds. We used one low-resolution dataset, two
small models of different size (11.2M vs 5.5M parameters), one-shot pruning with one recovery epoch
rather than gradual pruning @hooker2019forget, and five sparsity levels, so $D^*$ is interpolated
within coarse segments. The ambiguity cut is a convention and mixes ambiguity with label noise (for
79 test images the most common human label is not the CIFAR-10 label). The chance-floor lines and the
matched bonus analysis were added after seeing the results. Next steps: more seeds, and a finer
sparsity grid where the gap peaks (overall drops of 5--20 points); longer or gradual fine-tuning,
to test whether recovery favours clear images; structured pruning of heads and channels; larger
datasets with multiple annotators per image; and finding which internal features pruning removes
for ambiguous images, for example with sparse autoencoders.

#v(0.6em)
#text(size: 8.5pt)[*Reproducibility and AI use.* Code, notebooks, per-image test predictions and all
numbers are in the project repository (`01_vit.ipynb`, `02_cnn.ipynb`, `03_test.ipynb`). The
research question, hard-set definition, specifications and predictions were decided by the author
(with an AI assistant during planning) before any experiment. The assistant wrote most of
the code, ran the experiments, wrote the analysis rules marked in `03_test.ipynb` before the test
evaluation, and drafted this paper.]

#bibliography("refs.bib", style: "ieee", title: "References")
