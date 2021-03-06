# KiTS21 Evaluation
We refer to the [KiTS21](https://kits21.kits-challenge.org/) homepage for a detailed description of the metrics and 
ranking scheme used in the competition. This document only provides a brief overview.

## Evaluation metrics
KiTS21 uses two metrics for evaluation, the volumetric 
[Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and the 
[Surface Dice](https://arxiv.org/pdf/1809.04430.pdf). 

### Rationale
We refrain from using surface distance-based metrics such as the average symmetric surface distance or the 
Hausdorff distance because these measures penalize false positive/negative predictions by their distance, not their 
existence. While this property is useful for organ segmentation, it is not desired for tumor segmentation where 
lesions can be spread throughout the affected organ and thus do not have an expected spatial location. In addition, 
both Dice and SD have a predefined range (from 0 to 1 where 1 is best) which makes aggregation of these metrics through 
for example averaging more robust (we hereby refer to computing the average Dice or SD over cases, we are not averaging 
Dice and SD into a single number). ASSD and HD in contrast can have outlier values that, even if capped, can 
cause a single test case to dominate the performance of an algorithm (imagine having ASSD of ~1mm in most cases and 
a single case with very large ASSD (>100))).

## Ranking procedure
We will be comparing your test set predictions against realistic ground truth segmentations that were sampled from the 
individual annotations (see [below](#sampling-realistic-segmentations-from-individual-annotations)). The Dice and 
Surface Dice will be computed between your predictions and 
and all corresponding samples (resulting in `num_cases x num_samples x num_HECs x num_metrics` values). Then, each 
metric will be averaged over all cases, samples and HECs such 
that we end up with one average Dice and one average SD value per submission. Finally, we rank the algorithms 
independently for each of the metrics, resulting in two ranks per algorithm. The winner will be determined as the 
algorithm with the lowest average rank.


## Sampling realistic segmentations from individual annotations
Each kidney/cyst/tumor instance has been annotated multiple times by different annotators. We use these multiple annotations
to generate plausible complete annotations for each patient. These plausible annotations will be used to evaluate your 
test set submission (also see [above](#ranking-procedure)) and we thus recommend you use them for evaluating your 
models during development as well.
As a point of reference for human performance we will also be computing the inter-rater disagreement between these 
samples. In order to not underestimate the inter-rater disagreement, we need to sample the segmentations carefully. 
The following procedure is not strictly necessary for evaluating predictions, but we would like to be consistent and 
use the same segmentations for algorithm evaluation and inter-rater agreement.

When generating sampled segmentations with the intent of computing the inter-rater disagreement we cannot compare
samples segmentations that have an overlap between their instance annotations. To illustrate this, we use a simple
example that only has a kindey label (no tumor and cyst). We use `kidney_i1a1` as abbreviation for kidney instance 1
annotation 1.

- computing the inter-rater disagreement between `kidney_i1a1_i2a1` and `kidney_i1a2_i2a2` is valid because for none of the
  instances there are shared annotations
- computing the inter-rater disagreement between `kidney_i1a1_i2a1` and `kidney_i1a2_i2a1` is not valid because i2a1 was
  used to construct both segmentations. This would result in an underestimation of the inter-rater disagreement because
  parts of the segmentations perfectly overlap

To prevent underestimation of the inter-rater disagreement we generate 'groups' of sampled segmentations. Within each
group, none of the annotations are shared and members of each group can be evaluated against each other (therefore
each group has as many samples as there are annotations per instance). To get a more
robust estimate of the inter-rater disagreement of a case, we generate multiple groups of sampled segmentations and
average the inter-rater disagreement across groups.

You can generate the groups yourself by running `python kits21/annotation/sample_segmentations.py`. Sampling is
seeded to ensure that everyone uses the same samples.

## Code for metric computation
**Prerequisite**: You can only run our evaluation code after generating the segmentation samples yourself. Note that 
each dataset update requires you to rerun the sampling, so please make sure that you rerun it after every pull from 
GitHub! 

You can run the sampling of segmentations on the training set by executing:

`python kits21/annotation/sample_segmentations.py -num_processes XX`

where XX is the number of CPU cores to be used (as many as possible). Sampling is seeded to ensure that everyone uses 
the same samples.
Once it is completed you can evaluate your training set predictions using the code provided by this repository. Please 
run

`python kits21/evaluation/evaluate_predictions.py -h`

to see usage instructions. Since we will be using this code to evaluate the test cases as well, you are 
encouraged to use it for evaluating your own train:validation splits during model development (we recommend running 
5-fold CV on the provided training cases).


## Finding the tolerance for SD
This section is for documentation purposes only. You do not need to run this yourself and should just use the 
precomputed values located in `kits21/configuration/labels.py`. When running our evaluation code these values will 
be picked automatically.

We follow the procedure described by the paper that introduces the Surface Dice.

[https://arxiv.org/pdf/1809.04430.pdf](https://arxiv.org/pdf/1809.04430.pdf) page 5:

> We defined the organ-specific tolerance as the 95th percentile of the distances collected across multiple 
> segmentations from a subset of seven TCIA scans, where each segmentation was performed by a radiographer and then 
> arbitrated by an oncologist, neither of whom had seen the scan previously.

We use the same groups of sampled segmentations as are used to compute the inter-rater disagreement for computing the 
tolerance. The tolerance is computed for each HEC individually and is averaged over all cases. 

If you still desire to rerun the computation of the tolerances, you can do so by running 
`python kits21/evaluation/compute_tolerances.py`. Note that this requires you to have generated the sampled 
segmentations first (see [above](#sampling-realistic-segmentations-from-individual-annotations)).