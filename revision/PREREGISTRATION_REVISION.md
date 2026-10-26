# Pre-registration of the revision analyses (IJIES paper 20267047, first review round)

Recorded before any revision experiment (J1-J6 below) was run. Only smoke tests (two tasks per job, written to a
scratch folder outside the repository and deleted) preceded this record. The primary results of the original
submission (results/xgb_mh/full, full_ref, mirror; analysis/final) WERE known when this plan was written, so the
equivalence margins below are post hoc with respect to those results; the minimal equivalence margin (defined in
R4) is reported as well, so that readers can apply their own margins.

Platform: the revision runs use Linux (glibc / libstdc++) with the same pinned package versions as the original
Windows runs. XGBoost draws row and column subsamples through C++ standard-library distributions, whose output
differs between MSVC and libstdc++. Configurations with subsample < 1 or colsample_bytree < 1 therefore do not
reproduce bit for bit across platforms, while the deterministic default configuration does (verified: all five
default baselines reproduce exactly). Every revision comparison is therefore made within one platform: each
condition is compared with a reference condition run on Linux, never with a Windows run, except where stated.

## R4. Paired optimizer differences and equivalence (existing results, no new runs)

For every dataset x experiment cell and optimizer pair, primary metric, the paired difference d = goodness(A) -
goodness(B) over the 30 seeds. Report the mean and median of d, the 95% CI of the mean (t) and of the median
(percentile bootstrap, 10,000 resamples of seeds), and a TOST equivalence test (two one-sided paired t tests,
alpha = 0.05 each, i.e. the 90% CI of the mean inside (-delta, delta)). Margins delta: RMSE, 2% of the default
configuration's test RMSE on that dataset; ROC-AUC, 0.01. The TOST p value (the larger one-sided p) is
Holm-corrected over the 10 pairs of a cell. Also reported: the minimal equivalence margin, max(|L90|, |U90|) of the
90% CI, and the share of pairs equivalent at delta/2. The same for the mirrored-encoding paired changes (selected
fraction: delta = 0.03; test metric: the margins above) and for tuned-versus-default differences.
Wording rule for the manuscript: "equivalent within delta" only when TOST rejects after Holm; otherwise "no
difference detected".

## J1. Re-evaluation of the 1,500 selected configurations (Linux)

For each primary run: (a) pre-refit model, i.e. the selected configuration refitted on training data with early
stopping on validation exactly as during the search, model seed = tuning seed; validation and test scored from
this same model; (b) refit on training + validation with the logged tree count (Linux replicate of the primary
refit); (c) the same refit with model seeds 1000 and 1001. Default configuration: validation and test from the
training-only default model, and the refit.
Analyses: per cell and method, Spearman rho across seeds between (i) pre-refit validation and pre-refit test,
(ii) pre-refit validation and Linux refit test, with bootstrap 95% CIs; the change rho(ii) - rho(i) isolates
refitting on one platform. Distribution shift is quantified separately by the default configuration's
validation-test gap from one model. Model-seed control: for each run, the SD of test scores across the three
model seeds (b, c) for the fixed configuration, compared with the SD across tuning seeds; share of test-score
variance attributable to model seed = mean within-run variance / total variance.
Chemical dependence (Reviewer 2.4): parent structures standardized with RDKit rdMolStandardize (cleanup, largest
fragment, uncharger, canonical tautomer), stereochemistry removed, first InChIKey block. Exclusion set: validation
molecules whose parent occurs in training; test molecules whose parent occurs in training or validation; and all
molecules of parent groups whose labels conflict (classification: different classes; regression: range > 0.5 log
units or kcal/mol). The selected configurations are kept fixed; the J1 predictions are re-scored on the cleaned
validation and test sets. The RQ1-RQ3 analyses (Friedman, Holm pairwise Wilcoxon, tuning versus default, Spearman
validation-test) are repeated on the cleaned scores and compared with the same analyses on the uncleaned Linux
scores. Original protocol retained as the labelled reference.
Prediction-level stability (Reviewer 2.6): for each cell, the mean pairwise Pearson correlation of test
predictions between the 30 runs of one method.

## J2. Validation reliability control: configuration pool, single split vs inner cross-validation (Exp A)

Pool: 200 configurations, x ~ U[0,1]^8 from default_rng(20261025), identical for every dataset, plus the default.
Model seed fixed at 0 for every fit. For each configuration: (1) single-split criterion as in the benchmark;
pre-refit test from the same model; refit test (training + validation, tree count from the single split);
(2) 5-fold inner scaffold cross-validation on training + validation (scaffold groups, largest first, each to the
currently smallest fold; early stopping on the held fold), criterion = mean fold metric, refit test with the
median fold tree count. The outer test set is never used by either criterion.
Pre-specified analyses (primary metric, configurations as the resampling unit, 10,000 bootstrap resamples):
P1 range restriction: Spearman rho(single criterion, pre-refit test) over all 200 configurations and over the
best 20 by the criterion. Prediction: rho(all) > rho(best 20), with the bootstrap CI of the difference above 0, on
at least 3 of 5 datasets.
P2 refit: rho(single criterion, pre-refit test) versus rho(single criterion, refit test), all 200.
P3 criterion quality: rho(CV criterion, CV refit test) versus rho(single criterion, single refit test), all 200
and best 20.
P4 selection: 2,000 simulated random searches of 100 configurations drawn without replacement from the pool;
the best by each criterion is selected; outcome = its refit test score; regret = outcome minus the best test
score among the 100 drawn. Decision rule: inner CV improves selection on a dataset if the bootstrap 95% CI
(over simulated searches) of the paired difference in regret (single - CV) lies above 0. Also reported: share
of searches whose selection beats the default refit.
The control will not be altered after its results are seen.

## J3. Descriptor-only mirror (Reviewer 1.5)

Experiment B, BBBP and FreeSolv, seeds 0-29, budget 100, penalty 0.01. Conditions: S (standard), M (all
coordinates mirrored, as in the original check) and D (only the descriptor genes mirrored; the eight hyperparameter
coordinates and their part of the shared initial design unchanged). D for all five optimizers; S and M on Linux
for gwo_ref and random.
Prediction: if the descriptor genes alone carry GWO's subset-size bias, GWO's selected fraction is higher under D
than under S (paired Wilcoxon, Holm over the optimizers run under D vs S, p < 0.05) with median > 0.5 on both
datasets, and the paired D - M difference in fraction lies within +/-0.03 (TOST). Random search shows no
Holm-significant S-D difference. The hyperparameter-coordinate contribution is estimated as M - D (paired, 95% CI).
Test-metric changes reported as paired differences with 95% CIs and TOST at the R4 margins; no directional
prediction.

## J4. Subset-size penalty (Reviewer 1.6)

Experiment B, BBBP and FreeSolv, seeds 0-9, five optimizers. Penalty conditions: 0.01 absolute (original, Linux
reference), 0 (no penalty), and relative penalties 0.01 x m_default,val and 0.05 x m_default,val, where
m_default,val is the default configuration's validation metric on that dataset (RMSE or 1 - ROC-AUC), so that the
penalty is the same fraction of the baseline error on every dataset. Plus gwo_ref and random under the full mirror
with penalty 0.
Outcomes: per condition, Friedman test and mean ranks of the five optimizers on the test metric; Kendall tau
between the mean-rank vectors of each condition and the reference; GWO's median selected fraction under each
condition. Prediction: GWO's selected fraction stays below 0.5 under penalty 0 and moves above 0.5 under the
mirror with penalty 0 (paired Wilcoxon over the 10 seeds, p < 0.05, both datasets).

## J5. Larger budget (Reviewers 1.6 and 2.3)

Experiment B, BBBP and FreeSolv, seeds 0-9, five optimizers, budget 300 (population methods: 30 iterations of 10
agents), penalty 0.01; reference = the budget-100 Linux runs of J3/J4 with the same seeds. Outcomes: paired change
(300 - 100) in best validation fitness and in test metric per optimizer, with 95% CIs; Friedman across optimizers
at budget 300; unique configurations; elapsed compute. Descriptive; no directional prediction for test.
Elapsed compute curves for the primary runs are computed from the existing per-call logs (best-so-far validation
fitness against cumulative fitting time).

## J6. Independent scaffold partitions (Reviewer 2.1)

Experiment A, ESOL, FreeSolv, BBBP and BACE (Lipophilicity excluded for compute), five new randomized balanced
scaffold partitions (80/10/10; seeds 1-5; descriptors re-filtered on each training set), five optimizers x
tuning seeds 0-2, plus the default per partition (300 searches, 20 defaults).
Analysis per dataset, primary metric: two-way analysis with partition as a random block: variance components for
partition, optimizer x partition and residual (tuning seed); optimizer main effect tested against the
optimizer x partition interaction (F test); pairwise optimizer differences estimated on partition means with 95%
CIs (t, 4 df) and TOST at the R4 margins (the margins use each partition's default RMSE). Tuned versus default: the
mean improvement per partition and the number of partitions in which each optimizer beats the default.
Prediction (from the submitted paper's claim): the optimizer main effect is not significant (p >= 0.05) on any of
the four datasets, and the partition variance exceeds the tuning-seed variance.

## Reporting

All revision outputs are written to results/revision/; analyses to analysis/revision/. Deviations from this plan
will be listed in the revised manuscript.
