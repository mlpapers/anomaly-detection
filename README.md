# Anomaly and Novelty Detection

## Detection targets and supervision

- **Outlier versus novelty detection:** [outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html) fits potentially contaminated observations; novelty detection learns a reference population and scores new inputs. The training contamination and availability of labels define different tasks.
- **Anomaly structure:** unusual points, unusual values conditional on context, and unusual sequences or groups require different scoring rules. [Ruff et al.](https://arxiv.org/abs/2009.11732) (2020/2021) organize methods by assumptions about normality and available supervision.
- **Scores versus alarms:** an anomaly ranking does not specify an operational threshold or a probability of failure. [Detection interfaces](https://scikit-learn.org/stable/modules/outlier_detection.html) separate scores from thresholded predictions; contamination settings encode a cutoff choice.

## Neighborhoods, isolation and distribution tails

- **LOF:** compares local reachability density with neighboring densities, detecting points sparse relative to their neighborhood. [Breunig et al.](https://sigmodrecord.org/?download_id=6631&smd_process_download=1) (SIGMOD 2000). Neighborhood size changes the scale of abnormality.
- **Isolation Forest:** random feature/split partitions isolate observations; short average isolation paths indicate anomalies. [Liu et al.](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) (ICDM 2008). Isolation replaces explicit density or distance estimation.
- **Extended Isolation Forest:** uses randomly oriented hyperplanes instead of axis-aligned splits to reduce score artifacts. [Hariri et al.](https://arxiv.org/abs/1811.02141) (2018). The geometric change addresses partition bias rather than learning semantic anomaly labels.
- **ECOD:** estimates empirical marginal cumulative distributions, converts observations to tail scores and aggregates across features. [Li et al.](https://arxiv.org/abs/2201.00382) (2022). Its featurewise construction supplies a simple tabular baseline without learning arbitrary joint dependencies.
- **COPOD:** constructs an empirical copula and combines tail information to score extreme observations. [Li et al.](https://arxiv.org/abs/2009.09463) (2020). Distribution-tail scoring differs from LOF’s comparison against a local neighborhood.

## Support and robust distribution models

- **One-class SVM:** learns a function separating an estimated support region from its complement using kernels. [Schölkopf et al.](https://research.google/pubs/estimating-the-support-of-a-high-dimensional-distribution/) (2001). Kernel and regularization choices define the boundary; it is not a labeled anomaly classifier.
- **Robust covariance:** [minimum covariance determinant / EllipticEnvelope](https://scikit-learn.org/stable/modules/outlier_detection.html) fit a robust location and covariance for distance-based detection. Useful when an approximately elliptical normal population is plausible; multimodal structure needs other models.

## Learned one-class and reconstruction methods

- **Deep SVDD:** learns a representation concentrated near a hypersphere center. [Ruff et al.](https://proceedings.mlr.press/v80/ruff18a.html) (ICML 2018). Architecture and training restrictions address trivial collapse; a compact representation alone is not evidence of useful anomaly separation.
- **Deep SAD:** extends deep one-class learning to unlabeled data plus labeled normal or anomalous examples. [Ruff et al.](https://arxiv.org/abs/1906.02694) (2019/2020). Its label-informed setting differs from training only on presumed normal data.
- **DROCC:** trains against adversarially generated nearby negatives under a local low-dimensional-manifold assumption. [Goyal et al.](https://proceedings.mlr.press/v119/goyal20c.html) (ICML 2020). The generated contrast examples define the learned one-class boundary.
- **Reconstruction and MemAE:** ordinary autoencoders can reconstruct anomalies well. [Gong et al.](https://arxiv.org/abs/1904.02639) (2019) restrict reconstruction through a learned memory of normal patterns; low reconstruction error is not inherently a normality guarantee.

## Classifier-based and out-of-distribution detection

- **Maximum softmax probability:** uses classifier confidence as a baseline for misclassification and out-of-distribution detection. [Hendrycks and Gimpel](https://arxiv.org/abs/1610.02136) (2016/2017). This needs a trained classifier and does not turn confidence into a novelty guarantee.
- **Outlier Exposure:** trains detectors using auxiliary outlier data to improve detection of unseen anomalies. [Hendrycks et al.](https://arxiv.org/abs/1812.04606) (2018/2019). Auxiliary-data choice and overlap with evaluated anomalies are part of the supervision protocol.
- **Likelihood failure:** tested flows, VAEs and PixelCNNs trained on CIFAR-10 can assign higher likelihood to SVHN images. [Nalisnick et al.](https://arxiv.org/abs/1810.09136) (2018/2019). High generative density is not a general test of semantic in-distribution membership.
- **[Likelihood-ratio OOD detection](https://arxiv.org/abs/1906.02845)** — 2019. Compares a generative model with a background model to reduce nuisance-statistic effects. Ren et al. improve detection on genomic and image tasks; the result depends on what the background model removes, not simply on using a density model.

## Sequences and streaming thresholds

- **[Bayesian online changepoint detection](https://arxiv.org/abs/0710.3742)** — 2007. Maintains a posterior over time since the last change using a predictive model and hazard function. Detects changes in a sequence’s generating regime; an isolated unusual observation need not imply a persistent regime change.
- **Matrix-profile discords:** rank fixed-length subsequences by distance to their nearest nontrivial match. [Yeh et al.](https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf) (ICDM 2016). Window length and normalization define the anomaly; repeated faults may cease to be rare patterns.
- **OmniAnomaly:** uses stochastic recurrent representations and normalizing flows, scoring multivariate sequences by reconstruction probabilities. [Su et al.](https://netman.aiops.org/~peidan/ANM2023/5.KPIAnomalyDetection/LectureCoverage/2019KDD_OmniAnomaly.pdf) (KDD 2019). Per-channel scores localize discrepancies rather than identify causal fault sources.
- **Anomaly Transformer:** contrasts learned temporal attention with a locality prior using association discrepancy. [Xu et al.](https://arxiv.org/abs/2110.02642) (2021/2022). Its anomaly criterion assumes abnormal points have different temporal associations; this is a modeling hypothesis.
- **TranAD:** combines transformer reconstruction, self-conditioning and adversarial training for multivariate time series. [Tuli et al.](https://arxiv.org/abs/2201.07284) (2022). Detection scores and diagnostic rankings need evaluation under the chosen temporal and threshold protocol.
- **Extreme-value thresholds:** [Siffer et al.](https://kdd.org/kdd2017/papers/view/anomaly-detection-in-streams-with-extreme-value-theory) (KDD 2017) adapt streaming thresholds from tail behavior and a specified risk parameter. Threshold estimation is separate from learning representations; tail-model fit matters.

## Visual anomaly detection and localization

- **PaDiM:** fits multivariate Gaussian distributions to pretrained patch embeddings to detect and localize deviations from normal images. [Defard et al.](https://arxiv.org/abs/2011.08785) (2020/2021). The reference distribution is spatially indexed rather than one global image density.
- **PatchCore:** compares pretrained patch features with a representative memory bank of normal patches. [Roth et al.](https://arxiv.org/abs/2106.08265) (2021/2022). Nearest-reference scoring supports image detection and local defect maps without training a classifier on defect classes.
- **WinCLIP:** combines text prompts with window, patch and image features for zero-shot anomaly detection and segmentation. [Jeong et al.](https://arxiv.org/abs/2303.14814) (CVPR 2023). WinCLIP+ additionally uses normal reference images; these are different data-access regimes.

## Comparative evidence and evaluation

- **ADBench:** compares 30 detectors on 57 datasets across supervision, anomaly types and corruption. [Han et al.](https://arxiv.org/abs/2206.09426) (2022). No tested unsupervised method dominates statistically across datasets; strong methods depend on anomaly structure.
- **Label value in ADBench:** revealing 1% of anomaly labels lets most tested semi-supervised methods beat the best unsupervised baseline in that study. [Results](https://arxiv.org/html/2206.09426) (2022). This is a fraction of anomalies, not 1% of all observations.
- **Point adjustment:** crediting an entire true anomaly segment after one detected point can inflate F1, even for random scores. [Kim et al.](https://arxiv.org/abs/2109.05257) (2021/AAAI 2022). Report unadjusted scores and the exact event-credit rule.
- **Dataset artifacts:** [Wu and Keogh](https://arxiv.org/abs/2009.13807) (2020/2021) identify labeling and construction flaws in widely used time-series benchmarks and introduce the UCR anomaly archive. Their criticism concerns examined datasets; it does not invalidate all temporal detection.
- **MVTec AD 2:** adds eight industrial scenarios, including small defects and lighting changes. [Heckler-Kram et al.](https://arxiv.org/abs/2503.21622) (2025/2026) report tested methods below 60% average AU-PRO; earlier near-saturated benchmarks conceal these difficulties.
- **Rare-event metrics:** report class prevalence, precision–recall performance and false alarms at a threshold chosen without test labels. [ROC/PR analysis](https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf). High ROC-AUC alone does not specify an acceptable operational alert rate.
- **Temporal evaluation protocol:** separate point detection, event detection and delay; keep device/time splits intact and state whether future samples are available. [Evaluation critique](https://arxiv.org/abs/2109.05257) motivates reporting the rule that turns scores into credited events.

## Software and surveys

- **scikit-learn:** [outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html) includes Isolation Forest, LOF, one-class SVM and robust covariance. LOF’s novelty mode scores new observations; its training-score behavior differs from ordinary outlier mode.
- **PyOD:** provides a common Python interface for classical, ensemble and neural outlier detectors. [Zhao et al.](https://jmlr.org/papers/v20/19-011.html) (JMLR 2019). Shared interfaces aid comparisons but do not standardize preprocessing, tuning or label access automatically.
- **Anomalib:** [visual anomaly-detection toolkit](https://github.com/open-edge-platform/anomalib) with model implementations, data pipelines and evaluation support. Record backbone, preprocessing and reference-image access when comparing methods.
- **Unifying review:** [Ruff et al.](https://arxiv.org/abs/2009.11732) (2020/2021) connect density, support estimation and reconstruction approaches, including shallow/deep relationships. Their organizing assumptions help explain why the same score behaves differently across tasks.

## Related topics

[Evaluation](https://mlpapers.org/evaluation/) · [Clustering](https://mlpapers.org/clustering/) · [Feature extraction](https://mlpapers.org/feature-extraction/) · [Generative models](https://mlpapers.org/generative-models/) · [Computer vision](https://mlpapers.org/computer-vision/) · [Missing data](https://mlpapers.org/missing-data/).
