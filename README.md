# Anomaly and Novelty Detection

## Detection targets and supervision

- **Outlier versus novelty detection** - Outlier detection fits potentially contaminated observations; novelty detection learns a reference population and scores new inputs. The training contamination and availability of labels define different tasks. [docs](<https://scikit-learn.org/stable/modules/outlier_detection.html>).
- **Anomaly structure** - Unusual points, unusual values conditional on context, and unusual sequences or groups require different scoring rules. Ruff et al. organize methods by assumptions about normality and available supervision. [Ruff et al. 2021](<https://arxiv.org/abs/2009.11732> "A Unifying Review of Deep and Shallow Anomaly Detection; Proc. IEEE").
- **Scores versus alarms** - An anomaly ranking does not specify an operational threshold or a probability of failure. Detection interfaces separate scores from thresholded predictions; contamination settings encode a cutoff choice. [docs](<https://scikit-learn.org/stable/modules/outlier_detection.html>).

## Neighborhoods, isolation and distribution tails

- **LOF** - Compares local reachability density with neighboring densities, detecting points sparse relative to their neighborhood. Neighborhood size changes the scale of abnormality. [Breunig et al. 2000](<https://sigmodrecord.org/?download_id=6631&smd_process_download=1> "LOF: identifying density-based local outliers; SIGMOD Rec.").
- **Isolation Forest** - Random feature/split partitions isolate observations; short average isolation paths indicate anomalies. Isolation replaces explicit density or distance estimation. [Liu et al. 2008](<https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf> "Isolation Forest; ICDM").
- **Extended Isolation Forest** - Uses randomly oriented hyperplanes instead of axis-aligned splits to reduce score artifacts. The geometric change addresses partition bias rather than learning semantic anomaly labels. [Hariri et al. 2021](<https://arxiv.org/abs/1811.02141> "Extended Isolation Forest; TKDE").
- **ECOD** - Estimates empirical marginal cumulative distributions, converts observations to tail scores and aggregates across features. Its featurewise construction supplies a simple tabular baseline without learning arbitrary joint dependencies. [Li et al. 2023](<https://arxiv.org/abs/2201.00382> "ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions; TKDE").
- **COPOD** - Constructs an empirical copula and combines tail information to score extreme observations. Distribution-tail scoring differs from LOF's comparison against a local neighborhood. [Li et al. 2020](<https://arxiv.org/abs/2009.09463> "COPOD: Copula-Based Outlier Detection; ICDM").

## Support and robust distribution models

- **One-class SVM** - Learns a function separating an estimated support region from its complement using kernels. Kernel and regularization choices define the boundary; it is not a labeled anomaly classifier. [Schölkopf et al. 2001](<https://research.google/pubs/estimating-the-support-of-a-high-dimensional-distribution/> "Estimating the Support of a High-Dimensional Distribution; Neural Comput.").
- **Robust covariance** - Minimum covariance determinant / EllipticEnvelope fit a robust location and covariance for distance-based detection. Useful when an approximately elliptical normal population is plausible; multimodal structure needs other models. [docs](<https://scikit-learn.org/stable/modules/outlier_detection.html>).

## Learned one-class and reconstruction methods

- **Deep SVDD** - Learns a representation concentrated near a hypersphere center. Architecture and training restrictions address trivial collapse; a compact representation alone is not evidence of useful anomaly separation. [Ruff et al. 2018](<https://proceedings.mlr.press/v80/ruff18a.html> "Deep One-Class Classification; ICML").
- **Deep SAD** - Extends deep one-class learning to unlabeled data plus labeled normal or anomalous examples. Its label-informed setting differs from training only on presumed normal data. [Ruff et al. 2020](<https://arxiv.org/abs/1906.02694> "Deep Semi-Supervised Anomaly Detection; ICLR").
- **DROCC** - Trains against adversarially generated nearby negatives under a local low-dimensional-manifold assumption. The generated contrast examples define the learned one-class boundary. [Goyal et al. 2020](<https://proceedings.mlr.press/v119/goyal20c.html> "DROCC: Deep Robust One-Class Classification; ICML").
- **Reconstruction and MemAE** - Ordinary autoencoders can reconstruct anomalies well. Gong et al. restrict reconstruction through a learned memory of normal patterns; low reconstruction error is not inherently a normality guarantee. [Gong et al. 2019](<https://arxiv.org/abs/1904.02639> "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection; ICCV").

## Classifier-based and out-of-distribution detection

- **Maximum softmax probability** - Uses classifier confidence as a baseline for misclassification and out-of-distribution detection. This needs a trained classifier and does not turn confidence into a novelty guarantee. [Hendrycks and Gimpel 2017](<https://arxiv.org/abs/1610.02136> "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks; ICLR").
- **Outlier Exposure** - Trains detectors using auxiliary outlier data to improve detection of unseen anomalies. Auxiliary-data choice and overlap with evaluated anomalies are part of the supervision protocol. [Hendrycks et al. 2019](<https://arxiv.org/abs/1812.04606> "Deep Anomaly Detection with Outlier Exposure; ICLR").
- **Likelihood failure** - Tested flows, VAEs and PixelCNNs trained on CIFAR-10 can assign higher likelihood to SVHN images. High generative density is not a general test of semantic in-distribution membership. [Nalisnick et al. 2019](<https://arxiv.org/abs/1810.09136> "Do Deep Generative Models Know What They Don&#x27;t Know?; ICLR").
- **Likelihood-ratio OOD detection** - Compares a generative model with a background model to reduce nuisance-statistic effects. Ren et al. improve detection on genomic and image tasks; the result depends on what the background model removes, not simply on using a density model. [Ren et al. 2019](<https://arxiv.org/abs/1906.02845> "Likelihood Ratios for Out-of-Distribution Detection; NeurIPS").

## Sequences and streaming thresholds

- **Bayesian online changepoint detection** - Maintains a posterior over time since the last change using a predictive model and hazard function. Detects changes in a sequence's generating regime; an isolated unusual observation need not imply a persistent regime change. [Adams and MacKay 2007](<https://arxiv.org/abs/0710.3742> "Bayesian Online Changepoint Detection; arXiv").
- **Matrix-profile discords** - Rank fixed-length subsequences by distance to their nearest nontrivial match. Window length and normalization define the anomaly; repeated faults may cease to be rare patterns. [Yeh et al. 2016](<https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf> "Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View That Includes Motifs, Discords and Shapelets; ICDM").
- **OmniAnomaly** - Uses stochastic recurrent representations and normalizing flows, scoring multivariate sequences by reconstruction probabilities. Per-channel scores localize discrepancies rather than identify causal fault sources. [Su et al. 2019](<https://netman.aiops.org/~peidan/ANM2023/5.KPIAnomalyDetection/LectureCoverage/2019KDD_OmniAnomaly.pdf> "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network; KDD").
- **Anomaly Transformer** - Contrasts learned temporal attention with a locality prior using association discrepancy. Its anomaly criterion assumes abnormal points have different temporal associations; this is a modeling hypothesis. [Xu et al. 2021](<https://arxiv.org/abs/2110.02642> "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy; arXiv").
- **TranAD** - Combines transformer reconstruction, self-conditioning and adversarial training for multivariate time series. Detection scores and diagnostic rankings need evaluation under the chosen temporal and threshold protocol. [Tuli et al. 2022](<https://arxiv.org/abs/2201.07284> "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data; arXiv").
- **Extreme-value thresholds** - Siffer et al. adapt streaming thresholds from tail behavior and a specified risk parameter. Threshold estimation is separate from learning representations; tail-model fit matters. [Siffer et al. 2017](<https://kdd.org/kdd2017/papers/view/anomaly-detection-in-streams-with-extreme-value-theory> "Anomaly Detection in Streams with Extreme Value Theory; KDD").

## Visual anomaly detection and localization

- **PaDiM** - Fits multivariate Gaussian distributions to pretrained patch embeddings to detect and localize deviations from normal images. The reference distribution is spatially indexed rather than one global image density. [Defard et al. 2020](<https://arxiv.org/abs/2011.08785> "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization; arXiv").
- **PatchCore** - Compares pretrained patch features with a representative memory bank of normal patches. Nearest-reference scoring supports image detection and local defect maps without training a classifier on defect classes. [Roth et al. 2022](<https://arxiv.org/abs/2106.08265> "Towards Total Recall in Industrial Anomaly Detection; CVPR").
- **WinCLIP** - Combines text prompts with window, patch and image features for zero-shot anomaly detection and segmentation. WinCLIP+ additionally uses normal reference images; these are different data-access regimes. [Jeong et al. 2023](<https://arxiv.org/abs/2303.14814> "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation; CVPR").

## Comparative evidence and evaluation

- **ADBench** - Compares 30 detectors on 57 datasets across supervision, anomaly types and corruption. No tested unsupervised method dominates statistically across datasets; strong methods depend on anomaly structure. [Han et al. 2022](<https://arxiv.org/abs/2206.09426> "ADBench: Anomaly Detection Benchmark; NeurIPS").
- **Label value in ADBench** - Revealing 1% of anomaly labels lets most tested semi-supervised methods beat the best unsupervised baseline in that study. This is a fraction of anomalies, not 1% of all observations. [Han et al. 2022](<https://arxiv.org/html/2206.09426> "ADBench: Anomaly Detection Benchmark; NeurIPS").
- **Point adjustment** - Crediting an entire true anomaly segment after one detected point can inflate F1, even for random scores. Report unadjusted scores and the exact event-credit rule. [Kim et al. 2021](<https://arxiv.org/abs/2109.05257> "Towards a Rigorous Evaluation of Time-series Anomaly Detection; arXiv").
- **Dataset artifacts** - Wu and Keogh identify labeling and construction flaws in widely used time-series benchmarks and introduce the UCR anomaly archive. Their criticism concerns examined datasets; it does not invalidate all temporal detection. [Wu and Keogh 2021](<https://arxiv.org/abs/2009.13807> "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress; TKDE").
- **MVTec AD 2** - Adds eight industrial scenarios, including small defects and lighting changes. Heckler-Kram et al. report tested methods below 60% average AU-PRO; earlier near-saturated benchmarks conceal these difficulties. [Heckler-Kram et al. 2026](<https://arxiv.org/abs/2503.21622> "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection; IJCV").
- **Rare-event metrics** - Report class prevalence, precision-recall performance and false alarms at a threshold chosen without test labels. High ROC-AUC alone does not specify an acceptable operational alert rate. [Davis and Goadrich 2006](<https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf> "The relationship between Precision-Recall and ROC curves; ICML").
- **Temporal evaluation protocol** - Separate point detection, event detection and delay; keep device/time splits intact and state whether future samples are available. Evaluation critique motivates reporting the rule that turns scores into credited events. [Kim et al. 2021](<https://arxiv.org/abs/2109.05257> "Towards a Rigorous Evaluation of Time-series Anomaly Detection; arXiv").

<a id="software-and-surveys"></a>

## Software

- **scikit-learn** - Outlier detection includes Isolation Forest, LOF, one-class SVM and robust covariance. LOF's novelty mode scores new observations; its training-score behavior differs from ordinary outlier mode. [docs](<https://scikit-learn.org/stable/modules/outlier_detection.html>).
- **PyOD** - Provides a common Python interface for classical, ensemble and neural outlier detectors. Shared interfaces aid comparisons but do not standardize preprocessing, tuning or label access automatically. [Zhao et al. 2019](<https://jmlr.org/papers/v20/19-011.html> "PyOD: A Python Toolbox for Scalable Outlier Detection; JMLR").
- **Anomalib** - Visual anomaly-detection toolkit with model implementations, data pipelines and evaluation support. Record backbone, preprocessing and reference-image access when comparing methods. [code](<https://github.com/open-edge-platform/anomalib>).

## Surveys

- **Unifying review** - Ruff et al. connect density, support estimation and reconstruction approaches, including shallow/deep relationships. Their organizing assumptions help explain why the same score behaves differently across tasks. [Ruff et al. 2021](<https://arxiv.org/abs/2009.11732> "A Unifying Review of Deep and Shallow Anomaly Detection; Proc. IEEE").

## Related topics

- [Evaluation](https://mlpapers.org/evaluation/)
- [Clustering](https://mlpapers.org/clustering/)
- [Feature extraction](https://mlpapers.org/feature-extraction/)
- [Generative models](https://mlpapers.org/generative-models/)
- [Computer vision](https://mlpapers.org/computer-vision/)
- [Missing data](https://mlpapers.org/missing-data/)
