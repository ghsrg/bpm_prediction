## Title Page

**Article title:** Topology-Conditioned Candidate-Space Adaptation for Predictive Process Monitoring under Structural Drift

**Author:** Serhii A. Korotenko<sup>a</sup>

**Affiliation:** <sup>a</sup>Private Higher Educational Establishment "European University", 16-V Akademika Vernadskoho Blvd., Kyiv, 03115, Ukraine; e-mail: s.korotenko@e-u.edu.ua; ORCID: 0009-0003-9236-4775.

**Corresponding author:** Serhii A. Korotenko, Private Higher Educational Establishment "European University", 16-V Akademika Vernadskoho Blvd., Kyiv, 03115, Ukraine 

**E-mail:** s.korotenko@e-u.edu.ua.



## ABSTRACT

Structural drift is a major limitation for predictive process monitoring, where next-activity prediction models are usually trained on historical event logs and keep a fixed output vocabulary. This setting becomes problematic when a new process version changes admissible activities or routing before representative logs are available for retraining. Existing drift-aware studies mainly examine detection, resilience, and model update, while graph-based approaches often derive structure from past logs or use it as an auxiliary signal. The purpose of this study is to test whether a topology-conditioned candidate space can improve the robustness of next-activity prediction under zero-shot structural drift.

We propose EOPKG, a topology-conditioned candidate-space adaptation method that ranks candidates in the space of the current process version $C^{(v)}$, rather than only in the fixed training vocabulary $C_{\mathrm{train}}$. The model combines prefix-based execution context with process topology and uses impulse activation routing to score candidates relative to the current execution state. An ablation without impulse is also evaluated to isolate the role of the impulse state.

The empirical evaluation is performed on a versioned dataset with five consecutive process versions, parallel branches, and structural changes. The comparison includes fixed-vocabulary LSTM and GATv2 baselines, a topology-mask control, and an ablation variant. On the most distant future-version segment ($v_5$), used as the endpoint comparison, logs-only baselines generate more than 60% out-of-structure (OOS) predictions, GATv2+Mask reduces this level to 40.0%, and the proposed approach reduces it to 12.1%. At the same time, strict next-activity prediction on $v_5$ increases to $\mathrm{strict\_macro\_f1}=0.451$, compared with approximately 0.16-0.21 for fixed-vocabulary baselines. The results show that deployed process topology can be used as an explicit knowledge structure for safer release-time prediction under structural drift.

**Keywords:** Predictive process monitoring; Structural drift; Process mining; Knowledge engineering; Candidate-space adaptation; Graph neural networks.


## 1. Introduction

Predictive process monitoring supports operational decision-making by forecasting the future behavior of running business process cases, including the next activity, remaining time, service-level risks, and undesirable outcomes [3, 4, 8]. Such predictions help detect delays, bottlenecks, and compliance risks before a case is completed. Most predictive models learn from historical event logs and apply the learned execution patterns to new prefixes [3, 4]. This setting is effective when the process remains stable, but becomes fragile when the process structure itself evolves.

In modern BPMS and workflow systems, structural changes are introduced through process configuration: routing rules, task sets, gateways, and business regulations may change before enough executions of the new version are recorded. This setting can be viewed as a structural form of process drift within the broader concept drift problem in process mining [20], while the small number of new-version events limits reliable retraining [21]. The resulting release-time interval is illustrated in Fig. 1: the current topology may already be available as a deployed process model, but empirical logs of the new version are still sparse.

The technical limitation is that standard next-activity prediction usually keeps a fixed output vocabulary $C_{\mathrm{train}}$ learned from historical event logs [3, 4, 8]. Under structural drift, this vocabulary may no longer match the version-specific candidate space $C^{(v)}$: new activities can appear, obsolete ones can disappear, and valid routing constraints can change. A model may therefore remain confident while predicting activities that are no longer structurally admissible in the current process version.

This paper investigates whether the deployed topology of a new process version can be used as an explicit knowledge structure for prediction before sufficient logs of that version are accumulated. To this end, we propose EOPKG, a topology-conditioned candidate-space adaptation approach for next-activity prediction under structural drift. Instead of predicting only within the fixed training vocabulary, EOPKG ranks candidate activities in the topology-native space $C^{(v)}$ of the currently deployed process version and conditions candidate scoring on the observed execution state.

The main contributions of this paper are as follows:

1. It formulates topology-conditioned next-activity prediction as a candidate-space adaptation problem under structural drift.
2. It introduces a topology-native candidate-space contract in which new, removed, and rerouted activities are handled at the level of the current process version.
3. It proposes impulse activation routing as a mechanism for conditioning candidate scoring on the current execution state.
4. It evaluates the approach against sequential, graph-based, topology-mask, and no-impulse variants under future-version structural drift.

![](./outputs/article_figures/final/png/Fig1.png)

**Fig. 1. Release-time adaptation lag under structural process drift.** The topology of the newly deployed process version can serve as explicit structural context during the interval in which representative event logs are not yet available.


## 2. Related Work

### 2.1. Predictive Process Monitoring and Drift

Classical process mining studies established the methodological basis for event-log analysis, process model discovery, and conformance checking between actual execution and the normative process model [1, 2]. Predictive process monitoring (PPM) extends this setting from descriptive and conformance analysis to forecasting the future behavior of a running case from its incomplete prefix [3, 4, 8]. In next-activity prediction, the model receives a sequence of already executed events and estimates the most likely next activity. Early deep learning approaches to PPM used recurrent and sequential architectures, including LSTM/GRU models trained on historical event-log prefixes [3, 4]. Later reviews and benchmarks confirmed the effectiveness of such models in stable process environments, but also showed that results are sensitive to task formulation, dataset availability, and feature selection [8].

A common assumption of most sequential PPM models is a fixed prediction space: next-activity classes are defined by the vocabulary available during training. This assumption is natural for a stable process, but becomes a limitation under structural drift. If a new process version adds, removes, or reroutes activities, a fixed-vocabulary model may remain highly confident in predictions that belong to the outdated space $C_{\mathrm{train}}$, but no longer match the current candidate space $C^{(v)}$. Studies on small event logs further emphasize the practical difficulty of situations in which representative logs of the new version are not yet sufficient for reliable retraining [21]. Therefore, for release-time prediction it is important to distinguish between a model's historical ability to reproduce behavior from logs and its ability to operate in the current candidate space of the new process version.

A separate line of work studies concept drift in process mining and the resilience of PPM models to process changes [5, 20]. These studies address drift detection, degradation assessment, update timing, and retraining strategies. They are important for the long-term lifecycle of predictive monitoring, but usually assume that enough new logs are eventually accumulated after the process change for model adaptation or retraining. This paper focuses on a different interval: the release-time gap between deployment of a new process version and the availability of representative logs for that version. During this interval, BPMN/process topology may already be known from the information system, while empirical data from the new version are still limited. Thus, the problem is not only when to update the model, but also how to use the known version structure to constrain and rank the space of admissible next-activity candidates before new logs are accumulated.

### 2.2. Log-Derived Graph Representations

Event-log-derived graph abstractions are a standard and useful foundation in process mining: workflow mining, inductive discovery and block-structured process discovery infer control-flow relations from historical event logs [27, 28]. However, such graphs primarily reconstruct observed past behavior. Their directly-follows or discovered relations may deviate from the deployed process topology when event data are flattened, concurrent executions are linearized, object-centric behavior introduces divergence/convergence, or the process evolves between versions [29]. Recent preprint work has begun to explore DFG-to-GNN predictive process monitoring, suggesting that graph neural representations are an emerging direction for next-activity prediction [6]. EOPKG differs from this line by not deriving the prediction structure only from historical logs. Instead, it uses the currently deployed process topology as a version-conditioned knowledge structure that defines the admissible candidate space $C^{(v)}$ and conditions inference at release time.

### 2.3. Knowledge Structures and Constrained Prediction

A related but broader context is provided by the literature on knowledge graphs and relational reasoning. Knowledge graphs treat structured relations between entities as a basis for knowledge integration, search, reasoning, and machine representation learning [24]. Methods for reasoning over knowledge graphs, including path-based and subgraph-based approaches, show that structure can act not only as a set of features, but also as a carrier of constraints and reasoning paths [25]. Methodologically, this supports the EOPKG framing as a version-conditioned knowledge structure rather than another graph embedding layer over an event log: it describes current activities, transitions, parallel branches, and admissible candidates of the deployed process version. This framing is aligned with the data and knowledge engineering perspective: the process topology is treated as an explicit knowledge artifact that conditions prediction, rather than as an implicit pattern reconstructed from event frequencies. At the same time, the general KG reasoning literature does not formulate topology-known next-activity prediction for evolving business processes and does not solve the fixed-vocabulary PPM problem under structural drift.

The idea of constraining the prediction space also has analogies outside process mining. In neural machine translation, constrained decoding allows generation to account for external lexical constraints rather than operating over an unrestricted output space [26]. This analogy is useful only at the level of principle: prediction can be conditioned by external rules. In PPM, these rules are not lexical; they are defined by the topology of the current process version, gateways, reachability, and the candidate set $C^{(v)}$. EOPKG transfers this principle to process-topological candidate-space restriction rather than adopting NLP constrained decoding mechanically.

Existing PPM models primarily learn predictive behavior from historical event logs and usually retain a fixed output vocabulary. Drift-aware studies address detection, resilience and retraining, but they do not focus on the release-time interval in which a new process topology is already known while representative logs are still unavailable. Graph-based PPM demonstrates the value of structural representations, yet these structures are often log-derived, static, or used as auxiliary signals without redefining the prediction space. Knowledge graph reasoning and constrained prediction motivate the use of explicit structural constraints, but they do not formulate topology-known next-activity prediction for evolving business processes. EOPKG addresses this gap by using the current process version as a knowledge structure that defines and ranks the admissible candidate space $C^{(v)}$, rather than predicting only within the fixed training vocabulary $C_{\mathrm{train}}$.


## 3. Methodology

### 3.1. Problem Definition and Notation

This section formalizes the prediction setting used throughout the paper. The goal is to distinguish the conventional fixed-vocabulary formulation from the version-conditioned formulation required when the process topology changes between releases.

Let $x_t$ denote the execution prefix of a case up to time $t$, and let $G^{(v)}=(V^{(v)},E^{(v)})$ denote the process structure of version $v$ that corresponds to the current execution context. The conventional PPM formulation treats next-activity prediction as classification of the next event over an activity vocabulary formed from historical logs, as shown in Eq. (1) [3, 4, 8]:

$$
\hat{y}_{t+1}=\arg\max_{c_j \in C_{\mathrm{train}}} p(c_j \mid x_t). \tag{1}
$$

Here, $C_{\mathrm{train}}$ denotes the activity vocabulary observed during training. Under structural drift, this vocabulary may no longer match the set of activities and transitions allowed by the currently deployed process version. This mismatch motivates replacing fixed-vocabulary classification with version-conditioned candidate ranking.

Therefore, the task is formulated as version-conditioned candidate ranking according to Eq. (2):

$$
\hat{y}_{t+1}=\arg\max_{c_j \in C^{(v)}} s_j^{(v)}. \tag{2}
$$

where $C^{(v)}$ is the dynamic candidate set of the current process version. Thus, prediction is performed not over a historically fixed class pool, but over the current activity space of version $v$.

### 3.2. Topology-Native Candidate-Space Adaptation

In EOPKG, each candidate $c_j$ is treated as an element of the structure of a specific process version. This enables topology-native handling of new tasks, removed activities, routing changes, and cyclic returns. Unlike log-derived graph approaches, where graph structure is reconstructed from historical event logs [27, 28, 6], the version topology acts here as a prior condition for prediction.

Candidate relevance is defined by the scoring function $f_\theta$ according to Eq. (3):

$$
s_j^{(v)} = f_\theta(x_t, c_j, G^{(v)}), \quad c_j \in C^{(v)}. \tag{3}
$$

The scoring function $f_\theta$ is implemented as a two-tower architecture consisting of two encoders:

1. a prefix encoder (observed encoder), which compresses the current execution state into the latent representation $z_{\mathrm{prefix}}$;
2. a structural encoder, which generates a latent representation $h_j^{(v)}$ for each candidate in the context of the current version topology.

Because $z_{\mathrm{prefix}}$ and $h_j^{(v)}$ are produced by heterogeneous encoders, they are compared in a shared latent space following metric learning principles [11]. Final ranking is computed using cosine similarity with temperature scaling, as shown in Eq. (4):

$$
s_j^{(v)} = \frac{\cos(z_{\mathrm{prefix}}, h_j^{(v)})}{\tau}. \tag{4}
$$

where $\tau$ is the temperature parameter that controls the sharpness of the logit distribution and helps reduce overconfidence when the model selects a target candidate under structural drift [12]. Using cosine similarity instead of a raw dot product stabilizes gradients during joint training of the two network branches [11].

During training, EOPKG uses a topology-local candidate objective. Candidate scores are normalized over the version-specific candidate set $C^{(v)}$, and the observed next activity is mapped to a target candidate mask by matching candidate identifiers or labels. The loss is a set-aware cross-entropy: if several topology candidates correspond to the same activity label, their probability mass is treated jointly as the correct target. The process-state-aware topology mask is used as an additional structural penalty on invalid candidate mass, rather than as a replacement for the candidate-space objective.

### 3.3. Impulse Activation Routing

In impulse activation routing, the current execution state is projected onto the structure nodes. If the last completed activity or the current active set corresponds to specific structural nodes, these nodes receive an impulse state in the structural encoder input. Conceptually, this mechanism is related to instruction-pointer attention in program graphs [9], but here the pointer activates process-structure candidates rather than program instructions. Normalized impulse addition is defined by Eq. (5), and the initial node representation for the structural encoder is defined by Eq. (6):

$$
\begin{aligned}
\tilde{h}_u^{base}&=\operatorname{LN}_{base}(h_u^{base}),\\
\Delta h_u&=\gamma \cdot \operatorname{LN}_{imp}(\operatorname{Proj}(\mathbf{s}_u)).
\end{aligned} \tag{5}
$$

$$
h_u^{(0)} = \tilde{h}_u^{base} + \Delta h_u. \tag{6}
$$

where $\mathbf{s}_u(t)$ is the state vector of structural node $u$ relative to the current prefix $x_t$, defined in this study by Eq. (7):

$$
\mathbf{s}_u(t)=
\left[
\mathbb{1}(u \in L_t),
\mathbb{1}(u \in A_t),
\log(1+n_u(x_t)),
r_u(x_t)
\right]. \tag{7}
$$

Here, $L_t$ is the set of nodes corresponding to the last completed activity or the last known execution state; $A_t$ denotes active candidates; $n_u(x_t)$ is the number of occurrences of the activity associated with node $u$ in the prefix; and $r_u(x_t)$ is the normalized recency feature of the last occurrence. The function $\operatorname{Proj}$ is a learnable affine projection of the node state into the structural encoder dimension, as shown in Eq. (8):

$$
\operatorname{Proj}(\mathbf{s}_u)=W_s \mathbf{s}_u + b_s. \tag{8}
$$

Thus, $\mathbf{s}_u$ does not replace the structural node features, but adds a local execution impulse that indicates where the current case is located relative to the version structure $G^{(v)}$. Both components are normalized before addition, following a Pre-LN residual design that stabilizes the scale of the structural signal [16, 17]. After merging, the structural encoder propagates activation through the edges of the current version using the message-passing paradigm [23], according to Eq. (9):

$$
H^{struct} = \operatorname{GNN}(H^{(0)}, E^{(v)}). \tag{9}
$$

where $\gamma$ is the impulse scale. The interpretation is straightforward: the current version structure defines where the execution signal may propagate, while the observed encoder defines which prefix state should be compared with the activated candidates. The full information flow of the proposed mechanism is summarized in Fig. 2.

![](./outputs/article_figures/final/png/Fig2.png)

**Fig. 2. Core mechanism of topology-conditioned candidate scoring.** The observed prefix is encoded as execution context, impulse activation maps this context to the current process topology, and candidate scores are computed over the version-specific candidate space $C^{(v)}$.

This design separates execution-state encoding from version-specific candidate representation, while coupling them only at the scoring stage.

### 3.4. Topology Mask and Candidate Constraint

For a sequential process, admissible candidates can be approximated by the direct successors of the last completed activity. For a parallel process or a flattened XES log [22], this rule is insufficient because the next completed event may belong to a parallel branch that was already active earlier.

Therefore, this study uses a unified process-state-aware candidate constraint according to Eq. (10):

$$
M_t = A_t \cup S_t. \tag{10}
$$

where $A_t$ denotes active or potentially active candidates, and $S_t$ denotes structural successors reachable from the current prefix. For lifecycle-rich logs, $A_t$ is reconstructed from the active task set according to Eq. (11):

$$
A_t = \{c_j \in C^{(v)} \mid c_j \text{ has an open lifecycle instance at } t\}. \tag{11}
$$

For flattened XES logs [22] without full marking information, $A_t$ is approximated through relaxed reachability: from the last completed activity and several previous anchor nodes within a lookback window, the method retains candidates that remain reachable in the current version structure and were not explicitly closed by completed events. For parallel execution, this avoids restricting the mask only to direct successors of the last event and preserves candidates from other active parallel branches.

The set $S_t$ is formed as a bounded structural reachability neighborhood according to Eq. (12):

$$
S_t =
\{c_j \in C^{(v)} \mid \exists a \in L_t \cup A_t:
\operatorname{dist}_{G^{(v)}}(a,c_j) \le d\}. \tag{12}
$$

where $d$ is the maximum depth of the structural path to a candidate, and $L_t$ is the set of prefix anchor nodes.

In the practical implementation, Eq. (12) is complemented by heuristic constraints to prevent $S_t$ from degenerating into the full space $C^{(v)}$, especially in the presence of parallel branches and loops. The set $L_t$ is formed not from the entire execution history, but from a fixed window of recent events. In addition, an upper bound on the cardinality $|S_t|$ is imposed: if the neighborhood exceeds a predefined share of the total candidate set, it is narrowed to the closest structural neighbors. Completed activities are removed from $S_t$, except when they are direct successors of the current state, which corresponds to rework.

For all variants that use the topology mask $M_t$, these parameters are fixed identically: search depth $d=1$, an anchor-node lookback window of 8 events, and maximum cardinality of 35% of $|C^{(v)}|$. This design compensates for the limitations of flattened XES logs while keeping parallel tasks and allowed loops structurally reachable.

This mechanism for constructing the mask $M_t$ is an integral part of the topology-native contract and is not treated as the main ablation object. The main ablation focuses on the impulse parameter $\gamma$: $\gamma=0$ corresponds to basic candidate-space ranking, whereas $\gamma>0$ activates dynamic impulse routing.


## 4. Experimental Design

### 4.1. Research Questions

The experimental design is organized around four research questions:

**RQ1.** Does topology-conditioned candidate-space adaptation improve robustness under future-version structural drift?

**RQ2.** How much of the improvement can be explained by topology masking alone, and where does fixed-vocabulary post-filtering remain insufficient?

**RQ3.** Does impulse activation routing improve candidate ranking beyond topology-native scoring without impulse?

**RQ4.** Is the additional structural component computationally feasible for release-time predictive monitoring?

### 4.2. Dataset

Public benchmarks, including BPI Challenge 2012 [7], remain useful for general evaluation of PPM models. However, because they lack explicit versioned process topologies, they do not allow a clear measurement of how the known structure of a new version affects topology-known zero-shot structural drift adaptation. This study therefore uses a dedicated versioned dataset rather than only repeating standard PPM benchmark protocols [7, 8].

For this study, we constructed the semi-synthetic dataset `loan_v1_v5_complex_simulated` [13], which models the evolution of a loan process across five versions. The dataset is used as a controlled structural-drift benchmark because the evaluation requires aligned event logs, explicit process versions, and deployed topology for each version, which are not jointly available in standard public PPM benchmarks. The dataset contains structural drift, new tasks, routing changes, parallel execution, repetitions, technical incidents, and carryover between versions. Each later version adds new process modifications to the previous one, reflecting a realistic scenario of evolving business regulations. An important property is that some cases cross version boundaries, while concurrency breaks the simple assumption of a linear event order. Table 1 summarizes the dataset characteristics relevant to structural-drift evaluation.

**Table 1**

Summary of the versioned process dataset.

| Aspect | Dataset statistic | Relevance for evaluation |
| --- | ---: | --- |
| Cases and length | 5559 cases; mean length 27 ($\sigma=10.56$); max 95 events | Tests long-prefix prediction. |
| Process versions | Five versions; ~1111 cases per version; imbalance $<10\%$ | Supports version-wise drift evaluation. |
| Task nodes | 69 unique task nodes; growth from 39 in $v_1$ to 53 in $v_5$ | Expands $C^{(v)}$ across versions. |
| Version carryover | 19.8% of cases finish in a later version | Tests version-aware candidate spaces. |
| Loops and rework | 6.55% of cases contain repeated activities | Tests robustness to rework/retry behavior. |
| Parallel execution | 100% of cases include concurrency; 23.81% transitions are parallel | Challenges strict linear next-event prediction. |
| Incidents and retries | 28.91% of cases include incidents; 2337 automatic-task repeats | Adds operational noise. |
| Resource variability | 87.85% tasks use a non-dominant resource | Adds non-topological variability. |

A detailed description of the topological transformations for each process version ($v_1 \dots v_5$), including the full BPMN diagrams, is provided in the supplementary materials [15] to avoid overloading the main text.


### 4.3. Compared Models and Ablation Variants

The comparison is designed to isolate three effects: fixed-vocabulary sequence learning, graph-based prefix encoding, and topology-conditioned candidate-space adaptation. Table 2 summarizes the evaluated variants and their role in the experimental design.

The choice of LSTM as the sequential baseline and GATv2 as the graph baseline is motivated by a preliminary systematic comparison of architectures [10], where they showed stable behavior in logs-only and static-structure scenarios, respectively.

For a fair comparison, all variants receive the same observed-prefix input features: `concept:name`, `org:resource`, and `time:timestamp`. The categorical features are embedded from the training logs only; unseen future-version values are mapped to `<UNK>` / `UNKNOWN`. No textual task descriptions, organizational hierarchies, or rich semantic resource profiles are provided, so differences between models reflect the use of topology and candidate space rather than additional semantic context.

To support architectural comparability, all compared models use the same hidden-state dimensionality ($H=64$). The number of layers is fixed according to the architectural family: two recurrent layers for LSTM, two message-passing layers for GATv2, and a combined EOPKG structure with two observed-encoder layers and one structural propagation layer.

GATv2+Mask is included as a lightweight structural control: it applies the topology mask as post-filtering over a fixed-vocabulary GATv2 head, but does not create logits for candidates outside $C_{\mathrm{train}}$. EOPKG-WI is the internal ablation that keeps topology-native candidate scoring while disabling impulse activation. Detailed mappings between paper names, code classes, checkpoint names, and MLflow run exports are provided in the reproducibility materials [15].

**Table 2**

Model configurations for comparative evaluation and ablation.

| Variant               | Type           | Prediction space                               | Layers                          | Impulse           |
| --------------------- | -------------- | ---------------------------------------------- | ------------------------------- | ----------------- |
| GATv2                 | Graph baseline       | fixed $C_{\mathrm{train}}$                     | 2 observed layers               | -                 |
| GATv2+Mask            | Structural control        | fixed $C_{\mathrm{train}}$ + post-filter $M_t$ | 2 observed layers               | -                 |
| LSTM                  | Sequential baseline       | fixed $C_{\mathrm{train}}$                     | 2 LSTM layers                   | -                 |
| EOPKG without impulse | Ablation       | topology-native $C^{(v)}$                      | 2 observed + 1 structural layer |  $\gamma=0$        |
| EOPKG                 | Proposed model | topology-native $C^{(v)}$                      | 2 observed + 1 structural layer | $\gamma>0$        |

### 4.4. Evaluation Protocol

The evaluation protocol is divided into two stages to separate internal validation from adaptation assessment:

**Train-run validation**: Models are trained only on historical versions ($v_1$, $v_2$), which are supplied sequentially according to process evolution rather than mixed into a single inter-version stream. The data of each version are shuffled separately using a fixed seed and split into train/validation/test subsets with $\mathrm{split_ratio}=0.7/0.2/0.1$. Metrics at this stage measure generalization within known versions and do not include future versions $v_3$-$v_5$.

**Future-version structural drift evaluation**: To evaluate robustness to structural drift, the model state from the best validation epoch obtained during training on historical versions ($v_1$, $v_2$) is used. This fixed model is then tested on the consolidated evolutionary data stream ($v_1 \dots v_5$), without additional training on future versions. The future-version segments ($v_3, v_4, v_5$) are interpreted as topology-known zero-shot evaluation segments. In this mode, the model has access to the structural model of the corresponding version, but has no training experience on logs from these versions. Zero-shot in this paper means topology-known structural zero-shot: the BPMN topology of the future version is available during inference, but execution logs of new or changed activities are not used for training. New candidates enter the prediction space through $C^{(v)}$ and are ranked not through behavior learned from logs for a specific task identifier, but through structural position in $G^{(v)}$, reachability from the current prefix state, the topology-native mask, and impulse routing. This makes it possible to evaluate the initial state of the model before empirical execution data for the new versions are accumulated.

Evaluation on the $v_1 \dots v_5$ stream is evolutionary: the model is analyzed not only on separate future-version slices, but also through the dynamics of increasing structural drift. For this purpose, metrics are additionally computed over moving stream fragments, making it possible to trace how prediction quality changes along version transitions. This approach captures the dynamics of structural headroom: the model's ability to preserve predictive quality as the process moves further away from the versions used for training.

### 4.5. Metrics

The evaluation separates exact predictive accuracy, structural safety, topology-tolerant validity, calibration, and statistical reliability of comparisons. The primary metric for exact next-activity prediction is $\mathrm{strict\_macro\_f1}$: it counts only agreement with the next activity recorded in the log and is not replaced by other indicators.

$\mathrm{macro\_f1}$ is used as a secondary topology-tolerant metric for BPMN segments with parallel branches, where several activities may be structurally admissible next completions. This metric is not interpreted as a direct replacement for exact accuracy. The related metric $\mathrm{strict\_error\_but\_allowed\_rate}$ isolates cases where the prediction does not match the next log entry but remains admissible within the active parallel structure.

Structural safety is evaluated through $\mathrm{OOS}$, $\mathrm{target\_in\_mask\_rate}$, and $\mathrm{pred\_in\_mask\_rate}$. These metrics separate the quality of the topology mask itself from the behavior of the model after candidate ranking in the current structural context $M_t$.

Calibration is evaluated through $\mathrm{NLL}$ and $\mathrm{ECE}$ [12] to check whether models become overconfident when predicting under candidate-space shift. The main claims are statistically checked using a paired design with identical seeds and an exact paired sign-flip permutation test. This choice is suitable for the small number of independent runs ($n=7$) and does not require a normality assumption for model differences.

Additional diagnostic slices by prefix length, topology-mask cardinality, train/validation loss, gradient norms, and ambiguity rate are provided in the supplementary experimental details.

### 4.6. Implementation Details and Computational Complexity

Compared with a fixed-vocabulary baseline, EOPKG adds a structural encoder and candidate ranking in the space $C^{(v)}$. Therefore, the cost of one inference pass depends not only on the observed prefix length, but also on the size of the current process-version topology: $|V^{(v)}|$, $|E^{(v)}|$, and $|C^{(v)}|$. However, these operations are performed over the version-local process structure, not over the full historical log. The process structure is prepared in advance and used as a static artifact, so inference does not rebuild topology or statistical snapshots.

The experiments were conducted in a local Python environment (version 3.11) using PyTorch (2.12) and PyTorch Geometric (2.7) on Windows 25H2. The hardware configuration included an Intel Core i3-8145 processor and 24 GB of RAM. Approximate inference measurements are reported in the supplementary materials; they are used only to estimate the order of magnitude, not as an isolated hardware benchmark. The complete list of dependencies and libraries, fixed in `requirements-freeze.txt`, is provided in the reproducibility materials [15].


## 5. Results and Analysis

### 5.1. Training Convergence and Internal Stability

The dynamics of strict validation prediction quality ($\mathrm{strict\_val\_macro\_f1}$) show that the structural variants quickly reach a stable training regime (Fig. 3). Already at step 1, EOPKG and GATv2+Mask start at a similarly high level (0.747 and 0.748, respectively), whereas logs-only GATv2 and LSTM have lower initial values (0.593 and 0.542). By step 5, EOPKG reaches 0.768, and its final convergence plateau reaches 0.791. Logs-only solutions finish the trajectory at 0.716 (GATv2) and 0.761 (LSTM), while GATv2+Mask improves the graph baseline without moving to a topology-native candidate space.

Additional diagnostics of train/validation loss, gradient norms, and oversmoothing confirm a stable optimization regime and are provided in the supplementary experimental details.

![](./outputs/article_figures/final/png/Fig3.png)
**Fig. 3. Dynamics of strict validation accuracy $\mathrm{strict\_val\_macro\_f1}$ for the compared architectures during training (epochs 0-50).**

### 5.2. Known-Version Generalization

On the known-version holdout set ($v_1$, $v_2$), EOPKG remains competitive and achieves the highest $\mathrm{strict\_test\_macro\_f1}=0.775$ (see Table 3). LSTM, GATv2+Mask, GATv2, and EOPKG-WI remain in a close range from 0.729 to 0.750, indicating that topology-native candidate scoring does not degrade known-version prediction. At the same time, GATv2+Mask is locally competitive on short or structurally simple fragments, where lightweight topology post-filtering is effective because the candidate space still largely overlaps with $C_{\mathrm{train}}$. Detailed slices by prefix length and mask cardinality are provided in the supplementary experimental details.

**Table 3**

Aggregated test metrics on the known-version holdout set.

| Metric                                                    | GATv2         | GATv2+Mask        | LSTM              | EOPKG             | EOPKG-WI          |
| :-------------------------------------------------------- | :------------ | :---------------- | :---------------- | :---------------- | :---------------- |
| Strict test macro-F1 | 0.729 ± 0.014 | 0.749 ± 0.015     | 0.750 ± 0.011     | **0.775 ± 0.004** | 0.736 ± 0.018     |
| Test macro-F1                 | 0.938 ± 0.009 | **1.000 ± 0.000** | 0.999 ± 0.000     | **1.000 ± 0.000** | 0.982 ± 0.025     |
| Test OOS rate                      | 0.019 ± 0.003 | **0.000 ± 0.000** | 0.0001 ± 0.0002   | **0.000 ± 0.000** | 0.007 ± 0.008     |
| Test NLL                      | 0.230 ± 0.008 | 0.201 ± 0.006     | 0.194 ± 0.005     | **0.145 ± 0.001** | 0.200 ± 0.006     |
| Test ECE                           | 0.028 ± 0.006 | 0.032 ± 0.010     | 0.030 ± 0.007     | **0.011 ± 0.001** | 0.015 ± 0.004     |

### 5.3. Future-Version Structural Drift

This section evaluates the main stress-test scenario of the paper: future-version prediction after the process topology has changed but before representative logs of the new version are available for retraining. The analysis separates four effects: exact next-event accuracy, structural safety, topology-tolerant validity under parallelism, and calibration under candidate-space mismatch.

The drift trajectory is not limited to the final endpoint. Already in the first unseen segment ($v_3$), fixed-vocabulary baselines cannot assign dedicated logits to candidates that are present in $C^{(v_3)}$ but absent from $C_{\mathrm{train}}$. This immediately appears as a structural-safety gap: OOS for logs-only baselines reaches approximately 31%, while GATv2+Mask reduces early OOS to 11.8% and EOPKG to 3.1%. Thus, $v_3$ shows the onset of adaptation lag, while $v_5$ is used as the benchmark endpoint for accumulated drift in a scenario where a business introduces several structural changes over a short period before enough logs are available for retraining. Full per-version trajectory values are provided in Supplementary Table S5.

**Table 4**

Final robustness metrics on the most distant future-version segment $v_5$.

| Metric                        | GATv2          | GATv2+Mask     | LSTM           | EOPKG             | EOPKG-WI      |
| :---------------------------- | :------------- | :------------- | :------------- | :---------------- | :------------ |
| **Accuracy and validity**     |                |                |                |                   |               |
| Strict Macro F1               | 0.213 ± 0.020  | 0.337 ± 0.013  | 0.162 ± 0.062  | **0.451 ± 0.055** | 0.246 ± 0.023 |
| Macro F1                      | 0.280 ± 0.032  | 0.518 ± 0.014  | 0.218 ± 0.080  | **0.777 ± 0.058** | 0.416 ± 0.036 |
| **Structural safety**         |                |                |                |                   |               |
| OOS rate                      | 0.603 ± 0.045  | 0.400 ± 0.000  | 0.690 ± 0.110  | **0.121 ± 0.031** | 0.505 ± 0.059 |
| Prediction in Mask rate       | 0.397 ± 0.045  | 0.600 ± 0.000  | 0.310 ± 0.110  | **0.879 ± 0.031** | 0.495 ± 0.059 |
| Strict Error but Allowed rate | 0.079 ± 0.025  | 0.150 ± 0.018  | 0.060 ± 0.023  | **0.360 ± 0.046** | 0.165 ± 0.048 |
| **Calibration**               |                |                |                |                   |               |
| Test ECE                      | 0.467 ± 0.053  | 0.328 ± 0.066  | 0.425 ± 0.052  | **0.187 ± 0.044** | 0.233 ± 0.072 |
| Test Set NLL                  | 13.047 ± 1.163 | 10.440 ± 0.432 | 11.538 ± 0.644 | **2.555 ± 0.804** | 3.020 ± 0.401 |

After new topological structures are introduced, strict accuracy ($\mathrm{strict\_macro\_f1}$) decreases for all models, as shown in Fig. 4. This is explained not only by candidate-space mismatch, but also by the increasing natural ambiguity of the log: $\mathrm{ambiguous\_prefix\_rate}$ reaches 82.5%. On the most distant segment $v_5$, classical fixed-vocabulary models (LSTM and GATv2) decrease to approximately 0.16-0.21 strict-F1, GATv2+Mask reaches 0.337, and EOPKG maintains $\mathrm{strict\_macro\_f1}=0.451$ (see Table 4). Topology-tolerant macro-F1 does not replace strict-F1; it measures whether predictions remain structurally admissible when parallel branches may complete in different log orders. In this secondary dimension, EOPKG reaches $\mathrm{macro\_f1}=0.777$, indicating a stronger ability to remain within the admissible candidate space during parallel execution.


![](./outputs/article_figures/final/png/Fig4.png)
**Fig. 4. Dynamics of strict accuracy ($\mathrm{strict\_macro\_f1}$) under chronological structural drift.**

The key advantage of EOPKG becomes clear in the analysis of structural safety. On segment $v_5$, LSTM and GATv2 generate 69.0% and 60.3% OOS predictions, respectively (Fig. 6, Subfigure A). GATv2+Mask reduces this level to 40.0%, meaning that simple structural filtering reduces OOS by roughly one third relative to GATv2. In contrast, EOPKG reduces OOS to 12.1%, or by approximately 70% relative to GATv2+Mask. As shown in Fig. 5, GATv2+Mask keeps 60.0% of predictions inside the current BPMN scheme, while EOPKG keeps 87.9% ($\mathrm{pred\_in\_mask\_rate}$). This is the main evidence that topology-conditioned candidate-space adaptation improves not only numerical accuracy, but also structural safety.

The comparison with GATv2+Mask clarifies the role of structural information. The mask provides a clear structural-safety gain: it removes part of the predictions that are physically impossible in the current BPMN version. However, this mechanism is a negative constraint rather than positive ranking: it filters invalid classes, but does not create dedicated logits or embeddings for candidates from $C^{(v)} \setminus C_{\mathrm{train}}$, does not transfer the execution state to nodes of $G^{(v)}$, and does not distinguish between competing admissible branches. This is why GATv2+Mask occupies an intermediate position between logs-only GATv2 and EOPKG: it confirms the value of structural information, but also shows that post-filtering is not a sufficient replacement for topology-native candidate-space prediction.

![](./outputs/article_figures/final/png/Fig5.png)
**Fig. 5. Dynamics of keeping predictions inside the current topology mask ($\mathrm{pred\_in\_mask\_rate}$).**

Because the exact next log entry is often not the only admissible continuation of a parallel process, structurally admissible strict errors are analyzed separately. These are not arbitrary errors inside the mask, but predictions that do not match the next log event while still corresponding to an activity from a parallel branch that remains structurally active, does not violate precedence constraints, and must be completed before the corresponding BPMN join gateway. In most such scenarios, parallel tasks are not competing business alternatives: they differ in completion order, but both belong to a valid process execution. As a result, 36.0% of all EOPKG decisions are classified as structurally admissible strict errors ($\mathrm{strict\_error\_but\_allowed\_rate}$, Fig. 6, Subfigure B). These cases are still counted as errors under strict-F1; therefore, this metric characterizes the error profile rather than redefining exact predictive accuracy.

Calibration metrics are interpreted as supportive evidence of confidence behavior under candidate-space shift, not as a replacement for structural safety or exact prediction metrics. Logs-only models show signs of overconfidence in incorrect predictions: their NLL is in the range of 11.5-13.0 (Fig. 6, Subfigure C), and expected calibration error (ECE) is approximately 0.42-0.47 (Fig. 6, Subfigure D). GATv2+Mask partially improves the calibration profile ($\mathrm{NLL}=10.440$, $\mathrm{ECE}=0.328$) because it removes part of the clearly invalid predictions; at the same time, the zero dispersion of its OOS and prediction-in-mask rate across seed runs is an expected consequence of hard-mask post-filtering, not evidence that the models are identical. EOPKG has lower values of $\mathrm{NLL}=2.555$ and $\mathrm{ECE}=0.187$, which is consistent with better representation of uncertainty in the changed candidate space.

The ablation model EOPKG-WI clarifies the contribution of impulse routing. It has access to the topology-native candidate space, but without the impulse execution state it localizes the current execution state in the process topology less effectively. On $v_5$, its OOS reaches 50.5%, and $\mathrm{macro\_f1}$ decreases to 0.416, whereas full EOPKG maintains 12.1% OOS and 0.777 macro-F1 (Table 4). This shows that the novelty is not only the transition to $C^{(v)}$, but the combination of topology-native candidate space, process-state-aware mask, impulse routing, and candidate scoring. The comparison between EOPKG and EOPKG-WI suggests that impulse routing mainly contributes to candidate ranking and structural safety. Accordingly, the ablation claim is based primarily on Macro-F1 and OOS reduction, while the NLL difference is treated as descriptive rather than conclusive evidence.


![](./outputs/article_figures/final/png/Fig6.png)
**Fig. 6. Safety and calibration analysis under drift.** Subfigure A: structurally invalid prediction rate - OOS; Subfigure B: topology-safe strict errors - Strict Error but Allowed; Subfigure C: NLL calibration metric; Subfigure D: expected calibration error (ECE).

Overall, the drift stress-test shows that structural information helps at two different levels. A topology mask improves safety by removing part of the structurally impossible predictions, but remains limited by the fixed training vocabulary. EOPKG further reduces OOS because it ranks candidates directly in the version-specific space $C^{(v)}$ and conditions this ranking on the current execution state. Therefore, the main benefit of EOPKG is safer release-time prediction under candidate-space mismatch, not merely a higher aggregate accuracy score.

The inference-time diagnostics indicate that the structural component did not introduce an observable bottleneck in the tested setting. All compared models remained within a single-digit millisecond range per sample on the same local CPU environment, while EOPKG added roughly 1-2 ms/sample relative to the lightweight GATv2+Mask control (see Supplementary Table S4). These values should be interpreted as order-of-magnitude evidence rather than absolute latency benchmarks, because the CPU was not isolated from parallel operating-system tasks. The practical implication is that topology-native candidate scoring and impulse routing appear computationally feasible for release-time BPMS prediction when the candidate set is bounded as in Section 3.4.

### 5.4. Statistical Significance

Statistical testing was performed for the main drift metrics from Table 4, that is, for the endpoint comparison on the most distant segment $v_5$, using a paired design with identical seeds ($n=7$). For the comparisons EOPKG vs GATv2+Mask, EOPKG vs GATv2/LSTM, and EOPKG vs EOPKG-WI, an exact paired sign-flip permutation test was used, and effect size is reported through $\Delta$ and standardized paired effect size $d_z$. For metrics where lower values are better (OOS, ECE, NLL), $\Delta$ is computed as $\mathrm{baseline}-\mathrm{EOPKG}$; therefore, a positive value indicates an advantage for EOPKG. Full seed-level results, Holm-adjusted p-values, and MLflow run IDs are provided in the supplementary materials [15].

**Table 5**

Paired significance tests for key drift metrics.

| Metric       | EOPKG vs GATv2                        | EOPKG vs GATv2+Mask                   | EOPKG vs LSTM                         | EOPKG vs EOPKG-WI                     |
| :----------- | :------------------------------------ | :------------------------------------ | :------------------------------------ | :------------------------------------ |
| Strict Macro F1 | $\Delta=0.238$, $d_z=3.85$, $p=0.008$ | $\Delta=0.114$, $d_z=1.81$, $p=0.008$ | $\Delta=0.290$, $d_z=3.83$, $p=0.008$ | $\Delta=0.205$, $d_z=3.10$, $p=0.008$ |
| Macro F1     | $\Delta=0.497$, $d_z=9.15$, $p=0.008$ | $\Delta=0.259$, $d_z=3.73$, $p=0.008$ | $\Delta=0.559$, $d_z=11.04$, $p=0.008$ | $\Delta=0.361$, $d_z=6.55$, $p=0.008$ |
| OOS rate     | $\Delta=0.483$, $d_z=7.18$, $p=0.008$ | $\Delta=0.279$, $d_z=8.89$, $p=0.008$ | $\Delta=0.570$, $d_z=4.41$, $p=0.008$ | $\Delta=0.384$, $d_z=4.55$, $p=0.008$ |
| Test ECE     | $\Delta=0.281$, $d_z=3.09$, $p=0.008$ | $\Delta=0.141$, $d_z=2.35$, $p=0.008$ | $\Delta=0.238$, $d_z=2.73$, $p=0.008$ | $\Delta=0.047$, $d_z=0.74$, $p=0.039$ |
| Test Set NLL | $\Delta=10.492$, $d_z=7.16$, $p=0.008$ | $\Delta=7.885$, $d_z=7.78$, $p=0.008$ | $\Delta=8.983$, $d_z=6.89$, $p=0.008$ | $\Delta=0.465$, $d_z=0.53$, $p=0.109$ |

At the $v_5$ endpoint, compared with fixed-vocabulary baselines and GATv2+Mask, EOPKG shows a consistent advantage in Strict Macro F1, Macro F1, OOS, ECE, and NLL. For Strict Macro F1, the effect against GATv2+Mask is more moderate than for Macro F1 and OOS, but the direction of improvement is preserved across all seed runs. This supports the conclusion that exact prediction improves, while Macro F1 and OOS describe the broader effect of topology-native adaptation. The comparison with EOPKG-WI shows the contribution of impulse routing primarily for Strict Macro F1, Macro F1, and OOS; the NLL difference in this ablation pair is not statistically convincing and is treated as descriptive.



## 6. Threats to Validity

### 6.1. Internal Validity

Internal validity depends on whether the advantage of EOPKG could be explained by differences in configurations, initialization, or checkpoint selection. To reduce this risk, GATv2, GATv2+Mask, LSTM, EOPKG-WI, and EOPKG are compared within the same drift protocol, using the same split policy, seed policy, batch settings, and checkpoint selection rule. Results are reported as $\mathrm{mean} \pm \mathrm{std}$ over seven seeds and are checked using paired statistical testing.

Data leakage is an especially important risk in drift evaluation. The models are trained only on historical versions, and future-version evaluation is performed without using logs or statistics from those versions. Statistical artifacts are available under a point-in-time principle: a prefix cannot use data that chronologically belong to the future relative to the last event.

The hyperparameters of structural conditioning and impulse activation were not optimized as a separate research objective. Therefore, the results should be interpreted as empirical support for the architectural principle, not as proof of a globally optimal configuration. Detailed provenance checks and run-level artifacts are provided in the supplementary materials [15].

### 6.2. External Validity

The main limitation of external validity is that the evaluation is performed on one semi-synthetic versioned dataset, `loan_v1_v5_complex_simulated`. This setting is needed because there is no generally accepted public benchmark that jointly provides event logs, explicit process versions, and deployed topology for each version. Because the same design assumptions define both the generated topology and the evaluation constraints, the mask may reflect the benchmark construction more cleanly than would be possible in noisy real-world BPMS deployments. At the same time, one process does not represent the full diversity of BPMS settings: real logs may contain noisy timestamps, incomplete lifecycle events, implicit business rules, manual workarounds, and heterogeneous resource policies.

The data are intentionally limited to the standard flattened XES level with lifecycle transitions [22]. This makes the experiment closer to an interoperable BPMS scenario, but also means that full execution marking, object-centric links, and rich resource profiles are not used. The model also does not use semantic embeddings: unseen `concept:name` values in the observed prefix are encoded as `<UNK>`, while new activities are ranked through their position in $G^{(v)}$, reachability, and the topology-native candidate space. Therefore, the results demonstrate topology-known structural adaptation, not semantic zero-shot generalization. Real-world versioned logs or few-shot/incremental evaluation on new versions are needed to confirm industrial generalizability.

### 6.3. Construct Validity

Construct validity concerns whether the selected metrics measure the intended aspects of quality. $\mathrm{strict\_macro\_f1}$ remains the primary metric of exact next-activity prediction. $\mathrm{macro\_f1}$ and $\mathrm{strict\_error\_but\_allowed\_rate}$ are not replacements for accuracy; they are secondary topology-tolerant metrics for parallel BPMN segments where several branches are active, do not violate precedence constraints, and must be completed before a join gateway.

To avoid overstating quality, the paper reports exact predictive accuracy ($\mathrm{strict\_macro\_f1}$), structural safety ($\mathrm{OOS}$, $\mathrm{target\_in\_mask\_rate}$), and topology-tolerant validity ($\mathrm{macro\_f1}$, $\mathrm{strict\_error\_but\_allowed\_rate}$) separately. For the main endpoint comparison on $v_5$, the difference between $\mathrm{strict\_macro\_f1}=0.451$ and $\mathrm{macro\_f1}=0.777$ is not interpreted as hidden accuracy improvement. It means that part of the errors remains inside the structurally admissible parallel space. A separate construct assumption is the topology-known setting: the BPMN topology of the future version is available during inference. If the structure of the new version is unknown or unreliable in a real system, the applicability of the approach decreases.


## 7. Conclusions

This paper addressed the release-time adaptation lag in predictive process monitoring: a new process topology may already be deployed, while representative logs of that version are still unavailable for retraining. The proposed EOPKG approach treats the current process version as an explicit topology-conditioned candidate space $C^{(v)}$ and ranks admissible next activities in that space rather than only within the fixed training vocabulary $C_{\mathrm{train}}$.

The empirical results support the central claim that topology-conditioned candidate-space adaptation improves robustness under structural drift. The main endpoint comparison is reported on the most distant future-version segment $v_5$, while the per-version trajectory from $v_3$ to $v_5$ is provided in the supplementary materials. On $v_5$, EOPKG achieved $\mathrm{strict\_macro\_f1}=0.451$ and $\mathrm{macro\_f1}=0.777$, compared with approximately 0.16-0.21 strict-F1 for fixed-vocabulary baselines. Its OOS rate decreased to 0.121, compared with 0.400 for GATv2+Mask and more than 0.60 for logs-only baselines. This shows that topology masking alone improves structural safety, but does not resolve candidate-space mismatch; the strongest effect appears when the model both constrains and positively ranks candidates in the topology-native space. The ablation without impulse confirms that impulse routing contributes primarily to Macro-F1 and OOS reduction. The additional inference check showed single-digit millisecond latency in the tested CPU environment, so no evident runtime bottleneck was observed for online BPMS use cases, although these measurements are not an isolated hardware benchmark.

The main limitation is that the evaluation is based on one semi-synthetic versioned dataset. The results therefore demonstrate controlled structural adaptation under known process topology, not universal industrial generalization. The method also assumes that the deployed topology of the new version is available during inference and does not perform semantic zero-shot reasoning from task names. Future work should validate the approach on real-world versioned logs, study few-shot or incremental adaptation after the first executions of a new version become available, integrate semantic/resource context, and refine gateway-aware impulse routing and topology-mask construction.

## Data and Code Availability

To support full reproducibility, all research materials are made openly available:

**Dataset.** The semi-synthetic versioned event log `loan_v1_v5_complex_simulated` is published in the corresponding repository [13].

**Source code.** The source code of the `bpm_prediction` platform, including the EOPKG implementation and experimental pipeline, is available at the link in [14].

**Reproducibility materials.** Full BPMN diagrams of the process versions ($v_1 \dots v_5$), experiment configuration files, aggregated MLflow metrics, including drift-window slices, and detailed result tables for all seed runs are provided in the supplementary reproducibility package [15].

## Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process
During the preparation of this work, the author used OpenAI ChatGPT/Codex to support language editing, translation, structural revision, and consistency checks. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the article.

## Declaration of Competing Interest
The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## References

[1] IEEE Task Force on Process Mining. _Process Mining Manifesto_. In: Business Process Management Workshops. Springer, 2012. DOI: 10.1007/978-3-642-28108-2_19.

[2] van der Aalst, W. M. P. _Process Mining: Data Science in Action_. Springer, 2016. DOI: 10.1007/978-3-662-49851-4.

[3] Evermann, J., Rehse, J.-R., Fettke, P. Predicting process behaviour using deep learning. _Decision Support Systems_, 100, 129-140, 2017. DOI: 10.1016/j.dss.2017.04.003.

[4] Tax, N., Verenich, I., La Rosa, M., Dumas, M. Predictive Business Process Monitoring with LSTM Neural Networks. In: Advanced Information Systems Engineering. CAiSE 2017, LNCS, vol 10253, 477-492. Springer, Cham, 2017. DOI: 10.1007/978-3-319-59536-8_30.

[5] Rizzi, W., Di Francescomarino, C., Ghidini, C., Maggi, F. M. How do I update my model? On the resilience of predictive process monitoring models to change. _Knowledge and Information Systems_, 64(4), 1065-1108, 2022. DOI: 10.1007/s10115-022-01666-9.

[6] Lischka, A., Rauch, S., Stritzel, O. Directly Follows Graphs Go Predictive Process Monitoring With Graph Neural Networks. arXiv preprint arXiv:2503.03197, 2025.

[7] van Dongen, B. F. BPI Challenge 2012. 4TU.ResearchData, 2011. DOI: 10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f.

[8] Teinemaa, I., Dumas, M., La Rosa, M., Maggi, F. M. Outcome-oriented predictive process monitoring: Review and benchmark. _ACM Transactions on Knowledge Discovery from Data_, 13(2), 1-57, 2019. DOI: 10.1145/3301300.

[9] Bieber, D., Sutton, C., Larochelle, H., Tarlow, D. Learning to execute programs with instruction pointer attention graph neural networks. In: _Advances in Neural Information Processing Systems_, vol. 33, 8626-8637, 2020.

[10] Korotenko, S. A. Comparison of GNN architectures for activity prediction in business processes: Logs-only and BPMN approaches. _Information Technology: Computer Science, Software Engineering and Cyber Security_, Issue 2, 36-47, 2025. DOI: 10.32782/IT/2025-2-5.

[11] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In International conference on machine learning (pp. 1597-1607). PMLR.

[12] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On Calibration of Modern Neural Networks. ICML 2017.

[13] `loan_v1_v5_complex_simulated`: versioned semi-synthetic loan-process event log and process structures. Dataset DOI: [to be added after publication].

[14] `bpm_prediction`: source code for EOPKG implementation and experiment pipeline. Code DOI: [to be added after publication].

[15] Reproducibility materials: experiment configurations, aggregated MLflow metrics, drift-window exports, tables, and supplementary process-structure artifacts. DOI: [to be added after publication].

[16] Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., Liu, T.-Y. On Layer Normalization in the Transformer Architecture. In: _Proceedings of the 37th International Conference on Machine Learning_, PMLR 119, 10524-10533, 2020.

[17] Nguyen, T. Q., Salazar, J. Transformers without Tears: Improving the Normalization of Self-Attention. In: _Proceedings of the 16th International Workshop on Spoken Language Translation_, 2019.

[18] Hochreiter, S., Schmidhuber, J. Long Short-Term Memory. _Neural Computation_, 9(8), 1735-1780, 1997. DOI: 10.1162/neco.1997.9.8.1735.

[19] Brody, S., Alon, U., Yahav, E. How Attentive are Graph Attention Networks? In: _International Conference on Learning Representations_, 2022.

[20] Bose, R. P. J. C., van der Aalst, W. M. P., Zliobaite, I., Pechenizkiy, M. Dealing with Concept Drifts in Process Mining. _IEEE Transactions on Neural Networks and Learning Systems_, 25(1), 154-171, 2014. DOI: 10.1109/TNNLS.2013.2278313.

[21] Käppel, M., Jablonski, S., Schönig, S. Evaluating Predictive Business Process Monitoring Approaches on Small Event Logs. In: _Quality of Information and Communications Technology_, Communications in Computer and Information Science, vol. 1439, 167-182, Springer, Cham, 2021. DOI: 10.1007/978-3-030-85347-1_13.

[22] IEEE. _IEEE Standard for eXtensible Event Stream (XES) for Achieving Interoperability in Event Logs and Event Streams_. IEEE Std 1849-2023, 1-50, 2023. DOI: 10.1109/IEEESTD.2023.10041865.

[23] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. In International conference on machine learning (pp. 1263-1272). PMLR.

[24] Hogan, A., Blomqvist, E., Cochez, M., d'Amato, C., de Melo, G., Gutierrez, C., Labra Gayo, J. E., Kirrane, S., Neumaier, S., Polleres, A., Navigli, R., Ngonga Ngomo, A.-C., Rashid, S. M., Rula, A., Schmelzeisen, L., Sequeda, J., Staab, S., Zimmermann, A. Knowledge Graphs. _ACM Computing Surveys_, 54(4), Article 71, 2021. DOI: 10.1145/3447772.

[25] Zhang, Y., Yao, Q. Knowledge Graph Reasoning with Relational Digraph. In: _Proceedings of the ACM Web Conference 2022_, 912-924, 2022. DOI: 10.1145/3485447.3512008.

[26] Post, M., Vilar, D. Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation. In: _Proceedings of NAACL-HLT 2018_, 1314-1324, 2018. DOI: 10.18653/v1/N18-1119.

[27] van der Aalst, W. M. P., Weijters, A. J. M. M., Maruster, L. Workflow Mining: Discovering Process Models from Event Logs. _IEEE Transactions on Knowledge and Data Engineering_, 16(9), 1128-1142, 2004. DOI: 10.1109/TKDE.2004.47.

[28] Leemans, S. J. J., Fahland, D., van der Aalst, W. M. P. Discovering Block-Structured Process Models from Event Logs - A Constructive Approach. In: _Application and Theory of Petri Nets and Concurrency_, PETRI NETS 2013, LNCS 7927, 311-329. Springer, 2013. DOI: 10.1007/978-3-642-38697-8_17.

[29] van der Aalst, W. M. P. Object-Centric Process Mining: Dealing With Divergence and Convergence in Event Data. In: _Software Engineering and Formal Methods_, SEFM 2019, LNCS 11724, 3-25. Springer, 2019. DOI: 10.1007/978-3-030-30446-1_1.
