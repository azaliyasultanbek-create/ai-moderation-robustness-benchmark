# Domain model (ML track): formal definition of the task, loss and metrics

Week-9 studio artifact. It formalises the experiment of the methodology card
(`docs/methodology_card.md`): what is classified, how the models are trained, how the text is
obfuscated, and exactly how every reported number is computed.

## 1. Task and data

A moderation model decides whether a post is harmful. Let $\mathcal{V}^*$ be the set of all
strings and $\mathcal{Y}=\{0,1\}$ the labels (1 = harmful). The data are
$\mathcal{D}_{\text{train}}=\{(x_i,y_i)\}_{i=1}^{n}$ ($n=9000$) and
$\mathcal{D}_{\text{test}}=\{(x_j,y_j)\}_{j=1}^{m}$ ($m=2970$). Training uses clean text only.

A detector is a scoring function $f_\theta:\mathcal{V}^*\to[0,1]$ with
$f_\theta(x)=\hat P(y=1\mid x)$ and decision rule
$\hat y_\theta(x)=\mathbb{1}\,[\,f_\theta(x)\ge\tau\,]$, with fixed threshold $\tau=0.5$.

## 2. Obfuscation operator (independent variable)

For an obfuscation type $k\in\mathcal{K}=\{\text{none},\text{leet},\text{insert},\text{homoglyph},\text{swap}\}$ the operator
$T_k:\mathcal{V}^*\to\mathcal{V}^*$ rewrites a random share $\rho=0.5$ of the eligible words
(length $\ge 4$ letters; `@user`, hashtags and URLs are untouched). It is deterministic given
the seed and the item id, and $T_{\text{none}}(x)=x$. Every test item is evaluated in all
conditions, so the design is **paired**.

## 2b. Normalisation operator (proposed method, Task 3)

The proposed pipeline applies $N:\mathcal{V}^*\to\mathcal{V}^*$ before the detector, both when a TF-IDF model is trained
and when any model is used: $\hat y^{N}_\theta(x)=\mathbb{1}\,[\,f_\theta(N(x))\ge\tau\,]$. $N$ is a composition of five rules
applied in order: $N=r_5\circ r_4\circ r_3\circ r_2\circ r_1$ with $r_1$ = Unicode NFKC + lower case, $r_2$ = look-alike
Cyrillic/Greek letter $\to$ Latin letter, $r_3$ = remove `. * _ -` between two word characters, $r_4$ = digit/symbol used as a
letter $\to$ letter (`0→o, 1→i, 3→e, 4→a, 5→s, 7→t, 8→b, @→a, $→s`), $r_5$ = a run of $\ge 3$ identical characters $\to$ one.
Tokens without letters, `@mentions`, `#hashtags` and links are left unchanged.

## 3. Loss functions

**TF-IDF + logistic regression** (`tfidf_word_lr`, `tfidf_char_lr`). With $\phi(x)\in\mathbb{R}^d$ the
sub-linear TF-IDF vector and $f_\theta(x)=\sigma(w^\top\phi(x)+b)$, $\theta=(w,b)$, training minimises the
class-balanced, $\ell_2$-regularised log-loss

$$
\mathcal{L}(\theta)=-\frac{1}{n}\sum_{i=1}^{n}c_{y_i}\Big[y_i\log f_\theta(x_i)+(1-y_i)\log\big(1-f_\theta(x_i)\big)\Big]+\frac{1}{2Cn}\lVert w\rVert_2^2,
\qquad c_y=\frac{n}{2\,n_y},\; C=1 .
$$

The word model uses word 1–2-grams, the character model uses `char_wb` 2–5-grams as features $\phi$.

**Pre-trained transformer** (`roberta_hate`). The weights $\theta^\*$ are fixed and **not re-trained**
here; they come from minimising the cross-entropy over the original (external) training data,
$\mathcal{L}_{\text{CE}}=-\sum_{c}y_c\log \operatorname{softmax}(z_\theta(x))_c$. In this study only
inference is run, so no loss is optimised for this model.

## 4. Metrics

Let $H=\{j:y_j=1\}$ and $B=\{j:y_j=0\}$ be the harmful and benign test items.

| Role | Definition |
|---|---|
| **Primary: recall under condition $k$** | $\displaystyle R_k=\frac{1}{\lvert H\rvert}\sum_{j\in H}\mathbb{1}\,[\hat y_\theta(T_k(x_j))=1]$ |
| **Effect: paired drop** | $\Delta_k=100\,(R_{\text{none}}-R_k)$ percentage points |
| **Effect of normalisation** | $G_k=100\,(R^{N}_k-R_k)$ pp, where $R^{N}_k$ is recall of the normalised pipeline; share recovered $=G_k/\Delta_k$ |
| Guardrail 1: false-positive rate | $\displaystyle \mathrm{FPR}_k=\frac{1}{\lvert B\rvert}\sum_{j\in B}\mathbb{1}\,[\hat y_\theta(T_k(x_j))=1]\le 0.20$ |
| Guardrail 2: latency | $p_{95}\big(\ell\big)\le 100$ ms per text (CPU, batch size 1), $\ell_b=t_b/\lvert b\rvert$ for batch $b$ with wall time $t_b$ |
| Guardrail 3: memory | peak resident set size $\le 4096$ MB |
| Guardrail 4: errors | $\#\{\text{items without decision}\}/\#\{\text{items}\}=0$ |
| Throughput | $\mathrm{RPS}=N/T$ (texts per second), $N$ texts scored in wall time $T$ |

## 5. Statistical decision rule

For each model and each $k\neq\text{none}$ let $b$ be the number of harmful items detected when clean but
missed when obfuscated, and $c$ the reverse. Under $H_0$ ($R_k\ge R_{\text{none}}$) the discordant pairs are
$b\sim\mathrm{Bin}(b+c,\tfrac12)$, so the one-sided exact McNemar p-value is

$$
p_k=\Pr\big[\mathrm{Bin}(b+c,\tfrac12)\ge b\big].
$$

**H2 (Task 3).** The same test with the normalised pipeline as the first version and the raw pipeline as the second,
on the same obfuscated posts: $b$ = caught only after normalisation, $c$ = caught only without it.

The $p_k$ are Holm-adjusted over all model × obfuscation pairs, separately for H1 and H2. $H_0$ is rejected only if
$p_k^{\text{Holm}}<\alpha=0.05$ **and** $\Delta_k\ge 5$ pp. Uncertainty is quantified with the percentile bootstrap
($B=2000$ resamples of items, seed 42): the paired CI of $\Delta_k$ resamples the *pairs*; the CIs of the
median and of $p_{95}$ of latency resample the evaluation batches.

## 6. Notation

| Symbol | Meaning |
|---|---|
| $x,\ y$ | post (string) and its label, 1 = harmful |
| $\mathcal{D}_{\text{train}},\ \mathcal{D}_{\text{test}}$ | training split ($n=9000$), test split ($m=2970$) |
| $f_\theta$ | detector, output $\hat P(y=1\mid x)$ |
| $\theta=(w,b)$ | parameters of logistic regression |
| $\phi(x)$ | TF-IDF feature vector |
| $\tau$ | decision threshold, 0.5 |
| $C,\ c_y$ | inverse regularisation strength (1), class weight |
| $\mathcal{L}$ | training loss (balanced, regularised log-loss) |
| $T_k,\ \rho$ | obfuscation operator of type $k$, share of words changed (0.5) |
| $H,\ B$ | harmful / benign test items |
| $R_k,\ \Delta_k$ | recall under condition $k$, paired drop in pp |
| $\mathrm{FPR}_k$ | false-positive rate under condition $k$ |
| $b,\ c$ | discordant pairs of the McNemar test |
| $p_k,\ p_k^{\text{Holm}}$ | raw and Holm-adjusted p-value |
| $\ell,\ p_{95}$ | latency per text (ms), its 95th percentile |
| $\mathrm{RPS}$ | throughput, texts per second |
| $\alpha$ | significance level, 0.05 |
| $N,\ r_1..r_5$ | normalisation operator and its five rules |
| $R^{N}_k,\ G_k$ | recall with normalisation, gain from normalisation (pp) |
