# Master’s Thesis 

## 1.Thesis Title

**Forensic Linguistics and AI for the Detection of Machine-Generated Disinformation Campaigns**

## 2. Abstract

Large language models can now generate fluent, persuasive, and human-like English text at scale. Although these systems support many beneficial applications, they also create serious risks when used to produce synthetic disinformation, propaganda, misleading news, fake reviews, and coordinated influence content. Detecting such text is difficult because modern generated writing is often coherent, polished, and stylistically flexible.

This thesis proposes **FLAIRD: Forensic Linguistic and AI-Integrated Robust Detector**, an English-language hybrid framework for detecting machine-generated disinformation text. The system combines a pretrained Transformer encoder with computational forensic linguistic features. The Transformer captures contextual and semantic patterns, while the forensic component extracts interpretable evidence related to lexical richness, syntactic complexity, discourse coherence, stylometry, readability, entropy, and burstiness. These two evidence streams are integrated using a gated cross-attention fusion module.

The proposed system performs two related tasks. The main task is binary classification: distinguishing human-written text from machine-generated text. The auxiliary task is multi-class generator prediction: identifying the likely model family that generated a machine-written text. This auxiliary task is included because different language models may leave different linguistic and structural signatures. Learning these generator-specific patterns can strengthen the shared representation and improve the binary detector.

The system will be evaluated using in-domain, cross-domain, cross-generator, adversarial, calibration, and ablation experiments. The explanation module will present attention visualizations, fusion behavior, and ranked forensic linguistic indicators to support interpretable analysis.

The expected result is a robust, interpretable, and academically defensible detection framework for machine-generated disinformation text. Arabic-language support and multilingual detection are outside the main scope and will be considered future work.

## 3. Introduction

Large language models have changed the modern information environment. They can generate news-style articles, opinion pieces, social-media posts, educational explanations, and persuasive arguments that closely resemble human writing. This ability becomes risky when it is used to produce disinformation at scale.

Machine-generated disinformation is difficult to detect for three main reasons. First, generated text can be grammatically correct and semantically coherent. Second, a malicious actor can quickly create many variations of the same message across different domains and styles. Third, modern models can imitate human-like tone, structure, and argumentation, making simple rule-based detection insufficient.

Existing detection methods have important limitations. Purely neural detectors may achieve high performance in controlled settings, but they are often difficult to interpret and may fail when tested on unseen domains or generators. Feature-based forensic linguistic systems provide clearer evidence, but they may not capture deeper semantic patterns learned by modern language models.

This thesis proposes a hybrid approach that combines the strengths of both directions. The system integrates neural semantic representations with forensic linguistic evidence and uses a multi-task design that learns both binary machine-text detection and generator prediction.

## 4. Problem Statement

The core problem is that machine-generated English disinformation text is increasingly difficult to distinguish from human-written text, especially when it is persuasive, polished, paraphrased, or produced by different model families.

Current detectors face four major challenges:

1. **Limited interpretability:** many detectors output only a probability score without explaining the linguistic evidence behind the decision.
2. **Weak generalization:** detectors trained on one domain or generator may perform poorly on unseen domains or model families.
3. **Insufficient forensic grounding:** many systems do not use explicit linguistic features that support academic or investigative reasoning.
4. **Limited use of generator information:** binary labels ignore the fact that different models may produce different linguistic signatures.

This thesis addresses these challenges by proposing an interpretable hybrid architecture with binary classification and multi-class generator prediction.

## 5. Research Aim

The aim of this thesis is to design, implement, and evaluate an English-language hybrid AI system for detecting machine-generated disinformation text by integrating Transformer-based semantic modeling, forensic linguistic features, and supervised multi-task learning.



## 6. Research Objectives

The objectives are:



## 7. Research Questions



## 8. Research Hypotheses





## 9. Significance of the Study

This research is significant for both scientific and practical reasons.

Scientifically, it contributes to AI-generated text detection by proposing a hybrid architecture that integrates semantic representations, forensic linguistic features, and generator-aware multi-task learning. It also evaluates whether generator prediction can strengthen binary detection by helping the model learn finer-grained synthetic writing patterns.

Practically, the system can support researchers, analysts, educators, and security institutions by identifying potentially machine-generated disinformation text. The system is not intended to act as an automatic judge. It is a decision-support tool that provides probability scores, generator-related analysis, attention visualization, and interpretable linguistic evidence.

## 10. Proposed System Architecture

The proposed system is named:

**FLAIRD: Forensic Linguistic and AI-Integrated Robust Detector**

It consists of the following components:



## 11. Architecture Diagram

```mermaid
flowchart TD
  
```

## 12. Methodology



## 12.1 Data Collection



## 12.2 Data Cleaning and Splitting




## 13. Feature Engineering



## 14. Model Training




## 15. Loss Function








## 16. Explanation Module


## 17. Evaluation Plan






## 18. Conclusion


