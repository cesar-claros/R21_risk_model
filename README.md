# Repository for Integrative Post-Concussion Musculoskeletal Injury Risk Model for Collegiate Athletes
![timeline](figs/r21_timeline.svg)

Over the past twenty years, the early identification and diagnosis of concussions related to sports activities have significantly improved. Nevertheless, accurately determining when an athlete has fully recovered from a concussion continues to pose a challenge. Concussions can impair cognitive abilities, motor skills, vestibulo-ocular, and cardiovascular functions, and are linked to an increase in both somatic and psychological issues. While health care professionals employ clinical evaluations and neurological screening tools to detect concussions in their acute phase, athletes often experience lingering physiological effects even after they have been deemed clinically recovered. This suggests that athletes might be resuming their activities with underlying neurological deficiencies, having only completed a clinical recovery and exercise progression program.

This study aims to leverage an integrative statistical model using clinically applicable data to formulate a risk profile for musculoskeletal (MSK) injuries tailored to college athletes who have experienced concussions. We propose that a comprehensive risk score will be generated to distinguish athletes at risk of incurring a MSK injury post-concussion. By analyzing data from 194 athletes, our methodology—combining Weight of Evidence (WoE) transformation and logistic regression analysis—proved effective, as evidenced by achieving a 0.82 area under the curve (AUC) on the receiver operating characteristic curve.

## Data
The longitudinal dataset consists of 211 concussed student athletes at the University of Delaware—a NCAA Division I program, and includes demographic information, medical history, concussion injury and recovery information, and common data elements (CDEs) across clinical milestones collected 2015–2023 as described previously. Data collected between 2015 and 2021 were part of the Concussion Assessment, Research, and Education (CARE) Consortium}. 

Measurements are taken across four time points— Baseline, Acute ($ < $48 hours post-concussion), Asymptomatic  (when no concussion symptoms are reported), and Return to Play (when the student-athlete returns to full participation without restriction). The differences of measurements from baseline to each of the subsequent time points are included as additional variables. The remaining variables include information regarding athletes' previous injury information, demographic data, and psychological assessments. In total, $P=135$ predictors are considered for this study. All participants provided written and oral informed consent, and some participants consented to only a subset of access, as approved by the University of Delaware institutional review board. Out of the 211 athletes, only those who had information for at least one timepoint were considered for this analysis. In consequence, the analysis proceeded with 194 athletes, which were further divided into training and test sets: the training set is composed of $N=155$ athletes and the test set contains $N_{\text{test}}=34$ athletes. The test set is a held-out set that is only used for evaluation purposes and has no influence in the training process.

## Proposed methodology
- [Pre-processing](notebooks/preprocessing.ipynb)
- [Feature selection](notebooks/feature_selection.ipynb)
- [Model training](notebooks/model_training.ipynb)
- [Model evaluation](notebooks/model_evaluation.ipynb)

