# Repository for Integrative Post-Concussion Musculoskeletal Injury Risk Model for Collegiate Athletes
![timeline](figs/r21_timeline.svg)

Over the past twenty years, the early identification and diagnosis of concussions related to sports activities have significantly improved. Nevertheless, accurately determining when an athlete has fully recovered from a concussion continues to pose a challenge. Concussions can impair cognitive abilities, motor skills, vestibulo-ocular, and cardiovascular functions, and are linked to an increase in both somatic and psychological issues. While health care professionals employ clinical evaluations and neurological screening tools to detect concussions in their acute phase, athletes often experience lingering physiological effects even after they have been deemed clinically recovered. This suggests that athletes might be resuming their activities with underlying neurological deficiencies, having only completed a clinical recovery and exercise progression program.

This study aims to leverage an integrative statistical model using clinically applicable data to formulate a risk profile for musculoskeletal (MSK) injuries tailored to college athletes who have experienced concussions. We propose that a comprehensive risk score will be generated to distinguish athletes at risk of incurring a MSK injury post-concussion. By analyzing data from 211 athletes, our methodology—combining Weight of Evidence (WoE) transformation and logistic regression analysis—proved effective, as evidenced by achieving a 0.82 area under the curve (AUC) on the receiver operating characteristic curve.

## Proposed methodology
- Pre-processing [preprocessing](notebooks/preprocessing.ipynb)
- Feature selection [feature_selection](notebooks/feature_selection.ipynb)
- Model training [model training](notebooks/model_training.ipynb)
- Model evaluation [model evaluation](notebooks/model_evaluation.ipynb)
