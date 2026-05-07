# FLAME: Adaptive Mixture-of-Experts for Continual Multimodal Multi-Task Learning

## Instructions to Run:
Under your virtual environment, run
```
pip install -r requirements.txt
```
To train the models, go to dir `clinical-highmmt/src/` and run
```
./run.sh
```
The results will be saved under `clinical-highmmt/src/results/`.

To analyze the weights, go to `clinical-highmmt` and run 
```
./src/analysis/run_analysis.sh
```
Results will be saved under `clinical-highmmt/src/analysis/results/`.
