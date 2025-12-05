# Clinical Multimodal-Multitask Transformer

## Instructions to Run:
Under your virtual environment, run
```
pip install -r requirements.txt
```
To train the models, under dir `clinical-highmmt/src/`, run
```
run.sh
```
The results will be saved under `clinical-highmmt/src/results/`.

To analyze the weights, go to `clinical-highmmt` and run 
```
src/analysis/run_analysis.sh
```
Results will be saved under `clinical-highmmt/src/analysis/results/`.