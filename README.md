
### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate sail
```

---

### 3. Run detection
```bash
bash run_detection.sh
```
### 4. Evaluate detection results
```bash
python detect_eval.py
```
### Notes
The code for our metric can be seen in "call" function of "LocalStableDiffusionPipeline" class in local_model/pipe.py
