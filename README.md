### Demonstration of Antenna Autotuning by Machine Learning

# Introduction
This is a sub-project for antenna design automation. To nearly fully automate the antenna design process subject to basic antenna parameters specifications, please refer to my [main project](https://github.com/electronics10/Topology_Optimization). The computational power and effort are unreal for individual researchers to train a good enough model to replace electromagnetic solver. Thus, work and effort exerted to automate the antenna design by machine learning often result in unuseful or redundant small models that only applies to very specific problems. With this being said, I decided to focus on autotuning PIFA antenna that operates at 2.45 GHz. The reason is that PIFA antenna is an easy yet great and commonly used antenna design. Although PIFA is such a convenient and well studied antenna design, the surrounding materials, e.g. the cover or frame of a phone and the feed point postion, may change in different product designs. Therefore, a few proof of concepts models that can autotune the PIFA antenna parameters according to the surroundings are trained in this project to demonstrate the integrationg of ML in antenna design.

A tabular data `data\data.csv` would be used to train the model. The dataset is acquired by utilizing CST Studio Suite® optimizer to make sure the parameters setup of all PIFA antenna have S11 from 2.4~2.5 GHz below -6 dB. The acqusition process is automated by the script `pre_data_acqusition.py`, and the software setup is identical to my [main project](https://github.com/electronics10/Topology_Optimization). One can obtain their own data by modifying and running the script. The dataset is in the form of 15 numbers in each row/sample with `300` samples. The first eleven numbers are the input, with the first number representing the x postion of feed point and the remaining represents a binary sequence implying whether there are PEC surrounding blocks at some fixed positions. The rest 4 numbers are the output, representing the parameters of PIFA antenna shape. ~~A few models are trained on the data and can be compared parallely. To justify the comparison, hyperparameter optimization for each model is performed using random search with 5-fold cross-validation, evaluating 15 trials per model. The best configurations were used to train the final models, ensuring a fair comparison.~~

## Known Issues and Caveats
This code accompanies Chapter 3 of my master's thesis (NTU, 2025, doi:10.6342/NTU202501972). A review in Nov 2026 found the issues below. They have **not** been fixed yet. Please read them before reusing the code or citing the results.

**Results**
- The reported test MSE values (FNN 0.416, RNN 0.464, weighted FNN 0.578, TabTransformer 1.404) come from single runs without fixed seeds. Re-running the same FNN with 8 different seeds gives MSE from about 0.43 to 0.54. This spread is larger than the gaps between FNN, RNN and weighted FNN, so the ranking between these three models is not supported.
- TabTransformer (MSE 1.404) is worse than a trivial baseline that always predicts the training mean (MSE about 1.27).
- The test set is also used for early stopping, so the test MSE is optimistic.

**Data**
- The CST optimizer goal in `settings.py` is S11 < **-10 dB**, while the thesis states -6 dB.
- `pre_filter_data.py` keeps a sample if its *average* S11 over 2.4-2.5 GHz is at or below -4 dB. As a result, 39 of the 281 kept samples (about 14%) do not meet S11 < -6 dB over the whole band.
- In some test cases the CST optimizer itself does not reach -6 dB (e.g. thesis Case 2, Fig. 3.9). So "ground truth" here means "the CST optimizer's result", not "a design that meets the spec".
- Some samples sit at the parameter bounds (e.g. L or D = 1 mm). The inverse mapping is not unique, which inflates the MSE.

**Code bugs**
- `train_*.py`: `best_model_state = model.state_dict()` stores references, not a copy, and is never loaded. The saved model is the *last* model, not the best one. Fix: `copy.deepcopy(model.state_dict())` and load it before saving.
- `train_*.py`: `pd.read_csv('data/data.csv')` reads the first data row as a header, so one sample is silently dropped (281 -> 280).
- No `torch.manual_seed`, so results are not reproducible.
- Scalers are fit on the full dataset before the train/test split (minor leakage).
- TabTransformer is trained with mini-batches for 1000 epochs, while the other models use full-batch training for up to 10000 epochs, so the comparison is not like-for-like.
- `FeedbackGRUModel` applies one GRU cell 5 times to the same input. It is not a 2-layer GRU over a sequence, as the thesis describes.
- Predictions that violate the design-region limits are clipped afterwards by `post_check.py`. The thesis does not mention this step.
- The README says hyperparameters were chosen by random search with cross-validation (struck out above). This is not in the code; the hyperparameters are fixed.

**Thesis errata (Chapter 3)**
- Table 3.3, Case 3: the surrounding blocks should be `1010011101`, not `0110101010` (Fig. 3.7 and the data agree on `1010011101`).

---
## Prerequisites
Before running the code, ensure you have the following installed:
- **Git**: To clone this repository.
- **Miniconda** or **Anaconda**: To manage the Python environment and dependencies.

---

## Step-by-Step Setup Instructions

### 1. Install Git
- **Windows**: Download and install Git from [git-scm.com](https://git-scm.com/downloads). Follow the default installation options.
- **Mac**: Open a terminal and run `brew install git` (if you have Homebrew) or download from the website above.
- **Linux**: Open a terminal and run `sudo apt install git` (Ubuntu/Debian) or `sudo yum install git` (CentOS/RHEL).

Verify Git is installed by running:
```
git --version
```

---

### 2. Install Miniconda
Miniconda is a lightweight version of Anaconda that manages Python environments.
- Download Miniconda from [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html) (choose the version for your OS: Windows, Mac, or Linux).
- Follow the installer instructions:
  - **Windows**: Double-click the `.exe` file and follow the prompts. Check "Add Miniconda to my PATH" if available.
  - **Mac/Linux**: Run the downloaded `.sh` file in a terminal (e.g., `bash Miniconda3-latest-Linux-x86_64.sh`) and follow the prompts.
- Restart your terminal after installation.

Verify Miniconda is installed by running:
```
conda --version
```

---

### 3. Clone the Repository
In a terminal or command prompt, navigate to the folder where you want the project and run:
```
git clone https://github.com/electronics10/mlpifa.git
```
Then, enter the project directory:
```
cd mlpifa
```

---

### 4. Set Up the Conda Environment

This project uses a predefined environment file (environment.yml) to install all dependencies.

- Create the environment:
  - **Windows**:
  ```
  conda env create -f environment_win.yml
  ```

  - **Mac**:
  ```
  conda env create -f environment_mac.yml
  ```

- Activate the environment:
```
conda activate mlpifa
```

Please refer to my [main project](https://github.com/electronics10/Topology_Optimization) to create environment `autotune`.

---

### 5. Run the Code
- For new or more data acquirement:
  ```
  conda activate autotune
  python pre_data_acquisition.py
  ```
  Data is stored at `./data`
- After acquiring the data, filtered bad data samples:
  ```
  python pre_filter_data.py
  ```

- For training, change the the name of the filtered data `data_filtered.csv` to `data.csv` and run the script:
  ```
  conda deactivate
  conda activate mlpifa
  python train_FNN.py
  ```
  Trained models would be saved to `./artifacts_FNN`. Training and validation loss and Prediction vs. True value figures would also be stored. Most importantly, we'll get a file `post_prediction.csv` for further inspection or post processing with 15 randomly genereted inputs and corresponding predictions by the trained model.

- To show and compare the results of the models in post processing, copy `post_prediction.csv` to the folder `./data` and run the scripts:
     ```
     conda deactivate
     conda activate autotune
     python post_check.py
     python post_compare.py
     python post_plot_s11.py
     ```
  `post_check.py` file is used to map the predictions to its closest upper bound satisfying the physical constraints of the design region. `post_compare.py` is used to obtain the s11 data by CST.
---

## Troubleshooting
- If `conda` commands don’t work, ensure Miniconda is added to your system PATH or restart your terminal.
- For errors during `conda env create`, ensure you have an active internet connection, as it downloads packages.

---
