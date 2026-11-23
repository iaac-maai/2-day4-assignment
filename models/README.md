---
license: mit
library_name: pytorch
pipeline_tag: tabular-regression
tags:
  - pytorch
  - tabular-regression
  - mlp
  - house-prices
  - real-estate
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - r_squared
  - mae
---

# King County house price MLP: a screening estimate of a house's sale price

This model estimates the **sale price of a house in King County (Washington, USA, including Seattle)** from 13 facts about the building and its location (floor area, rooms, floors, grade, condition, age, renovation, latitude/longitude and ZIP code). It is meant for **architects, developers, buyers and real-estate professionals who want a quick first estimate**, for example to compare design options for a plot in a feasibility study, **before** a professional valuation. It is a small **multilayer perceptron** (82 inputs → 64 → 32 → 1, ReLU, 7,425 parameters), trained from scratch with PyTorch on 2014–2015 sales, that predicts `log10(price)`. On the test set it is typically **12.6% off** the true price (R² = 0.877 on log10(price)). It comes close to our Day 2 gradient boosting model but does **not** beat it (12.2%, R² = 0.886). It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/2-day4-assignment
- **Team:** Karthikeyan Saravanan, Emmanuel Marquez, Bars Babila, Samiha Abeda Nur

## Model at a glance

![Predicted vs actual price on the test set](https://huggingface.co/iaac-maai/day4-group2-house-price-mlp/resolve/main/07_test_predicted_vs_actual.png)

*Predicted against actual price for the 4,147 test houses, for the Day 2 gradient boosting model (left) and this network (right). Both follow the diagonal and about 80% of the houses fall inside the ±20% band; the largest misses are expensive houses that are under-estimated and a few cheap houses that are over-estimated.*

![Screening estimate for two design options](https://huggingface.co/iaac-maai/day4-group2-house-price-mlp/resolve/main/09_application_design_options.png)

*What the model does: two hand-made design options on the same plot in ZIP 98117, keeping a 1952 house of 1,500 sqft (about 504,000 USD) or replacing it with a new 2,800 sqft house (about 825,000 USD). The shaded bands show the test-set uncertainty: the ranking is clear, but each estimate is typically ±12.6% off.*

| Model (test set, 4,147 houses) | R² on log10(price) | Typical error | Mean error (USD) |
|---|---|---|---|
| Baseline: dummy (always the average price) | 0.000 | 47.0% | 196,281 |
| Baseline: Day 2 gradient boosting | 0.886 | 12.2% | 60,201 |
| **This model** | **0.877** | **12.6%** | **62,797** |

*Typical error = 10^MAE(log10) − 1: how far, in percent, a prediction usually is from the true price.*

## Intended use

- **Intended users and use:** architects, developers, buyers and real-estate professionals who need a **quick, rough screening estimate** of the sale value of a house in King County, for example to compare a few design options for one plot (keep, extend or replace a house) before deciding whether a project is worth studying further. The estimate should be read together with its uncertainty (typically ±12.6%, and one house in five more than 20% off) and next to a cost estimate.
- **Out-of-scope uses:**
  - **Any decision about a specific property or person:** setting a sale price or an offer, mortgage lending, insurance, property tax assessment or any official valuation. These need a professional valuation with current market data.
  - **Other regions or periods:** the model only knows King County sales from May 2014 to May 2015. It says nothing about other places, and prices have changed a lot since.
  - **Other building types and atypical properties:** apartments, commercial or mixed-use buildings, land deals, unfinished buildings, distressed sales, waterfront or view properties, and houses outside the ranges of the training data (see *Input features*), e.g. over 4,970 sqft, more than 6 bedrooms or very large rural lots.
  - **Judging design quality:** the model does not see the design, the interior, the materials or the energy performance, so it cannot say which design is better, only roughly what houses with these facts sold for.

## Model details

| | |
|---|---|
| **Task** | Tabular regression: `log10(price)` of one house sale (price in USD) |
| **Architecture** | MLP: 82 inputs → Linear 64 → ReLU → Linear 32 → ReLU → Linear 1 (no dropout) |
| **Parameters** | 7,425 (all trainable) |
| **Base model** | None, trained from scratch |
| **Input** | 13 columns (see below): 12 numeric columns standardised with a `StandardScaler` + `zipcode` one-hot encoded (70 ZIP codes) = 82 inputs. The fitted preprocessor is `scaler.joblib` |
| **Output** | One number: the predicted `log10(price)`. The price in USD is `10 ** output` |
| **Files** | `model.pt` (the `state_dict`, 32 KB), `scaler.joblib` (the fitted scikit-learn `ColumnTransformer`, 3 KB), figures |
| **Library versions** | Saved with Python 3.14.7, torch 2.14.1, scikit-learn 1.9.1, joblib 1.6.0, pandas 3.0.6, numpy 2.5.3 |

### Input features

The columns must be given **in this order**: `bedrooms, bathrooms, sqft_living, sqft_lot, floors, condition, grade, sqft_basement, yr_built, yr_renovated, lat, long, zipcode`. The range is what the model saw in the data (min – max).

| Column | Type | Unit / meaning | Range in the data | Preprocessing |
|---|---|---|---|---|
| `bedrooms` | integer | number of bedrooms | 2 – 6 | standardised |
| `bathrooms` | number | bathrooms (0.25 steps: a toilet only counts 0.5) | 0.5 – 5.25 | standardised |
| `sqft_living` | integer | interior living area, sqft | 720 – 4,970 | standardised |
| `sqft_lot` | integer | lot area, sqft | 520 – 1,651,359 | standardised |
| `floors` | number | number of floors (0.5 = split level) | 1 – 3.5 | standardised |
| `condition` | integer | condition of the house, 1 (poor) – 5 (very good) | 1 – 5 | standardised |
| `grade` | integer | King County building grade, quality of construction and design, 1 – 13 | 4 – 13 | standardised |
| `sqft_basement` | integer | basement area, sqft (0 = no basement) | 0 – 2,350 | standardised |
| `yr_built` | integer | year built | 1900 – 2015 | standardised |
| `yr_renovated` | integer | year of the last renovation, **0 if never renovated** | 0 or up to 2015 | standardised |
| `lat` | number | latitude, degrees | 47.156 – 47.778 | standardised |
| `long` | number | longitude, degrees | −122.519 – −121.316 | standardised |
| `zipcode` | integer (category) | ZIP code | 70 ZIP codes of King County | one-hot; an unknown ZIP code gets all zeros |

## Training data

| | |
|---|---|
| **Dataset** | House Sales in King County, USA, as cleaned in our [Day 1 / Day 2 repository](https://github.com/iaac-maai/2-day2-assignment) |
| **Source** | [Kaggle: harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) |
| **Licence** | CC0 1.0 (public domain) |
| **Size** | 20,703 sales (one row = one sale): train 13,244 / validation 3,312 / test 4,147 |
| **Splits** | The test set is exactly our Day 2 test set (`GroupShuffleSplit`, 20%, `random_state=42`, grouped by house `id` so both sales of a house sold twice stay on the same side). Validation = 20% of the Day 2 training set, split the same way |
| **Coverage** | Houses sold in King County (Washington, USA, including Seattle) between May 2014 and May 2015 |
| **Cleaning** | Done on Day 1: dates converted, `waterfront` and `view` columns removed, extreme prices, living areas and bedroom counts filtered out. The preprocessor is fitted on the training set only |

## Training procedure

- **Settings:** Adam, learning rate 0.001 halved when the validation error has not improved for 5 epochs (`ReduceLROnPlateau`), weight decay 1e-4, Huber loss (δ = 0.05) on `log10(price)`, batches of 64, up to 300 epochs, seed 42. The output bias starts at the average `log10(price)` of the training set.
- **Regularisation and model selection:** weight decay and early stopping. After each epoch we compute the typical error on the validation set and keep the epoch with the lowest one; training stops after 30 epochs without progress. The published weights kept **epoch 90 of 120** (validation typical error 12.29%). The test set was used once, at the end.
- **How the settings were chosen:** 27 experiments in three rounds, changing one thing at a time, each trained with 3 seeds and compared on the validation set. Starting the output bias at the mean price was by far the best single change (14.4% → 12.7%); weight decay, Huber loss and the learning-rate schedule then brought it to 12.3%. Bigger networks and a larger learning rate made it worse.
- **Hardware and time:** laptop CPU, no GPU. The final network trains in about 20 seconds; the whole notebook, including the 81 training runs of the experiments, takes about 6 minutes on an Apple M-series laptop.
- **Note on reproducibility:** the published weights were re-trained with the same code, seed and settings on a Windows laptop CPU (one thread). PyTorch is not bit-identical across machines, so they differ slightly from the network shown in section 6 of the notebook (that one kept epoch 75 of 105, with a mean error of 62,682 USD; R² and typical error are the same). All numbers in this card are those of the published weights. The figures of the training curves, the predicted vs actual plot and the price bands come from the notebook run; per price band the published weights differ from them by at most 0.3 point.

![Training curves](https://huggingface.co/iaac-maai/day4-group2-house-price-mlp/resolve/main/06_training_curves_final.png)

*Typical error after each epoch. Left: the first settings start at 154% error and zig-zag around 14–17%. Right: the final settings start at a sensible level and become smooth after each halving of the learning rate. The training error ends only about 0.5 point below the validation error, so there is little overfitting.*

## Evaluation

Test set: 4,147 houses, used once, next to the two baselines.

| Metric (test set) | Dummy | Day 2 gradient boosting | This model |
|---|---|---|---|
| R² on log10(price) | 0.000 | 0.886 | **0.877** |
| Typical error | 47.0% | 12.2% | **12.6%** |
| Mean absolute error (USD) | 196,281 | 60,201 | **62,797** |
| Houses within ±10% | – | 55.8% | **54.4%** |
| Houses within ±20% | – | 80.6% | **79.8%** |

On validation the gap was the same (12.3% against 11.9% for the Day 2 model), so the experiments did not overfit the validation set. For a single house the two models are about equally good: this network is closer to the true price than the Day 2 model for 48% of the test houses.

**Typical error per price band** (each band = 20% of the test houses):

| Price band (USD) | Day 2 gradient boosting | This model |
|---|---|---|
| 154k – 300k | 13.5% | 13.7% |
| 300k – 398k | 11.1% | 11.4% |
| 398k – 515k | 11.3% | 11.3% |
| 515k – 685k | 10.4% | 10.9% |
| 685k – 1,950k | 14.7% | 15.9% |

![Test error per price band](https://huggingface.co/iaac-maai/day4-group2-house-price-mlp/resolve/main/08_test_error_by_price_band.png)

*In the middle of the market the network is about as good as gradient boosting. It loses at both ends, most of all for the most expensive 20% of houses (the figure comes from the notebook run; the table above gives the numbers of the published weights).*

**Where it fails:** the largest errors are houses that sold far below what their description suggests, which the model cannot recognise: for example a new 4,460 sqft house sold for 490,000 USD and predicted at 1.4 million, and old houses on very large lots. It is least reliable for houses under about 300,000 USD and over about 685,000 USD, where the training data is thinnest.

## Limitations and biases

- **Period and place:** sales in King County in 2014–2015 only. Prices have changed since, so the estimates are 2014–2015 prices.
- **Inherited cleaning:** extreme prices, areas and bedroom counts were removed on Day 1, as were the `waterfront` and `view` columns, so the most unusual and most expensive houses are under-represented.
- **Missing information:** no interior quality, architectural design, materials, energy performance, neighbourhood amenities or market conditions, which also drive prices.
- **Worse at the ends of the market:** typically 14–16% off below about 300,000 USD and above about 685,000 USD.
- **Special sales:** it cannot recognise unfinished buildings, land deals or distressed sales, and should not be used to value an atypical property.
- **Location bias:** latitude, longitude and ZIP code are strong inputs, so the model reproduces the price differences between neighbourhoods as they were in 2014–2015, including any historical inequalities between them.
- **Not more accurate than a simpler model:** gradient boosting is slightly better, needs no tuning and trains faster. This network is a second opinion, not a replacement.

## Ethics and privacy

The data contains no people and no personal information beyond public sale records of houses. Because a price estimate can affect people (a seller, a buyer, a borrower), the model must not be used for any decision about a specific property or person (see *Out-of-scope uses*).

## How to use

A `state_dict` only contains the numbers, so the network must first be rebuilt with **exactly** this `nn.Sequential`. The preprocessor (`scaler.joblib`) turns the 13 columns, in this order, into the 82 inputs.

```python
import joblib
import pandas as pd
import torch
from torch import nn
from huggingface_hub import hf_hub_download

REPO = "iaac-maai/day4-group2-house-price-mlp"
COLUMNS = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "condition", "grade",
           "sqft_basement", "yr_built", "yr_renovated", "lat", "long", "zipcode"]   # this order

model = nn.Sequential(                       # exactly the trained architecture: 82 -> 64 -> 32 -> 1
    nn.Linear(82, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
model.load_state_dict(torch.load(hf_hub_download(REPO, "model.pt"), map_location="cpu"))
model.eval()
preprocess = joblib.load(hf_hub_download(REPO, "scaler.joblib"))   # StandardScaler + one-hot ZIP code

house = pd.DataFrame([{
    "bedrooms": 3, "bathrooms": 1.0, "sqft_living": 1500, "sqft_lot": 4432, "floors": 1.0,
    "condition": 3, "grade": 7, "sqft_basement": 0, "yr_built": 1952, "yr_renovated": 0,
    "lat": 47.686, "long": -122.379, "zipcode": 98117,
}])[COLUMNS]

x = torch.tensor(preprocess.transform(house), dtype=torch.float32)
with torch.no_grad():
    log_price = model(x).item()
print(f"Screening estimate: {10 ** log_price:,.0f} USD (2014-2015 prices, typically ±12.6%)")
```

This prints `Screening estimate: 504,490 USD (2014-2015 prices, typically ±12.6%)`: option A of the application above (the notebook gives 504,491 USD with unrounded coordinates and lot size).

⚠️ **Security note:** `.pt` and `.joblib` files are Python pickles and can execute code when they are loaded: only load models from sources you trust. `torch.load` uses `weights_only=True` by default for the weights; the scaler needs a full unpickle, so load it only from this repository. Use the library versions listed above (a different scikit-learn version may fail to load `scaler.joblib` or warn).

## Reproducibility

The full pipeline is in the GitHub repository: https://github.com/iaac-maai/2-day4-assignment. Run `notebook/neural_network.ipynb` (from the `notebook/` folder) with the packages pinned in `requirements.txt`; the cleaned data is downloaded automatically. Sections 7 and 8 of the notebook save, reload and apply this model.

## AI usage disclosure

- **Claude (chat)** helped draft the GitHub README and `.gitignore`, explain errors and adapt the class notebook to our data. We wrote the notebook code ourselves and checked every answer.
- **Claude Code** (an agent that edited files and ran code by itself) was used to add the save/reload and Application cells to the notebook (sections 7 and 8), write this model card, replace the remaining TODOs in the README and publish the files to Hugging Face. We reviewed the changes.

## Authors and licence

Karthikeyan Saravanan, Emmanuel Marquez, Bars Babila, Samiha Abeda Nur. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The data is CC0 1.0 (public domain): [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).
