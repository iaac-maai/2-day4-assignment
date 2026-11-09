# Predicting King County house prices with a neural network

**Team:** Karthikeyan Saravanan, Emmanuel Marquez, Bars Babila, Samiha Abeda Nur
**Course:** AI Methods I – Day 4: Neural networks and computer vision (IAAC MaAI 2026/27)
**Track:** A1 – MLP on tabular data
**Model on Hugging Face:** TODO (Person 4): `https://huggingface.co/iaac-maai/day4-group2-house-price-mlp`

TODO (Person 2 or 4): one figure at the top that shows the model at work (e.g. one house travelling through the network, or predicted vs. actual price).

## Use case

Real estate professionals, buyers and lenders who want a **quick first estimate** of what a house in King County (Washington, USA, including Seattle) is worth, from a few facts about the building and its location, before a full professional valuation. Over-estimating the price can make a buyer overpay or a lender lend too much; under-estimating it can make a seller sell too cheaply. Because an error of 50,000 USD matters much more for a cheap house than for an expensive one, we predict `log10(price)` and judge the model by its **typical error in percent**.

The question of this project: **does a neural network beat our Day 2 gradient boosting model on the same data, inputs and test set?**

## Dataset

| | |
|---|---|
| **Name** | House Sales in King County, USA |
| **Source** | [Kaggle: harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction), cleaned in our [Day 1 / Day 2 repository](https://github.com/iaac-maai/2-day2-assignment) |
| **Coverage** | Sales of houses in King County, May 2014 to May 2015 |
| **Size** | 20,703 sales (one row = one sale), 13 input columns; train 13,244 / validation 3,312 / test 4,147 |
| **Target** | `log10(price)` (price in USD), as on Day 2 |
| **Licence** | CC0 1.0 (public domain) |

The test set is **exactly the Day 2 test set** (`GroupShuffleSplit`, 20%, `random_state=42`, grouped by house `id` so that both sales of a house sold twice stay on the same side). The validation set is 20% of the Day 2 training set, split the same way.

## Repository structure

```
├── data/          # cleaned Day 2 CSV, downloaded by the notebook (not committed)
├── figures/       # plots exported by the notebook
├── models/        # model card (README.md) + its figures; model.pt and scaler.joblib live on Hugging Face
├── notebook/      # neural_network.ipynb: the whole pipeline
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Open `notebook/neural_network.ipynb`, select the `.venv` kernel and run all cells. Run it from the `notebook/` folder, because all paths are relative to it. The cleaned data is downloaded automatically into `data/` the first time.

TODO (Person 2): where the training ran (laptop CPU / Colab) and how long it took.

## Approach

- **Inputs:** 13 columns, the same as on Day 2: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `condition`, `grade`, `sqft_basement`, `yr_built`, `yr_renovated`, `lat`, `long` (standardised) and `zipcode` (one-hot encoded, 70 ZIP codes), giving **82 inputs** to the network. The preprocessing is fitted on the training set only.
- **Baselines** (section 2 of the notebook): a dummy model that always predicts the average price, and our **Day 2 model** (`HistGradientBoostingRegressor`, default settings), retrained on the Day 2 training set. It reproduces our published Day 2 test result (R² = 0.886).
- **Architecture:** TODO (Person 2): layers, activation, number of parameters.
- **Training:** TODO (Person 2): loss, optimiser, learning rate, batch size, epochs.
- **Experiments:** TODO (Person 3): what was changed and compared on the validation set.

## Results

| Model | R² on log10(price) (test) | Typical error (test) | MAE in USD (test) |
|---|---|---|---|
| Dummy (always the average price) | 0.000 | 47.0% | 196,281 |
| Day 2: gradient boosting | 0.886 | 12.2% | 60,201 |
| **Our network** | TODO (Person 3) | TODO | TODO |

*Typical error = 10^MAE(log10) − 1: how far, in percent, a prediction usually is from the true price.*

![Baselines on the validation set](figures/00_baselines.png)

*The Day 2 model is typically 11.9% off the true price on the validation set, against 47.6% for the dummy model: a network has to stay below about 12% to beat it.*

TODO (Persons 2–4): at least one figure showing the training (training curves), one showing the performance (predicted vs. actual, or network vs. Day 2 vs. dummy) and one showing what the model does, each with one or two sentences.

## Limitations

- **Period and place:** sales in King County in 2014–2015. The model says nothing about other regions, and prices have changed since.
- **Inherited cleaning:** extreme prices, areas and bedroom counts were removed on Day 1, as were the `waterfront` and `view` columns, so the most unusual and most expensive houses are under-represented.
- **Missing information:** the data has no interior quality, architectural design, neighbourhood amenities or market conditions, which also drive prices.
- TODO (Persons 3 and 4): limitations of the network itself (where it fails, what it should not be used for).

## AI usage disclosure

| | |
|---|---|
| **Tools** | Claude (chat) |
| **Why** | To save time on setup files and to understand errors and unclear concepts |
| **Where** | First drafts of the README and `.gitignore`, debugging, and adapting parts of the class notebook to our data |
| **How** | We wrote the notebook code ourselves following the class notebook. We asked Claude to draft the README and `.gitignore`, then checked and edited them. In some places we asked how to adapt the class code to our dataset, and when a cell failed we asked about the error. We applied the changes ourselves and re-ran the cells |

**Lessons learned:** Claude saved time on routine files and on errors, but we always checked its answers against our own code and the brief.

## License

Code and model released under the [MIT License](LICENSE). The data is CC0 (public domain).
