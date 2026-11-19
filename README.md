# Natural Gas Forecasting with Attention-Based Deep Learning

Code and data for the paper **[Forecasting Natural Gas for Power Generation Using Attention-Based Deep Learning Model](https://link.springer.com/chapter/10.1007/978-3-032-23407-0_18)**, published in *Data Science and Emerging Technologies*, Lecture Notes on Data Engineering and Communications Technologies (Springer, 2026), pp. 250–264.

The project compares three sequence models (**LSTM**, **GRU** and **TCN**), each with and without a **multi-head self-attention** layer, for one-step-ahead forecasting of daily natural gas power generation in Mexico.

## Results (as published)

Evaluation-set performance from the original experiments:

| Model | MAPE (%) | MAE (GWh) | RMSE (GWh) | R² |
|---|---:|---:|---:|---:|
| LSTM | 1.1750 | 6.8954 | 8.5329 | 0.9538 |
| LSTM + Attention | 0.9167 | 5.3124 | 6.9602 | 0.9693 |
| GRU | 2.2669 | 13.1905 | 16.9785 | 0.8173 |
| GRU + Attention | 1.0667 | 6.1992 | 7.9328 | 0.9601 |
| TCN | 1.2404 | 7.2368 | 9.6663 | 0.9408 |
| **TCN + Attention** | **0.7782** | **4.5300** | **6.0220** | **0.9770** |

Adding attention improved every architecture. TCN + Attention gave the best result overall.

> **Reproducing the paper exactly:** the original notebooks behind these numbers are preserved at the git tag [`paper-v1`](../../tree/paper-v1) (`git checkout paper-v1`).
> The notebooks on `main` have been cleaned up and use a stricter evaluation protocol (see below), so re-running them will give somewhat different numbers.

## Dataset

[`data/dataset.csv`](data/dataset.csv) contains 1,215 daily observations of natural gas electricity generation in Mexico from **1 January 2022 to 30 April 2025**.

| Column | Description |
|---|---|
| `country` | Country (`Mexico`) |
| `date` | Date (`DD/MM/YYYY`) |
| `sector` | Generation source (`Gas`) |
| `GWh` | Daily generation in gigawatt-hours |

## Method

- **Input:** sliding windows of the previous 30 days, used to predict the next day.
- **Split:** chronological, with 70% train, 15% validation and 15% test.
- **Scaling:** min-max scaling fitted on the training period only.
- **Tuning:** Keras Tuner random search (10 trials) selected on validation loss, with early stopping.
- **Models:**
  - *Vanilla:* an LSTM, GRU or TCN encoder followed by a dense output.
  - *Attention:* the same encoder returns the full sequence and feeds a `MultiHeadAttention` self-attention layer. This is followed by layer normalisation, global average pooling, dropout and a linear output.
- **Metrics:** MAPE, MAE, RMSE and R² in original units (GWh). The test set is used only for the final evaluation.
- **Forecast:** a recursive 90-day forecast beyond the last observation.

## Repository structure

```
├── data/dataset.csv          # daily natural gas generation, Mexico
├── docs/paper-slides.pdf     # presentation slides of the paper
├── notebooks/                # one notebook per model
│   ├── LSTM_Vanilla.ipynb    LSTM_Attention.ipynb
│   ├── GRU_Vanilla.ipynb     GRU_Attention.ipynb
│   └── TCN_Vanilla.ipynb     TCN_Attention.ipynb
├── src/forecasting.py        # shared pipeline: data, models, tuning, evaluation, plots
└── requirements.txt
```

Running a notebook writes its metrics to `results/<model>.json` and its figure to `results/figures/<model>.png`. Tuning logs (`tuning/`) and trained models (`models/`) are git-ignored.

## Getting started

```bash
git clone https://github.com/Syukrondzeko/natural-gas-forecasting.git
cd natural-gas-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

On Google Colab, clone the repository in the first cell, `cd` into `notebooks/` and run `pip install -r ../requirements.txt`.

Random seeds are fixed (`SEED = 42` in `src/forecasting.py`). Results may still vary slightly across hardware and TensorFlow versions.

## Citation

```bibtex
@inproceedings{syukron2026forecasting,
  title     = {Forecasting Natural Gas for Power Generation Using Attention-Based Deep Learning Model},
  author    = {Syukron, Muhamad and Anwar, Muhammad Rifqy Rezvany and Nareswari, Putri Hanna and Putri, Anisa Ardiani and Gaya, Joycelin Gracelda Resi},
  booktitle = {Data Science and Emerging Technologies},
  series    = {Lecture Notes on Data Engineering and Communications Technologies},
  pages     = {250--264},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  doi       = {10.1007/978-3-032-23407-0_18}
}
```
