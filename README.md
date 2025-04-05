# F12025Predictor

A machine learning model to predict Formula 1 Grand Prix finishing order using qualifying times and historical wet/dry performance.

---

## Overview

This project predicts F1 race results — specifically finishing times and driver rankings — based on:

- **Qualifying performance** (latest lap times)
- **Wet performance score** (how well drivers adapt to wet races, calculated using historic comparisons)
- **Historical race results** (used as training labels)

You can apply this predictor to **any upcoming Grand Prix**, given a qualifying order and a matching past race to train on.

---

## Key Features

- **Gradient Boosting Regressor** trained on any recent race data (e.g., Japan 2024)
- **Wet performance score** computed from performance delta between a dry and wet race (e.g., Canada 2023 vs 2024)
- Full mapping between driver names and FastF1 3-letter codes
- Customizable input: manually enter qualifying times for any upcoming race
- Works with FastF1 caching to minimize network calls and support offline runs

---
