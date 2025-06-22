# Smart Home System

A machine learning-powered application that forecasts household energy consumption using pre-trained models, and helps users optimize their usage based on selected goals such as cost savings, noise minimization, load balancing, and appliance longevity.

---

## Features

* **Forecasting**: Predicts energy consumption of home appliances using pre-trained ML models.
* **Goal-Based Optimization**: Users can choose from multiple energy optimization goals:

  * Minimize Standby Power Consumption
  * Load Balancing
  * Minimize Noise Disruption
  * Appliance Lifetime Extension
  * Minimize Energy Costs
* **Interactive Dashboard**: Visualize consumption trends and optimization impact.
* **User-Centric Recommendations**: Generates actionable suggestions tailored to the selected goal.

---

## How It Works

1. **Data Input**: Uses historical appliance-level energy consumption data.
2. **Forecasting**: Loads pre-trained gradient boosting models (saved via `joblib`) to predict future appliance energy use.
3. **Optimization Engine**: Based on user-selected goals, applies custom logic to suggest more efficient usage strategies.
4. **Visualization**: Displays current trends, predictions, and recommendations through an interactive Streamlit UI.


## Technologies Used

* **Python** — Core language for backend logic and data handling
* **Streamlit** — Lightweight frontend for interactive web UI
* **Joblib** — Model serialization for loading pre-trained models
* **Gradient Boosting Models** — Used for energy consumption forecasting
* **Matplotlib** — For plotting and visualizing energy trends
* **Pandas & NumPy** — For data manipulation and numerical operations

---

## Sample Optimization Goals Explained

| Goal                                   | Description                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------- |
| **Minimize Standby Power Consumption** | Identifies idle appliances consuming energy and suggests actions.           |
| **Load Balancing**                     | Distributes energy usage across time periods to reduce peak demand.         |
| **Minimize Noise Disruption**          | Schedules noisy appliances (like washing machines) during low-impact times. |
| **Appliance Lifetime Extension**       | Avoids overuse and suggests cooldown periods.                               |
| **Minimize Energy Costs**              | Recommends usage during off-peak hours to save on electricity bills.        |

---

## Installation

```bash
git clone https://github.com/opeyemiorugun/energy-optimizer.git
cd energy-optimizer
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```bash
energy-optimizer/
├── models/             # Pre-trained model files
├── data/               # Sample input data
├── app.py              # Main application
├── utils.py        # Logic for each optimization goal
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Real-time energy data integration
* Integration with smart devices / IoT sensors
* User authentication and history tracking
* More granular control over optimization goals
* Deploy as a full-fledged web app (e.g., Flask + React or Django)

---

## Contributing

Feel free to fork this repository and suggest improvements via pull requests.



