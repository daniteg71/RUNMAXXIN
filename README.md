# 🏃‍♂️ RUNMAXXIN

**RUNMAXXIN** is an intelligent system that recommends music tracks in real-time during a running session. 

Starting from a short user prompt and analyzing biometric data from sensors (heart rate, speed, effort), the system dynamically adapts the music selection to support the workout. The goal is to balance the athletic target with the desired mood and, most importantly, the physiological safety of the runner.

This project was developed for the AI-Lab course (Integration of NLP, Symbolic Layer, and Recommendation Systems).

## 🧠 High-Level Pipeline

The system reasons song by song following this flow:

1. **NLP (`intent.py`):** Analyzes the text prompt to extract the intent (goal, mood) and numerical parameters (speed, distance).
2. **Sensors (`physiological_state.py`):** Evaluates the runner's physical state in real-time (e.g., effort, HRR, heart rate trend).
3. **Controller (`controller.py`):** Merges the user's desires with their physical state to generate a **Target Vector** (BPM, Energy, Valence).
4. **Recommender & Symbolic Gate:** The recommender finds the closest music tracks to the target. Before playback, a symbolic layer (SPARQL/SHACL) filters the candidates, strictly verifying physiological compatibility to prevent dangerous overexertion.

## ⚙️ Quickstart

Ensure you have Python installed, then clone the repository and install the dependencies:

```bash
# Install requirements
pip install -r requirements.txt

# 1. Train the NLP models (SetFit)
python train_intent.py

# 2. Generate simulated session archetypes and process data
python simulate_sessions.py
python build_dataset.py --input data/simulated/bpm_sessions.csv --output data/processed/physiological_windows.csv

# 3. Launch the Interactive Generative Tester (On-the-fly sessions)
python tester.py

# (Optional) Launch a pre-recorded demo
python run_demo.py "oggi voglio spingere tantissimo" --session push_fatigue
