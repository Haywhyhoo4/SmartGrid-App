import streamlit as st
import simpy
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Smart Grid Dashboard", layout="wide")
st.title("⚡ Smart Grid SDN + PSO‑GA Unified Dashboard")

# -------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------
st.sidebar.header("Simulation Parameters")

sim_duration = st.sidebar.number_input("Simulation Duration (s)", 10, 3600, 120)
dispatch_interval = st.sidebar.number_input("Dispatch Interval (s)", 1, 300, 10)
event_interval = st.sidebar.number_input("Event Interval (s)", 1, 300, 7)
controller_interval = st.sidebar.number_input("Controller Interval (s)", 1, 300, 5)

st.sidebar.header("Frequency Bands (Hz)")
low_min = st.sidebar.number_input("Low Band Min", 45.0, 60.0, 49.2)
low_max = st.sidebar.number_input("Low Band Max", 45.0, 60.0, 50.0)
nominal_max = st.sidebar.number_input("Nominal Max", 45.0, 60.0, 50.2)

st.sidebar.header("Initial Conditions")
init_gen = st.sidebar.number_input("Initial Generation (MW)", 0.0, 200.0, 50.0)
init_load = st.sidebar.number_input("Initial Load (MW)", 0.0, 200.0, 45.0)
init_der = st.sidebar.number_input("Initial DER Output (MW)", 0.0, 200.0, 5.0)
init_freq = st.sidebar.number_input("Initial Frequency (Hz)", 48.0, 52.0, 50.0)

start_btn = st.sidebar.button("Start Simulation")
stop_btn = st.sidebar.button("Stop Simulation")

# -------------------------------------------------------
# FREQUENCY CLASSIFIER
# -------------------------------------------------------
def classify_frequency(f):
    if f < low_min:
        return "UNDER"
    elif low_min <= f < low_max:
        return "LOW"
    elif low_max <= f <= nominal_max:
        return "NOMINAL"
    else:
        return "OVER"

# -------------------------------------------------------
# GRID MODEL
# -------------------------------------------------------
class GridModel:
    def __init__(self, env, gen, load, der, freq):
        self.env = env
        self.generation = gen
        self.load = load
        self.der_output = der
        self.frequency = freq

    def apply_event(self):
        self.load += random.uniform(-5, 5)
        self.der_output += random.uniform(-2, 2)

        imbalance = (self.load - self.der_output) - self.generation
        self.frequency += imbalance * 0.01 + random.uniform(-0.05, 0.05)
        self.frequency = max(48.0, min(52.0, self.frequency))

    def get_state(self):
        return {
            "time": float(self.env.now),
            "generation": float(self.generation),
            "load": float(self.load),
            "der_output": float(self.der_output),
            "net_demand": float(self.load - self.der_output),
            "frequency": float(self.frequency),
            "band": classify_frequency(self.frequency)
        }

# -------------------------------------------------------
# PSO-GA ENGINE
# -------------------------------------------------------
class PSOGAEngine:
    def __init__(self, env, grid):
        self.env = env
        self.grid = grid

    def run(self):
        state = self.grid.get_state()
        net_demand = state["net_demand"]
        return max(0, net_demand + 2)

# -------------------------------------------------------
# SDN CONTROLLER
# -------------------------------------------------------
class SDNController:
    def __init__(self, env, grid):
        self.env = env
        self.grid = grid

    def apply(self, gen):
        self.grid.generation = gen

# -------------------------------------------------------
# NETWORK TOPOLOGY
# -------------------------------------------------------
def build_topology():
    G = nx.Graph()
    G.add_nodes_from(["Controller", "PSO-GA", "PMU1", "PMU2", "DER1", "DER2"])
    G.add_edges_from([
        ("Controller", "PSO-GA"),
        ("Controller", "PMU1"),
        ("Controller", "PMU2"),
        ("PMU1", "DER1"),
        ("PMU2", "DER2")
    ])
    return G

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = []

# -------------------------------------------------------
# MAIN SIMULATION FUNCTION
# -------------------------------------------------------
def run_simulation():
    st.session_state.data = []
    env = simpy.Environment()
    grid = GridModel(env, init_gen, init_load, init_der, init_freq)
    pso = PSOGAEngine(env, grid)
    controller = SDNController(env, grid)

    data_log = st.session_state.data

    def event_process(env):
        while True:
            yield env.timeout(event_interval)
            grid.apply_event()
            data_log.append(grid.get_state())

    def controller_process(env):
        while True:
            yield env.timeout(controller_interval)
            controller.apply(grid.generation)
            data_log.append(grid.get_state())

    def optimisation_process(env):
        while True:
            yield env.timeout(dispatch_interval)
            gen = pso.run()
            controller.apply(gen)
            state = grid.get_state()
            state["generation"] = gen
            data_log.append(state)

    env.process(event_process(env))
    env.process(controller_process(env))
    env.process(optimisation_process(env))

    placeholder_chart = st.empty()
    placeholder_heatmap = st.empty()
    placeholder_voltage = st.empty()
    placeholder_topology = st.empty()

    step = 1
    while env.now < sim_duration and st.session_state.running:
        env.run(until=env.now + step)

        df = pd.DataFrame(data_log)

        # ------------------ LIVE TIME SERIES ------------------
        with placeholder_chart.container():
            st.subheader("📈 Live Frequency / Net Demand / Generation")
            if len(df) > 0 and "time" in df.columns:
                st.line_chart(df.set_index("time")[["frequency", "net_demand", "generation"]])
            else:
                st.info("Waiting for simulation data…")

        # ------------------ FREQUENCY HEATMAP ------------------
        with placeholder_heatmap.container():
            st.subheader("🌡 Frequency Heatmap")
            if len(df) > 0:
                heatmap, xedges, yedges = np.histogram2d(
                    df["time"], df["frequency"], bins=20
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(heatmap, cmap="viridis")
                st.pyplot(fig)

        # ------------------ VOLTAGE PLOT ------------------
        with placeholder_voltage.container():
            st.subheader("🔌 Voltage Plot (Simulated)")
            if len(df) > 0:
                voltage = 1.0 + (df["frequency"] - 50.0) * 0.01
                fig2, ax2 = plt.subplots()
                ax2.plot(df["time"], voltage)
                ax2.set_ylabel("Voltage (p.u.)")
                st.pyplot(fig2)

        # ------------------ NETWORK TOPOLOGY ------------------
        with placeholder_topology.container():
            st.subheader("🌐 Network Topology Map")
            G = build_topology()
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", ax=ax3)
            st.pyplot(fig3)

        time.sleep(0.1)

    st.session_state.running = False

# -------------------------------------------------------
# BUTTON LOGIC
# -------------------------------------------------------
if start_btn:
    st.session_state.running = True
    run_simulation()

if stop_btn:
    st.session_state.running = False

st.markdown("---")
st.markdown("Adjust parameters in the sidebar and click **Start Simulation** to run the unified dashboard.")

