"""
Radar Signal Processing Simulator
Main application entry point
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("Radar Signal Processing Simulator")
    st.write("Welcome to the radar simulator!")
    
    # Test plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.ones_like(theta)
    ax.plot(theta, r)
    ax.set_title("Basic Radar Display")
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()