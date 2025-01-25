import streamlit as st
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import tempfile

# Streamlit app
st.set_page_config(page_title="EEG Signal and PSD Visualization", layout="wide")
st.title("🌙 EEG Signal and PSD Visualization App")

# Upload .edf file
st.sidebar.header("Upload EEG File")
uploaded_file = st.sidebar.file_uploader("Upload an EEG .edf file", type=["edf"])

# Function to calculate band power
def band_power(data, sf, band):
    """Calculate power in a specific frequency band."""
    from mne.time_frequency import psd_array_multitaper
    if data.ndim == 1:
        data = data[np.newaxis, :]  # Reshape to 2D if single-channel
    psd, freqs = psd_array_multitaper(data, sfreq=sf, fmin=band[0], fmax=band[1], adaptive=True, normalization="full", verbose=0)
    power = np.sum(psd, axis=1)  # Sum across frequencies for each channel
    return power

if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Read the EDF file
        raw = mne.io.read_raw_edf(temp_file_path, preload=True)
        eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch.upper()]

        if not eeg_channels:
            st.error("The file does not contain any EEG channels.")
        else:
            st.sidebar.success("EEG file loaded successfully!")
            st.write("## File Summary")
            st.write(raw.info)

            # Select EEG channels
            raw.pick_channels(eeg_channels)

            # Extract EEG Data
            eeg_data = raw.get_data()
            sf = raw.info["sfreq"]
            time = np.arange(eeg_data.shape[1]) / sf

            # Select a channel for visualization
            selected_channel = st.sidebar.selectbox("Select Channel for Visualization", eeg_channels)
            selected_idx = eeg_channels.index(selected_channel)

            # Plot signal and PSD
            st.write(f"### Signal and PSD for {selected_channel}")
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

            # Plot signal
            axs[0].plot(time, eeg_data[selected_idx], color="blue", lw=0.8)
            axs[0].set_title("EEG Signal")
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Amplitude (μV)")

            # Plot PSD
            psd, freqs = mne.time_frequency.psd_array_multitaper(
                eeg_data[selected_idx],
                sfreq=sf,
                fmin=0.5,
                fmax=50,
                adaptive=True,
                normalization="full",
                verbose=0,
            )
            axs[1].semilogy(freqs, psd, color="green", lw=0.8)
            axs[1].set_title("Power Spectral Density (PSD)")
            axs[1].set_xlabel("Frequency (Hz)")
            axs[1].set_ylabel("Power (μV²/Hz)")

            st.pyplot(fig)

            # Extract Features
            st.write("## Extracted Features")
            features = []
            for i, ch_name in enumerate(eeg_channels):
                channel_data = eeg_data[i]

                # Compute band power for sleep stages
                delta_power = band_power(channel_data, sf, band=(0.5, 4))
                theta_power = band_power(channel_data, sf, band=(4, 8))
                alpha_power = band_power(channel_data, sf, band=(8, 12))
                beta_power = band_power(channel_data, sf, band=(12, 30))
                gamma_power = band_power(channel_data, sf, band=(30, 50))

                # Compute signal variance and entropy
                signal_variance = np.var(channel_data)
                signal_entropy = entropy(np.abs(channel_data))

                # Append features
                features.append({
                    "Channel": ch_name,
                    "Delta Power": delta_power[0],
                    "Theta Power": theta_power[0],
                    "Alpha Power": alpha_power[0],
                    "Beta Power": beta_power[0],
                    "Gamma Power": gamma_power[0],
                    "Variance": signal_variance,
                    "Entropy": signal_entropy,
                })

            # Convert features to DataFrame
            features_df = pd.DataFrame(features)
            st.dataframe(features_df.style.background_gradient(cmap="viridis"))

            # Allow feature download
            csv_data = features_df.to_csv(index=False)
            st.download_button(
                label="Download Features as CSV",
                data=csv_data,
                file_name="eeg_features.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload an EEG .edf file to start.")
