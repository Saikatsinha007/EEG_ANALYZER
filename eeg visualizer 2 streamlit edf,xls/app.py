import streamlit as st
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, lfilter, welch
from scipy.ndimage import gaussian_filter1d
from PIL import Image
import tempfile

# Set page configuration
st.set_page_config(page_title="Advanced EEG & Image Analysis", layout="wide")

# Function to apply a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to filter data using the Butterworth bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to calculate Power Spectral Density (PSD)
def calculate_psd(signal, fs):
    freqs, psd = welch(signal, fs, nperseg=1024)
    return freqs, psd

# Function to detect artifacts (e.g., eye blinks, muscle activity)
def detect_artifacts(signal, fs):
    threshold = 3 * np.std(signal)
    artifacts = np.abs(signal) > threshold
    return artifacts

# Define the pages of the app
pages = ["EEG Signal Analysis", "Image Analysis"]

# Sidebar for page selection
page_selection = st.sidebar.selectbox("Choose Analysis Type", pages)

# EEG Signal Analysis Page
if page_selection == "EEG Signal Analysis":
    st.title('Advanced EEG Signal Analysis and Visualization')

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "edf"])

    default_fs = 256  # Default sample rate if not provided in EDF file
    default_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }

    fs = st.sidebar.number_input('Sampling Frequency (Hz)', value=default_fs)

    st.sidebar.subheader('Frequency Bands')
    bands = {}
    for band, (low, high) in default_bands.items():
        bands[band] = (
            st.sidebar.number_input(f'{band} Wave Lower Bound (Hz)', value=low),
            st.sidebar.number_input(f'{band} Wave Upper Bound (Hz)', value=high)
        )

    st.sidebar.subheader('Signal Smoothing')
    smoothing = st.sidebar.checkbox('Apply Smoothing')
    smoothing_sigma = st.sidebar.slider('Smoothing Sigma', min_value=0.1, max_value=10.0, value=2.0)

    st.sidebar.subheader('Artifacts Detection')
    detect_artifacts_button = st.sidebar.button('Detect Artifacts')

    # Define a maximum number of data points to load for large EDF files
    MAX_SAMPLE_POINTS = 10000  # Adjust this limit based on memory constraints

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.edf'):
            # Save uploaded EDF file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load EDF file using MNE
            raw = mne.io.read_raw_edf(tmp_file_path, preload=True)
            data = raw.get_data()

            # Limit the data to a smaller sample if it's too large
            total_points = data.shape[1]
            if total_points > MAX_SAMPLE_POINTS:
                st.warning(f"The file contains {total_points} data points, which is too large for full analysis. Only the first {MAX_SAMPLE_POINTS} data points will be analyzed.")
                data = data[:, :MAX_SAMPLE_POINTS]  # Take only the first MAX_SAMPLE_POINTS

            df = pd.DataFrame(data.T, columns=raw.ch_names)
            fs = raw.info['sfreq']  # Sampling frequency from EDF file

        st.subheader('EEG Signal Data Preview')
        st.write(df.head())

        st.subheader('EEG Signal Visualization')
        columns = df.columns.tolist()
        selected_channel = st.sidebar.selectbox('Select EEG Channel', columns)

        if selected_channel:
            signal = df[selected_channel].values
            time = np.arange(len(signal)) / fs

            if smoothing:
                signal = gaussian_filter1d(signal, sigma=smoothing_sigma)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='Signal'))
            fig.update_layout(title=f'{selected_channel} Signal',
                              xaxis_title='Time (s)',
                              yaxis_title='Amplitude')
            st.plotly_chart(fig)

            st.subheader('Power Spectral Density (PSD) Analysis')
            if selected_channel:
                freqs, psd = calculate_psd(signal, fs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name='PSD'))
                fig.update_layout(title=f'{selected_channel} Power Spectral Density',
                                  xaxis_title='Frequency (Hz)',
                                  yaxis_title='Power/Frequency (dB/Hz)')
                st.plotly_chart(fig)

            if detect_artifacts_button:
                st.subheader('Artifacts Detected')
                artifacts = detect_artifacts(signal, fs)
                artifact_times = time[artifacts]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='Signal'))
                fig.add_trace(go.Scatter(x=artifact_times, y=signal[artifacts], mode='markers', name='Artifacts', marker=dict(color='red')))
                fig.update_layout(title=f'{selected_channel} Signal with Artifacts',
                                  xaxis_title='Time (s)',
                                  yaxis_title='Amplitude')
                st.plotly_chart(fig)

            for band, (low, high) in bands.items():
                st.subheader(f'{band} Wave Visualization ({low}-{high} Hz)')
                if selected_channel:
                    filtered_signal = bandpass_filter(signal, low, high, fs)
                    if smoothing:
                        filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=time, y=filtered_signal, mode='lines', name=f'{band} Waves'))
                    fig.update_layout(title=f'{selected_channel} {band} Waves ({low}-{high} Hz)',
                                      xaxis_title='Time (s)',
                                      yaxis_title='Amplitude')
                    st.plotly_chart(fig)

            st.subheader('Overlay Multiple EEG Waves')
            overlay_waves = st.multiselect('Select Waves to Overlay', bands.keys(), default=list(bands.keys()))
            if overlay_waves:
                fig = go.Figure()
                for wave in overlay_waves:
                    low, high = bands[wave]
                    filtered_signal = bandpass_filter(signal, low, high, fs)
                    if smoothing:
                        filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
                    fig.add_trace(go.Scatter(x=time, y=filtered_signal, mode='lines', name=f'{wave} ({low}-{high} Hz)'))
                fig.update_layout(title=f'{selected_channel} Overlayed EEG Waves',
                                  xaxis_title='Time (s)',
                                  yaxis_title='Amplitude')
                st.plotly_chart(fig)

            st.subheader('Custom Time Range Selection')
            time_min, time_max = st.slider('Select Time Range (seconds)', min_value=0.0, max_value=float(time[-1]), value=(0.0, float(time[-1])))
            time_range_signal = signal[int(time_min*fs):int(time_max*fs)]
            time_range_time = time[int(time_min*fs):int(time_max*fs)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_range_time, y=time_range_signal, mode='lines', name='Time Range Signal'))
            fig.update_layout(title=f'{selected_channel} Signal from {time_min} to {time_max} seconds',
                              xaxis_title='Time (s)',
                              yaxis_title='Amplitude')
            st.plotly_chart(fig)

            st.subheader(f'Summary Statistics for {selected_channel}')
            st.write(df[selected_channel].describe())

            st.sidebar.subheader('Download Filtered Signals')
            download_options = {}
            for band in bands.keys():
                download_options[band] = st.sidebar.checkbox(f'Download {band} Waves')

            for band, (low, high) in bands.items():
                if download_options[band]:
                    filtered_signal = bandpass_filter(signal, low, high, fs)
                    if smoothing:
                        filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
                    signal_csv = pd.DataFrame({f'{selected_channel}_{band.lower()}': filtered_signal})
                    st.sidebar.download_button(f'Download {band} CSV', signal_csv.to_csv(index=False), f'{selected_channel}_{band.lower()}.csv')

# # Image Analysis Page
# elif page_selection == "Image Analysis":
#     st.title('Image Analysis and Visualization')

#     uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

#     if uploaded_image is not None:
#         img = Image.open(uploaded_image)
#         st.image(img, caption='Uploaded Image', use_column_width=True)
#         st.write(f"Image Size: {img.size}")
#         st.write(f"Image Format: {img.format}")
        
#         st.sidebar.subheader('Image Filters')
#         if st.sidebar.button('Apply Grayscale'):
#             img_gray = img.convert('L')
#             st.image(img_gray, caption='Grayscale Image', use_column_width=True)

#         if st.sidebar.button('Apply Gaussian Blur'):
#             img_blur = img.filter(ImageFilter.GaussianBlur(radius=5))
#             st.image(img_blur, caption='Gaussian Blurred Image', use_column_width=True)

#         st.sidebar.subheader('Download Processed Image')
#         img_download = st.sidebar.checkbox('Download Image')
#         if img_download:
#             img.save('processed_image.png')
#             st.sidebar.download_button('Download Image', 'processed_image.png', 'processed_image.png')
