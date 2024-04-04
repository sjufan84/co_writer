""" Demo page for visualizing audio features """ 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import librosa
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from scipy.spatial import distance
from scipy.spatial.distance import cosine
from utils.audio_processing import extract_features, read_audio

st.set_page_config(page_title="Artist Vault Demo", page_icon=":microphone:",
                initial_sidebar_state="auto", layout="wide")

def get_3d_chart_fcar():
    """ Get the 3D chart for Jeremiah's Clip """
    # Read and preprocess audio signals
    jenny_signal = read_audio('./audio_samples/clones/jenny_clone.wav')
    jeremiah_signal = read_audio('./audio_samples/originals/jeremiah_10s.wav')
    joel_signal = read_audio('./audio_samples/clones/joel_clone.wav')
    min_length = min(len(jenny_signal), len(jeremiah_signal), len(joel_signal))
    jenny_signal = jenny_signal[:min_length]
    jeremiah_signal = jeremiah_signal[:min_length]
    joel_signal = joel_signal[:min_length]
    
    # Extract features
    if len(jenny_signal) < 2048 or len(jeremiah_signal) < 2048 or len(joel_signal) < 2048:
        st.warning("One of the audio signals is too short for FFT. Skipping...")
        return

    jenny_features = extract_features(jenny_signal)
    jeremiah_features = extract_features(jeremiah_signal)
    joel_features = extract_features(joel_signal)

    # Transpose to have features as columns
    jenny_features = jenny_features.T
    jeremiah_features = jeremiah_features.T
    joel_features = joel_features.T

    # Create a DataFrame by concatenating the two feature sets
    df = pd.concat([pd.DataFrame(jenny_features), 
    pd.DataFrame(jeremiah_features), pd.DataFrame(joel_features)], axis=0)
    # Standardize the data before applying PCA and KMeans
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform PCA to capture 95% of the variance
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(scaled_df)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(pca_df, columns=['PC1', 'PC2', 'PC3'])

    # Create segments (you may need to adjust this to match your actual segmentation logic)
    jenny_segments = np.array_split(jenny_signal, len(plot_df)//3)
    jeremiah_segments = np.array_split(jeremiah_signal, len(plot_df)//3)
    joel_segments = np.array_split(joel_signal, len(plot_df)//3)

   # Create labels for the segments
    jenny_labels = [f"Jenny - Segment {i+1}" for i in range(len(jenny_segments))]
    jeremiah_labels = [f"Jeremiah - Segment {i+1}" for i in range(len(jeremiah_segments))]
    joel_labels = [f"Joel - Segment {i+1}" for i in range(len(joel_segments))]
    segment_labels = jenny_labels + jeremiah_labels + joel_labels

    # Create segment numbers (e.g., Segment 1, Segment 2, ...)
    segment_numbers = [f"Segment {i+1}" for i in range(len(jeremiah_segments))] * 3

    # Add segment names and numbers to the DataFrame
    plot_df['segment_name'] = segment_labels
    plot_df['segment_number'] = segment_numbers

    # Create two columns, one for the chart and one for the audio
    col1, col2 = st.columns([1.75, 1], gap='large') 
    with col2:  # Add a Streamlit multiselect widget to allow users to select artists
        st.text("")
        st.text("")
        st.text("")
        # Display the original clips
        # Convert the clips to bytes using librosa

        st.markdown("**Original Clip:**")
        jeremiah_bytes = librosa.to_mono(jeremiah_signal)
        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif;
                    color: #76B900; font-size: 15px; font-weight: 550;">
                    Jeremiah</p>
                    """, unsafe_allow_html=True)
        st.audio(jeremiah_bytes, format='audio/wav', start_time=0, sample_rate=16000)

        st.markdown("**Cloned Clips:**")
        joel_bytes = librosa.to_mono(joel_signal)
        jenny_bytes = librosa.to_mono(jenny_signal)

        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif;
                    color: #76B900; font-size: 15px; font-weight: 550;">
                    Jenny</p>
                    """, unsafe_allow_html=True)
        st.audio(jenny_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif; color: #76B900;
                    font-size: 15px; font-weight: 550;">
                    Joel</p>
                    """, unsafe_allow_html=True)
        st.audio(joel_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        
        selected_artists = st.multiselect(
        "Select Artists to Display:",
        options=['Jenny', 'Jeremiah', 'Joel'],
        default=['Jenny', 'Jeremiah', 'Joel'],
        )
        # Filter the DataFrame based on selected artists
        filtered_plot_df = plot_df[plot_df['segment_name'].
                        str.contains('|'.join(selected_artists))]
        st.text("")
        # Create a button to continue to the next page
        continue_secure_button = st.button("Continue to next page",
                        type='secondary', use_container_width=True)
        if continue_secure_button:
            switch_page("new_main")
        st.markdown("---")
        # st.markdown("Curious about the cloning process? [Try it out for yourself](https://huggingface.co/spaces/dthomas84/RVC_RULE1)\
        #        with Jenny and Joel's voices in our First Rule AI playground!")

    with col1:  # Plot using the filtered DataFrame
        fig = px.scatter_3d(
        filtered_plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='segment_number',
        color_continuous_scale='rainbow',
        title='3D Representation of Vocal Features -- Jeremiah, Joel, Jenny',
        text='segment_name',
    )

        fig.update_layout(
            width=750,  # Width of the plot in pixels
            height=750,  # Height of the plot in pixels
            scene=dict(
                xaxis=dict(title='PC1'),
                yaxis=dict(title='PC2'),
                zaxis=dict(title='PC3')
            ),
            showlegend=False,
        )
        fig.update_traces(
        textposition='top center',  # Position of the text labels
        textfont_size=10,
        marker_size=8            # Font size of the text labels
    )
        st.plotly_chart(fig, use_container_width=True)

def home():
    """ Home page for the application. Display the mechanism """
    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #76B900;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">So what's the problem?</h4>
        <br>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Never before has someone
        been able to clone an artist's voice and then be able to <i>generate new content with it.</i>
        This is not about simply copying and misusing actual recordings.  This is about
        the ability to create vocals virtually <i>indistinguishable</i> from the artist's, and have them
        "sing" literally anything, with no quick way to verify it's authenticity.  The stakes
        in the age of social media, where disinformation spreads like wildfire and very few people
        would even think to check whether or not a voice is authentic, could not be higher.</h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">In order to prove the point, we trained
        models using Joel and Jenny's voices, with less than 10 minutes of data each.  We then separated vocals
        from downloaded YouTube audio of Jeremiah Harmon's "Almost Heaven" (with his permission), fed it to our model,
        and within approximately 30 seconds generated a deepfake.  While it isn't perfect, it's important to remember the little amount
        of data we trained Joel and Jenny's models with, and with some fine-tuning and a little more
        training time we could produce whole songs or even albums that could pass for an actual artist's.
        </h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">In the 3d chart below, we have mapped out the high level
            features of each artist's vocals.  You will notice that for each point in the song, the artists' voices
            cluster together, but are not identical.  It is this difference that distinguishes each voiceprint
            and allows for security measures to be taken to protect the artist and their authenticity.</h3>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
                -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
                -o-animation: fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">
                </div>""", unsafe_allow_html=True)
            
    get_3d_chart_fcar()

if __name__ == "__main__":
    home()

    # Display the "Business Chat" sidebar
    # from utils.bplan_utils import chat_state_sidebar
    # chat_state_sidebar()