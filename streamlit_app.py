import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main():
    st.title('Φόρτωση Δεδομένων')

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Check if the file type is CSV or Excel
        if uploaded_file.type == 'application/vnd.ms-excel':
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)

        # Display the first few rows of the dataset
        st.write("Πρώτες 5 γραμμές του αρχείου:")
        st.write(df.head())

        # Display the dimensions of the dataset
        st.write(f"Διαστάσεις του πίνακα: {df.shape}")

        # Create a 2D Visualization tab
        st.title('2D Οπτικοποιήσεις')

        # Dropdown for dimensionality reduction algorithms
        dimensionality_reduction = st.selectbox(
            "Επιλέξτε αλγόριθμο μείωσης διαστάσεων:",
            ['PCA', 't-SNE']
        )

        # Perform dimensionality reduction
        if dimensionality_reduction == 'PCA':
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(df.iloc[:, :-1])
        elif dimensionality_reduction == 't-SNE':
            tsne = TSNE(n_components=2)
            reduced_data = tsne.fit_transform(df.iloc[:, :-1])

        # Scatter plot
        st.write(f"2D Visualization using {dimensionality_reduction}")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=df.iloc[:, -1])
        st.pyplot()

        # EDA Plots
        st.title('Exploratory Data Analysis (EDA)')

        # Example EDA plot (e.g., histogram)
        st.write("Histogram of a feature:")
        sns.histplot(df.iloc[:, 0])
        st.pyplot()

if __name__ == "__main__":
    main()