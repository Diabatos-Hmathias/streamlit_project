import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Φόρτωση Δεδομένων
uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Μη υποστηριζόμενος τύπος αρχείου.")

    # Προδιαγραφές Πίνακα
    st.subheader("Προδιαγραφές Πίνακα")
    st.write("Ο πίνακας δεδομένων έχει διαμορφωθεί ως εξής:")
    st.write("Γραμμές: Αντιπροσωπεύουν τα {} δείγματα που αποτελούν το σύνολο δεδομένων.".format(data.shape[0]))
    st.write("Στήλες: Καταγράφουν τα {} χαρακτηριστικά που περιγράφουν κάθε δείγμα.".format(data.shape[1] - 1))
    st.write("Μεταβλητή Εξόδου: Η στήλη {} περιέχει την ετικέτα (label) για κάθε δείγμα.".format(data.columns[-1]))

    # Εμφάνιση των δεδομένων
    st.subheader("Προβολή των Δεδομένων")
    st.write(data)

    # 2D Visualization Tab
    st.subheader("2D Visualization Tab")

    # Εκτέλεση PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.iloc[:, :-1])

    # Εκτέλεση t-SNE
    tsne = TSNE(n_components=2, perplexity=30)  # Μπορείτε να προσαρμόσετε την παράμετρο perplexity όπως εσείς θέλετε
    tsne_result = tsne.fit_transform(data.iloc[:, :-1])

    # Οπτικοποίηση PCA
    st.subheader("PCA Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=data.iloc[:, -1], ax=ax)
    st.pyplot(fig)

    # Οπτικοποίηση t-SNE
    st.subheader("t-SNE Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=data.iloc[:, -1], ax=ax)
    st.pyplot(fig)

    # Διαγράμματα EDA
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("Παρακάτω παρουσιάζονται μερικά διαγράμματα EDA για την εξερεύνηση των δεδομένων:")

    # Ιστογράμματα για τα χαρακτηριστικά
    st.write("Ιστογράμματα για τα χαρακτηριστικά:")
    for column in data.columns[:-1]:
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

    # Διαγράμματα boxplot
    st.write("Διαγράμματα boxplot για τα χαρακτηριστικά:")
    for column in data.columns[:-1]:
        fig, ax = plt.subplots()
        sns.boxplot(y=data[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)
