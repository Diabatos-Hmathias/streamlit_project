import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

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

    # Add Tabs
    task = st.sidebar.selectbox("Επιλέξτε τον τύπο εργασίας", ("Κατηγοριοποίηση", "Ομαδοποίηση"))

    if task == "Κατηγοριοποίηση":
        st.subheader("Αλγόριθμοι Κατηγοριοποίησης")
        classifier = st.sidebar.selectbox("Επιλέξτε έναν αλγόριθμο κατηγοριοποίησης",
                                          ("Logistic Regression", "Random Forest"))

        if classifier == "Logistic Regression":
            # Διαχωρισμός σε χαρακτηριστικά και ετικέτες
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            # Διαχωρισμός σε train και test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Εκπαίδευση του μοντέλου
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Πρόβλεψη στο test set
            y_pred = model.predict(X_test)

            # Αξιολόγηση του μοντέλου
            st.write("Αποτελέσματα Logistic Regression:")
            st.write(classification_report(y_test, y_pred))

        elif classifier == "Random Forest":
            # Διαχωρισμός σε χαρακτηριστικά και ετικέτες
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            # Διαχωρισμός σε train και test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Εκπαίδευση του μοντέλου
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Πρόβλεψη στο test set
            y_pred = model.predict(X_test)

            # Αξιολόγηση του μοντέλου
            st.write("Αποτελέσματα Random Forest:")
            st.write(classification_report(y_test, y_pred))

    elif task == "Ομαδοποίηση":
        st.subheader("Αλγόριθμοι Ομαδοποίησης")
        clustering_algorithm = st.sidebar.selectbox("Επιλέξτε έναν αλγόριθμο ομαδοποίησης",
                                                    ("KMeans", "Hierarchical Clustering"))

        if clustering_algorithm == "KMeans":
            # Διαχωρισμός σε χαρακτηριστικά
            X = data.iloc[:, :-1]

            # Εύρεση του βέλτιστου αριθμού ομάδων με χρήση του silhouette score
            silhouette_scores = []
            for n_clusters in range(2, 11):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(X)
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_scores.append(silhouette_avg)

            optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

            # Εκπαίδευση του μοντέλου με τον βέλτιστο αριθμό ομάδων
            model = KMeans(n_clusters=optimal_n_clusters, random_state=42)
            cluster_labels = model.fit_predict(X)

            # Αποθήκευση των ετικετών στα δεδομένα
            data['Cluster'] = cluster_labels

            # Οπτικοποίηση των ομάδων
            st.subheader("Οπτικοποίηση Ομάδων με KMeans")
            fig, ax = plt.subplots()
            sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=data['Cluster'], palette='viridis', ax=ax)
            st.pyplot(fig)

        elif clustering_algorithm == "Hierarchical Clustering":
            # Διαχωρισμός σε χαρακτηριστικά
            X = data.iloc[:, :-1]

            # Εκτέλεση ιεραρχικής ομαδοποίησης
            model = AgglomerativeClustering()
            cluster_labels = model.fit_predict(X)

            # Αποθήκευση των ετικετών στα δεδομένα
            data['Cluster'] = cluster_labels

            # Οπτικοποίηση των ομάδων
            st.subheader("Οπτικοποίηση Ομάδων με Ιεραρχική Ομαδοποίηση")
            fig, ax = plt.subplots()
            sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=data['Cluster'], palette='viridis', ax=ax)
            st.pyplot(fig)
