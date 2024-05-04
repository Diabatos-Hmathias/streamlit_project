import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering

# Load Data
uploaded_file = st.file_uploader("Select a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")

    # Data Specifications
    st.subheader("Data Specifications")
    st.write("The dataset has the following specifications:")
    st.write("Rows: Represent {} samples.".format(data.shape[0]))
    st.write("Columns: Describe {} features for each sample.".format(data.shape[1] - 1))
    st.write("Output Variable: The column {} contains the label for each sample.".format(data.columns[-1]))

    # Display Data
    st.subheader("View Data")
    st.write(data)

    # Streamlit Tabs
    tabs = st.sidebar.radio("Navigation", ["2D Visualization", "Exploratory Data Analysis", "Comparison", "Info"])

    if tabs == "2D Visualization":
        # 2D Visualization Tab
        st.subheader("2D Visualization Tab")

        # Streamlit Sub-Tabs for PCA and t-SNE Visualization
        subtabs_2d = st.radio("Select Algorithm To Visualise", ["PCA", "t-SNE"])

        if subtabs_2d == "PCA":
            # PCA Execution
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data.iloc[:, :-1])

            # PCA Visualization
            st.subheader("PCA Visualization")
            fig, ax = plt.subplots()
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=data.iloc[:, -1], ax=ax)
            st.pyplot(fig)

        elif subtabs_2d == "t-SNE":
            # t-SNE Execution
            tsne = TSNE(n_components=2, perplexity=30)
            tsne_result = tsne.fit_transform(data.iloc[:, :-1])

            # t-SNE Visualization
            st.subheader("t-SNE Visualization")
            fig, ax = plt.subplots()
            sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=data.iloc[:, -1], ax=ax)
            st.pyplot(fig)

    elif tabs == "Exploratory Data Analysis":
        # Exploratory Data Analysis (EDA) Plots
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Below are some EDA plots for exploring the data:")

        # Streamlit Sub-Tabs for Histograms and Boxplots
        subtabs_eda = st.radio("Select EDA Plot Type", ["Histograms", "Boxplots"])

        if subtabs_eda == "Histograms":
            # Histograms for Features
            st.write("Histograms for Features:")
            for column in data.columns[:-1]:
                fig, ax = plt.subplots()
                sns.histplot(data[column], kde=True, ax=ax)
                ax.set_title(f"Histogram of {column}")
                st.pyplot(fig)

        elif subtabs_eda == "Boxplots":
            # Boxplots for Features
            st.write("Boxplots for Features:")
            for column in data.columns[:-1]:
                fig, ax = plt.subplots()
                sns.boxplot(y=data[column], ax=ax)
                ax.set_title(f"Boxplot of {column}")
                st.pyplot(fig)

    elif tabs == "Comparison":
        # t-SNE Execution for Comparison Tab
        tsne = TSNE(n_components=2, perplexity=30)
        tsne_result = tsne.fit_transform(data.iloc[:, :-1])

        # Classification Comparison
        st.subheader("Comparison of Classification Algorithms")

        # Logistic Regression vs. Random Forest
        st.write("Logistic Regression vs. Random Forest:")
        regularization_param_lr = st.slider("Regularization Parameter (C) for Logistic Regression", 0.01, 10.0, 1.0)
        num_estimators_rf = st.slider("Number of Estimators for Random Forest", 1, 100, 10)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_lr = LogisticRegression(C=regularization_param_lr)
        model_rf = RandomForestClassifier(n_estimators=num_estimators_rf)

        model_lr.fit(X_train, y_train)
        model_rf.fit(X_train, y_train)

        y_pred_lr = model_lr.predict(X_test)
        y_pred_rf = model_rf.predict(X_test)

        st.write("Logistic Regression Results:")
        lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
        st.write(lr_report)

        st.write("Random Forest Results:")
        rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
        st.write(rf_report)

        if lr_report['accuracy'] > rf_report['accuracy']:
            st.write("Logistic Regression performed better.")
        elif lr_report['accuracy'] < rf_report['accuracy']:
            st.write("Random Forest performed better.")
        else:
            st.write("Both models performed equally well.")

        # Clustering Comparison
        st.subheader("Comparison of Clustering Algorithms")

        # K-Means vs. Hierarchical Clustering
        st.write("K-Means vs. Hierarchical Clustering:")
        num_clusters_km = st.slider("Number of Clusters (k) for K-Means", 2, 10, 5)
        num_clusters_hc = st.slider("Number of Clusters for Hierarchical Clustering", 2, 10, 5)

        X = data.iloc[:, :-1]

        model_km = KMeans(n_clusters=num_clusters_km, random_state=42)
        model_hc = AgglomerativeClustering(n_clusters=num_clusters_hc)

        cluster_labels_km = model_km.fit_predict(X)
        cluster_labels_hc = model_hc.fit_predict(X)

        silhouette_score_km = silhouette_score(X, cluster_labels_km)
        silhouette_score_hc = silhouette_score(X, cluster_labels_hc)

        st.write("Silhouette Score for K-Means:", silhouette_score_km)
        st.write("Silhouette Score for Hierarchical Clustering:", silhouette_score_hc)

        if silhouette_score_km > silhouette_score_hc:
            st.write("K-Means performed better based on silhouette score.")
        elif silhouette_score_km < silhouette_score_hc:
            st.write("Hierarchical Clustering performed better based on silhouette score.")
        else:
            st.write("Both clustering algorithms performed equally well based on silhouette score.")

        # Visualization of Clustering Results
        st.subheader("Visualization of Clustering Results")
        st.write("Below are scatter plots visualizing the clustering results:")

        # K-Means Visualization
        st.write("K-Means Clustering Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=cluster_labels_km, palette='viridis', ax=ax)
        st.pyplot(fig)

        # Hierarchical Clustering Visualization
        st.write("Hierarchical Clustering Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=cluster_labels_hc, palette='viridis', ax=ax)
        st.pyplot(fig)

    #info tab
    elif tabs == "Info":
        st.subheader("About This Application")
        st.write("This web-based application is designed for data mining and analysis.")
        st.write("It allows users to load tabular data, perform 2D visualizations, conduct exploratory data analysis, and compare classification and clustering algorithms.")

        st.subheader("How It Works")
        st.write("Users can upload CSV or Excel files containing their data.")
        st.write("Then, they can navigate between different tabs to visualize data, explore its distribution, and compare machine learning algorithms.")

        st.subheader("Development Team")
        st.write("This application was developed by [Diabatos Hmathias].")
        st.write("Team Members:")
        st.write("- Member 1: [Ευφραιμίδης Χρήστος]")
        st.write("- Member 2: [Πυρινός Παύλος]")

        st.subheader("Tasks Performed")
        st.write("Each team member contributed to different aspects of the project.")
        st.write("- Ευφραιμίδης Χρήστος: tbd")
        st.write("- Πυρινός Παύλος: tbd")

