import streamlit as st
import pandas as pd
import numpy as np
import string, re
import matplotlib.pyplot as plt
import nltk
import joblib
from datetime import datetime
from google_play_scraper import Sort, reviews
from streamlit_option_menu import option_menu
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from PIL import Image

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Stemmer initialization
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Check if 'df' is not in session state, initialize it to None
if 'df' not in st.session_state:
    st.session_state['df'] = None

def cleansing(data):
    if isinstance(data, str):
        positive_words = ["baik", "bagus", "puas", "senang", "terbaik", "nyaman", "lancar", "mudah", "aman", "ramah", "cepat"]
        negative_words = ["buruk", "jelek", "tidak puas", "marah", "kecewa", "sulit", "rusak", "lambat", "bermasalah", "mengecewakan", "tidak aman"]
        
        words_to_keep = positive_words + negative_words
        
        data = data.lower()
        words = nltk.word_tokenize(data)
        words = [word for word in words if word not in string.punctuation]
        words = [word for word in words if not re.search(r'^\\U[0-9a-fA-F]{8}$', word)]
        stopword_factory = StopWordRemoverFactory()
        stopword_remover = stopword_factory.create_stop_word_remover()
        words = stopword_remover.remove(' '.join(words)).split()
        stemmed_words = [stemmer.stem(word) for word in words]
        cleaned_data = ' '.join([word for word in stemmed_words if word in words_to_keep])
        return cleaned_data
    else:
        return ""

# Function to load data and perform preprocessing
@st.cache_data()
def load_data(df):
    df['content'] = df['content'].apply(cleansing)
    df = df.drop(columns=['at', 'userName'])
    return df

def train_svm_model_from_csv(csv_file_path):
    # Read the CSV file using pandas
    df_latih = pd.read_csv(csv_file_path)

    # Define the feature and target variables
    X_latih = df_latih['review']
    y_latih = df_latih['label']

    # Define TfidfVectorizer for feature extraction
    vectorizer = TfidfVectorizer()
    X_latih_transformed = vectorizer.fit_transform(X_latih)

    # SVM model training
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_latih_transformed, y_latih)

    return svm_model, vectorizer

# Path to the CSV file
csv_file_path = 'data_latih.csv'

# Train the model
svm_model, vectorizer = train_svm_model_from_csv(csv_file_path)

# Function to resize images
def resize_image(image, width):
    aspect_ratio = image.width / image.height
    new_height = int(width / aspect_ratio)
    resized_image = image.resize((width, new_height))
    return resized_image

# Load and resize images
image_analisis1 = resize_image(Image.open("pilihan analisis.png"), 600)
image_analisis2 = resize_image(Image.open("search analisis.png"), 600)
image_analisis3 = resize_image(Image.open("tampilan df .png"), 600)
image_analisis4 = resize_image(Image.open("tampilan df clean.png"), 600)
image_analisis5 = resize_image(Image.open("tampilan grafik .png"), 600)
image_analisis6 = resize_image(Image.open("grafik per bulan.png"), 600)
image_prediksi1 = resize_image(Image.open("prediksi.png"), 600)

# Streamlit app starts here
st.title("Ul-ecom")

# Streamlit options menu for navigation
selected = st.sidebar.radio("Pilih Fitur", ["Beranda", "Analisis", "Prediksi Sentimen"])

# Home page
if selected == "Beranda":
    st.write("""
    Selamat datang di website analisis sentimen untuk ulasan kumpulan e-commerce yang ada di Indonesia. 
    Dengan adanya website ini, diharapkan dapat membantu pengguna dalam menganalisis ulasan pengguna dari berbagai platform e-commerce.
    """)

    st.header("Fitur Analisis")
    st.write("""
    Berikut adalah langkah-langkah untuk menggunakan fitur Analisis:
    """)

    st.image(image_analisis1, caption="Langkah 1 : Pilih menu 'Analisis' di sidebar", use_column_width=True)
    st.image(image_analisis2, caption="Langkah 2 : Pilih E-commerce dan banyak ulasan yang dicari", use_column_width=True)
    st.image(image_analisis3, caption="Langkah 3 : Berikut merupakan hasil setelah user scrape ulasan dalam bentuk tabel", use_column_width=True)
    st.image(image_analisis4, caption="Langkah 4 : Untuk melihat full content di dalam tabel tekan tombol scrape ulasan sekali lagi", use_column_width=True)
    st.image(image_analisis5, caption="Langkah 5 : Berikut merupakan grafik persentase sentimen dari hasil yang sudah user cari", use_column_width=True)
    st.image(image_analisis6, caption="Langkah 6 : Berikut merupakan grafik persentase sentimen per bulan yang sudah user cari ", use_column_width=True)

    st.header("Fitur Prediksi Sentimen")
    st.write("""
    Berikut adalah langkah-langkah untuk menggunakan fitur Prediksi Sentimen:
    """)

    st.image(image_prediksi1, caption="Untuk fitur prediksi sentimen masukan ulasan yang ingin dicari tahu nilainya dan klik tombol prediksi sentimen ", use_column_width=True)

elif selected == "Analisis":
    # Side bar menu
    with st.sidebar:
        # Radio button to choose between uploading CSV or scraping data
        data_source = st.radio("Pilih Sumber Data", ["Scrape Data"])

        if data_source == "Scrape Data":
            e_commerce = st.selectbox("Pilih E-commerce", ["Blibli","Bukalapak", "Shopee", "Tokopedia", "Lazada", "Zalora", "OLX"])
            number = st.number_input(label="Masukkan Jumlah", min_value=1, max_value=500)  # Set max_value to 500

            with st.form(key="scrape_form"):  # Form to encapsulate the button
                st.write("Klik tombol di bawah untuk melakukan scraping ulasan:")
                submit = st.form_submit_button("Scrape Ulasan")
                if submit:
                    try:
                        # Berdasarkan pilihan e-commerce, Anda dapat melakukan scraping dari URL yang sesuai
                        if e_commerce == "Bukalapak":
                            selected_url = "com.bukalapak.android"
                        elif e_commerce == "Shopee":
                            selected_url = "com.shopee.id"
                        elif e_commerce == "Tokopedia":
                            selected_url = "com.tokopedia.tkpd"
                        elif e_commerce == "Lazada":
                            selected_url = "com.lazada.android"
                        elif e_commerce == "Zalora":
                            selected_url = "com.zalora.android"
                        elif e_commerce == "Blibli":
                            selected_url = "blibli.mobile.commerce" 
                        elif e_commerce == "OLX":
                            selected_url = "com.app.tokobagus.betterb"   

                        # Proses Scraping dari web
                        result, continuation_token = reviews(
                            selected_url,
                            lang='id',  # defaultnya adalah 'en'
                            country='id',  # defaultnya adalah 'us'
                            sort=Sort.MOST_RELEVANT,  # mengambil ulasan terbaru
                            count=min(number, 500),  # Limit the count to 500
                            filter_score_with=None  # defaultnya adalah None (artinya semua skor) Gunakan 1, 2, 3, 4, atau 5 untuk memilih skor tertentu)
                        )

                        df_busu = pd.DataFrame(np.array(result), columns=['review'])
                        df_busu = df_busu.join(pd.DataFrame(df_busu.pop('review').tolist()))
                        st.session_state['df'] = df_busu[['userName', 'at', 'content', 'score']]
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat scraping: {str(e)}")
                    if number > 500:
                        st.warning("Jumlah ulasan yang diminta harus kurang dari atau sama dengan 500. Hanya 500 ulasan yang akan diambil.")


    if st.session_state.df is None:
        st.warning("Anda harus scraping ulasan terlebih dahulu!")
    else:
        df = st.session_state['df']
        load_data(df)
        
        # Sentimen
        # Load the TfidfVectorizer and SVM model for Sentiment
        model_folder2 = 'Trained_Model_Sentiment'
        vectorizer_path2 = model_folder2 + '/vectorizer2.pkl'
        svm_model_path2 = model_folder2 + '/svm_model2.pkl'

        # Load the TfidfVectorizer and SVM model
        vectorizer2 = joblib.load(vectorizer_path2)
        svm_model2 = joblib.load(svm_model_path2)

        # Replace missing values (NaN) with empty strings
        df['content'] = df['content'].fillna("")

        # Prepare new data for prediction
        data = df['content']  # Replace with your actual new data

        # Use the loaded vectorizer to transform the new data
        data_transformed2 = vectorizer2.transform(data)

        # Use the loaded SVM model to make predictions on the transformed new data
        predict2 = svm_model2.predict(data_transformed2)

        # Create a dataframe to store intermediate results
        df_result = pd.DataFrame({'Sentiment': predict2, 'Content': data})

        # Create a list to store the final predictions
        predicted_sentiment = []

        # Loop through the intermediate results and make predictions
        for index, row in df_result.iterrows():
            predicted_sentiment_label = row['Sentiment']
            if predicted_sentiment_label == 1:
                predicted_sentiments = 'Positif'
            else:
                predicted_sentiments = 'Negatif'

            predicted_sentiment.append(predicted_sentiments)

        # Add the final predictions to the result_df
        df_result['Sentiment'] = predicted_sentiment

        st.title(f"Ulasan Pengguna untuk {selected}")

        # Get the 'at' column from the original DataFrame
        df_result['at'] = st.session_state['df'].loc[df_result.index, 'at']

        # Calculate the percentage based on the count of "Positive" values in the "Sentiment" column
        total_count = len(df_result)
        positive_count = (df_result['Sentiment'] == 'Positif').sum()
        negative_count = total_count - positive_count

        # Ensure positive and negative counts don't exceed the total count
        positive_count = min(positive_count, total_count)
        negative_count = min(negative_count, total_count - positive_count)

        # Calculate percentages based on capped counts
        positive_percentage = (positive_count / total_count) * 100
        negative_percentage = (negative_count / total_count) * 100

        # Create a horizontal bar chart using matplotlib with green and lighter gray bars
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh([0], [positive_percentage], color='lightblue', height=0.2)  # Blue bar for the positive percentage
        ax.barh([0], [negative_percentage], left=[positive_percentage], color='lightcoral', height=0.2)  # Red bar for the negative percentage

        # Display "Positive" label inside the green bar and above the percentage
        ax.text(positive_percentage / 2, 0, f"Positif\n{positive_percentage:.2f}%", fontsize=8, va='center', color='black', ha='center', rotation=0)

        # Display "Negative" label inside the light gray bar and above the percentage
        ax.text(positive_percentage + (negative_percentage / 2), 0, f"Negatif\n{negative_percentage:.2f}%", fontsize=8, va='center', color='black', ha='center', rotation=0)

        ax.set_xlim(0, 100)
        ax.axis('off')  # Remove axis

        # Display the chart in Streamlit
        st.pyplot(fig)

        # Display each review along with its sentiment and 'at' value
        st.subheader("Ulasan Pengguna, Tanggal, dan Sentimennya")
        st.dataframe(df_result[['at', 'Content', 'Sentiment']])

        st.subheader("Catatan")
        st.write("Total Data = ", total_count)
        st.write("Sentimen Positif = ", positive_count)
        st.write("Sentimen Negatif = ", negative_count)

        # Convert the 'at' column to datetime
        df_result['at'] = pd.to_datetime(df_result['at'])

        # Group the DataFrame by month and calculate the percentage of positive and negative sentiments per month
        df_result['Month'] = df_result['at'].dt.to_period('M')
        monthly_sentiments = df_result.groupby('Month')['Sentiment'].value_counts(normalize=True).unstack().fillna(0) * 100

        # Add missing months if any
        min_month = df_result['Month'].min()
        max_month = df_result['Month'].max()
        all_months = pd.period_range(start=min_month, end=max_month, freq='M')
        monthly_sentiments = monthly_sentiments.reindex(all_months, fill_value=0)

        # Plot the line chart for monthly sentiments
        st.subheader("Persentase Sentimen Positif dan Negatif per Bulan")
        fig, ax = plt.subplots(figsize=(10, 6))

        if 'Positif' in monthly_sentiments.columns:
            ax.plot(monthly_sentiments.index.astype(str), monthly_sentiments['Positif'], marker='o', linestyle='-', color='lightblue', label='Positif')

        if 'Negatif' in monthly_sentiments.columns:
            ax.plot(monthly_sentiments.index.astype(str), monthly_sentiments['Negatif'], marker='o', linestyle='-', color='lightcoral', label='Negatif')

        ax.set_xlabel('Bulan')
        ax.set_ylabel('Persentase Sentimen')
        ax.set_title('Persentase Sentimen Positif dan Negatif per Bulan')
        ax.set_xticklabels(all_months.astype(str), rotation=45)  # Use all_months for x-axis labels
        ax.set_ylim(0, 100)  
        ax.legend()
        st.pyplot(fig)

elif selected == "Prediksi Sentimen":
    st.subheader("Prediksi Sentimen")

    # Sentiment prediction form
    with st.form(key='sentiment_form'):
        review_text = st.text_area(label='Masukkan ulasan Anda:')
        submit_button = st.form_submit_button(label='Prediksi Sentimen')

    # Perform sentiment prediction
    if submit_button:
        if review_text.strip() == '':
            st.warning("Silakan masukkan ulasan!")
        elif not any(char.isalnum() for char in review_text):
            st.warning("Ulasan tidak terdeteksi. Silakan masukkan ulasan yang valid.")
        else:
            try:
                # Load the TfidfVectorizer and SVM model for Sentiment
                model_folder2 = 'Trained_Model_Sentiment'
                vectorizer_path2 = model_folder2 + '/vectorizer2.pkl'
                svm_model_path2 = model_folder2 + '/svm_model2.pkl'

                # Load the TfidfVectorizer and SVM model
                vectorizer2 = joblib.load(vectorizer_path2)
                svm_model2 = joblib.load(svm_model_path2)

                # Transform the input text using the loaded vectorizer
                transformed_review = vectorizer2.transform([review_text])

                # Use the loaded SVM model to make predictions on the transformed text
                prediction = svm_model2.predict(transformed_review)

                # Display prediction result
                if prediction == 1:
                    st.success("Sentimen ulasan adalah positif!")
                else:
                    st.error("Sentimen ulasan adalah negatif.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")
