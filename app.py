import streamlit as st
import pickle
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io

# NLTK setup
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download stopwords and punkt if not already available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ps = PorterStemmer()

st.set_page_config(page_title="Email/SMS Spam Classifier", layout="wide", page_icon="📧")

@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
    return tfidf, model

tfidf, model = load_models()

@st.cache_data
def load_data():
    try:
        # Many spam datasets use latin-1 encoding
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df.iloc[:, :2] # keep only first two columns
        df.columns = ['target', 'text']
        return df
    except FileNotFoundError:
        return None

def transform_text(text):
    text = str(text).lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("📧 Email/SMS Spam Classifier")
st.markdown("Use this AI-powered tool to detect whether a message is **Spam** or **Ham** (Not Spam).")

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction", "📊 Data Analytics"])

with tab1:
    st.subheader("Single Message Prediction")
    input_sms = st.text_area("Enter the email or SMS message:", height=150)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        predict_btn = st.button('Predict', type="primary")
        
    if predict_btn and input_sms:
        with st.spinner("Analyzing message..."):
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            
            # Predict Probabilities if available
            try:
                probs = model.predict_proba(vector_input)[0]
                spam_prob = probs[1] * 100
                ham_prob = probs[0] * 100
            except AttributeError:
                spam_prob = None
                ham_prob = None
            
            st.markdown("---")
            if result == 1:
                st.error("🚨 **Classification: SPAM**")
                if spam_prob is not None:
                    st.write(f"Confidence: **{spam_prob:.2f}%**")
            else:
                st.success("✅ **Classification: NOT SPAM (HAM)**")
                if ham_prob is not None:
                    st.write(f"Confidence: **{ham_prob:.2f}%**")

with tab2:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV file containing a column named **`text`** (or **`message`**) to predict multiple items at once.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            # Find the text column
            text_col = None
            for col in ['text', 'message', 'sms', 'email', 'v2']:
                if col in batch_df.columns.str.lower():
                    # get the actual column name that matches
                    text_col = batch_df.columns[batch_df.columns.str.lower() == col][0]
                    break
                    
            if text_col is None:
                st.error("Could not find a text column. Please ensure your CSV has a column named 'text' or 'message'.")
            else:
                st.success(f"Found text column: `{text_col}` with {len(batch_df)} rows.")
                
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch predictions... this may take a moment."):
                        # Process texts
                        processed_texts = batch_df[text_col].apply(transform_text)
                        vector_input = tfidf.transform(processed_texts)
                        predictions = model.predict(vector_input)
                        
                        # Map 1 to Spam, 0 to Ham
                        batch_df['Prediction'] = ['Spam' if p == 1 else 'Ham' for p in predictions]
                        
                        # Show preview
                        st.subheader("Prediction Preview")
                        st.dataframe(batch_df[[text_col, 'Prediction']].head(10))
                        
                        # Download button
                        csv_output = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_output,
                            file_name='predictions_output.csv',
                            mime='text/csv',
                        )
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab3:
    st.subheader("Data & Model Analytics")
    
    df = load_data()
    if df is not None:
        st.markdown("Visualizations based on the training dataset (`spam.csv`).")
        
        # 1. Distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Message Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            # Pie chart
            target_counts = df['target'].value_counts()
            ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
            ax.axis('equal')
            st.pyplot(fig)
            
        with col2:
            st.markdown("#### Dataset Sample")
            st.dataframe(df.head(10), use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### Word Clouds")
        st.markdown("Most frequent words appearing in Ham versus Spam messages.")
        
        wc_col1, wc_col2 = st.columns(2)
        
        # We need to create WordCloud, let's process some texts
        # To avoid being too slow, we'll just take a smaller sample or use raw words
        wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
        
        with wc_col1:
            st.markdown("**Spam Messages**")
            spam_texts = df[df['target'] == 'spam']['text'].str.cat(sep=" ")
            spam_wc = wc.generate(spam_texts)
            fig_spam, ax_spam = plt.subplots(figsize=(5, 5))
            ax_spam.imshow(spam_wc)
            ax_spam.axis('off')
            st.pyplot(fig_spam)
            
        with wc_col2:
            st.markdown("**Ham Messages**")
            ham_texts = df[df['target'] == 'ham']['text'].str.cat(sep=" ")
            ham_wc = wc.generate(ham_texts)
            fig_ham, ax_ham = plt.subplots(figsize=(5, 5))
            ax_ham.imshow(ham_wc)
            ax_ham.axis('off')
            st.pyplot(fig_ham)
            
    else:
        st.info("The file `spam.csv` was not found. Analytics cannot be displayed.")
