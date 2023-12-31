import streamlit as st
import pandas as pd
from Bio.Seq import Seq
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://free4kwallpapers.com/uploads/originals/2021/03/01/dna.-wallpaper_.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

with open('model_0.pkl', 'rb') as model_file:
    loaded_mnb_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

def generate_kmers(sequence, k):
    return [str(sequence[i:i+k]) for i in range(len(sequence) - k + 1)]

def create_dataframe_column_kmers(df, sequence_column, k):
    df['kmers'] = df[sequence_column].apply(lambda x: generate_kmers(Seq(x), k))
    return df
st.text('Created by: Rakesh MK on 31/12/2023')
with st.expander("Learn More"):
    st.write("This is additional information you want to provide.")


st.title('🧬GENOMIC Classification')

sequence_pred = st.text_area("Enter a valid  DNA Sequence:", "")
st.text("(eg:ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCCATACTCCTTACACTATT....)")                            



# Assuming sequence is entered as a string, not a list
data = pd.DataFrame({'sequence': [sequence_pred]})

if st.button('Predict'):
    if sequence_pred:
        k_value = 6

        # Create dataframe with sequences and their k-mers
        data_dna = create_dataframe_column_kmers(data, 'sequence', k_value)

        data_dna['text'] = data_dna['kmers'].apply(lambda x: ' '.join(x))

        X_pred = loaded_vectorizer.transform(data_dna['text'])

        X_array_pred = X_pred.toarray()

        predN = loaded_mnb_model.predict(X_array_pred)

        numerical_to_nucleotide = {
            0: 'G protein-coupled receptors',
            1: 'Tyrosine kinase',
            2: 'Tyrosine phosphatase',
            3: 'Synthetase',
            4: 'Synthase',
            5: 'Ion channel',
            6: 'Transcription factor'
        }
        predictedN = [numerical_to_nucleotide[label] for label in predN]
        st.text("Predicted gene family is:")
        for label in predictedN:
            st.write(label)
