import streamlit as st
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

#Model to convert sentences to embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step-1: Upload PDFs
def uploading():
    # Create a folder to store uploaded PDFs if it doesn't exist
    UPLOAD_FOLDER = "uploaded_pdfs"
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Initialize the list to store PDF file paths
    if 'pdf_paths' not in st.session_state:
        st.session_state.pdf_paths = []
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add the file path to the list if it's not already there
            if file_path not in st.session_state.pdf_paths:
                st.session_state.pdf_paths.append(file_path)
            
            st.success(f"File {uploaded_file.name} has been uploaded and saved.")
    return st.session_state.pdf_paths

def extract_headings_from_pdf(pdf_paths):
    all_headings=[]
    pdf_sources = []
    for path in pdf_paths:
        doc = fitz.open(path)
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["flags"] & 2**4:  # Check if bold
                                all_headings.append(span["text"])
                                pdf_sources.append(os.path.basename(path))
        doc.close()
    return all_headings,pdf_sources

def convert_to_embeddings_and_store(headings, pdf_sources):
    # Convert headings to embeddings
    embeddings = model.encode(headings, show_progress_bar=True)
    
    # Convert embeddings to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings)
    
    # Save the index
    faiss.write_index(index, "embeddings.faiss")
    
    # Save the headings and PDF sources
    with open("metadata.npy", "wb") as f:
        np.save(f, np.array(headings))
        np.save(f, np.array(pdf_sources))
    
    st.success(f"Stored {len(headings)} headings and embeddings in the FAISS index.")
    #return embeddings

def search_similar_headings(query, top_k=3):
    # Load the FAISS index
    index = faiss.read_index("embeddings.faiss")
    
    # Load the headings and PDF sources
    with open("metadata.npy", "rb") as f:
        headings = np.load(f, allow_pickle=True)
        pdf_sources = np.load(f, allow_pickle=True)
    
    # Convert query to embedding
    query_embedding = model.encode([query]).astype('float32')    
    distances, indices = index.search(query_embedding, top_k)
    
    # Return results
    results = [(headings[i], pdf_sources[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# Main Streamlit app
def main():
    st.title("Knowledge Base Uploader and Search")
    pdf_paths = uploading()
    if st.button("Process PDFs"):
        if pdf_paths:
            headings, pdf_sources = extract_headings_from_pdf(pdf_paths)
            if headings:
                convert_to_embeddings_and_store(headings, pdf_sources)
            else:
                st.warning("No headings found in the uploaded PDFs.")
        else:
            st.warning("No PDFs uploaded yet.")
                
    # Separate search functionality
    st.subheader("Query")
    query = st.text_input("Enter an IT issue:")
    if st.button("Search"):
        if query:
            st.write("Process starting")
            if os.path.exists("embeddings.faiss") and os.path.exists("metadata.npy"):
                results = search_similar_headings(query)
                st.write("The searched issues are in",end=" ")
                for heading, pdf_source, distance in results:
                    st.write(f"Heading: {heading}")
                    st.write(pdf_source,end=",")
                    st.write(f"Distance: {distance:.2f}")
                    st.write("---")
            else:
                st.warning("No index found. Please process PDFs first.")
        else:
            st.warning("Please enter a query.")
            
        

if __name__ == "__main__":
    main()

