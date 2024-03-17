# Information Retrieval - Boolean_Model

This project implements a basic boolean model for information retrieval on a static set of documents. It supports two types of queries: boolean queries and proximity queries.

# Features

1) **Indexing:** The system either loads an existing index or creates a new one if none exists.

2) **Document Parsing:** The system reads a specified set of documents.

3) **Tokenization:** Text from the documents undergoes tokenization.

4) **Text Cleaning:** The system removes stop words, punctuation, alphanumeric words, and more to clean the tokens.

5) **Storage:** Cleaned tokens are stored in the index, which is saved locally.

6) **Query Interface:** The system provides a user-friendly interface for querying.

# Getting Started

To execute the project, you'll need to compile it on your own. Ensure that the file and folder structure remains unchanged, and all files from the repository are available. You'll require Python and the following libraries: nltk, re, string, os, pickle, and Tkinter. Run the main.py file. If you want to work with custom files, include them in the code by adding their filenames to the "Files" list within main.py. If the program has been executed previously, a file named "term_frequency_and_postings.csv" will be generated. To update the index for new files, you must delete this file.
