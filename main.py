import tkinter as tk
import nltk
import string
import re
import os
import csv
import pickle
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('punkt')


# Function to remove URLs from text
def removing_url(text):
    url_pattern = re.compile(r'https?://\S+|http?://\S+|http\.\S+|www\.\S+')
    return url_pattern.sub(' ', text)

# Tokenization function
def tokenization(content_Without_URL):
    #removing numbers from the word that combine together Like this 1faculty
    content_Without_URL = content_Without_URL.replace("1", " ")
    content_Without_URL = content_Without_URL.replace("2", " ") 
    content_Without_URL = content_Without_URL.replace("3", " ")
    content_Without_URL = content_Without_URL.replace("4", " ")
    content_Without_URL = content_Without_URL.replace("5", " ")
    content_Without_URL = content_Without_URL.replace("6", " ")
    content_Without_URL = content_Without_URL.replace("7", " ")
    content_Without_URL = content_Without_URL.replace("8", " ")
    content_Without_URL = content_Without_URL.replace("9", " ")
    content_Without_URL = content_Without_URL.replace("0", " ")
    content_Without_URL = content_Without_URL.replace("�", " ")
    content_Without_URL = content_Without_URL.replace("", " ")
    content_Without_URL = content_Without_URL.replace("¨", " ")
    content_Without_URL = content_Without_URL.replace("´", " ")
    content_Without_URL = content_Without_URL.replace("¼", " ")
    content_Without_URL = content_Without_URL.replace("¸", " ")
    content_Without_URL = content_Without_URL.replace("···", " ")
    content_Without_URL = content_Without_URL.replace("·", " ")
    content_Without_URL = content_Without_URL.replace("······", " ")
    content_Without_URL = content_Without_URL.replace("×", " ")
    content_Without_URL = content_Without_URL.replace("ß", " ")
    content_Without_URL = content_Without_URL.replace("ü", " ")
    content_Without_URL = content_Without_URL.replace("þ", " ")
    content_Without_URL = content_Without_URL.replace("˜", " ")
    content_Without_URL = content_Without_URL.replace("š", " ")
    content_Without_URL = content_Without_URL.replace("ž", " ")
    content_Without_URL = content_Without_URL.replace("ý", " ")
    content_Without_URL = content_Without_URL.replace("ˆ", " ")
    content_Without_URL = content_Without_URL.replace("–", " ")
    content_Without_URL = content_Without_URL.replace("•", " ")
    content_Without_URL = content_Without_URL.replace("…", " ")
    content_Without_URL = content_Without_URL.replace("——", " ")
    content_Without_URL = content_Without_URL.replace("—", " ")
    # Tokenize the content into words
    words = word_tokenize(content_Without_URL)
    return words

# Function to process a file
def process_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        content_Without_URL = removing_url(content)
    words = tokenization(content_Without_URL)
    normalized_words = normalization(words)
    return normalized_words

# Function to remove punctuation and special characters from each word and filter out empty strings
def removing_punctuation(words):
    table = str.maketrans('', '', string.punctuation + '�')
    return [word.translate(table) for word in words if word.translate(table)]

# Load stop words from Stopword-List.txt
with open("Stopword-List.txt","r") as file:
    stopwords = file.read().split()

# Function to remove stop words from the tokens
def remove_stop_words(words):
    return [token for token in words if token.lower() not in stopwords]

# Function to remove numbers and single character tokens from the tokens
def remove_single_character_tokens(words):
    return [token for token in words if not token.isdigit() and len(token) > 1 ]

# Case Folding: Making the Tokens into Small letters
def case_fold(tokens):
    return [token.lower() for token in tokens]
# Function to normalize the tokens
def normalization(tokens):
    tokens = removing_punctuation(tokens)
    tokens = remove_stop_words(tokens)
    tokens = remove_single_character_tokens(tokens)
    tokens = case_fold(tokens)
    tokens = stemming(tokens)  # Fix: Defined the stemming function before using it
    #tokens = lemmatization(tokens)
    return tokens

# Stemming function: To make the Tokens into Root form. Like classification into classifi
def stemming(words):
    ps = PorterStemmer()
    stemmed_word = []
    for w in words:
        stemmed_word.append(ps.stem(w))
    return stemmed_word


#lemmatization
def lemmatization(words):
    wnl = WordNetLemmatizer()
    lemmatized_words = []
    for w in words:
        lemma_word = wnl.lemmatize(w)
        lemmatized_words.append(lemma_word)
    return lemmatized_words


# Directory containing all text files
directory = r"C:\Users\HP\OneDrive\Desktop\IR ASSIGNMENT\Assignment Of IR\ResearchPapers"
total_word_count = 0
all_tokens = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        words = process_file(file_path)
        num_words = len(words)
        #print the total no of words/Tokens in each file 
        #print(f"File: {filename}, Word Count: {num_words}")
        total_word_count += num_words
        all_tokens.extend(words)
        
#printing the Total Number of Words/Tokens in all files
#print("Total number of words in all files:", total_word_count)

# Making Text file for all tokens for checking purpose
# with open('all_tokens.txt', 'w') as file:
#     for token in all_tokens:
#         file.write(token.lower() + '\n')
    
all_tokens_set = set(all_tokens)

# Writing distinct tokens to a file
# Storing the all distinct Tokens for checking purpose
# with open('distinct_tokens.txt', 'w') as file:
#     for token in sorted(all_tokens_set):
#         file.write(token.lower() + '\n')
        
        
# Function to create a new node
def create_node(doc_id, position):
    return {"doc_id": doc_id, "position": position, "next": None}

# Function to insert a new node into the linked list
def insert_node(head, doc_id, position):
    new_node = create_node(doc_id, position)
    if head is None:
        return new_node
    current = head
    while current["next"]:
        current = current["next"]
    current["next"] = new_node
    return head

# Function to build inverted index
def build_indexes(docs_dir):
    inverted_index = defaultdict(dict)
    term_frequency = defaultdict(int)
    for filename in os.listdir(docs_dir):
        if filename.endswith('.txt'):
            doc_id = filename[:-4]
            doc_path = os.path.join(docs_dir, filename)
            tokens = process_file(doc_path)
            for term in set(tokens):  # Only unique terms
                term_frequency[term] += 1
                if doc_id not in inverted_index[term]:
                    inverted_index[term][doc_id] = [i for i, t in enumerate(tokens) if t == term]
    return inverted_index, term_frequency

# Function to save term frequency and posting list to CSV file
def save_term_frequency_and_postings(term_frequency, inverted_index, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Term', 'Frequency', 'PostingList'])
        for term in sorted(term_frequency.keys()):  # Sort terms alphabetically
            frequency = term_frequency[term]
            posting_list = inverted_index[term]
            writer.writerow([term, frequency, posting_list])

# The load_term_frequency_and_postings function reads the term frequency and postings from a CSV file and loads them into memory
def load_term_frequency_and_postings(input_file):
    term_frequency = {}
    inverted_index = {}

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            term, frequency, posting_list_str = row
            frequency = int(frequency)
            # Convert posting list from string to list
            posting_list = eval(posting_list_str)
            inverted_index[term] = posting_list
            term_frequency[term] = frequency
    
    return term_frequency, inverted_index

# Example usage:
# where all documents placed is the docs_dir varaible means
docs_dir = r'C:\Users\HP\OneDrive\Desktop\IR ASSIGNMENT\Assignment Of IR\ResearchPapers'
#where to store the inverted index file the output_file means
output_file = r'C:\Users\HP\OneDrive\Desktop\IR ASSIGNMENT\Assignment Of IR\term_frequency_and_postings.csv'

#building inverted index
inverted_index, term_frequency = build_indexes(docs_dir)
#storing the inverted index
save_term_frequency_and_postings(term_frequency, inverted_index, output_file)
# Loading it back to memory for use
term_frequency, inverted_index = load_term_frequency_and_postings(r"C:\Users\HP\OneDrive\Desktop\IR ASSIGNMENT\Assignment Of IR\term_frequency_and_postings.csv")

# Function to build positional index
def build_positional_index(docs_dir):
    positional_index = defaultdict(dict)
    for filename in os.listdir(docs_dir):
        if filename.endswith('.txt'):
            doc_id = filename[:-4]
            doc_path = os.path.join(docs_dir, filename)
            tokens = process_file(doc_path)
            for position, term in enumerate(tokens):
                if term not in positional_index:
                    positional_index[term] = defaultdict(list)
                positional_index[term][doc_id].append(position)
    # Sort the positional index by position
    for term in positional_index:
        for doc_id in positional_index[term]:
            positional_index[term][doc_id].sort()
    return positional_index

# Function to save positional index to CSV file
def save_positional_index(positional_index, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Term', 'DocID', 'Positions'])
        for term, postings in positional_index.items():
            for doc_id, positions in postings.items():
                positions_str = '[' + ', '.join(map(str, positions)) + ']'
                writer.writerow([term, doc_id, positions_str])
                
#  The load_positional_index function reads a positional index from a CSV file and loads it into memory.
def load_positional_index(file_path):
    positional_index = {}
    with open(file_path, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            term = row[0]
            doc_id = int(row[1])
            positions = [int(pos) for pos in row[2][1:-1].split(',')]  # Extract positions from string
            if term not in positional_index:
                positional_index[term] = {}
            positional_index[term][doc_id] = positions
    return positional_index


# Example usage:
# where all documents placed is the docs_dir varaible means
docs_dir = r'C:\Users\HP\OneDrive\Desktop\IR ASSIGNMENT\Assignment Of IR\ResearchPapers'
#where to store the Positional index file the output_file means
output_file = r'C:\Users\HP\OneDrive\Desktop\IR ASSIGNMENT\Assignment Of IR\positional_index.csv'

#building Positional index
positional_index = build_positional_index(docs_dir)
#storing the Positional index
save_positional_index(positional_index, output_file)
# Loading it back to memory for use
loaded_positional_index = load_positional_index(output_file)


# Function to parse the user query
def parse_query(user_query):
    query_terms = []
    operators = []
    for term in user_query.split():
        if term.upper() in {'AND', 'OR', 'NOT'}:
            operators.append(term.upper())
        else:
            query_terms.append(term)
    return query_terms, operators

# Function to evaluate the cost of the query (for simplicity, we'll consider all terms equally)
def evaluate_cost(query_terms):
    return len(query_terms)

# Function to check the complex query
def is_complex_query(query_terms, operators):
    # Check if the query contains any boolean operators
    return any(op in operators for op in ['AND', 'OR', 'NOT'])


# Function to get all document IDs from the files in a directory
def get_all_document_ids(docs_dir):
    document_ids = set()
    for filename in os.listdir(docs_dir):
        if filename.endswith('.txt'):
            doc_id = filename[:-4]  # Remove the file extension to get the document ID
            document_ids.add(doc_id)
    return document_ids

docs_dir = r'C:\Users\HP\OneDrive\Desktop\IR ASSIGNMENT\Assignment Of IR\ResearchPapers'

# Get all document IDs
all_documents = get_all_document_ids(docs_dir)

# Function for attending NOT operator Queries
def process_query_result(search_terms, operators, inverted_index, all_documents):
    # Initialize an empty set for documents containing the search terms
    documents_containing_terms = set()

    # Add documents containing the first search term to the set
    if search_terms[0] in inverted_index:
        documents_containing_terms = set(inverted_index[search_terms[0]])

    # Perform intersection, union, or exclusion operation for the remaining search terms
    i = 1
    while i < len(search_terms):
        if search_terms[i] in inverted_index:
            if operators[i - 1] == 'AND':
                documents_containing_terms = documents_containing_terms.intersection(inverted_index[search_terms[i]])
            elif operators[i - 1] == 'OR':
                documents_containing_terms = documents_containing_terms.union(inverted_index[search_terms[i]])
            elif operators[i - 1] == 'NOT':
                if search_terms[i] in inverted_index:
                    documents_containing_terms = documents_containing_terms.difference(inverted_index[search_terms[i]])
        i += 1

    # Perform the NOT operation to exclude documents containing the search terms
    documents_not_containing_terms = all_documents - documents_containing_terms

    return documents_not_containing_terms

#Function to attended the Complex Queries
def complex_execute_query(inverted_index, query_terms, operators):
    if len(query_terms) == 0:
        return set()  # Return empty set if there are no query terms
    
    result = set()
    if query_terms[0] in inverted_index:
        result = set(inverted_index[query_terms[0]])
    else:
        if operators and operators[0] == 'AND':  # If the first operator is AND, return empty set
            return set()
    
    i = 0
    while i < len(operators) and i < len(query_terms) - 1:  # Ensure not to exceed the length of query_terms
        if operators[i] == 'AND' or operators[i] == 'OR':
            if query_terms[i+1] in inverted_index:
                if operators[i] == 'AND':
                    result = result.intersection(inverted_index[query_terms[i+1]])
                else:
                    result = result.union(inverted_index[query_terms[i+1]])
                i += 1
            else:
                if operators[i] == 'AND':  # If term not found and operator is AND, return empty set
                    return set()
        elif operators[i] == 'NOT':
            if query_terms[i+1] in inverted_index:
                result = result.difference(inverted_index[query_terms[i+1]])
                i += 1
            else:
                return set()  # Return empty set if the term after NOT is not found

    # Handle remaining query terms without operators
    while i < len(query_terms):
        if query_terms[i] == 'NOT':
            if i + 1 < len(query_terms):
                if query_terms[i + 1] in inverted_index:
                    result = result.difference(inverted_index[query_terms[i + 1]])
                else:
                    return set()  # Return empty set if the term after NOT is not found
            else:
                return set()  # Return empty set if NOT is the last query term
        elif query_terms[i] in inverted_index:
            result = result.intersection(inverted_index[query_terms[i]])
        else:
            return set()  # Return empty set if any query term is not found
        i += 1
    
    return result


#Function To attend the Simple Queries
def execute_query(inverted_index, query_terms, operators):
    if len(query_terms) == 0:
        return set()  # Return empty set if there are no query terms
    
    result = set()
    if query_terms[0] in inverted_index:
        result = set(inverted_index[query_terms[0]])
    else:
        if operators and operators[0] == 'AND':  # If the first operator is AND, return empty set
            return set()
    
    i = 0
    while i < len(operators) and i < len(query_terms) - 1:  # Ensure not to exceed the length of query_terms
        if operators[i] == 'AND' or operators[i] == 'OR':
            if query_terms[i+1] in inverted_index:
                if operators[i] == 'AND':
                    result = result.intersection(inverted_index[query_terms[i+1]])
                else:
                    result = result.union(inverted_index[query_terms[i+1]])
                i += 1
            else:
                if operators[i] == 'AND':  # If term not found and operator is AND, return empty set
                    return set()
        elif operators[i] == 'NOT':
            if query_terms[i+1] in inverted_index:
                result = result.difference(inverted_index[query_terms[i+1]])
                i += 1
            else:
                return set()  # Return empty set if the term after NOT is not found

    # Handle remaining query terms without operators
    while i < len(query_terms):
        if query_terms[i] == 'NOT':
            if i + 1 < len(query_terms):
                if query_terms[i + 1] in inverted_index:
                    result = result.difference(inverted_index[query_terms[i + 1]])
                else:
                    return set()  # Return empty set if the term after NOT is not found
            else:
                return set()  # Return empty set if NOT is the last query term
        elif query_terms[i] in inverted_index:
            result = result.intersection(inverted_index[query_terms[i]])
        else:
            return set()  # Return empty set if any query term is not found
        i += 1
    
    return result

#Function For handling Proximity Search
def execute_proximity_query(positional_index, term1, term2, k):
    # Check if both terms exist in the positional index
    if term1 not in positional_index or term2 not in positional_index:
        return []

    # Initialize result list to store matching document IDs
    result = []

    # Iterate through documents containing both terms
    for doc_id in positional_index[term1]:
        if doc_id in positional_index[term2]:
            positions_term1 = positional_index[term1][doc_id]
            positions_term2 = positional_index[term2][doc_id]

            # Check for proximity between terms
            for pos1 in positions_term1:
                for pos2 in positions_term2:
                    if abs(pos1 - pos2) <= k:
                        result.append(doc_id)
                        break  # Break inner loop if proximity condition is met
                if doc_id in result:
                    break  # Break outer loop if document already added to result

    return result

# The search_query function with the conditional execution based on whether the query is complex or not:
def search_query():
    user_query = query_entry.get()
    query_terms, operators = parse_query(user_query)
    cost = evaluate_cost(query_terms)
    if 'NOT' in operators:
        result_docs = process_query_result(query_terms, operators, inverted_index, all_documents)
    else:
        if is_complex_query(query_terms, operators):  # Assuming you have a function is_complex_query to identify complex queries
            result_docs = complex_execute_query(inverted_index, query_terms, operators)
        else:
            result_docs = execute_query(inverted_index, query_terms, operators)
    result_text.delete(1.0, tk.END)  # Clear previous results
    result_text.insert(tk.END, "Documents matching the query:\n")
    if result_docs:
        for doc_id in result_docs:
            result_text.insert(tk.END, doc_id + "\n")
    else:
        result_text.insert(tk.END, "No documents found")


# This  function handles the GUI window closing event and saves any changes made before exiting
def search_positional_index():
    def search():
        query = proximity_entry.get()
        terms = query.split()
        if len(terms) != 3:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "Invalid input format. Please enter a query in the format: term1 term2 k")
            return
        term1, term2, k = terms
        try:
            k = int(k)
        except ValueError:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "Invalid input format. 'k' must be an integer representing the maximum distance between terms.")
            return
        result = execute_proximity_query(loaded_positional_index, term1, term2, k)
        result_text.delete(1.0, tk.END)  # Clear previous results
        if result:
            result_text.insert(tk.END, f"Documents containing both '{term1}' and '{term2}' with distance {k} apart:\n")
            for doc_id in result:
                result_text.insert(tk.END, f"{doc_id}\n")
        else:
            result_text.insert(tk.END, f"No documents found for the proximity query")

    search()

# Function is used to clear the text entry field when the clear button is clicked
def clear_query():
    query_entry.delete(0, tk.END)
    proximity_entry.delete(0,tk.END)
    result_text.delete(1.0, tk.END)
    
# Create the main window
root = tk.Tk()
root.title("Simple Search Engine")

# Create and place the widgets for the normal query
query_label = tk.Label(root, text="Enter your query:")
query_label.grid(row=0, column=0, padx=10, pady=10)

query_entry = tk.Entry(root, width=50)
query_entry.grid(row=0, column=1, padx=10, pady=10)

# This Button is used for Searching Simple and Complex Query
search_button = tk.Button(root, text="Search", command=search_query)
search_button.grid(row=0, column=2, padx=10, pady=10)

# Create and place the widgets for the proximity query
proximity_label = tk.Label(root, text="Enter proximity query (term1 term2 k):")
proximity_label.grid(row=1, column=0, padx=10, pady=10)

proximity_entry = tk.Entry(root, width=50)
proximity_entry.grid(row=1, column=1, padx=10, pady=10)

# This Button is used for Searching Proximity Query
proximity_search_button = tk.Button(root, text="Proximity Search", command=search_positional_index)
proximity_search_button.grid(row=1, column=2, padx=10, pady=10)

# This Button is used for Clearing the input field and output field also
clear_button = tk.Button(root, text="Clear", command=clear_query)
clear_button.grid(row=0, column=3, padx=10, pady=10)

result_text = tk.Text(root, width=60, height=20)
result_text.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

# Run the application
root.mainloop()
