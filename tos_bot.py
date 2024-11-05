
#this is my first project, extremely proud of it.

#import nltk as n
from nltk.tokenize import sent_tokenize
from transformers import pipeline


# Function to load text from a file or input string
def load_text(input_text=None, file_path=None):
    if file_path:
        with open(file_path, 'r') as file:
            text = file.read()
    elif input_text:
        text = input_text
    else:
        raise ValueError("No input text or file path provided.")
    
    if not text.strip():
        raise ValueError("The provided text is empty.")
    
    return text


# Function to preprocess the text (e.g., removing special characters)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.replace('\n', ' ')  # Remove newlines
    return text


# Function to tokenize the text into sentences
def tokenize_sentences(text):
    return sent_tokenize(text)


# Function to summarize the text using a pre-trained model
def summarize_text(text, summarizer, chunk_size=512):

    # Handle large texts by splitting them into chunks
    tokens = text.split()
    chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    
    summary_points = []
    for chunk in chunks:
        summarized = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary_points.extend([point['summary_text'] for point in summarized])
    
    return summary_points

# Function to format the summary into bullet points
def format_summary(summary):
    return "\n".join([f"- {point}" for point in summary])

# Main function to put everything together
def summarize_terms_of_service(input_text=None, file_path=None):
    try:
        text = load_text(input_text, file_path)
        processed_text = preprocess_text(text)
        sentences = tokenize_sentences(processed_text)
        
        # Join sentences for summarization, or handle them individually if needed
        joined_text = " ".join(sentences)
        
        # Initialize the summarization pipeline
        summarizer = pipeline("summarization")
        
        summary = summarize_text(joined_text, summarizer)
        return format_summary(summary)
    
    except ValueError as e:
        return str(e)

# Example usage
terms_of_service_text = """Your long terms of service text goes here."""
summary = summarize_terms_of_service(input_text=terms_of_service_text)
print(summary)
