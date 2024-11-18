import os
import sys
import platform
import subprocess
import logging
import warnings

# Suppress logging and warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("nltk").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.tokenization_utils_base")
warnings.simplefilter(action='ignore', category=FutureWarning)


def is_in_virtualenv_check():
    # Check if the script is running inside a virtual environment
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)


def install_virtualenv():
    try:
        import virtualenv
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "-q", "virtualenv"],
            check=True
        )


def setup_virtual_environment():
    # Set up the virtual environment and install required packages
    install_virtualenv()
    # Virtual environment directory
    venv_dir = 'env'

    # Path to the Python executable in the virtual environment
    if platform.system() == 'Windows':
        python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe')
    else:
        python_executable = os.path.join(venv_dir, 'bin', 'python')

    # Check if the virtual environment already exists
    if not os.path.exists(venv_dir):
        # Create the virtual environment
        subprocess.run([sys.executable, "-m", "virtualenv", venv_dir], check=True)

    # Upgrade pip
    subprocess.run([python_executable, "-m", "pip", "install", "--upgrade", "-q", "pip"], check=True)
    # Install compatible versions of torch and torchtext
    subprocess.run([python_executable, "-m", "pip", "install", "-q", "torch==1.13.1", "torchtext==0.14.1"],check=True)
    # Uninstall NLTK to ensure the correct version is installed
    subprocess.run([python_executable, "-m", "pip", "uninstall", "-y", "nltk"],check=True )
    # Install required packages with specific versions
    packages = ["numpy<2", "datasets", "pyvi", "nltk==3.8.1", "tqdm", "transformers"]
    subprocess.run([python_executable, "-m", "pip", "install", "-q", "--force-reinstall"] + packages, check=True)
    # Download NLTK data
    subprocess.run([python_executable, "-c", "import nltk; nltk.download('punkt', quiet=True)"], check=True)
    # Run the script inside the virtual environment
    subprocess.run([python_executable] + sys.argv, check=True)
    sys.exit()


if not is_in_virtualenv_check():
    setup_virtual_environment()

import re
import html
import argparse
from torch.cuda.amp import autocast, GradScaler
import itertools
from functools import partial
import nltk
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset, DatasetDict
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from pyvi import ViTokenizer


# -------------------- Model Architecture --------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx):
        super(Encoder, self).__init__()
        # Size of the vocabulary
        self.vocab_size = vocab_size
        # Dimensionality of the word embeddings
        self.embedding_dim = embedding_dim
        # Dimensionality of the hidden states in the GRU
        self.hidden_dim = hidden_dim
        # Embedding layer that maps input tokens to dense vectors of a specified dimension
        # padding_idx specifies the index of the padding token so that its embeddings are zeroed out
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx)
        # Bidirectional GRU layer that processes the embedded tokens.
        # This layer outputs both forward and backward hidden states, which allows the model to
        # capture contextual information from both directions in the sequence
        self.bi_gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        # Fully connected layer to project the concatenated bidirectional hidden states
        # into a single hidden state, with a dimensionality compatible with the decoder
        self.hidden_fc = nn.Linear(
            in_features=self.hidden_dim * 2,
            out_features=self.hidden_dim
        )

    def forward(self, input_tokens):
        # Pass the input tokens through the embedding layer to obtain their dense representations
        embedded_tokens = self.embedding_layer(input_tokens)
        # Pass the embedded tokens through the bidirectional GRU.
        # gru_outputs contains the hidden states for each time step
        # gru_hidden contains the final hidden states of both directions
        gru_outputs, gru_hidden = self.bi_gru(embedded_tokens)
        # Extract the final hidden state of the forward GRU
        forward_hidden = gru_hidden[0, :, :]
        # Extract the final hidden state of the backward GRU
        backward_hidden = gru_hidden[1, :, :]
        # Concatenate forward and backward hidden states to form a single vector.
        encoder_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        # Pass the concatenated hidden state through the fully connected layer to adjust its dimensionality.
        # Apply tanh activation to introduce non-linearity and enhance representational capacity.
        encoder_hidden = torch.tanh(self.hidden_fc(encoder_hidden))
        # Extra dimension to match the expected shape for initializing the decoder's hidden state.
        encoder_hidden = encoder_hidden.unsqueeze(0)
        # Return the sequence of GRU outputs and the processed hidden state for the decoder.
        return gru_outputs, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim, padding_idx):
        super(Decoder, self).__init__()
        # Dimensionality of the GRU hidden states.
        self.hidden_dim = hidden_dim
        # Size of the output vocabulary.
        self.vocab_size = vocab_size
        # Dimensionality of the word embeddings.
        self.embedding_dim = embedding_dim
        # # Embedding layer to convert input tokens to dense vectors
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_idx)
        # Attention mechanism layers
        # Transform decoder hidden state
        self.attn_hidden = nn.Linear(hidden_dim, hidden_dim)
        # Transform encoder outputs
        self.attn_encoder = nn.Linear(hidden_dim * 2, hidden_dim)
        # Compute attention scores
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        # GRU layer to process the context vector concatenated with embedded input
        self.gru = nn.GRU(self.embedding_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        # Linear layer to project GRU output to vocabulary size for predictions
        self.output_linear = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        # Determine sequence length from encoder outputs
        seq_len = encoder_outputs.size(1)
        # Pass decoder input token through embedding layer and add time dimension
        embedded_input = self.embedding_layer(decoder_input).unsqueeze(1)
        # Prepare decoder hidden state for attention by transposing and expanding
        decoder_hidden_t = decoder_hidden.permute(1, 0, 2).contiguous()
        decoder_hidden_expanded = decoder_hidden_t.expand(-1, seq_len, -1)
        # Transform the decoder hidden state and encoder outputs for attention computation
        attn_hidden = self.attn_hidden(decoder_hidden_expanded)
        attn_encoder = self.attn_encoder(encoder_outputs)
        # Compute alignment scores for each encoder output with respect to the decoder hidden state
        align_scores = torch.tanh(attn_hidden + attn_encoder)
        # Calculate unnormalized attention scores
        attn_scores = self.attn_score(align_scores).squeeze(2)
        # Normalize scores using softmax to obtain attention weights
        attn_weights = torch.softmax(attn_scores, dim=1)
        # Compute context vector as weighted sum of encoder outputs
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # Concatenate embedded input and context vector to form GRU input
        # Pass concatenated input through the GRU to obtain new hidden state
        gru_input = torch.cat((embedded_input, context_vector), dim=2)
        # gru_output represents the output of the GRU for the current time step
        # decoder_hidden is the updated-hidden state of the GRU
        gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
        # Project GRU output to the vocabulary space to obtain scores for each word in the vocabulary
        output_vocab_scores = self.output_linear(gru_output)
        # Return the scores for the current time step and updated hidden state
        return output_vocab_scores, decoder_hidden


# -------------------- Dataset Utilities --------------------
def fix_punctuation_spacing_english(text):
    """
    Corrects spacing issues around punctuation in English text, including contractions, possessives, and general punctuation spacing.
    :param text: english text to correct
    :return: english text with corrected punctuation spacing
    """
    text = re.sub(r"\b(\w+)\s*'\s*([a-zA-Z])", r"\1'\2", text)  # "I' m" -> "I'm"
    text = re.sub(r"\b(\w+)\s*'\s*s\b", r"\1's", text)  # "world' s" -> "world's"
    text = re.sub(r"\b(\w+)\s*'\s*d\b", r"\1'd", text)  # "I' d" -> "I'd"
    text = re.sub(r"\b(\w+)\s*'\s*ll\b", r"\1'll", text)  # "I' ll" -> "I'll"
    text = re.sub(r"\b(\w+)\s*'\s*re\b", r"\1're", text)  # "you' re" -> "you're"
    text = re.sub(r"\b(\w+)\s*'\s*ve\b", r"\1've", text)  # "I' ve" -> "I've"
    text = re.sub(r"\b(\w+)\s*'\s*em\b", r"\1'em", text)  # "let' em" -> "let'em"
    # Standardize spacing around single and double quotations
    text = re.sub(r"\s*'\s*([^']*?)\s*'\s*", r"'\1'", text)  # " ' example ' " -> "'example'"
    text = re.sub(r'\s*"\s*([^"]*?)\s*"\s*', r'"\1"', text)  # ' " example " ' -> '"example"'
    text = re.sub(r"\b'\s*(\d{2}s)\b", r"'\1", text)  # "' 80s" -> "'80s"
    # Handle general punctuation spacing rules
    # Remove spaces before punctuation marks
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Ensure space after punctuation if followed by a word
    text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)
    # Collapse multiple spaces into one and trim leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text_english(text):
    """
    Preprocesses english text by handling html entities, punctuation, spacing, and currency formatting
    :param text: english text to preprocess
    :return: preprocessed English text
    """
    # Replace HTML entities with corresponding characters
    text = re.sub(r'(&amp;\s?lt\s?;)+', '<', text, flags=re.IGNORECASE)
    text = re.sub(r'(&amp;\s?gt\s?;)+', '>', text, flags=re.IGNORECASE)
    text = re.sub(r'(&amp;\s?amp\s?;)+', '&', text, flags=re.IGNORECASE)
    text = re.sub(r'&lt;', '<', text, flags=re.IGNORECASE)
    text = re.sub(r'&gt;', '>', text, flags=re.IGNORECASE)
    text = re.sub(r'&quot;', '"', text, flags=re.IGNORECASE)
    text = re.sub(r'&apos;', "'", text, flags=re.IGNORECASE)
    text = re.sub(r'&amp;', '&', text, flags=re.IGNORECASE)
    # Replace special quotes with standard quotes
    text = re.sub(r'[“”]', '"', text)
    text = html.unescape(text)
    # Remove unwanted characters
    text = text.replace('[', '').replace(']', '')
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove commas within numbers
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    # Handle currency formatting
    text = re.sub(r'\$\s*([\d]+(?:[\d.,]*\d)?)', r'\1 dollars', text)
    text = re.sub(r'\b([\d]+(?:[\d.,]*\d)?)\s*dollars?\b', replace_dollar_plural, text, flags=re.IGNORECASE)
    # Fix punctuation spacing
    text = fix_punctuation_spacing_english(text)
    return text


def replace_dollar_plural(match):
    """
    Ensures correct pluralization of 'dollar' based on the numerical value.
    :param match: a regex match object containing the number as a string
    :return: number followed by 'dollar' or 'dollars' as appropriate
    """
    # Extract the numerical string from the match and remove commas for parsing
    number_str = match.group(1)
    number_value = float(number_str.replace(',', ''))
    # Return the correct pluralization based on the numerical value
    return f"{number_str} dollar" if number_value == 1 else f"{number_str} dollars"


def create_dataset_from_file(file):
    # Dictionary to store data
    data = {'en': []}
    # Open the file in read mode
    with open(file, 'r', encoding='utf-8') as f:
        # Iterate through each line in the file
        for line in f:
            # Remove any leading/trailing whitespace and add line
            data['en'].append(line.strip())
    # Convert the dictionary to a Dataset
    return Dataset.from_dict(data)


# Set up the device to use GPU if available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model and tokenizer from pretrained NLLB model
nllb_model_name = "facebook/nllb-200-3.3B"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to(device)
# Define the target language for translation (Vietnamese)
tgt_language = "vie_Latn"
# Get the token ID for the target language to set in the model for forced decoding
tgt_language_id = nllb_tokenizer.convert_tokens_to_ids(tgt_language)


def translate_batch_to_vietnamese(text_list):
    """
    Translates a batch of English text strings to Vietnamese
    :param text_list: List of text strings in English to translate
    :return: List of translated text strings in Vietnamese
    """
    # Encode the input texts using the tokenizer, with padding and truncation for uniform input sizes
    encoded_inputs = nllb_tokenizer(
        text_list, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    # Generate translated tokens, setting the target language to Vietnamese
    translated_tokens = nllb_model.generate(
        **encoded_inputs, forced_bos_token_id=tgt_language_id
    )

    # Decode the translated tokens back into text, skipping any special tokens
    translated_texts = nllb_tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True
    )

    return translated_texts


def preprocess_and_load_dataset(dataset, split, batch_size=32):
    """
    Preprocesses and translates a dataset split and creates a Vietnamese text file
    :param dataset: Dataset containing text entries to be processed and translated
    :param split: Dataset split to process ("train", "tst2012", or "tst2013")
    :param batch_size: Number of samples to process in each batch
    :return: None
    """
    # Define file paths based on the split
    en_file = f"{split}.en.txt"
    vi_file = f"{split}.vi.txt"  # Create a new Vietnamese text file

    with open(en_file, "w", encoding="utf-8") as f_en, open(vi_file, "w", encoding="utf-8") as f_vi:
        english_texts = []
        for example in tqdm(dataset[split], desc=f"Processing {split} dataset"):
            # Preprocess English text
            english_text = preprocess_text_english(example['en'])
            english_texts.append(english_text)

            # Process in batches
            if len(english_texts) == batch_size:
                vietnamese_texts = translate_batch_to_vietnamese(english_texts)
                for en_text, vi_text in zip(english_texts, vietnamese_texts):
                    f_en.write(en_text + "\n")
                    f_vi.write(vi_text + "\n")
                english_texts = []

        # Process any remaining texts
        if english_texts:
            vietnamese_texts = translate_batch_to_vietnamese(english_texts)
            for en_text, vi_text in zip(english_texts, vietnamese_texts):
                f_en.write(en_text + "\n")
                f_vi.write(vi_text + "\n")


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_file, tgt_file, max_seq_length):
        """
        Initializes the TranslationDataset with source and target sequences.
        :param src_file: Source language file
        :param tgt_file: Target language file.
        :param max_seq_length: Maximum sequence length
        """
        # Load and process source and target sequences with a maximum length constraint
        self.src_sequences = self.load_sequences(src_file, max_seq_length)
        self.tgt_sequences = self.load_sequences(tgt_file, max_seq_length)

    def load_sequences(self, file, max_seq_length):
        """
        Loads sequences from a file and truncates them to the max sequence length.
        :param file: File containing text data
        :param max_seq_length: Maximum sequence length
        :return:
        """
        sequences = []

        # Open the file and read line by line
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()  # Remove surrounding whitespace
                tokens = text.split()  # Split text into tokens
                # Truncate the token list if it exceeds max_seq_length
                if len(tokens) > max_seq_length:
                    tokens = tokens[:max_seq_length]
                    text = ' '.join(tokens)  # Rejoin tokens into a truncated sentence
                sequences.append(text)  # Append the processed text to the sequences list
        return sequences

    def __getitem__(self, index):
        """
        Retrieves the source and target sequence at a specific index.
        :param index: Index of the sequence pair
        :return: Tuple containing the source and target sequences
        """
        return self.src_sequences[index], self.tgt_sequences[index]

    def __len__(self):
        """
        Returns the number of sequence pairs in the dataset.
        :return: Length of the dataset.
        """
        return len(self.src_sequences)


# -------------------- Dataset Preprocessing --------------------
def tokenize_text(text, lang):
    """
    Tokenizes a given text based on the specified language.
    :param text: Input text to tokenize
    :param lang: Language of the text ('english' or 'vietnamese')
    :return: List of tokens extracted from the text
    """
    # Convert text to lowercase for consistent tokenization
    text = text.lower()

    # Tokenize based on the specified language
    if lang == 'english':
        tokens = nltk.word_tokenize(text)
    elif lang == 'vietnamese':
        tokens = ViTokenizer.tokenize(text).split()
    else:
        # Default tokenization by simple whitespace splitting
        tokens = text.split()

    return tokens


def tokenize_batch(texts, lang):
    """
    Tokenizes a batch of texts
    :param texts: List of texts to tokenize
    :param lang: Language of the texts
    :return: List of tokens for each text in the batch
    """
    for text in texts:
        # Yield the tokens of each text for efficient batch processing
        yield tokenize_text(text, lang)


def build_vocabulary(texts, lang):
    """
    Builds a vocabulary from a list of texts with special tokens included.
    :param texts: List of texts to build the vocabulary from
    :param lang: Language of the texts for tokenization
    :return: Vocabulary with tokens indexed, including special tokens
    """
    # Special tokens for padding, unknown, beginning of sentence, and end of sentence
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
    # Create a vocabulary iterator from tokenized text with special tokens added
    vocabulary = build_vocab_from_iterator(
        tokenize_batch(texts, lang),
        specials=special_tokens,
        special_first=True
    )
    # Set a default index for unknown tokens
    vocabulary.set_default_index(vocabulary['<unk>'])
    return vocabulary


def collate_batch(batch, src_vocab, tgt_vocab, src_lang, tgt_lang):
    """
    Processes and batches a list of source and target text pairs, converting them to tensor indices
    with padding for uniformity.
    :param batch: Batch of (source_text, target_text) pairs.
    :param src_vocab: Vocabulary for the source language.
    :param tgt_vocab: Vocabulary for the target language.
    :param src_lang: Identifier for the source language
    :param tgt_lang: Identifier for the target language
    :return: A tuple containing Padded tensor of source and target sequences with indices
    """
    # Initialize lists to store the tokenized and indexed source and target sequences
    src_batch, tgt_batch = [], []

    for src_text, tgt_text in batch:
        # Tokenize source and target texts, adding BOS and EOS tokens
        src_tokens = ['<bos>'] + tokenize_text(src_text, src_lang) + ['<eos>']
        tgt_tokens = ['<bos>'] + tokenize_text(tgt_text, tgt_lang) + ['<eos>']

        # Convert tokens to vocabulary indices for both source and target
        src_indices = torch.tensor(src_vocab(src_tokens), dtype=torch.long)
        tgt_indices = torch.tensor(tgt_vocab(tgt_tokens), dtype=torch.long)

        # Append the indexed sequences to the batch lists
        src_batch.append(src_indices)
        tgt_batch.append(tgt_indices)

    # Pad the source and target batches to create uniform sequence lengths
    src_batch = pad_sequence(src_batch, padding_value=src_vocab['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab['<pad>'], batch_first=True)

    return src_batch, tgt_batch


# -------------------- Model Evaluation --------------------
def validate(data_loader, encoder, decoder, target_vocab, device, print_translations=False):
    """
    Validates the encoder-decoder model on a validation dataset by generating translations and calculating BLEU scores.
    :param data_loader: DataLoader for the validation dataset
    :param encoder: Encoder model that encodes the input sequences
    :param decoder: Decoder model that generates translations from encoded sequences
    :param target_vocab: Target language vocabulary for token-to-id and id-to-token conversions.
    :param device: Device to perform validation on, such as CPU or GPU. Defaults to CPU.
    :param print_translations: If True, prints each translated sentence. Defaults to False.
    :return: Tuple containing average sentence-level BLEU score and corpus-level BLEU score for the validation set
    """
    # Set the encoder and decoder to evaluation mode
    encoder.eval()
    decoder.eval()
    # Lists to store reference and hypothesis sentences for BLEU score calculation
    references = []
    hypotheses = []
    # Variables for tracking total sentence BLEU score and number of sentences
    total_sentence_bleu_score = 0.0
    num_sentences = 0
    # List to store translated sentences if printing is enabled
    translated_sentences = [] if print_translations else None
    # Smoothing function for BLEU score calculation
    smoothing = SmoothingFunction().method1

    # Disable gradient calculations for validation
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation", leave=False)
        # Iterate over each batch in the validation data loader
        for src_text, tgt_text in progress_bar:
            # Move source and target tensors to the specified device (CPU/GPU)
            src_text = src_text.to(device)
            tgt_text = tgt_text.to(device)
            # Pass the source text through the encoder to get encoder outputs and hidden states
            encoder_output, encoder_hidden = encoder(src_text)
            # Initialize the decoder hidden state with the encoder's final hidden state
            decoder_hidden = encoder_hidden
            # Initialize the decoder input with the <bos> token for each sentence in the batch
            decoder_input = torch.full((src_text.size(0),), target_vocab['<bos>'], dtype=torch.long, device=device)
            # Get batch size and maximum sequence length from target text
            batch_size = src_text.size(0)
            max_sequence_length = tgt_text.size(1)
            # Store decoded token IDs for each sentence in the batch
            decoded_tokens = [[] for _ in range(batch_size)]
            # Loop over each time step in the sequence
            for t in range(max_sequence_length):
                # Pass the decoder input through the decoder to get output probabilities and new hidden state
                output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                # Remove the extra dimension from output
                output = output.squeeze(1)
                # Get the token IDs with the highest probability
                decoder_input = output.argmax(1)
                # Append the decoded token IDs to the list for each sentence
                for i in range(batch_size):
                    token_id = decoder_input[i].item()
                    decoded_tokens[i].append(token_id)
            # Process each sentence in the batch to calculate BLEU scores
            for i in range(batch_size):
                # Get the reference sequence by excluding the <bos> token
                reference_seq = tgt_text[i, 1:].tolist()
                # Get the hypothesis decoded tokens sequence
                hypothesis_seq = decoded_tokens[i]
                # Convert token IDs to tokens, ignoring padding tokens
                reference_tokens = [target_vocab.lookup_token(token) for token in reference_seq if
                                    token != target_vocab['<pad>']]
                hypothesis_tokens = [target_vocab.lookup_token(token) for token in hypothesis_seq if
                                     token != target_vocab['<pad>']]
                # Truncate sequences at the <eos> (end of sentence) token if present
                if '<eos>' in reference_tokens:
                    reference_tokens = reference_tokens[:reference_tokens.index('<eos>')]
                if '<eos>' in hypothesis_tokens:
                    hypothesis_tokens = hypothesis_tokens[:hypothesis_tokens.index('<eos>')]
                # Append the processed reference and hypothesis to lists
                references.append([reference_tokens])
                hypotheses.append(hypothesis_tokens)

                # Calculate sentence-level BLEU score
                sentence_bleu_score = sentence_bleu(
                    [reference_tokens],
                    hypothesis_tokens,
                    smoothing_function=smoothing
                )
                # Accumulate the total sentence BLEU score and increment the sentence count
                total_sentence_bleu_score += sentence_bleu_score
                num_sentences += 1

                # Store the translated sentence
                if print_translations:
                    # Join the hypothesis tokens into a single string and clean formatting
                    translated_sentence = ' '.join(hypothesis_tokens).replace('_', ' ')
                    translated_sentences.append(translated_sentence)

        # Calculate the average sentence BLEU score across all sentences
        average_sentence_bleu_score = total_sentence_bleu_score / num_sentences if num_sentences > 0 else 0.0
        # Calculate the corpus-level BLEU score
        corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing)

        # Print translated sentences
        if print_translations:
            print("\nTranslated Sentences:")
            for sentence in translated_sentences:
                print(sentence)

    # Return the average sentence BLEU score and the corpus BLEU score
    return average_sentence_bleu_score, corpus_bleu_score


# -------------------- Model Training --------------------
def train_model(encoder, decoder, train_data_loader, val_data_loader, optimizer, loss_function, target_vocab,
                lr_scheduler, num_epochs, device):
    """
    Trains the encoder and decoder models with mixed precision and gradient accumulation.
    :param encoder: The encoder model
    :param decoder: The decoder model
    :param train_data_loader: DataLoader for the training data
    :param val_data_loader: DataLoader for the validation data
    :param optimizer: Optimizer for updating model parameters
    :param loss_function: Loss function to compute the training loss
    :param target_vocab: Target language vocabulary for token-to-id and id-to-token conversions
    :param lr_scheduler: Learning rate scheduler
    :param num_epochs: Number of training epochs
    :param device: Device to perform validation on, such as CPU or GPU. Defaults to CPU
    :return: None
    """
    # Directory where the model will be saved
    model_directory = "model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename = "trained_model.pth"
    best_model_filepath = os.path.join(model_directory, model_filename)

    # Move encoder and decoder models to the specified device (CPU/GPU)
    encoder.to(device)
    decoder.to(device)

    # Keep track of the best BLEU score achieved
    best_bleu_score = 0

    # Initialize the GradScaler for mixed precision
    scaler = GradScaler()

    # Number of steps for gradient accumulation
    accumulation_steps = 4

    print(
        f"{'Epoch':>6} | {'Loss':>10} | {'Validation Corpus BLEU Score':>28} | {'Validation Avg Sentence BLEU Score':>34}")

    for epoch in range(num_epochs):
        # Set encoder and decoder to training mode
        encoder.train()
        decoder.train()
        epoch_loss = 0  # Accumulate loss over the epoch

        optimizer.zero_grad()  # Zero the gradients at the start of each epoch
        num_batches = len(train_data_loader)  # Total number of batches in training data

        # Loop through each batch
        progress_bar = tqdm(enumerate(train_data_loader, 1), total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}",
                            leave=False)
        for batch_idx, (src_text, tgt_text) in progress_bar:
            src_text = src_text.to(device)  # Move source text to device
            tgt_text = tgt_text.to(device)  # Move target text to device

            # Mixed precision training
            with autocast():
                # Forward pass through encoder
                encoder_outputs, encoder_hidden = encoder(src_text)
                decoder_hidden = encoder_hidden  # Initialize decoder hidden state with encoder's final hidden state

                # Set up target sequence lengths
                batch_size = src_text.size(0)
                max_target_length = tgt_text.size(1)

                # Prepare inputs and targets for decoder
                decoder_inputs = tgt_text[:, :-1]
                decoder_targets = tgt_text[:, 1:]
                decoder_targets_flat = decoder_targets.contiguous().view(-1)

                # Initialize the first input for decoder
                decoder_input = decoder_inputs[:, 0]
                decoder_outputs = torch.zeros(batch_size, max_target_length - 1, decoder.vocab_size, device=device)

                for t in range(max_target_length - 1):
                    output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    output = output.squeeze(1)  # Remove extra dimension
                    decoder_outputs[:, t, :] = output  # Store output

                    # Set up the next input to decoder
                    if t + 1 < decoder_inputs.size(1):
                        decoder_input = decoder_inputs[:, t + 1]
                    else:
                        decoder_input = decoder_inputs[:, -1]

                # Flatten the decoder outputs for loss computation
                decoder_outputs_flat = decoder_outputs.view(-1, decoder.vocab_size)
                # Calculate loss
                loss = loss_function(decoder_outputs_flat, decoder_targets_flat) / accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Accumulate loss over the epoch
            epoch_loss += loss.item() * accumulation_steps
            # Gradient accumulation and optimization step
            if batch_idx % accumulation_steps == 0 or batch_idx == num_batches:
                scaler.unscale_(optimizer)
                parameters = itertools.chain(encoder.parameters(), decoder.parameters())
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'batch': f"{batch_idx}/{num_batches}"
            })

        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / len(train_data_loader)
        lr_scheduler.step()  # Step the learning rate scheduler
        # Validation to calculate BLEU scores
        average_sentence_bleu_score, bleu_score = validate(val_data_loader, encoder, decoder, target_vocab, device)

        print(f"{epoch + 1:>6} | {avg_epoch_loss:>10.6f} | {bleu_score:>28.6f} | {average_sentence_bleu_score:>34.6f}")

        # Save model if BLEU score improves
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, best_model_filepath)

    print(f"Model saved in file: {best_model_filepath}.")


# -------------------- CLI --------------------

def main():
    epilog = """Usage examples:
      python3 NMT.py train       Train the model 
      python3 NMT.py test        Test the model
    """

    # Argument parser for CLI
    parser = argparse.ArgumentParser(
        usage='python3 NMT.py {train,test} ...',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
        epilog=epilog
    )

    # Subparsers for 'train' and 'test' commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute: train or test")
    subparsers.add_parser('train', help="Train the model")
    subparsers.add_parser('test', help="Test the model")

    # Parse the command-line arguments
    args = parser.parse_args()
    if len(sys.argv) == 1 or args.command is None:
        parser.print_help()
        sys.exit(1)

    # Set hyperparameters and configs
    MAX_SEQ_LENGTH = 100  # Maximum sequence length for sentences
    BATCH_SIZE = 64  # Batch size for training
    EMBEDDING_DIM = 512  # Embedding dimension for embeddings
    HIDDEN_SIZE = 1024  # Hidden size for encoder and decoder
    NUM_EPOCHS = 10  # Number of training epochs
    LEARNING_RATE = 0.001  # Learning rate for the optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device (GPU if available)

    # Define source and target languages
    src_language = 'english'
    tgt_language = 'vietnamese'

    # Files for training and testing data
    src_language_train_file = 'train.en.txt'
    src_language_tst2013_file = 'tst2013.en.txt'
    src_language_tst2012_file = 'tst2012.en.txt'
    tgt_language_train_file = 'train.vi.txt'
    # Paths to save or load vocabulary files
    src_vocab_path = 'model/src_vocab.pth'
    trg_vocab_path = 'model/trg_vocab.pth'

    if args.command == 'train':
        # Load datasets from files
        train_dataset = create_dataset_from_file(src_language_train_file)
        dataset = DatasetDict({'train': train_dataset})

        # Preprocess and load datasets
        preprocess_and_load_dataset(dataset, 'train')

        # Create a TranslationDataset for training data
        full_train_dataset = TranslationDataset(
            src_file=src_language_train_file,
            tgt_file=tgt_language_train_file,
            max_seq_length=MAX_SEQ_LENGTH
        )

        # Calculate sizes for training and validation splits
        train_split_size = int(0.9 * len(full_train_dataset))
        validation_split_size = len(full_train_dataset) - train_split_size

        # Split full dataset into training and validation sets
        train_dataset, validation_dataset = random_split(
            full_train_dataset,
            [train_split_size, validation_split_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Build vocabularies from training data sequences
        src_vocab = build_vocabulary(full_train_dataset.src_sequences, src_language)
        tgt_vocab = build_vocabulary(full_train_dataset.tgt_sequences, tgt_language)

        # Define collate function for batching data
        collate_fn = partial(
            collate_batch,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            src_lang=src_language,
            tgt_lang=tgt_language
        )

        # DataLoaders for training and validation datasets
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Initialize encoder and decoder models
        encoder = Encoder(
            vocab_size=len(src_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_SIZE,
            padding_idx=src_vocab['<pad>']
        ).to(device)

        decoder = Decoder(
            hidden_dim=HIDDEN_SIZE,
            vocab_size=len(tgt_vocab),
            embedding_dim=EMBEDDING_DIM,
            padding_idx=tgt_vocab['<pad>']
        ).to(device)

        # Set up optimizer with parameters from encoder and decoder
        optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=LEARNING_RATE)
        # Learning rate scheduler to adjust learning rate
        lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.2)
        # Loss function, ignoring padding index
        loss_function = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

        # Train the model
        train_model(
            encoder=encoder,
            decoder=decoder,
            train_data_loader=train_data_loader,
            val_data_loader=validation_data_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            target_vocab=tgt_vocab,
            lr_scheduler=lr_scheduler,
            num_epochs=NUM_EPOCHS,
            device=device
        )

        # Save vocabularies for future use in testing
        torch.save(src_vocab, src_vocab_path)
        torch.save(tgt_vocab, trg_vocab_path)

    elif args.command == 'test':
        tst2013_dataset = create_dataset_from_file(src_language_tst2013_file)
        tst2012_dataset = create_dataset_from_file(src_language_tst2012_file)
        dataset = DatasetDict({'tst2013': tst2013_dataset, 'tst2012': tst2012_dataset})
        preprocess_and_load_dataset(dataset, 'tst2013')
        preprocess_and_load_dataset(dataset, 'tst2012')

        # Load vocabularies if they exist
        if os.path.exists(src_vocab_path) and os.path.exists(trg_vocab_path):
            src_vocab = torch.load(src_vocab_path)
            tgt_vocab = torch.load(trg_vocab_path)
            print("Loaded vocabularies.")
        else:
            print("Vocabulary files not found. Please run model training first.")
            sys.exit(1)

        # Initialize encoder and decoder models
        encoder = Encoder(
            vocab_size=len(src_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_SIZE,
            padding_idx=src_vocab['<pad>']
        ).to(device)

        decoder = Decoder(
            hidden_dim=HIDDEN_SIZE,
            vocab_size=len(tgt_vocab),
            embedding_dim=EMBEDDING_DIM,
            padding_idx=tgt_vocab['<pad>']
        ).to(device)

        # Load trained model if available
        best_model_filepath = 'model/trained_model.pth'
        if os.path.exists(best_model_filepath):
            checkpoint = torch.load(best_model_filepath, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            print("Loaded trained model.")
        else:
            print(f"No trained model found at {best_model_filepath}. Please run model training first.")
            sys.exit(1)

        # Test datasets for evaluation (tst2013 and tst2012)
        test_datasets = {
            'tst2013': {
                'src_file': 'tst2013.en.txt',
                'tgt_file': 'tst2013.vi.txt',
            },
            'tst2012': {
                'src_file': 'tst2012.en.txt',
                'tgt_file': 'tst2012.vi.txt',
            }
        }

        # Loop through test datasets and evaluate
        for test_name, files in test_datasets.items():
            print(f"\nTesting on {test_name}")
            test_dataset = TranslationDataset(
                src_file=files['src_file'],
                tgt_file=files['tgt_file'],
                max_seq_length=MAX_SEQ_LENGTH
            )

            # Collate function for batching test data
            collate_fn = partial(
                collate_batch,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                src_lang=src_language,
                tgt_lang=tgt_language
            )

            # DataLoader for test dataset
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn
            )

            # Validate the model on the test dataset
            average_sentence_bleu, corpus_bleu_score = validate(
                test_loader,
                encoder,
                decoder,
                tgt_vocab,
                device=device,
                print_translations=True
            )

            # Print BLEU scores for the test dataset
            print(f'{test_name} - Average Sentence BLEU: {average_sentence_bleu * 100:.3f}')
            print(f'{test_name} - Corpus BLEU: {corpus_bleu_score * 100:.3f}')


if __name__ == '__main__':
    main()
