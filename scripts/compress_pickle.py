import pickle, gzip, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_compressed_pickle(data, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_compressed_pickle(filename):
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def get_file_size(filepath):
    return os.path.getsize(filepath)


# Example usage
pickle_file = "data/sys3_template_bank_mcz40.pkl"
compressed_pickle_file = "data/sys3_template_bank_mcz40.pkl.gz"

# Load pickle file
with open(pickle_file, "rb") as f:
    data = pickle.load(f)

# Save data to a compressed pickle file
save_compressed_pickle(data, compressed_pickle_file)

# Get file sizes
original_size = get_file_size(pickle_file)
compressed_size = get_file_size(compressed_pickle_file)

print(f"Original size: {original_size} bytes")
print(f"Compressed size: {compressed_size} bytes")
print(f"Compression ratio: {original_size / compressed_size:.2f}:1")
