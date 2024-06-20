import time
from collections import deque
import numpy as np
from sentence_transformers import SentenceTransformer, util
import heapq

QUEUE_SIZE = 10

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_sentence_embedding(sentence):
    # Encode the sentence to get its embedding
    embedding = model.encode(sentence, convert_to_tensor=True)
    return embedding

def compute_cosine_similarity(embedding1, embedding2):
    # Compute cosine similarity between two embeddings
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()

# Example sentences
sentence1 = "Fried Chicken is good food."
sentence2 = "KFC hits the feels."

# Measure time to get embeddings
start_time = time.time()
embedding1 = get_sentence_embedding(sentence1)
embedding2 = get_sentence_embedding(sentence2)
embedding_time = time.time() - start_time

# Measure time to compute similarity
start_time = time.time()
similarity = compute_cosine_similarity(embedding1, embedding2)
similarity_time = time.time() - start_time

print(f"Embedding Generation Time: {embedding_time:.4f} seconds")
print(f"Cosine Similarity Computation Time: {similarity_time:.4f} seconds")
print(f"Cosine Similarity between the sentences: {similarity:.4f}")

messageQueue = deque(maxlen=QUEUE_SIZE)
userInput = ""

while userInput != "stop":
    userInput = input("Enter message into chat (type 'stop' to exit): ")
    if userInput == "stop":
        break

    inputTuple = [userInput, 0]
    input_embedding = get_sentence_embedding(userInput)

    if len(messageQueue) == QUEUE_SIZE:
        messageQueue.popleft()

    for i in messageQueue:
        stored_embedding = get_sentence_embedding(i[0])
        similarity = compute_cosine_similarity(stored_embedding, input_embedding)
        i[1] += similarity
        inputTuple[1] += similarity

    messageQueue.append(inputTuple)
    print(messageQueue)