import time
from collections import deque
import numpy as np
from sentence_transformers import SentenceTransformer, util
import heapq

QUEUE_SIZE = 10
TOP_SIZE = 3

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
sentence1 = "white"
sentence2 = "black"

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
    topMessages = []
    userInput = input("Enter message into chat (type 'stop' to exit): ")
    if userInput == "stop":
        break

    inputTuple = [0,userInput]
    input_embedding = get_sentence_embedding(userInput)

    remMessage = None
    if len(messageQueue) == QUEUE_SIZE:
        remMessage = messageQueue.popleft()[1]

    for i in messageQueue:
        stored_embedding = get_sentence_embedding(i[1])
        similarity = compute_cosine_similarity(stored_embedding, input_embedding)
        if remMessage != None:
            remSimilarity = compute_cosine_similarity(stored_embedding, get_sentence_embedding(remMessage))
        else:
            remSimilarity = 0
        i[0] -= similarity-remSimilarity
        inputTuple[0] -= similarity
    messageQueue.append(inputTuple)
    messageHeap = list(messageQueue)
    heapq.heapify(messageHeap)
    topMessages = list(map(lambda x: x[1], heapq.nsmallest(TOP_SIZE, messageHeap)))
    print(topMessages)
