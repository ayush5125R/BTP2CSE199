from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import chromadb

# ── Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ── Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ── ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ── Create or load collection
collection = chroma_client.get_or_create_collection(
    name="iilm_knowledge",
    metadata={"hnsw:space": "cosine"}
)

app = Flask(__name__)
CORS(app)
app.secret_key = "iilm_secret_key"  # needed for session to work

# ── Load knowledge base from text file
with open("iilm_data.txt", "r") as f:
    raw_text = f.read()

chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]

# ── Load chunks into ChromaDB only if collection is empty
if collection.count() == 0:
    print("Loading chunks into ChromaDB...")
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Loaded {len(chunks)} chunks successfully!")
else:
    print(f"ChromaDB already has {collection.count()} chunks loaded!")

# ── Chat histories per user
chat_histories = {}

# ── Routes

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    # give each browser tab its own ID
    if "uid" not in session:
        session["uid"] = os.urandom(8).hex()

    uid = session["uid"]
    if uid not in chat_histories:
        chat_histories[uid] = []

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()

    if not user_message:
        return jsonify({"reply": "Please type a valid question."}), 400
    if len(user_message) > 500:
        return jsonify({"reply": "Please keep your question under 500 characters."}), 400

    try:
        # Step 1: Convert user question to vector embedding
        query_embedding = embedding_model.encode(user_message).tolist()

        # Step 2: Search ChromaDB for top 3 most relevant chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Step 3: Extract top chunks and build context
        top_chunks = results['documents'][0]
        context = "\n\n".join(top_chunks)

        # Step 4: Build system prompt with retrieved context
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful and friendly enquiry assistant for IILM University Greater Noida.
Answer the user's question using ONLY the information provided below.
If the answer is not in the information, politely say you don't know and suggest contacting admissions@iilm.edu or calling +91-8860427537.

INFORMATION:
{context}"""
            }
        ]

        # Step 5: Add last 6 turns of THIS user's conversation history
        for turn in chat_histories[uid][-6:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["bot"]})

        # Step 6: Add current user message and call LLaMA via Groq
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        reply = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error: {e}")
        reply = "I'm not sure about that. Please contact us at admissions@iilm.edu or call +91-8860427537."

    # ── Save turn to this user's history
    chat_histories[uid].append({"user": user_message, "bot": reply})
    return jsonify({"reply": reply, "source": "rag"})


@app.route("/reset", methods=["POST"])
def reset():
    if "uid" in session:
        chat_histories[session["uid"]] = []
    return jsonify({"status": "Chat history cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)