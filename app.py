import os
import json
import torch
import logging
from flask import Flask, request, jsonify, Response, render_template
from transformers import AutoTokenizer, AutoModel
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from sklearn.preprocessing import normalize
from flask_caching import Cache
from sklearn.metrics.pairwise import cosine_similarity

# Flask 앱 생성
app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple' 
cache = Cache(app)

# 전역 설정
FOLDER_PATH = "db"  # 벡터 저장소 디렉토리
PDF_FOLDER = "pdf"  # PDF 파일 업로드 디렉토리
LLM_MODEL = "llama3"  # 사용할 LLM 모델

# KoE5 임베딩 클래스
class KoE5Embedding(Embeddings):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlpai-lab/KoE5")
        self.model = AutoModel.from_pretrained("nlpai-lab/KoE5")

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return normalize(embeddings, norm="l2").tolist()

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return normalize(embeddings, norm="l2").squeeze().tolist()

embedding_function = KoE5Embedding()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100, separators=["\n\n", "\n"," "])
cached_llm = OllamaLLM(model=LLM_MODEL)

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        file_name = file.filename
        save_path = os.path.join(PDF_FOLDER, file_name)
        os.makedirs(PDF_FOLDER, exist_ok=True)
        file.save(save_path)

        loader = PDFPlumberLoader(save_path)
        docs = loader.load_and_split()

        if not docs:
            return jsonify({"error": "No valid text found in PDF"}), 400

        chunks = text_splitter.split_documents(docs)
        if not chunks:
            return jsonify({"error": "Failed to split text"}), 400

        vector_store = Chroma.from_documents(chunks, embedding_function, persist_directory=FOLDER_PATH)
        
        response = {"status": "success", "filename": file_name, "chunk_len": len(chunks)}
        return Response(json.dumps(response, ensure_ascii=False), mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 로그 설정
logging.basicConfig(level=logging.DEBUG)

@app.route("/ask_pdf", methods=["POST"])
def handle_pdf_query():
    json_content = request.json
    query = json_content.get("query")
    
    if not query:
        return jsonify({"error": "쿼리를 입력하세요."}), 400

    vector_store = Chroma(persist_directory=FOLDER_PATH, embedding_function=embedding_function)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold":0.3})

    retrieved_docs = retriever.get_relevant_documents(query)
    if not retrieved_docs:
        return jsonify({"error": "관련 문서를 찾을 수 없습니다."}), 400

    def calculate_cosine_similarity(query, document_content, embedding_model):
        query_embedding = embedding_model.embed_documents([query])[0]
        document_embedding = embedding_model.embed_documents([document_content])[0]
        similarity = cosine_similarity([query_embedding], [document_embedding])
        return similarity[0][0]

    context_text = ""
    for doc in retrieved_docs:
        score = calculate_cosine_similarity(query, doc.page_content, embedding_function)
        context_text += f"유사도: {score:.4f} - {doc.page_content[:1024]}<br><br>"

    prompt_template = f"""
    사용자가 요청한 사항: {query}
    ### 참고 문서:
    {context_text}
    출처: 해당 문서들에서 제공된 정보.
    ---
    
    아래 문서를 참고하여 사용자의 질문에 대한 답변을 작성하세요. 답변은 무조건 한국어로만 해야합니다.  
    반드시 **문서의 내용을 기반으로 논리적인 답변**을 제공해야 하며, **단순한 인용을 피하고 상세한 설명**을 포함하세요.  
    질문과 관련된 정보가 있다면, 해당 내용을 요약하여 사용자가 쉽게 이해할 수 있도록 작성하세요.  
    만약 문서에서 관련 정보를 찾을 수 없다면 "문서에서 해당 내용을 찾을 수 없습니다."라고 답변하세요. 
    """

    result = cached_llm.invoke(prompt_template)

    response_data = {
        "answer": result,
        "context_text": context_text
    }

    return app.response_class(
        response=json.dumps(response_data, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs(FOLDER_PATH, exist_ok=True)
    app.run(host="0.0.0.0", port=8000, debug=True)
