# 📘 RAG 기반 PDF 처리 및 질의응답 시스템

이 프로젝트는 Flask를 기반으로 한 웹 애플리케이션으로, 사용자가 PDF 파일을 업로드하면 이를 분석하고 AI 모델을 이용하여 문서 기반 질의응답을 수행할 수 있도록 설계되었습니다. PDF에서 텍스트를 추출한 후, 이를 벡터화하여 데이터베이스에 저장하며, 사용자의 질문과 가장 관련성이 높은 내용을 찾아 답변을 생성합니다.

## 📌 설치 환경

이 애플리케이션을 실행하기 위해서는 아래의 환경이 필요합니다.
- Python 3.x 이상 
- Flask
- PyTorch
- Transformers
- LangChain
  

## 🔧 설치 방법

### 1️⃣ 프로젝트 클론

먼저 GitHub에서 프로젝트를 클론합니다.
```sh
git clone https://github.com/ML-TANGO/llm-rag-test
cd llm-rag-test
```

### 2️⃣ 가상 환경 생성 및 활성화

의존성 충돌을 방지하기 위해 가상 환경을 생성하고 활성화합니다.
```sh
python3 -m venv venv
source venv/bin/activate  # Linux
```

### 3️⃣ 필수 패키지 설치

다음 명령어를 실행하여 프로젝트에 필요한 모든 라이브러리를 설치합니다.
```sh
  pip install flask torch transformers langchain_ollama langchain langchain_community chromadb langchain_chroma flask_caching scikit-learn pdfplumber
```
또는 
```sh
pip install -r requirements.txt
```

## 🚀 실행 방법

### 1️⃣ 서버 실행

애플리케이션을 실행하기 위해 아래 명령어를 입력합니다.
```sh
python app.py
```

### 2️⃣ 웹 브라우저에서 접속

서버가 실행되면 아래 주소로 접속하여 애플리케이션을 사용할 수 있습니다.
```
http://127.0.0.1:8000
```

## 📂 프로젝트 구조

이 프로젝트는 다음과 같은 구조로 이루어져 있습니다.
```
flask_app/
│-- app.py              # Flask 애플리케이션 메인 파일
│-- requirements.txt    # 필요한 라이브러리 목록
│-- templates/
│   ├── index.html      # 기본 웹 인터페이스
│-- static/
│-- pdf/                # PDF 파일 저장 폴더
│-- db/                 # 벡터 저장소
```
pdf와 db 디렉토리는 코드 실행 시 자동생성 됩니다.


### 📥 1. PDF 업로드

배쉬 쉘로 PDF 파일을 업로드하려면 다음과 같은 요청을 보냅니다.
```sh
curl -X POST http://127.0.0.1:8000/upload_pdf -F "file=@/input/your/file/path.pdf"
```

- **설명:** 사용자가 PDF 파일을 업로드하면 해당 파일이 저장되고, 텍스트가 추출되어 벡터화된 후 데이터베이스에 저장됩니다.
- **응답 예시:**
```json
{
  "status": "success",
  "filename": "example.pdf",
  "chunk_len": 20
}
```
- **실행 예시:**

![image](https://github.com/user-attachments/assets/6d21cce6-ed0c-4411-b61d-8c19eb5c1e1a)
![image](https://github.com/user-attachments/assets/ef2be0bf-0f3e-438c-95d7-4e6f276873df)


### 🔍 2. PDF 기반 질의응답

배쉬 쉘을 통해 저장된 PDF 내용을 기반으로 질문을 하면 답변을 생성합니다.
```sh
curl -X POST "http://127.0.0.1:8000/ask_pdf" \
     -H "Content-Type: application/json" \
     -d '{"query": "사용자가 입력한 질문"}'
```
- **설명:** 벡터 DB에 업로드 된 문서 조각들 중 입력 쿼리 벡터값과 가장 유사한 값을 찾아 LLM이 받아서 반환합니다. 
- **응답 예시:**
```json
{
  "answer": "이 문서는 개인정보보호법에 관한 내용을 다루고 있습니다.",
  "context_text": "유사도: 0.85 - 개인정보보호법은..."
}
```
- **실행 예시:**

![image](https://github.com/user-attachments/assets/b1bfc712-8733-45ac-ac6a-9545ba437e72)

  
