# ChefMate - Trợ Lý Nấu Ăn Thông Minh 👨‍🍳

Chatbot AI hỗ trợ tìm kiếm công thức ăn, tư vấn nấu ăn bằng **Retrieval-Augmented Generation (RAG)**, với hỗ trợ **tiếng Việt và tiếng Anh**.

## 🎯 Giới Thiệu

**ChefMate** là một chatbot thông minh được xây dựng để giúp bạn:
- 🔍 Tìm kiếm công thức ăn từ kho dữ liệu 10,000+ công thức
- 🔪 Nhận các chỉ dẫn nấu ăn chi tiết từ các chuyên gia được tổng hợp từ nguồn uy tín 
- 🍎 Tìm món ăn dựa trên nguyên liệu có sẵn
- 🌐 Hỏi đáp bằng tiếng Việt hoặc tiếng Anh tự động

Chatbot sử dụng công nghệ **RAG** để kết hợp kho kiến thức từ vector database với khả năng sinh văn bản của LLM hiện đại, đảm bảo câu trả lời chính xác, liên quan và hữu ích.

## 🚀 Demo

**Thử ngay:** https://huggingface.co/spaces/Lippovn04/ChefMate-Chatbot

## 🛠️ Công Nghệ Sử Dụng

| Thành Phần | Mục Đích |
|-----------|---------|
| **Chainlit** | Framework chatbot web tương tác |
| **LangChain** | Orchestration framework cho LLM applications |
| **FAISS** | Vector database cho tìm kiếm semantic nhanh |
| **HuggingFace Embeddings** | Mô hình embedding đa ngôn ngữ (50+ ngôn ngữ) |
| **Groq LLM** | Mô hình Llama 3.3 70B để sinh câu trả lời |
| **Pandas** | Xử lý và làm sạch dữ liệu |
| **Python-dotenv** | Quản lý biến môi trường an toàn |

**Stack:** Python 3.8+ | LLM (Groq) | Vector DB (FAISS) | Web UI (Chainlit)

## ✨ Các Tính Năng Chính

### 1. 🔍 Tìm Kiếm Công Thức (Recipe Search)
```
Bạn: "pizza recipe"
ChefMate: Tìm kiếm trong vector database, xếp hạng lại công thức phù hợp nhất, hiển thị ảnh và hướng dẫn chi tiết
```

### 2. 🍎 Tìm Kiếm Theo Nguyên Liệu (Ingredient-Based Search)
```
Bạn: "tôi có gà và tỏi, nấu gì?"
ChefMate: Trích xuất ingredient, tìm công thức chứa cả hai, loại trừ ingredient không muốn
```

### 3. 🔪 Tư Vấn Nấu Ăn (Cooking Advice)
```
Bạn: "cách luộc trứng sao cho vàng ươm?"
ChefMate: Cung cấp hướng dẫn chi tiết từ kiến thức LLM + context từ vector database
```

### 4. 💬 Đa Ngôn Ngữ (Multilingual Support)
- Tự động detect tiếng Việt hoặc tiếng Anh
- Xử lý follow-up questions trong cùng ngôn ngữ
- Chức năng dịch các câu trả lời

### 5. 🖼️ Hiển Thị Ảnh (Image Display)
- Tự động tải ảnh công thức tương ứng

### 6. 💾 Lưu Lịch Sử (Conversation History)
- Ghi nhớ các công thức trong phiên chat
- Xử lý follow-up hỏi tiếp về cùng công thức

### 7. 📊 Xếp Hạng Lại Công Thức (Reranking)
- Sử dụng LLM để xếp hạng lại kết quả FAISS
- Đảm bảo công thức phù hợp nhất được hiển thị trước

## 📊 Dataset

Dự án sử dụng bộ dữ liệu **Food Ingredients and Recipe Dataset with Images**:

- **Số lượng**: 10,000+ công thức
- **Ngôn ngữ**: Tiếng Anh
- **Bao gồm**: Tên công thức, nguyên liệu, hướng dẫn, ảnh
- **Nguồn**: [Kaggle - Food Ingredients and Recipe Dataset](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images/)

### Cấu Trúc Dữ Liệu
```
recipes.xlsx / recipes_cleaned.xlsx
├── Title: Tên công thức
├── Ingredients: Danh sách nguyên liệu thô
├── Cleaned_Ingredients: Nguyên liệu đã làm sạch (ưu tiên sử dụng)
├── Instructions: Hướng dẫn nấu ăn
└── Image_Name: Tên file ảnh công thức
```

## 📋 Hướng Dẫn Cài Đặt

### Điều Kiện Tiên Quyết
- Python 3.8+
- pip hoặc conda
- API Key từ Groq (https://console.groq.com)
- Optional: CPU mạnh hoặc GPU (để embedding nhanh)

### 1️⃣ Clone Repository & Setup

```bash
# Clone dự án
git clone <repository-url>
cd RAG\ chatbot

# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2️⃣ Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Các thư viện chính:**
```
chainlit==2.9.6          # Web UI chatbot
langchain==1.2.8         # LLM orchestration
langchain-groq==1.1.2    # Groq LLM integration
faiss-cpu==1.13.2        # Vector database (dùng faiss-gpu nếu có NVIDIA GPU)
sentence-transformers==5.2.2  # Embedding model
python-dotenv==1.0.1    # Environment variables
pandas==2.2.3            # Data processing
openpyxl==3.1.5         # Excel file handling
```

### 3️⃣ Configurate Environment

Tạo file `.env` trong thư mục gốc:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

**Cách lấy GROQ_API_KEY:**
1. Truy cập https://console.groq.com
2. Đăng ký hoặc đăng nhập
3. Tạo API key mới
4. Copy key vào file `.env`

### 4️⃣ Chuẩn Bị Dữ Liệu

#### A. Tải Dataset & Cấu Trúc Thư Mục

```bash
# Tải dataset từ Kaggle
# https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images/

# Tạo cấu trúc thư mục (nếu chưa có):
mkdir -p "Food Ingredients and Recipes"
mkdir -p public/images

# Đặt file recipes.xlsx vào thư mục gốc
# Đặt ảnh vào public/images/ hoặc giải nén images.zip
```

#### B. Làm Sạch Dữ Liệu (Optional nhưng Khuyến Nghị)

```bash
python clean_data.py
```

**Tác dụng:**
- Loại bỏ các dòng không có ảnh
- Xóa dữ liệu lỗi (#NAME?, NaN)
- Chuẩn hóa tên file ảnh
- Tạo file `recipes_cleaned.xlsx`

### 5️⃣ Tạo Vector Database (FAISS)

```bash
python createdb_faiss.py
```

**Quá trình này sẽ:**
1. Đọc file `recipes_cleaned.xlsx`
2. Tạo embeddings cho mỗi công thức (sử dụng paraphrase-multilingual-MiniLM-L12-v2)
3. Lưu vào FAISS index tại thư mục `./faiss_index/`
4. Mất vài phút tùy thuộc vào máy (có thể 10-30 phút với 10,000 recipes)

**Output:**
```
faiss_index/
├── index.faiss      # Vector index file
└── index.pkl        # Metadata
```

### 6️⃣ Chạy Chatbot

```bash
chainlit run app.py -w
```

**Options:**
- `-w`: Watch mode (reload khi có thay đổi file)
- `--port 7860`: Chỉ định port (mặc định: 8000)
- `--host 0.0.0.0`: Cho phép truy cập từ ngoài localhost

### 7️⃣ Truy Cập Chatbot

Mở trình duyệt và truy cập:
```
http://localhost:8000
```

## 📁 Cấu Trúc Dự Án

```
RAG chatbot/
├── app.py                   # Main chatbot application (Chainlit)
├── createdb_faiss.py        # Script tạo FAISS vector database
├── clean_data.py            # Script làm sạch dữ liệu
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (tạo thủ công)
├── Dockerfile               # Docker setup (optional)
│
├── faiss_index/             # Vector database (tạo sau khi chạy createdb_faiss.py)
│   ├── index.faiss
│   └── index.pkl
│
├── public/
│   └── images/              # Thư mục ảnh công thức
│
└── README.md               # File này
```

## 🔧 Nguyên Lý Hoạt Động (RAG)

### English

```
┌─────────────────────────────────────────────────────────┐
│                       User Query                        │
│                     "pizza recipe"                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │      Language Detection      │
        │    (Vietnamese / English)    │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │   Query Analysis & Routing   │
        │   - Intent Classification    │
        │   - Recipe vs. Cooking Tips  │
        │   - Ingredient Extraction    │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │        Vector Search (FAISS)         │
        │   - Multilingual Query Embedding     │
        │   - Retrieve Top-K Candidates        │
        │   - Semantic Similarity Scoring      │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │        Reranking (LLM-based)         │
        │   - Relevance Evaluation             │
        │   - Re-order by Context Match        │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │          Context Assembly            │
        │   - Recipe Data Extraction           │
        │   - System Prompt Integration        │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │        LLM Generation (Groq)         │
        │   - Llama 3.3 70B Orchestration      │
        │   - Detailed Instruction Synthesis   │
        │   - Dynamic Cooking Advice           │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │      Post-Processing & UI            │
        │   - Image Metadata Retrieval         │
        │   - Markdown Message Formatting      │
        │   - Chat History Synchronization     │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │           Final Response             │
        │   - Rich Content + Recipe Image      │
        │   - Interactive Follow-ups           │
        └──────────────────────────────────────┘
```

### Tiếng Việt

```
┌─────────────────────────────────────────────────────────┐
│                    Truy vấn người dùng                  │
│                   "cách làm bánh pizza"                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │      Nhận diện ngôn ngữ      │
        │    (Tiếng Việt / Tiếng Anh)  │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────────┐
        │     Phân tích & Điều hướng       │
        │   - Phân loại ý định người dùng  │
        │   - Trích xuất nguyên liệu       │
        └──────────────┬───────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │       Tìm kiếm Vector (FAISS)        │
        │   - Nhúng truy vấn đa ngôn ngữ       │
        │   - Truy xuất Top-K món ăn phù hợp   │
        │   - Tính điểm tương đồng ngữ nghĩa   │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │      Sắp xếp lại (Dùng LLM)          │
        │   - Đánh giá lại độ liên quan        │
        │   - Tối ưu thứ tự kết quả            │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │       Xây dựng ngữ cảnh (Prompt)     │
        │   - Trích xuất dữ liệu công thức     │
        │   - Kết hợp cấu trúc chỉ dẫn         │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │        Sinh văn bản (Groq)           │
        │   - Mô hình Llama 3.3 70B            │
        │   - Tổng hợp chỉ dẫn chi tiết        │
        │   - Thêm mẹo vặt & lưu ý nấu ăn      │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │      Hậu xử lý & Hiển thị            │
        │   - Lấy hình ảnh minh họa            │
        │   - Định dạng hiển thị Markdown      │
        │   - Lưu lịch sử trò chuyện           │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │           Phản hồi cuối cùng         │
        │   - Công thức đầy đủ + Hình ảnh      │
        │   - Sẵn sàng cho câu hỏi tiếp theo   │
        └──────────────────────────────────────┘
```

## 📚 Thư Viện & Module Chính

### `app.py`
- **ThreadPoolExecutor**: Xử lý blocking operations (FAISS search, LLM calls)
- **@cl.on_chat_start**: Khởi tạo session, component checks
- **@cl.on_message**: Main message handler với RAG logic
- **Chainlit Messages/Images**: Display kết quả với ảnh

### `createdb_faiss.py`
- **HuggingFaceEmbeddings**: Tạo embeddings đa ngôn ngữ
- **FAISS.from_documents()**: Build vector index
- **langchain Document**: Format dữ liệu

### `clean_data.py`
- **pandas DataFrame**: Xử lý Excel file
- **Filtering logic**: Loại bỏ records không hợp lệ
- **Image validation**: Đối chiếu với ảnh thực tế

## ⚙️ Advanced Configuration

### Thay Đổi Model LLM
Trong `app.py` dòng 96:
```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Hoặc "llama-3.1-70b-versatile", "gemma-2-9b-it"
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)
```

### Thay Đổi Embedding Model
Trong `createdb_faiss.py`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # Hoặc: "sentence-transformers/multilingual-e5-large"
)
```

### Cách Chọn Số K (top-k results)
```python
results = vector_db.similarity_search_with_score(query, k=10)
```
Tăng `k` để tìm kiếm rộng hơn, giảm để focus hơn.

## 🐛 Troubleshooting

| Vấn Đề | Giải Pháp |
|--------|---------|
| **ImportError: No module named 'faiss'** | `pip install faiss-cpu` (hoặc `faiss-gpu` nếu có NVIDIA) |
| **FAISS index not found** | Chạy `python createdb_faiss.py` để tạo index |
| **Groq API Key error** | Kiểm tra file `.env`, validate API key tại https://console.groq.com |
| **Embedding model slow** | Dùng GPU: `pip install faiss-gpu` và cấu hình CUDA |
| **Memory error với 10k recipes** | Dùng CPU, FAISS đã tối ưu cho CPU. Hoặc giảm dataset |

## 📖 Tài Liệu Tham Khảo

- **Chainlit Docs**: https://docs.chainlit.io/
- **LangChain Docs**: https://python.langchain.com/
- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **Groq Console**: https://console.groq.com/
- **HuggingFace Models**: https://huggingface.co/models

## 📝 License

Dự án sử dụng dataset công khai từ Kaggle. Kiểm tra license của dataset trước khi sử dụng thương mại.

---

**Hạnh phúc nấu ăn! 🍽️** 

*Made with ❤️ using Chainlit + LangChain + FAISS + Groq*
