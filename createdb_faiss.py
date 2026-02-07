import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import pickle

load_dotenv()

def create_db():
    """
    Tạo Vector Database từ recipes.xlsx với FAISS
    """
    
    print("🔄 Bắt đầu tạo FAISS Vector Database...")
    
    # === 1. ĐỌC DỮ LIỆU ===
    data_file = "recipes_cleaned.xlsx"
    
    if not os.path.exists(data_file):
        print(f"❌ Không tìm thấy file {data_file}")
        return
    
    print(f"📖 Đang đọc file {data_file}...")
    
    # Đọc Excel file
    df = pd.read_excel(data_file, engine='openpyxl')
    
    print(f"✅ Đã đọc {len(df)} recipes")
    
    # === 2. XỬ LÝ DỮ LIỆU ===
    docs = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            # Kiểm tra dữ liệu hợp lệ
            title = str(row.get('Title', 'Unknown Recipe')).strip()
            
            # ƯU TIÊN Cleaned_Ingredients, fallback về Ingredients
            ingredients_cleaned = str(row.get('Cleaned_Ingredients', '')).strip()
            ingredients_raw = str(row.get('Ingredients', '')).strip()
            
            # Logic ưu tiên
            if ingredients_cleaned and ingredients_cleaned != 'nan':
                ingredients = ingredients_cleaned
            elif ingredients_raw and ingredients_raw != 'nan':
                ingredients = ingredients_raw
            else:
                ingredients = ''
            
            instructions = str(row.get('Instructions', '')).strip()
            image_name = str(row.get('Image_Name', '')).strip()
            
            # Bỏ qua nếu thiếu thông tin quan trọng
            if not title or title == 'nan':
                skipped += 1
                continue
            
            # Tạo nội dung để embedding (TIẾNG ANH)
            content = f"""Recipe: {title}

Ingredients:
{ingredients if ingredients else 'Not specified'}

Instructions:
{instructions if instructions and instructions != 'nan' else 'Not provided'}"""
            
            formatted_image = None
            if image_name and image_name != 'nan':
                formatted_image = f"{image_name}.jpg"
            
            metadata = {
                "title": title,
                "image": formatted_image,
                "source": "recipes.xlsx",
                "row_index": idx
            }
            
            docs.append(Document(page_content=content, metadata=metadata))
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"   Đã xử lý {idx + 1}/{len(df)} recipes...")
                
        except Exception as e:
            print(f"⚠️ Lỗi tại dòng {idx}: {e}")
            skipped += 1
            continue
    
    print(f"✅ Đã tạo {len(docs)} documents (bỏ qua {skipped} dòng lỗi)")
    
    if len(docs) == 0:
        print("❌ Không có document nào được tạo. Kiểm tra lại file!")
        return
    
    # === 3. KHỞI TẠO EMBEDDING MODEL ===
    print("🔧 Đang tải embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
    print("   Model này hỗ trợ 50+ ngôn ngữ, tốt cho cả tiếng Việt và tiếng Anh")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("✅ Đã tải embedding model")
    
    # === 4. TẠO FAISS VECTOR DATABASE ===
    print("💾 Đang tạo FAISS index...")
    
    faiss_dir = "./faiss_index"
    
    # Xóa database cũ nếu có
    if os.path.exists(faiss_dir):
        import shutil
        print(f"🗑️ Xóa FAISS index cũ tại {faiss_dir}...")
        shutil.rmtree(faiss_dir)
    
    os.makedirs(faiss_dir, exist_ok=True)
    
    # Tạo FAISS index với batch processing
    print("   Đang embedding documents...")
    
    # FAISS xử lý tất cả documents cùng lúc
    vector_db = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    
    print("✅ Đã tạo FAISS index")
    
    # === 5. LƯU FAISS INDEX ===
    print(f"💾 Đang lưu FAISS index vào {faiss_dir}...")
    
    # Lưu FAISS index
    vector_db.save_local(faiss_dir)
    
    print(f"✅ Đã lưu FAISS index tại {faiss_dir}")
    print(f"📊 Tổng số documents: {len(docs)}")
    
    # === 6. KIỂM TRA DATABASE ===
    print("\n🧪 Kiểm tra FAISS index...")
    
    # Load lại để test
    test_db = FAISS.load_local(
        faiss_dir, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Test search
    test_queries = ["pizza", "chocolate cake", "pasta"]
    
    for query in test_queries:
        results = test_db.similarity_search(query, k=1)
        if results:
            print(f"   ✓ Query '{query}' → Found: {results[0].metadata['title']}")
        else:
            print(f"   ✗ Query '{query}' → No results")
    
    print("\n🎉 Hoàn tất! FAISS Database đã sẵn sàng sử dụng.")
    print(f"📁 Files created:")
    print(f"   - {faiss_dir}/index.faiss (vector index)")
    print(f"   - {faiss_dir}/index.pkl (metadata)")

if __name__ == "__main__":
    create_db()