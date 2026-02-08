import pandas as pd
import os

def clean_recipes():
    # 1. CẤU HÌNH
    input_file = "recipes.xlsx"
    output_file = "recipes_cleaned.xlsx"
    image_dir = "./public/images"
    
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file {input_file}")
        return

    # 2. QUÉT THƯ MỤC ẢNH THỰC TẾ
    print("📂 Đang quét thư mục ảnh để làm đối chiếu...")
    # Tạo set để tìm kiếm với tốc độ O(1)
    actual_images = set(os.listdir(image_dir))
    
    # 3. ĐỌC VÀ LỌC DỮ LIỆU
    print(f"📖 Đang đọc {input_file}...")
    df = pd.read_excel(input_file)
    initial_count = len(df)
    
    print("🧹 Bắt đầu thanh lọc dữ liệu...")

    def is_valid_row(row):
        img_name = str(row.get('Image_Name', '')).strip()
        
        # Loại bỏ nếu lỗi #NAME? hoặc trống
        if img_name.lower() in ['#name?', 'nan', '']:
            return False
            
        # Chuẩn hóa tên file để đối chiếu
        formatted_name = img_name if img_name.endswith('.jpg') else f"{img_name}.jpg"
        
        # Thử cả 2 trường hợp: có dấu gạch ngang và không có
        variant1 = formatted_name
        variant2 = f"-{formatted_name}" if not formatted_name.startswith('-') else formatted_name
        
        if variant1 in actual_images or variant2 in actual_images:
            return True
        
        return False

    # Lọc các dòng thỏa mãn điều kiện
    df_cleaned = df[df.apply(is_valid_row, axis=1)].copy()
    
    # 4. LƯU FILE MỚI
    df_cleaned.to_excel(output_file, index=False)
    
    # 5. TỔNG KẾT
    removed_count = initial_count - len(df_cleaned)
    print("\n" + "="*50)
    print("🎉 HOÀN TẤT THANH LỌC!")
    print("="*50)
    print(f"✅ Số lượng ban đầu:  {initial_count}")
    print(f"🗑️ Số dòng đã loại bỏ: {removed_count}")
    print(f"🚀 Số lượng còn lại:  {len(df_cleaned)}")
    print(f"💾 File sạch đã lưu tại: {output_file}")
    print("="*50)

if __name__ == "__main__":
    clean_recipes()