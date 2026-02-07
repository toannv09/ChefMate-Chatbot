import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import json
import traceback
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import zipfile
from pathlib import Path

load_dotenv()

# =====================================================
# AUTO-EXTRACT IMAGES FROM ZIP
# =====================================================
def setup_images():
    """Tự động giải nén images.zip nếu thư mục public/images chưa tồn tại"""
    images_dir = Path("./public/images")
    zip_file = Path("./images.zip")
    
    # Kiểm tra xem thư mục images đã tồn tại chưa
    if not images_dir.exists():
        print("📁 Thư mục public/images chưa tồn tại...")
        
        # Kiểm tra xem có file images.zip không
        if zip_file.exists():
            print(f"📦 Tìm thấy {zip_file}, đang giải nén...")
            
            try:
                # Tạo thư mục public nếu chưa có
                images_dir.parent.mkdir(parents=True, exist_ok=True)
                
                # Giải nén
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall("./public/images")
                
                print(f"✅ Đã giải nén {len(os.listdir(images_dir))} files vào {images_dir}")
                
            except Exception as e:
                print(f"❌ Lỗi khi giải nén images.zip: {e}")
                traceback.print_exc()
        else:
            print(f"⚠️ Không tìm thấy {zip_file}")
            print("   Tạo thư mục trống public/images...")
            images_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"✅ Thư mục {images_dir} đã tồn tại với {len(os.listdir(images_dir))} files")

# Chạy setup images trước khi khởi tạo components
setup_images()

# =====================================================
# THREAD POOL FOR BLOCKING OPERATIONS
# =====================================================
executor = ThreadPoolExecutor(max_workers=4)

def run_in_executor(func, *args, **kwargs):
    """Chạy blocking function trong thread pool"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, partial(func, *args, **kwargs))

# =====================================================
# INITIALIZATION WITH ERROR HANDLING
# =====================================================

def initialize_components():
    """Khởi tạo các components với error handling"""
    try:
        print("🔧 Initializing multilingual embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("🔧 Loading FAISS index...")
        faiss_dir = "./faiss_index"
        
        if not os.path.exists(faiss_dir):
            print(f"❌ FAISS index not found at {faiss_dir}")
            print("   Please run create_db.py first!")
            return None, None, None
        
        vector_db = FAISS.load_local(
            faiss_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        print("🔧 Initializing LLM...")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=60,
            max_retries=2
        )
        
        print("✅ All components initialized successfully")
        return embeddings, vector_db, llm
    
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        traceback.print_exc()
        return None, None, None

# Khởi tạo global
embeddings, vector_db, llm = initialize_components()

# =====================================================
# ASYNC-WRAPPED UTILITY FUNCTIONS
# =====================================================

def _llm_call_sync(prompt, default_response="Xin lỗi, tôi gặp lỗi khi xử lý."):
    """Sync wrapper cho LLM calls"""
    try:
        if not llm:
            return default_response
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        return default_response

async def safe_llm_call(prompt, default_response="Xin lỗi, tôi gặp lỗi khi xử lý."):
    """Async wrapper an toàn cho LLM calls"""
    return await run_in_executor(_llm_call_sync, prompt, default_response)

def safe_json_parse(text, default=None):
    """Parse JSON an toàn"""
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error: {e}")
        print(f"Raw text: {text[:200]}")
        return default if default else {}

def _vector_search_sync(query, k=3):
    """Sync wrapper cho FAISS search"""
    try:
        if not vector_db:
            print("❌ Vector DB not initialized")
            return []
        
        results = vector_db.similarity_search_with_score(query, k=k)
        
        # Convert FAISS scores (L2 distance) to similarity scores (0-1)
        converted_results = []
        for doc, distance in results:
            similarity = max(0, 1 - (distance / 2.0))
            converted_results.append((doc, similarity))
        
        return converted_results
    
    except Exception as e:
        print(f"⚠️ FAISS search failed: {e}")
        traceback.print_exc()
        return []

async def safe_vector_search(query, k=3):
    """Async wrapper cho FAISS search"""
    return await run_in_executor(_vector_search_sync, query, k)

def get_image_path(image_name):
    """Tìm đường dẫn hình ảnh"""
    if not image_name or image_name == 'nan':
        return None
    
    possible_paths = [
        f"./public/images/{image_name}",
        f"./images/{image_name}",
        f"./{image_name}"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

# =====================================================
# CHAINLIT HANDLERS
# =====================================================

@cl.on_chat_start
async def start():
    """Khởi động chatbot với error checking"""
    try:
        if not llm or not vector_db:
            await cl.Message(
                content="❌ **Lỗi khởi động chatbot**\n\n"
                       "Các thành phần chưa sẵn sàng.\n"
                       "Vui lòng chạy `python create_db.py` trước!"
            ).send()
            return
        
        # Khởi tạo session
        cl.user_session.set("history", [])
        cl.user_session.set("conversation_count", 0)
        cl.user_session.set("waiting_for_selection", False)
        cl.user_session.set("pending_recipes", [])
        cl.user_session.set("last_recipe_context", "")
        cl.user_session.set("last_assistant_message", "")
        cl.user_session.set("waiting_for_recipe_confirm", False)
        cl.user_session.set("suggested_recipe", None)
        
        await cl.Message(
            content="👨‍🍳 **Xin chào! Mình là ChefMate - Trợ lý nấu ăn thông minh.**\n\n"
                   "Bạn có thể:\n"
                   "Tìm công thức món ăn (vd: \"pizza recipe\")\n"
                   "Hỏi cách nấu ăn (vd: \"cách luộc trứng\")\n"
                   "Tìm món từ nguyên liệu (vd: \"tôi có gà và tỏi, nấu gì?\")\n\n"
                   "Hỏi bằng tiếng Việt hoặc tiếng Anh đều được nhé! 😊"
        ).send()
    
    except Exception as e:
        print(f"❌ Error in start: {e}")
        traceback.print_exc()
        await cl.Message(content="❌ Lỗi khởi động. Vui lòng refresh trang.").send()

@cl.on_message
async def main(message: cl.Message):
    """Main handler với FAISS search"""
    
    user_query = message.content.strip()
    
    # =====================================================
    # INPUT VALIDATION
    # =====================================================
    
    if not user_query:
        await cl.Message(content="🤔 Bạn chưa nhập gì. Hỏi mình đi!").send()
        return
    
    if len(user_query) > 500:
        await cl.Message(content="😅 Câu hỏi hơi dài. Bạn rút gọn lại được không?").send()
        return
    
    try:
        # Component check
        if not llm or not vector_db:
            await cl.Message(
                content="❌ Chatbot đang gặp sự cố. Vui lòng thử lại sau."
            ).send()
            return
        
        # =====================================================
        # HANDLE RECIPE CONFIRMATION
        # =====================================================
        
        affirmative_keywords = ["ok", "yes", "có", "được", "ừ", "cho xin", "show", "xem", "đồng ý", "agree"]
        is_affirmative = any(kw in user_query.lower() for kw in affirmative_keywords)
        
        if cl.user_session.get("waiting_for_recipe_confirm") and is_affirmative:
            cl.user_session.set("waiting_for_recipe_confirm", False)
            suggested_recipe = cl.user_session.get("suggested_recipe")
            
            if suggested_recipe:
                doc, score = suggested_recipe
                detected_lang = cl.user_session.get("last_detected_lang", "vi")
                
                # Generate detailed recipe
                context = f"Recipe: {doc.metadata.get('title', 'N/A')}\n{doc.page_content}"
                
                recipe_prompt = f"""You are a professional chef. Provide the COMPLETE recipe with detailed instructions.

Recipe:
{context}

Provide:
1. Full ingredient list with measurements
2. Step-by-step cooking instructions (numbered)
3. Cooking time and serving size
4. Any tips or variations

Answer in {"VIETNAMESE" if detected_lang == "vi" else "ENGLISH"}:"""
                
                msg = cl.Message(content="🤔 Đang chuẩn bị công thức chi tiết...")
                await msg.send()
                
                answer = await safe_llm_call(recipe_prompt)
                
                cl.user_session.set("last_recipe_context", context)
                cl.user_session.set("last_assistant_message", answer)
                
                # Update history
                history = cl.user_session.get("history", [])
                history.append({"role": "user", "content": "Show me the recipe"})
                history.append({"role": "assistant", "content": answer[:200]})
                cl.user_session.set("history", history)
                
                image_path = get_image_path(doc.metadata.get("image"))
                
                await msg.remove()
                
                if image_path:
                    elements = [cl.Image(path=image_path, name="recipe_image", display="inline")]
                    await cl.Message(
                        content=f"👨‍🍳 **{doc.metadata.get('title', 'Recipe')}**\n\n{answer}",
                        elements=elements
                    ).send()
                else:
                    await cl.Message(content=f"👨‍🍳 **{doc.metadata.get('title', 'Recipe')}**\n\n{answer}").send()
                
                return
        
        # Show thinking indicator
        msg = cl.Message(content="🤔 Đang suy nghĩ...")
        await msg.send()
        
        # =====================================================
        # TRANSLATION HANDLING
        # =====================================================
        
        translation_keywords_vi = ["dịch", "dich", "translate", "chuyển sang", "chuyen sang"]
        translation_keywords_en = ["translate", "in english", "in vietnamese", "dịch sang"]
        
        is_translation_request = any(kw in user_query.lower() for kw in translation_keywords_vi + translation_keywords_en)
        
        if is_translation_request:
            last_message = cl.user_session.get("last_assistant_message", "")
            
            if last_message:
                target_lang = "en" if any(kw in user_query.lower() for kw in ["english", "tiếng anh", "tieng anh"]) else "vi"
                
                translation_prompt = f"""Translate this message to {"ENGLISH" if target_lang == "en" else "VIETNAMESE"}.

Message to translate:
{last_message}

Rules:
- Keep the same formatting and structure
- Translate naturally, not word-by-word
- Preserve recipe names if they're proper nouns
- Keep measurements and numbers the same

Translation:"""
                
                translated = await safe_llm_call(translation_prompt, default_response=last_message)
                
                cl.user_session.set("last_assistant_message", translated)
                
                await msg.remove()
                await cl.Message(content=f"🌐 {translated}").send()
                return
            else:
                await msg.remove()
                if "english" in user_query.lower() or "tiếng anh" in user_query.lower():
                    await cl.Message(content="😅 I don't have anything to translate yet. Ask me about a recipe first!").send()
                else:
                    await cl.Message(content="😅 Chưa có gì để dịch. Hỏi mình về công thức trước nhé!").send()
                return
        
        # =====================================================
        # MULTI-RECIPE SELECTION HANDLING
        # =====================================================
        
        if cl.user_session.get("waiting_for_selection", False):
            pending_recipes = cl.user_session.get("pending_recipes", [])
            
            if user_query.isdigit():
                choice = int(user_query) - 1
                
                if 0 <= choice < len(pending_recipes):
                    selected_doc, selected_score = pending_recipes[choice]
                    
                    cl.user_session.set("waiting_for_selection", False)
                    cl.user_session.set("pending_recipes", [])
                    
                    history = cl.user_session.get("history", [])
                    detected_lang = "vi"
                    if history and len(history) > 0:
                        last_msg = history[-1].get('content', '')
                        if 'found' in last_msg.lower() or 'recipe' in last_msg.lower():
                            detected_lang = "en"
                    
                    context = f"Recipe: {selected_doc.metadata.get('title', 'N/A')}\n{selected_doc.page_content}"
                    
                    final_prompt = f"""You are a professional chef assistant.

Recipe:
{context}

Provide detailed cooking instructions and tips.

Answer in {"VIETNAMESE" if detected_lang == "vi" else "ENGLISH"}:"""
                    
                    answer = await safe_llm_call(final_prompt, "Đây là công thức bạn chọn!")
                    
                    cl.user_session.set("last_recipe_context", context)
                    cl.user_session.set("last_assistant_message", answer)
                    
                    history = cl.user_session.get("history", [])
                    history.append({"role": "user", "content": f"Selected recipe {choice + 1}"})
                    history.append({"role": "assistant", "content": answer[:200]})
                    cl.user_session.set("history", history)
                    
                    image_path = get_image_path(selected_doc.metadata.get("image"))
                    
                    await msg.remove()
                    
                    if image_path:
                        elements = [cl.Image(path=image_path, name="recipe_image", display="inline")]
                        await cl.Message(
                            content=f"👨‍🍳 **{selected_doc.metadata.get('title', 'Recipe')}**\n\n{answer}",
                            elements=elements
                        ).send()
                    else:
                        await cl.Message(
                            content=f"👨‍🍳 **{selected_doc.metadata.get('title', 'Recipe')}**\n\n{answer}"
                        ).send()
                    
                    return
                else:
                    await msg.remove()
                    await cl.Message(
                        content=f"😅 Số {user_query} không hợp lệ. Chọn từ 1 đến {len(pending_recipes)} nhé!"
                    ).send()
                    return
        
        # =====================================================
        # CONVERSATION HISTORY
        # =====================================================
        
        history = cl.user_session.get("history", [])
        
        if len(history) > 6:
            history = history[-6:]
        
        history_context = ""
        if history:
            history_context = "Previous conversation:\n" + "\n".join([
                f"- {h['role']}: {h['content'][:100]}..." 
                for h in history[-4:]
            ]) + "\n\n"
        
        count = cl.user_session.get("conversation_count", 0)
        cl.user_session.set("conversation_count", count + 1)
        
        # =====================================================
        # IMPROVED LANGUAGE DETECTION
        # =====================================================
        
        language_prompt = f"""Analyze the CURRENT query ONLY. Ignore conversation history when detecting language.

Current query: "{user_query}"

⚠️ LANGUAGE DETECTION RULE:
- If current query is in ENGLISH → "language": "en"
- If current query is in VIETNAMESE → "language": "vi"
- DO NOT let conversation history influence language detection

Previous context (for category and followup detection only):
{history_context}

Analyze and respond in JSON:
{{
  "language": "vi" or "en",
  "english_query": "translated to English",
  "category": "recipe_search" or "cooking_advice" or "greeting" or "off_topic" or "ingredient_based",
  "is_followup": true or false,
  "excluded_ingredients": ["ingredient1", "ingredient2"]
}}

Categories:
- recipe_search: Finding specific recipes (e.g., "pizza recipe", "cách làm phở")
- cooking_advice: General cooking questions (e.g., "how to boil eggs", "cách luộc trứng")
- ingredient_based: Has ingredients and asks what to cook (e.g., "tôi có gà và tỏi, nấu gì?")
- greeting: Hi, hello, xin chào
- off_topic: NOT cooking related

⚠️ CRITICAL RULE FOR is_followup:
- is_followup = true ONLY IF the query refers to the SAME topic/recipe from previous conversation
- is_followup = false if user provides COMPLETELY NEW ingredients or changes topic

⚠️ EXCLUDED INGREDIENTS:
- If user says "KHÔNG có X", "NO X", "without X", "except X" → add X to excluded_ingredients
"""

        analysis_text = await safe_llm_call(
            language_prompt, 
            default_response='{"language": "vi", "english_query": "' + user_query + '", "category": "recipe_search", "is_followup": false, "excluded_ingredients": []}'
        )
        
        data = safe_json_parse(analysis_text, default={
            "language": "vi",
            "english_query": user_query,
            "category": "recipe_search",
            "is_followup": False,
            "excluded_ingredients": []
        })
        
        detected_lang = data.get("language", "vi")
        english_query = data.get("english_query", user_query)
        category = data.get("category", "recipe_search")
        is_followup = data.get("is_followup", False)
        excluded_ingredients = data.get("excluded_ingredients", [])
        
        print(f"🔍 Detected: lang={detected_lang}, category={category}, followup={is_followup}, excluded={excluded_ingredients}")
        
        cl.user_session.set("last_detected_lang", detected_lang)
        
        # =====================================================
        # FOLLOW-UP QUESTION HANDLING
        # =====================================================
        
        if is_followup and history:
            last_recipe_context = cl.user_session.get("last_recipe_context", "")
            
            if last_recipe_context:
                followup_prompt = f"""Previous recipe context:
{last_recipe_context}

User's follow-up question: "{user_query}"

Answer in {"VIETNAMESE" if detected_lang == "vi" else "ENGLISH"}:"""
                
                answer = await safe_llm_call(followup_prompt)
                
                cl.user_session.set("last_assistant_message", answer)
                
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": answer[:200]})
                cl.user_session.set("history", history)
                
                await msg.remove()
                await cl.Message(content=f"👨‍🍳 {answer}").send()
                return
        
        # =====================================================
        # GREETING HANDLING
        # =====================================================
        
        if category == "greeting":
            if detected_lang == "vi":
                response = "👋 Xin chào! Tôi là trợ lý nấu ăn. Bạn cần tìm công thức gì không?"
            else:
                response = "👋 Hello! I'm your cooking assistant. What recipe are you looking for?"
            
            cl.user_session.set("last_assistant_message", response)
            
            await msg.remove()
            await cl.Message(content=response).send()
            
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": response})
            cl.user_session.set("history", history)
            
            return
        
        # =====================================================
        # OFF-TOPIC HANDLING
        # =====================================================
        
        if category == "off_topic":
            if detected_lang == "vi":
                response = """Xin lỗi, tôi chỉ chuyên về nấu ăn thôi! 🍳

Tôi có thể giúp bạn:
Tìm công thức món ăn
Hướng dẫn nấu ăn
Tư vấn nguyên liệu
Mẹo vặt bếp núc

Bạn muốn hỏi gì về nấu ăn không? 😊"""
            else:
                response = """Sorry, I only specialize in cooking! 🍳

I can help you with:
Recipe recommendations
Cooking instructions
Ingredient advice
Cooking tips

Would you like to ask about cooking? 😊"""
            
            cl.user_session.set("last_assistant_message", response)
            
            await msg.remove()
            await cl.Message(content=response).send()
            
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": response})
            cl.user_session.set("history", history)
            
            return
        
        # =====================================================
        # INGREDIENT-BASED SEARCH WITH IMPROVED EXTRACTION
        # =====================================================
        
        if category == "ingredient_based":
            try:
                ingredient_prompt = f"""Extract ONLY the ingredients the user HAS and wants to use.

⚠️ CRITICAL RULES:
- DO NOT include ingredients if user says: "không có", "no", "without", "except", "besides", "but not"
- Only extract ingredients they WANT to cook with
- Return comma-separated list in English

Query: "{user_query}"

Examples:
- "tôi có gà và tỏi" → "chicken, garlic"
- "tôi có gà nhưng không có hành tây" → "chicken"
- "salmon but no chocolate" → "salmon"
- "I have eggs except milk" → "eggs"

Ingredients user HAS:"""
                
                ingredients_text = await safe_llm_call(ingredient_prompt, default_response="chicken")
                ingredients_list = [i.strip() for i in ingredients_text.split(",") if i.strip()]
                
                print(f"🥕 Extracted ingredients: {ingredients_list}")
                print(f"🚫 Excluded ingredients: {excluded_ingredients}")
                
                # Build compound query
                if len(ingredients_list) == 1:
                    search_query = f"recipe with {ingredients_list[0]}"
                elif len(ingredients_list) == 2:
                    search_query = f"{ingredients_list[0]} and {ingredients_list[1]} recipe"
                else:
                    search_query = f"{ingredients_list[0]} {ingredients_list[1]} {ingredients_list[2]} recipe"
                
                print(f"🔍 Search query: {search_query}")
                
                # Primary search
                results = await safe_vector_search(search_query, k=10)
                
                # Fallback searches
                search_tasks = [
                    safe_vector_search(f"recipe with {ingredient}", k=5)
                    for ingredient in ingredients_list[:3]
                ]
                search_results = await asyncio.gather(*search_tasks)
                
                all_results = list(results)
                for results in search_results:
                    all_results.extend(results)
                
                if not all_results:
                    await msg.remove()
                    if detected_lang == "vi":
                        await cl.Message(content="😅 Không tìm thấy món nào với nguyên liệu này. Thử nguyên liệu khác nhé!").send()
                    else:
                        await cl.Message(content="😅 No recipes found with these ingredients. Try different ones!").send()
                    return
                
                # Remove duplicates
                unique_recipes = {}
                for doc, score in all_results:
                    title = doc.metadata.get('title')
                    if title not in unique_recipes or score > unique_recipes[title][1]:
                        unique_recipes[title] = (doc, score)
                
                sorted_results = sorted(unique_recipes.values(), key=lambda x: x[1], reverse=True)[:10]
                
                # Filter with hard exclusion check
                filtered_results = []
                for doc, score in sorted_results:
                    recipe_content = doc.page_content.lower()
                    recipe_title = doc.metadata.get('title', '').lower()
                    
                    # Check exclusions
                    contains_excluded = False
                    for excluded in excluded_ingredients:
                        excluded_lower = excluded.lower()
                        if excluded_lower in recipe_content or excluded_lower in recipe_title:
                            contains_excluded = True
                            print(f"   ✗ EXCLUDED: {doc.metadata['title']} (contains {excluded})")
                            break
                    
                    if contains_excluded:
                        continue
                    
                    # Count ingredient matches
                    ingredient_match_count = sum(
                        1 for ing in ingredients_list 
                        if ing.lower() in recipe_content
                    )
                    
                    min_matches = 1 if len(ingredients_list) <= 2 else 2
                    
                    if ingredient_match_count >= min_matches:
                        coverage_boost = ingredient_match_count / len(ingredients_list)
                        adjusted_score = score * (1 + coverage_boost * 0.3)
                        filtered_results.append((doc, adjusted_score, ingredient_match_count))
                        print(f"   ✓ {doc.metadata['title']}: {ingredient_match_count}/{len(ingredients_list)} ingredients (score: {adjusted_score:.2f})")
                
                if not filtered_results:
                    await msg.remove()
                    if detected_lang == "vi":
                        if excluded_ingredients:
                            await cl.Message(content=f"😅 Không tìm thấy món nào có {', '.join(ingredients_list)} mà KHÔNG có {', '.join(excluded_ingredients)}. Thử nguyên liệu khác nhé!").send()
                        else:
                            await cl.Message(content="😅 Không tìm thấy món nào có đủ nguyên liệu này. Thử nguyên liệu khác nhé!").send()
                    else:
                        if excluded_ingredients:
                            await cl.Message(content=f"😅 No recipes found with {', '.join(ingredients_list)} WITHOUT {', '.join(excluded_ingredients)}. Try different ones!").send()
                        else:
                            await cl.Message(content="😅 No recipes found with these ingredients. Try different ones!").send()
                    return
                
                sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:5]
                
                # Rerank
                if len(sorted_results) >= 2:
                    rerank_prompt = f"""User has: {', '.join(ingredients_list)}
User does NOT want: {', '.join(excluded_ingredients) if excluded_ingredients else 'nothing excluded'}

Rank by best ingredient match (1 = best):
{chr(10).join([f"{i+1}. {doc.metadata['title']} (contains {match_count}/{len(ingredients_list)} ingredients)" for i, (doc, _, match_count) in enumerate(sorted_results)])}

Return ONLY numbers, comma-separated:"""
                    
                    ranking_text = await safe_llm_call(rerank_prompt, default_response="1,2,3,4,5")
                    
                    try:
                        ranking = [int(x.strip()) - 1 for x in ranking_text.split(",") if x.strip().isdigit()]
                        reranked = [sorted_results[i] for i in ranking if i < len(sorted_results)]
                        if reranked:
                            sorted_results = reranked
                    except:
                        pass
                
                best_doc, best_score, best_match_count = sorted_results[0]
                
                context = f"Recipe: {best_doc.metadata.get('title', 'N/A')}\n{best_doc.page_content}"
                
                recipe_ingredients_lower = best_doc.page_content.lower()
                has_ingredients = [ing for ing in ingredients_list if ing.lower() in recipe_ingredients_lower]
                missing_ingredients = [ing for ing in ingredients_list if ing.lower() not in recipe_ingredients_lower]
                
                # Generate suggestion and ask for confirmation
                suggestion_prompt = f"""You are a chef assistant.

User has: {', '.join(ingredients_list)}
⚠️ User does NOT want: {', '.join(excluded_ingredients) if excluded_ingredients else 'no exclusions'}

Best matching recipe:
{context}

The recipe contains: {', '.join(has_ingredients) if has_ingredients else 'none'}
Missing: {', '.join(missing_ingredients) if missing_ingredients else 'none'}

⚠️ CRITICAL CHECK:
- If recipe contains ANY excluded ingredient, respond: "Xin lỗi, công thức này có [ingredient], không phù hợp."
- Otherwise, give a 2-3 sentence summary explaining why this recipe is a good match

Answer in {"VIETNAMESE" if detected_lang == "vi" else "ENGLISH"}:"""
                
                answer = await safe_llm_call(suggestion_prompt, default_response="Đây là món phù hợp!")
                
                # Save and ask for confirmation
                cl.user_session.set("suggested_recipe", (best_doc, best_score))
                cl.user_session.set("waiting_for_recipe_confirm", True)
                cl.user_session.set("last_recipe_context", context)
                
                if detected_lang == "vi":
                    suggestion = f"""🥗 **Gợi ý món ăn phù hợp:**

**{best_doc.metadata.get('title', 'Recipe')}**

{answer}

---
💡 **Bạn có muốn xem công thức chi tiết không?** (Trả lời: "ok", "có", "yes")"""
                else:
                    suggestion = f"""🥗 **Recipe Suggestion:**

**{best_doc.metadata.get('title', 'Recipe')}**

{answer}

---
💡 **Would you like to see the full recipe?** (Reply: "ok", "yes")"""
                
                cl.user_session.set("last_assistant_message", suggestion)
                
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": suggestion})
                cl.user_session.set("history", history)
                
                await msg.remove()
                
                image_path = get_image_path(best_doc.metadata.get("image"))
                if image_path:
                    elements = [cl.Image(path=image_path, name="recipe_image", display="inline")]
                    await cl.Message(content=suggestion, elements=elements).send()
                else:
                    await cl.Message(content=suggestion).send()
                
                return
                
            except Exception as e:
                print(f"⚠️ Error in ingredient search: {e}")
                traceback.print_exc()
                category = "recipe_search"
        
        # =====================================================
        # COOKING ADVICE
        # =====================================================
        
        if category == "cooking_advice":
            results = await safe_vector_search(english_query, k=3)
            
            if results and results[0][1] >= 0.5:
                context = "\n\n".join([
                    f"Recipe: {d.metadata.get('title', 'N/A')}\n{d.page_content}" 
                    for d, score in results[:2] if score >= 0.5
                ])
                
                prompt = f"""You are a professional chef assistant.

Related recipes:
{context}

User's question: "{user_query}"

Instructions:
- Answer in {"VIETNAMESE" if detected_lang == "vi" else "ENGLISH"}
- Use recipe context if relevant, BUT also add your general cooking knowledge
- Be practical and helpful
- Give clear steps if needed

Answer:"""
            else:
                prompt = f"""You are a professional chef assistant.

User's question: "{user_query}"

Instructions:
- Answer in {"VIETNAMESE" if detected_lang == "vi" else "ENGLISH"}
- Use your cooking expertise
- Be practical and helpful
- Give clear steps if needed

Answer:"""
            
            answer = await safe_llm_call(prompt)
            
            cl.user_session.set("last_assistant_message", answer)
            
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": answer[:200]})
            cl.user_session.set("history", history)
            
            await msg.remove()
            await cl.Message(content=f"👨‍🍳 {answer}").send()
            return
        
        # =====================================================
        # RECIPE SEARCH
        # =====================================================
        
        results = await safe_vector_search(english_query, k=10)
        
        if not results or results[0][1] < 0.4:
            await msg.remove()
            if detected_lang == "vi":
                await cl.Message(content="😅 Xin lỗi, tôi không tìm thấy công thức phù hợp. Bạn thử hỏi cách khác nhé!").send()
            else:
                await cl.Message(content="😅 Sorry, I couldn't find a matching recipe. Try asking differently!").send()
            return
        
        # Reranking
        try:
            if len(results) >= 3:
                rerank_prompt = f"""You are a recipe relevance expert.

User query: "{user_query}"

Recipes:
{chr(10).join([f"{i+1}. {doc.metadata['title']} (score: {score:.2f})" for i, (doc, score) in enumerate(results[:10])])}

Rerank by TRUE relevance. Return ONLY numbers, comma-separated:"""
                
                ranking_text = await safe_llm_call(rerank_prompt, default_response="1,2,3,4,5,6,7,8,9,10")
                
                try:
                    ranking_indices = [int(x.strip()) - 1 for x in ranking_text.split(",") if x.strip().isdigit()]
                    
                    reranked = []
                    for idx in ranking_indices:
                        if 0 <= idx < len(results):
                            reranked.append(results[idx])
                    
                    for i, result in enumerate(results):
                        if i not in ranking_indices:
                            reranked.append(result)
                    
                    results = reranked[:5]
                    print(f"✅ Reranked: {[doc.metadata['title'] for doc, _ in results[:3]]}")
                
                except Exception as e:
                    print(f"⚠️ Reranking failed: {e}")
        
        except Exception as e:
            print(f"⚠️ Reranking error: {e}")
        
        # =====================================================
        # MULTI-RECIPE DISPLAY
        # =====================================================
        
        top_recipes = results[:3]
        good_recipes = [r for r in top_recipes if r[1] >= 0.5]
        
        if len(good_recipes) > 1:
            recipe_list = "\n\n".join([
                f"**{i+1}. {doc.metadata.get('title', 'Recipe')}** (độ phù hợp: {score:.0%})"
                for i, (doc, score) in enumerate(good_recipes)
            ])
            
            if detected_lang == "vi":
                intro = f"🍳 **Tôi tìm thấy {len(good_recipes)} công thức phù hợp:**\n\n{recipe_list}\n\n💡 Bạn muốn xem công thức nào? (gõ số 1, 2, 3...)"
            else:
                intro = f"🍳 **I found {len(good_recipes)} matching recipes:**\n\n{recipe_list}\n\n💡 Which one would you like? (type 1, 2, 3...)"
            
            cl.user_session.set("pending_recipes", good_recipes)
            cl.user_session.set("waiting_for_selection", True)
            cl.user_session.set("last_assistant_message", intro)
            
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": intro})
            cl.user_session.set("history", history)
            
            await msg.remove()
            await cl.Message(content=intro).send()
            return
        
        # Single best match
        best_doc, best_score = good_recipes[0] if good_recipes else top_recipes[0]
        
        context = f"Recipe: {best_doc.metadata.get('title', 'N/A')}\n{best_doc.page_content}"
        
        final_prompt = f"""You are a professional chef assistant.

Recipe:
{context}

User's question: "{user_query}"

Instructions:
- Answer in {"VIETNAMESE" if detected_lang == "vi" else "ENGLISH"}
- Focus on recipe details
- Be concise and clear
- Cite the recipe name

Answer:"""
        
        answer = await safe_llm_call(final_prompt)
        
        cl.user_session.set("last_recipe_context", context)
        cl.user_session.set("last_assistant_message", answer)
        
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": answer[:200]})
        cl.user_session.set("history", history)
        
        image_path = get_image_path(best_doc.metadata.get("image"))
        
        await msg.remove()
        
        if image_path:
            elements = [cl.Image(path=image_path, name="recipe_image", display="inline")]
            await cl.Message(
                content=f"👨‍🍳 **{best_doc.metadata.get('title', 'Recipe')}**\n\n{answer}",
                elements=elements
            ).send()
        else:
            await cl.Message(content=f"👨‍🍳 {answer}").send()
    
    except Exception as e:
        print(f"❌ Unexpected error in main: {e}")
        traceback.print_exc()
        
        try:
            await msg.remove()
        except:
            pass
        
        detected_lang = cl.user_session.get("last_detected_lang", "vi")
        
        if detected_lang == "vi":
            await cl.Message(
                content="😅 **Xin lỗi, có lỗi xảy ra.**\n\n"
                       "Bạn thử:\n"
                       "• Hỏi lại bằng cách khác\n"
                       "• Refresh trang nếu lỗi vẫn còn"
            ).send()
        else:
            await cl.Message(
                content="😅 **Sorry, something went wrong.**\n\n"
                       "Please try:\n"
                       "• Rephrasing your question\n"
                       "• Refreshing the page if error persists"
            ).send()