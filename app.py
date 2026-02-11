import streamlit as st
from rag import ingest_documents, retrieve_answer, llm, get_stored_files, delete_all_embeddings, delete_embedding_by_file_name
from tools import finance_advisor_tool, yfinance_tool, web_search_full
import logging
from datetime import datetime
import re
import time
try:
    import sympy as sp
except Exception:
    sp = None

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('rag_app.log', encoding='utf-8')  # File with UTF-8
    ]
)
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("[START] RAG Application Started")
logger.info("=" * 80)

st.set_page_config(page_title="FINWISE", layout="wide")

# Session state initialization
if "web_permission" not in st.session_state: st.session_state.web_permission = None
if "pending_query" not in st.session_state: st.session_state.pending_query = None
if "uploaded_files" not in st.session_state: 
    st.session_state.uploaded_files = get_stored_files()  # Load from Chroma DB
if "force_web" not in st.session_state: st.session_state.force_web = False

st.title("🏦 FINWISE")

with st.sidebar:
    st.header("[FILE] Document Management")
    logger.info("[FOLDER] Document sidebar opened")
    
    # File uploader for txt and pdf
    uploaded = st.file_uploader(
        "Upload files (PDF or TXT)", 
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    
    if uploaded:
        logger.info(f"[UPLOAD] User uploaded {len(uploaded)} file(s)")

        # Prepare lists
        texts = []
        file_names = []
        successful_uploads = []
        failed_uploads = []

        # Show simple filename list immediately
        st.write("**Upload Status:**")
        for file in uploaded:
            st.write(f"- {file.name}")

        # Process files (no per-file success badges)
        for file in uploaded:
            try:
                logger.info(f"[FILE] Processing file: {file.name} (Type: {file.type})")
                if file.type == "application/pdf":
                    logger.info(f"[SEARCH] Extracting text from PDF: {file.name}")
                    from PyPDF2 import PdfReader
                    pdf_reader = PdfReader(file)
                    text = " ".join([page.extract_text() for page in pdf_reader.pages])
                    texts.append(text)
                    file_names.append(file.name)
                    successful_uploads.append(file.name)
                elif file.type == "text/plain":
                    logger.info(f"[SEARCH] Reading text file: {file.name}")
                    text = file.read().decode('utf-8')
                    texts.append(text)
                    file_names.append(file.name)
                    successful_uploads.append(file.name)
            except Exception as e:
                logger.error(f"[ERROR] Error processing {file.name}: {str(e)}")
                failed_uploads.append((file.name, str(e)))

        # Ingest successfully uploaded documents
        if texts:
            logger.info(f"[PROCESS] Ingesting {len(texts)} document(s) into vector store...")
            with st.spinner(f"Indexing {len(successful_uploads)} file(s)..."):
                added, duplicates, successfully_ingested = ingest_documents(texts, file_names)
                
                # Show duplicate alerts as toasts ONLY during upload
                if duplicates:
                    for dup in duplicates:
                        st.toast(f"⚠️ Already exists: '{dup['name']}'", icon="❌")
                
                if added and added > 0:
                    logger.info(f"[OK] Successfully indexed {len(successfully_ingested)} file(s) (added {added} embeddings)")
                    # Reload files from Chroma DB
                    st.session_state.uploaded_files = get_stored_files()
                    # Show success toasts for newly added files
                    for file in successfully_ingested:
                        st.toast(f"✅ Embedded and stored: '{file}'", icon="✅")
                else:
                    if duplicates:
                        logger.warning("[WARN] All files were duplicates - no new embeddings added")

        if failed_uploads:
            logger.warning(f"[WARN] Failed to upload {len(failed_uploads)} file(s)")
    
    # Always show stored files section at the bottom of sidebar
    st.divider()
    st.subheader("📁 Stored Files")
    
    if st.session_state.uploaded_files:
        for stored_file in st.session_state.uploaded_files:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f"• **{stored_file}**")
            with col2:
                if st.button("🗑️", key=f"del_{stored_file}", help="Delete this file", use_container_width=True):
                    if delete_embedding_by_file_name(stored_file):
                        st.session_state.uploaded_files = get_stored_files()
                        st.toast(f"✅ Deleted: '{stored_file}'", icon="✅")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"Failed to delete '{stored_file}'")
        
        # Delete all embeddings button
        st.divider()
        if st.button("🗑️ Delete All Embeddings", key="delete_all", use_container_width=True):
            if delete_all_embeddings():
                st.session_state.uploaded_files = []
                st.toast("⚠️ All embeddings deleted!", icon="🗑️")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Failed to delete all embeddings")
    else:
        st.info("No files stored. Upload files to get started!")

# Intent Logic
def get_intent(q):
    q = q.lower()
    if any(k in q for k in ['price', 'ticker', '$']): return "price"
    if any(k in q for k in ['should i', 'invest', 'advice', 'plan']): return "advisor"
    return "general"

def solve_any_math_problem(query: str):
    """Comprehensive math solver for ANY type of math problem: arithmetic, algebraic, geometric, financial."""
    import re
    import math
    
    query_lower = query.lower()
    logger.info(f"[MATH_SOLVER] Attempting to solve: {query_lower}")
    
    # Extract all numbers from query
    numbers = re.findall(r'(\d+(?:[.,]\d+)?)', query)
    numbers = [float(n.replace(',', '')) for n in numbers]
    
    # ============ ALGEBRAIC & SYMBOLIC MATH ============
    if any(k in query_lower for k in ['solve', 'simplify', 'expand', 'factor', 'derivative', 'integral']):
        logger.info("[ALGEBRA] Attempting algebraic solution via SymPy")
        if sp is None:
            return False
        
        try:
            equation_text = query_lower
            for word in ['solve', 'simplify', 'expand', 'factor', 'calculate', 'find']:
                equation_text = equation_text.replace(word + ':', '').replace(word, '', 1)
            
            equation_text = equation_text.replace('𝑥', 'x').replace('𝑋', 'X')
            equation_text = re.sub(r'[^a-z0-9\-+*/.()=x]', '', equation_text)
            equation_text = re.sub(r'\s+', '', equation_text)
            
            if '=' in equation_text:
                lhs, rhs = equation_text.split('=', 1)
                expr_l = sp.sympify(lhs)
                expr_r = sp.sympify(rhs)
                sol = sp.solve(expr_l - expr_r)
                st.write("**Algebraic Solution:**")
                st.write(f"Given equation: {lhs} = {rhs}")
                st.write(f"Solutions: ")
                for s in sol:
                    st.write(f"- x = {s}")
                return True
            else:
                expr = sp.sympify(equation_text)
                simplified = sp.simplify(expr)
                st.write("**Simplified Expression:**")
                st.write(f"Original: {expr}")
                st.write(f"Simplified: {simplified}")
                return True
        except Exception as e:
            logger.error(f"[ALGEBRA_ERROR] {e}")
            return False
    
    # ============ SIMPLE INTEREST ============
    if any(k in query_lower for k in ['simple interest', 'si =']) and len(numbers) >= 3:
        logger.info("[SI] Simple Interest Problem")
        try:
            P, R, T = numbers[0], numbers[1], numbers[2]
            SI = (P * R * T) / 100
            Amount = P + SI
            st.write("**Simple Interest Formula:** SI = (P × R × T) / 100")
            st.write(f"- Principal (P) = ₹{P:,.2f}")
            st.write(f"- Rate (R) = {R}%")
            st.write(f"- Time (T) = {T} years")
            st.write(f"\n**Solution:**")
            st.write(f"SI = ({P} × {R} × {T}) / 100 = ₹{SI:,.2f}")
            st.write(f"Total Amount = ₹{P:,.2f} + ₹{SI:,.2f} = **₹{Amount:,.2f}**")
            return True
        except: return False
    
    # ============ COMPOUND INTEREST ============
    if any(k in query_lower for k in ['compound interest', 'ci =', 'compounded']) and len(numbers) >= 3:
        logger.info("[CI] Compound Interest Problem")
        try:
            P, R, T = numbers[0], numbers[1], numbers[2]
            A = P * ((1 + R/100) ** T)
            CI = A - P
            st.write("**Compound Interest Formula:** A = P(1 + R/100)^T")
            st.write(f"- Principal (P) = ₹{P:,.2f}")
            st.write(f"- Rate (R) = {R}%")
            st.write(f"- Time (T) = {T} years")
            st.write(f"\n**Solution:**")
            st.write(f"A = {P}(1 + {R}/100)^{T} = **₹{A:,.2f}**")
            st.write(f"CI = A - P = **₹{CI:,.2f}**")
            return True
        except: return False
    
    # ============ PROFIT & LOSS ============
    if ('profit' in query_lower or 'loss' in query_lower) and len(numbers) >= 2:
        logger.info("[P&L] Profit/Loss Problem")
        try:
            if 'profit' in query_lower:
                cost, profit = numbers[0], numbers[1]
                sp_price = cost + profit
                profit_pct = (profit / cost) * 100
                st.write(f"**Profit Calculation:**")
                st.write(f"- Cost Price = ₹{cost:,.2f}")
                st.write(f"- Profit = ₹{profit:,.2f}")
                st.write(f"- Selling Price = ₹{sp_price:,.2f}")
                st.write(f"- Profit % = {profit_pct:.2f}%")
                return True
            elif 'loss' in query_lower:
                cost, loss = numbers[0], numbers[1]
                sp_price = cost - loss
                loss_pct = (loss / cost) * 100
                st.write(f"**Loss Calculation:**")
                st.write(f"- Cost Price = ₹{cost:,.2f}")
                st.write(f"- Loss = ₹{loss:,.2f}")
                st.write(f"- Selling Price = ₹{sp_price:,.2f}")
                st.write(f"- Loss % = {loss_pct:.2f}%")
                return True
        except: return False
    
    # ============ PERCENTAGE ============
    if (any(k in query_lower for k in ['percent', '%']) or 'of' in query_lower) and len(numbers) >= 2:
        logger.info("[%] Percentage Problem")
        try:
            if 'percent of' in query_lower or '% of' in query_lower:
                percent, value = numbers[0], numbers[1]
                result = (percent * value) / 100
                st.write(f"**Percentage Calculation:** {percent}% of {value}")
                st.write(f"Result = ({percent} × {value}) / 100 = **{result:,.2f}**")
                return True
        except: return False
    
    # ============ DISCOUNT ============
    if 'discount' in query_lower and len(numbers) >= 2:
        logger.info("[DISCOUNT] Discount Problem")
        try:
            price, discount_pct = numbers[0], numbers[1]
            discount_amt = (discount_pct * price) / 100
            final_price = price - discount_amt
            st.write(f"**Discount Calculation:**")
            st.write(f"- Original Price = ₹{price:,.2f}")
            st.write(f"- Discount % = {discount_pct}%")
            st.write(f"- Discount Amount = ₹{discount_amt:,.2f}")
            st.write(f"- Final Price = **₹{final_price:,.2f}**")
            return True
        except: return False
    
    # ============ CIRCLE - AREA ============
    if ('circle' in query_lower and 'area' in query_lower) and len(numbers) >= 1:
        logger.info("[CIRCLE] Circle Area")
        try:
            r = numbers[0]
            area = math.pi * r * r
            circumference = 2 * math.pi * r
            st.write(f"**Circle - Radius: {r}**")
            st.write(f"- Area = πr² = π × {r}² = **{area:.2f} sq units**")
            st.write(f"- Circumference = 2πr = 2π × {r} = **{circumference:.2f} units**")
            return True
        except: return False
    
    # ============ CIRCLE - PERIMETER/CIRCUMFERENCE ============
    if ('circle' in query_lower and ('perimeter' in query_lower or 'circumference' in query_lower)) and len(numbers) >= 1:
        logger.info("[CIRCLE_PERI] Circle Circumference")
        try:
            r = numbers[0]
            circumference = 2 * math.pi * r
            st.write(f"**Circle - Circumference (Radius: {r})**")
            st.write(f"Circumference = 2πr = 2π × {r} = **{circumference:.2f} units**")
            return True
        except: return False
    
    # ============ RECTANGLE ============
    if 'rectangle' in query_lower and len(numbers) >= 2:
        logger.info("[RECT] Rectangle")
        try:
            length, width = numbers[0], numbers[1]
            area = length * width
            perimeter = 2 * (length + width)
            diagonal = math.sqrt(length**2 + width**2)
            st.write(f"**Rectangle - Length: {length}, Width: {width}**")
            st.write(f"- Area = length × width = {length} × {width} = **{area:.2f} sq units**")
            st.write(f"- Perimeter = 2(l + w) = 2({length} + {width}) = **{perimeter:.2f} units**")
            st.write(f"- Diagonal = √(l² + w²) = **{diagonal:.2f} units**")
            return True
        except: return False
    
    # ============ SQUARE ============
    if 'square' in query_lower and len(numbers) >= 1:
        logger.info("[SQUARE] Square")
        try:
            side = numbers[0]
            area = side * side
            perimeter = 4 * side
            diagonal = side * math.sqrt(2)
            st.write(f"**Square - Side: {side}**")
            st.write(f"- Area = side² = {side}² = **{area:.2f} sq units**")
            st.write(f"- Perimeter = 4 × side = 4 × {side} = **{perimeter:.2f} units**")
            st.write(f"- Diagonal = side√2 = **{diagonal:.2f} units**")
            return True
        except: return False
    
    # ============ TRIANGLE ============
    if 'triangle' in query_lower and len(numbers) >= 2:
        logger.info("[TRIANGLE] Triangle")
        try:
            if 'area' in query_lower:
                base, height = numbers[0], numbers[1]
                area = (base * height) / 2
                st.write(f"**Triangle Area - Base: {base}, Height: {height}**")
                st.write(f"Area = (base × height) / 2 = ({base} × {height}) / 2 = **{area:.2f} sq units**")
            else:
                a, b, c = numbers[0], numbers[1], numbers[2] if len(numbers) >= 3 else numbers[1]
                s = (a + b + c) / 2
                area = math.sqrt(s * (s - a) * (s - b) * (s - c))
                st.write(f"**Triangle - Sides: {a}, {b}, {c}**")
                st.write(f"Using Heron's Formula: Area = **{area:.2f} sq units**")
            return True
        except: return False
    
    # ============ CUBE ============
    if 'cube' in query_lower and len(numbers) >= 1:
        logger.info("[CUBE] Cube")
        try:
            side = numbers[0]
            volume = side ** 3
            surface_area = 6 * side * side
            st.write(f"**Cube - Side: {side}**")
            st.write(f"- Volume = side³ = {side}³ = **{volume:.2f} cubic units**")
            st.write(f"- Surface Area = 6 × side² = 6 × {side}² = **{surface_area:.2f} sq units**")
            return True
        except: return False
    
    # ============ SPHERE ============
    if 'sphere' in query_lower and len(numbers) >= 1:
        logger.info("[SPHERE] Sphere")
        try:
            r = numbers[0]
            volume = (4 / 3) * math.pi * r ** 3
            surface_area = 4 * math.pi * r * r
            st.write(f"**Sphere - Radius: {r}**")
            st.write(f"- Volume = (4/3)πr³ = **{volume:.2f} cubic units**")
            st.write(f"- Surface Area = 4πr² = **{surface_area:.2f} sq units**")
            return True
        except: return False
    
    # ============ CYLINDER ============
    if 'cylinder' in query_lower and len(numbers) >= 2:
        logger.info("[CYLINDER] Cylinder")
        try:
            r, h = numbers[0], numbers[1]
            volume = math.pi * r * r * h
            surface_area = 2 * math.pi * r * (r + h)
            st.write(f"**Cylinder - Radius: {r}, Height: {h}**")
            st.write(f"- Volume = πr²h = **{volume:.2f} cubic units**")
            st.write(f"- Surface Area = 2πr(r + h) = **{surface_area:.2f} sq units**")
            return True
        except: return False
    
    # ============ DISTANCE, SPEED, TIME ============
    if any(k in query_lower for k in ['distance', 'speed', 'velocity', 'time']) and len(numbers) >= 2:
        logger.info("[MOTION] Distance/Speed/Time")
        try:
            if 'distance' in query_lower and 'speed' in query_lower and 'time' in query_lower:
                speed, time = numbers[0], numbers[1]
                distance = speed * time
                st.write(f"**Distance Calculation:**")
                st.write(f"- Speed = {speed} km/h")
                st.write(f"- Time = {time} hours")
                st.write(f"- Distance = Speed × Time = {speed} × {time} = **{distance:.2f} km**")
                return True
        except: return False
    
    return False
    """Solve financial, geometric, algebraic, and word math problems with formulas and step-by-step explanation."""
    import re
    import math
    
    query_lower = query.lower()
    
    # Simple Interest: P = Principal, R = Rate, T = Time
    if any(k in query_lower for k in ['simple interest', 'si =', 'principal', 'rate', 'annum']):
        logger.info("[FINANCE_MATH] Detected Simple Interest problem")
        
        # Extract numbers from the query
        numbers = re.findall(r'(\d+(?:[.,]\d+)?)', query)
        
        if len(numbers) >= 3:
            try:
                P = float(numbers[0].replace(',', ''))  # Principal
                R = float(numbers[1].replace(',', ''))  # Rate
                T = float(numbers[2].replace(',', ''))  # Time
                
                SI = (P * R * T) / 100
                Amount = P + SI
                
                st.write("**Formula:** SI = (P × R × T) / 100")
                st.write(f"**Where:**")
                st.write(f"- P (Principal) = ₹{P:,.2f}")
                st.write(f"- R (Rate per annum) = {R}%")
                st.write(f"- T (Time in years) = {T}")
                st.write(f"\n**Solution Steps:**")
                st.write(f"1. Simple Interest = (P × R × T) / 100")
                st.write(f"2. SI = ({P} × {R} × {T}) / 100")
                st.write(f"3. SI = {P * R * T} / 100")
                st.write(f"4. **SI = ₹{SI:,.2f}**")
                st.write(f"\n**Final Answer:**")
                st.write(f"- Simple Interest = ₹{SI:,.2f}")
                st.write(f"- Total Amount = Principal + SI = ₹{P:,.2f} + ₹{SI:,.2f} = **₹{Amount:,.2f}**")
                return True
            except Exception as e:
                logger.error(f"[FINANCE_MATH_ERROR] {str(e)}")
                return False
    
    # Compound Interest
    elif any(k in query_lower for k in ['compound interest', 'ci =', 'compounded']):
        logger.info("[FINANCE_MATH] Detected Compound Interest problem")
        
        numbers = re.findall(r'(\d+(?:[.,]\d+)?)', query)
        
        if len(numbers) >= 3:
            try:
                P = float(numbers[0].replace(',', ''))
                R = float(numbers[1].replace(',', ''))
                T = float(numbers[2].replace(',', ''))
                
                # Assuming annual compounding
                Amount_CI = P * ((1 + R/100) ** T)
                CI = Amount_CI - P
                
                st.write("**Formula:** A = P(1 + R/100)^T")
                st.write(f"**Where:**")
                st.write(f"- P (Principal) = ₹{P:,.2f}")
                st.write(f"- R (Rate per annum) = {R}%")
                st.write(f"- T (Time in years) = {T}")
                st.write(f"\n**Solution Steps:**")
                st.write(f"1. Amount = P(1 + R/100)^T")
                st.write(f"2. A = {P}(1 + {R}/100)^{T}")
                st.write(f"3. A = {P} × (1 + {R/100})^{T}")
                st.write(f"4. A = {P} × ({1 + R/100})^{T}")
                st.write(f"5. **A = ₹{Amount_CI:,.2f}**")
                st.write(f"\n**Final Answer:**")
                st.write(f"- Compound Interest = Amount - Principal")
                st.write(f"- CI = ₹{Amount_CI:,.2f} - ₹{P:,.2f} = **₹{CI:,.2f}**")
                return True
            except Exception as e:
                logger.error(f"[FINANCE_MATH_ERROR] {str(e)}")
                return False
    
    # Percentage problems
    elif any(k in query_lower for k in ['percent', '%', 'of']):
        logger.info("[FINANCE_MATH] Detected Percentage problem")
        
        numbers = re.findall(r'(\d+(?:[.,]\d+)?)', query)
        
        if len(numbers) >= 2:
            try:
                if 'percent of' in query_lower or '% of' in query_lower:
                    percent = float(numbers[0].replace(',', ''))
                    value = float(numbers[1].replace(',', ''))
                    result = (percent * value) / 100
                    
                    st.write(f"**Problem:** Find {percent}% of {value}")
                    st.write(f"\n**Formula:** Result = (Percentage × Value) / 100")
                    st.write(f"**Solution:**")
                    st.write(f"1. Result = ({percent} × {value}) / 100")
                    st.write(f"2. Result = {percent * value} / 100")
                    st.write(f"3. **Result = {result:,.2f}**")
                    return True
            except Exception as e:
                logger.error(f"[FINANCE_MATH_ERROR] {str(e)}")
                return False
    
    # Profit/Loss problems
    elif any(k in query_lower for k in ['profit', 'loss', 'cost price', 'selling price']):
        logger.info("[FINANCE_MATH] Detected Profit/Loss problem")
        
        numbers = re.findall(r'(\d+(?:[.,]\d+)?)', query)
        
        if len(numbers) >= 2:
            try:
                if 'profit' in query_lower:
                    cost = float(numbers[0].replace(',', ''))
                    profit_amt = float(numbers[1].replace(',', ''))
                    selling_price = cost + profit_amt
                    profit_pct = (profit_amt / cost) * 100
                    
                    st.write(f"**Problem:** Calculate profit and profit %")
                    st.write(f"**Given:**")
                    st.write(f"- Cost Price = ₹{cost:,.2f}")
                    st.write(f"- Profit = ₹{profit_amt:,.2f}")
                    st.write(f"\n**Formulas:**")
                    st.write(f"- Selling Price = Cost Price + Profit")
                    st.write(f"- Profit % = (Profit / Cost Price) × 100")
                    st.write(f"\n**Solution:**")
                    st.write(f"1. Selling Price = ₹{cost} + ₹{profit_amt} = **₹{selling_price:,.2f}**")
                    st.write(f"2. Profit % = ({profit_amt} / {cost}) × 100 = **{profit_pct:.2f}%**")
                    return True
                elif 'loss' in query_lower:
                    cost = float(numbers[0].replace(',', ''))
                    loss_amt = float(numbers[1].replace(',', ''))
                    selling_price = cost - loss_amt
                    loss_pct = (loss_amt / cost) * 100
                    
                    st.write(f"**Problem:** Calculate loss and loss %")
                    st.write(f"**Given:**")
                    st.write(f"- Cost Price = ₹{cost:,.2f}")
                    st.write(f"- Loss = ₹{loss_amt:,.2f}")
                    st.write(f"\n**Formulas:**")
                    st.write(f"- Selling Price = Cost Price - Loss")
                    st.write(f"- Loss % = (Loss / Cost Price) × 100")
                    st.write(f"\n**Solution:**")
                    st.write(f"1. Selling Price = ₹{cost} - ₹{loss_amt} = **₹{selling_price:,.2f}**")
                    st.write(f"2. Loss % = ({loss_amt} / {cost}) × 100 = **{loss_pct:.2f}%**")
                    return True
            except Exception as e:
                logger.error(f"[FINANCE_MATH_ERROR] {str(e)}")
                return False
    
    # Geometric problems (Area, Volume, Perimeter)
    elif any(k in query_lower for k in ['area', 'perimeter', 'volume', 'radius', 'diameter', 'circle', 'rectangle', 'square', 'triangle', 'sphere', 'cube']):
        logger.info("[GEOMETRY] Detected geometry problem")
        
        numbers = re.findall(r'(\d+(?:[.,]\d+)?)', query)
        
        try:
            if 'circle' in query_lower and 'area' in query_lower and len(numbers) >= 1:
                radius = float(numbers[0].replace(',', ''))
                area = math.pi * radius ** 2
                st.write(f"**Problem:** Area of Circle")
                st.write(f"**Formula:** A = πr²")
                st.write(f"**Given:** Radius = {radius}")
                st.write(f"**Solution:**")
                st.write(f"1. A = π × {radius}²")
                st.write(f"2. A = 3.14159 × {radius**2}")
                st.write(f"3. **A = {area:,.2f} square units**")
                return True
            
            elif 'rectangle' in query_lower and 'area' in query_lower and len(numbers) >= 2:
                length = float(numbers[0].replace(',', ''))
                width = float(numbers[1].replace(',', ''))
                area = length * width
                st.write(f"**Problem:** Area of Rectangle")
                st.write(f"**Formula:** A = length × width")
                st.write(f"**Given:** Length = {length}, Width = {width}")
                st.write(f"**Solution:**")
                st.write(f"1. A = {length} × {width}")
                st.write(f"2. **A = {area:,.2f} square units**")
                return True
            
            elif 'triangle' in query_lower and 'area' in query_lower and len(numbers) >= 2:
                base = float(numbers[0].replace(',', ''))
                height = float(numbers[1].replace(',', ''))
                area = (base * height) / 2
                st.write(f"**Problem:** Area of Triangle")
                st.write(f"**Formula:** A = (base × height) / 2")
                st.write(f"**Given:** Base = {base}, Height = {height}")
                st.write(f"**Solution:**")
                st.write(f"1. A = ({base} × {height}) / 2")
                st.write(f"2. A = {base * height} / 2")
                st.write(f"3. **A = {area:,.2f} square units**")
                return True
            
            elif 'cube' in query_lower and 'volume' in query_lower and len(numbers) >= 1:
                side = float(numbers[0].replace(',', ''))
                volume = side ** 3
                st.write(f"**Problem:** Volume of Cube")
                st.write(f"**Formula:** V = side³")
                st.write(f"**Given:** Side = {side}")
                st.write(f"**Solution:**")
                st.write(f"1. V = {side}³")
                st.write(f"2. V = {side} × {side} × {side}")
                st.write(f"3. **V = {volume:,.2f} cubic units**")
                return True
        except Exception as e:
            logger.error(f"[GEOMETRY_ERROR] {str(e)}")
            return False
    
    # Distance, Speed, Time problems
    elif any(k in query_lower for k in ['distance', 'speed', 'time', 'velocity']):
        logger.info("[MOTION] Detected distance/speed/time problem")
        
        numbers = re.findall(r'(\d+(?:[.,]\d+)?)', query)
        
        try:
            if len(numbers) >= 2:
                if 'distance' in query_lower and ('speed' in query_lower or 'time' in query_lower):
                    if 'speed' in query_lower and 'time' in query_lower:
                        speed = float(numbers[0].replace(',', ''))
                        time = float(numbers[1].replace(',', ''))
                        distance = speed * time
                        st.write(f"**Problem:** Calculate Distance")
                        st.write(f"**Formula:** Distance = Speed × Time")
                        st.write(f"**Given:** Speed = {speed} km/h, Time = {time} hours")
                        st.write(f"**Solution:**")
                        st.write(f"1. Distance = {speed} × {time}")
                        st.write(f"2. **Distance = {distance:,.2f} km**")
                        return True
        except Exception as e:
            logger.error(f"[MOTION_ERROR] {str(e)}")
            return False
    
    return False

def preprocess_math_expression(expr: str) -> str:
    """Convert implicit multiplication to explicit (*) for SymPy parsing.
    Examples: 3x -> 3*x, 2(x+1) -> 2*(x+1), (2)(3) -> (2)*(3)
    """
    expr = expr.strip()
    result = []
    i = 0
    while i < len(expr):
        current = expr[i]
        next_char = expr[i + 1] if i + 1 < len(expr) else None
        
        result.append(current)
        
        # Add * between: digit and letter
        if current.isdigit() and next_char and next_char.isalpha():
            result.append('*')
        # Add * between: letter and digit
        elif current.isalpha() and next_char and next_char.isdigit():
            result.append('*')
        # Add * between: ) and (
        elif current == ')' and next_char == '(':
            result.append('*')
        # Add * between: digit and (
        elif current.isdigit() and next_char == '(':
            result.append('*')
        # Add * between: ) and letter
        elif current == ')' and next_char and next_char.isalpha():
            result.append('*')
        # Add * between: letter and (
        elif current.isalpha() and next_char == '(':
            result.append('*')
        # Add * between: ) and digit
        elif current == ')' and next_char and next_char.isdigit():
            result.append('*')
        
        i += 1
    
    return ''.join(result)

def is_simple_arithmetic(s: str) -> bool:
    """Check if query is simple arithmetic (numbers with math operators)."""
    s = s.lower()
    # Remove words and special chars, keep only math
    expr_part = re.sub(r'[a-z\s\.,:]', '', s).strip()
    # Check if what's left is mostly numbers and operators
    math_chars = set('0123456789+-*/(). ')
    if expr_part and all(c in math_chars for c in expr_part):
        # Has numbers and operators
        if any(op in expr_part for op in ['+', '-', '*', '/', '(', ')']):
            return True
    # Also check if query has numbers separated by operators even without decimals
    if re.search(r'\d+\s*[+\-*/]\s*\d+', s):
        return True
    return False

def is_math_query(s: str) -> bool:
    """Check if query is symbolic math (equations/simplifications, not word problems)."""
    s = s.lower()
    
    # First check for simple arithmetic
    if is_simple_arithmetic(s):
        return True
    
    # Explicit math keywords that indicate symbolic math problems
    math_keywords = [
        'solve', 'simplify', 'equation', 'expand', 'factor', 'differentiate', 'integrate',
        'derivative', 'integral', 'limit', 'series', 'polynomial', 'expression',
        'evaluate', 'formula'
    ]
    
    # Check if query has explicit math keywords
    if any(k in s for k in math_keywords):
        return True
    
    # Also check for mathematical operators with equations (ignore financial P, R, T patterns)
    if '=' in s and not any(x in s for x in ['₹', '% p.a', 'principal', 'rate', 'interest']):
        return True
    
    return False

def is_word_problem(s: str) -> bool:
    """Check if query is a real-world word problem (financial/geometry/motion/algebraic)."""
    s_lower = s.lower()
    
    # SPECIFIC keywords for all problem types
    word_problem_keywords = [
        # Financial
        'simple interest', 'compound interest', 'principal', 'annum',
        'percent', 'percentage', 'profit', 'loss', 'discount', 'markup', 'marked price',
        # Geometry
        'area', 'perimeter', 'volume', 'circumference', 'radius', 'diameter', 'side',
        'circle', 'rectangle', 'square', 'triangle', 'sphere', 'cube', 'cylinder',
        'diagonal', 'surface area',
        # Motion
        'distance', 'speed', 'velocity', 'time',
        # Algebraic
        'solve', 'simplify', 'expand', 'factor', 'equation', 'expression',
        # General
        'find', 'compute', 'calculate'
    ]
    
    if any(k in s_lower for k in word_problem_keywords):
        return True
    
    # Detect patterns like "P = ₹8,000" or "r = 5"
    if re.search(r'\b[p|r|a|b|c|h|l|w|s|x|y|z]\s*=', s_lower):
        return True
    
    # Detect currency patterns with numbers
    if re.search(r'[₹$€]\s*[\d,]+', s):
        return True
    
    return False

# Track last query to avoid re-processing on keystrokes
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

query = st.text_input("Ask a question:")

# Process query only if it's new OR if force_web is active
if st.session_state.force_web and st.session_state.web_permission == 'yes':
    # Continue forced web search - pending_query already set
    pass
elif query and query != st.session_state.last_query:
    logger.info(f"[Q] User query received: {query}")
    st.session_state.pending_query = query
    st.session_state.web_permission = None
    st.session_state.force_web = False
    st.session_state.last_query = query

if st.session_state.pending_query:
    query = st.session_state.pending_query
    # Math handling - Check math before word problems (simple arithmetic comes first)
    if is_math_query(query):
        logger.info("[MATH] Detected math query")
        
        # Try simple arithmetic first
        if is_simple_arithmetic(query):
            logger.info("[MATH] Detected simple arithmetic")
            try:
                # Extract just the math parts, preserve the expression
                import re as re_module
                # Try to extract the numeric expression (numbers + operators)
                expr_part = query.lower()
                # Remove common math words but keep the expression
                for word in ['calculate', 'compute', 'what is', 'find', 'result of', 'solve', 'evaluate']:
                    expr_part = expr_part.replace(word + ':', '', 1)  # Handle "Calculate:" format
                    expr_part = expr_part.replace(word, '', 1).strip()
                
                # Remove all punctuation except math operators
                expr_part = re_module.sub(r'[^0-9+\-*/.()^]', '', expr_part)
                
                # Clean spaces around operators for parsing
                expr_part = re_module.sub(r'\s+', '', expr_part)
                
                # Support exponentiation using ** or ^
                expr_part = expr_part.replace('^', '**')
                
                # Security: ensure only valid characters before eval
                if not all(c in '0123456789+-*/.()* ' for c in expr_part):
                    raise ValueError("Invalid expression")
                
                # Try to evaluate
                result = eval(expr_part)
                st.subheader('🧮 Math Solution')
                st.write(f"**Expression:** {query}")
                st.write(f"**Result:** {result}")
                st.session_state.pending_query = None
                st.session_state.force_web = False
            except Exception as e:
                logger.error(f"[ARITHMETIC_ERROR] {e}")
                st.error(f"I couldn't evaluate that expression. Please try again with a simpler calculation.")
                st.session_state.pending_query = None
        elif sp is None:
            st.error("Math support requires 'sympy' package. See requirements.txt")
            st.session_state.pending_query = None
        else:
            try:
                st.subheader('🧮 Math Solution')
                
                if '=' in query:
                    # Equation solving
                    logger.info("[MATH] Solving equation")
                    # Extract the equation part
                    equation_text = query
                    for keyword in ['solve', 'what is', 'find', 'calculate', 'solve:']:
                        equation_text = equation_text.replace(keyword, '', 1).strip()
                    
                    # Normalize special characters
                    equation_text = equation_text.replace('𝑥', 'x').replace('𝑋', 'X')
                    equation_text = re.sub(r'\s+', '', equation_text)  # Remove all spaces
                    
                    lhs, rhs = equation_text.split('=', 1) if '=' in equation_text else (equation_text, '0')
                    
                    # Preprocess to add explicit multiplication
                    lhs = preprocess_math_expression(lhs)
                    rhs = preprocess_math_expression(rhs)
                    
                    st.write("**Problem:** Solve for x")
                    st.write(f"**Equation:** {lhs.strip()} = {rhs.strip()}")
                    
                    try:
                        expr_l = sp.sympify(lhs)
                        expr_r = sp.sympify(rhs)
                        eq = sp.Eq(expr_l, expr_r)
                        sol = sp.solve(eq)
                        
                        st.write("**Solution Steps:**")
                        st.write(f"1. Given equation: {eq}")
                        st.write(f"2. Rearrange: {expr_l} - ({expr_r}) = 0")
                        st.write(f"3. Solutions found:")
                        for i, s in enumerate(sol, 1):
                            st.write(f"   **x = {s}**")
                    except Exception as parse_err:
                        logger.error(f"[MATH_PARSE] {parse_err}")
                        st.error(f"I couldn't parse that equation. Try simpler format like: 2*x + 3 = 7 or 3x+5=20")
                else:
                    # Expression simplification or evaluation
                    logger.info("[MATH] Simplifying expression")
                    expr_text = query
                    for keyword in ['simplify', 'solve', 'calculate', 'compute', 'find', 'evaluate']:
                        expr_text = expr_text.replace(keyword, '', 1).strip()
                    
                    # Normalize special characters
                    expr_text = expr_text.replace('𝑥', 'x').replace('𝑋', 'X')
                    expr_text = re.sub(r'\s+', '', expr_text)  # Remove all spaces
                    
                    # Preprocess to add explicit multiplication
                    expr_text = preprocess_math_expression(expr_text)
                    
                    st.write(f"**Problem:** Simplify/Evaluate: {expr_text}")
                    
                    try:
                        expr = sp.sympify(expr_text)
                        simplified = sp.simplify(expr)
                        
                        st.write("**Solution Steps:**")
                        st.write(f"1. Original expression: {expr}")
                        st.write(f"2. Simplified form: {simplified}")
                        
                        # Try to evaluate numerically
                        try:
                            evaluated = float(sp.N(expr))
                            st.write(f"3. Numeric value: {evaluated}")
                        except:
                            st.write(f"3. (Cannot evaluate to decimal)")
                        
                        # Show expanded form if different
                        expanded = sp.expand(expr)
                        if expanded != simplified:
                            st.write(f"4. Expanded form: {expanded}")
                    except Exception as e:
                        logger.error(f"[MATH_PARSE] {e}")
                        st.error(f"I couldn't parse that expression. Check formatting and try again.")
            except Exception as e:
                logger.error(f"[MATH_ERR] {e}")
                st.error(f"Error solving math query: {e}")
            st.session_state.pending_query = None
            st.session_state.force_web = False
    elif is_word_problem(query):
        logger.info("[MATH] Detected word problem")
        st.subheader('🧮 Math Solution')
        
        # Try to solve as comprehensive math problem
        if solve_any_math_problem(query):
            st.session_state.pending_query = None
            st.session_state.force_web = False
        else:
            st.warning("Could not solve this problem. Please rephrase with specific numbers.")
            st.info("Examples: 'Simple Interest on ₹10,000 at 8% for 3 years' or 'Area of circle with radius 5'")
            st.session_state.pending_query = None
    else:
        intent = get_intent(query)
        logger.info(f"[AIM] Query intent detected: {intent}")

        # Check for stock price queries and handle them specially with yfinance for real-time data
        if intent == "price":
            logger.info("[FINANCE] Stock price query detected - using yfinance for real-time data")
            with st.spinner("Fetching real-time stock price..."):
                price_result = yfinance_tool(query)
                if price_result and price_result.strip():
                    st.subheader('💰 Stock Price')
                    st.write(price_result)
                    st.session_state.pending_query = None
                else:
                    # Fallback to web search for real-time data
                    logger.info("[FINANCE] Ticker not found, falling back to web search")
                    ans, sources, sources_text = web_search_full(query, llm)
                    st.subheader('💰 Stock Price')
                    st.write(ans)
                    if sources:
                        st.markdown("---")
                        st.subheader('📚 Sources Used')
                        for url in sources:
                            st.markdown(f"🔗 [{url}]({url})")
                    if sources_text:
                        st.info(sources_text)
                    st.session_state.pending_query = None
        else:
            # Force web path
            if st.session_state.force_web and st.session_state.web_permission == 'yes':
                logger.info("[WEB] Force web search active - performing web search")
                with st.spinner("Searching the Web..."):
                    ans, sources, sources_text = web_search_full(st.session_state.pending_query, llm)
                    st.subheader('🌐 Web Answer')
                    st.write(ans)
                    
                    # Show sources below answer
                    if sources:
                        st.markdown("---")
                        st.subheader('📚 Sources Used')
                        cols = st.columns(len(sources)) if len(sources) <= 3 else [st.columns(3) for _ in range((len(sources) + 2) // 3)]
                        for i, url in enumerate(sources):
                            st.markdown(f"🔗 [{url}]({url})")
                    if sources_text:
                        st.info(sources_text)
                st.session_state.pending_query = None
                st.session_state.web_permission = None
                st.session_state.force_web = False
            else:
                if st.session_state.web_permission is None:
                    logger.info("[DOCS] Searching in document database...")
                    with st.spinner('Searching Documents...'):
                        doc_ans = retrieve_answer(query)
                    logger.info(f"[FILE] Document search result: {doc_ans[:100]}..." if len(doc_ans) > 100 else f"[FILE] Document search result: {doc_ans}")

                    if "NOT_FOUND" not in doc_ans:
                        logger.info("[OK] Answer found in documents")
                        st.subheader('Document Answer')
                        st.write(doc_ans)
                        st.session_state.pending_query = None
                    else:
                        logger.info("[ERROR] Answer NOT found in documents - asking for web search permission")
                        st.warning('Answer not found in documents.')
                        st.info('Would you like to search the web for an answer?')
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button('Yes, Search Web', key='btn_yes'):
                                logger.info('[OK] User granted web search permission - YES (force web)')
                                st.session_state.web_permission = 'yes'
                                st.session_state.force_web = True
                                st.rerun()
                        with col2:
                            if st.button('No, Thanks', key='btn_no'):
                                logger.info('[ERROR] User declined web search permission - NO')
                                st.session_state.web_permission = 'no'
                                st.session_state.force_web = False
                                st.rerun()
                else:
                    if st.session_state.web_permission == 'yes':
                        logger.info('[WEB] Web search permission confirmed - proceeding with web search')
                        with st.spinner('Searching the Web...'):
                            ans, sources, sources_text = web_search_full(st.session_state.pending_query, llm)
                            st.subheader('🌐 Web Answer')
                            st.write(ans)
                            
                            # Show sources below answer
                            if sources:
                                st.markdown("---")
                                st.subheader('📚 Sources Used')
                                for url in sources:
                                    st.markdown(f"🔗 [{url}]({url})")
                            if sources_text:
                                st.info(sources_text)
                        st.session_state.web_permission = None
                        st.session_state.pending_query = None
                        st.session_state.force_web = False
                    elif st.session_state.web_permission == 'no':
                        logger.info('[BLOCKED] Web search cancelled by user')
                        st.error("Search cancelled. Answer stays 'Not Found' in local documents.")
                        st.session_state.web_permission = None
                        st.session_state.pending_query = None