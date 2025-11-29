# Dating Image Search Engine (CLIP-Based)

## Installation

### 1. Create and activate a virtual environment

#### Windows
```sh
python -m venv mtb_aryan
mtb_aryan\Scripts\activate
```

#### macOS / Linux
```sh
python3 -m venv mtb_aryan
source mtb_aryan/bin/activate
```

---

### 2. Upgrade pip and install dependencies
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Verify installation
```sh
python -c "import torch, open_clip; print('OK')"
```

Expected:
```
OK
```

---

## Features

### Text Search (CLIP Text Encoder)
Enter queries such as:
- guy with beard
- man at the beach
- gym selfie

Returns the top-5 relevant images.

### Image Search (CLIP Image Encoder)
Upload an image:
- selfie
- travel picture
- gym photo

Returns the top-5 visually similar images.

### Streamlit Frontend
Provides:
- Text search input
- Image upload
- Grid display of results

### FastAPI Backend (optional)
Endpoints:
- /search/text
- /search/image
- /health

---

## Project Structure

```
MTB Pipeline/
│
├── app.py                        # Streamlit frontend
│
├── api.py (optional)             # FastAPI backend
│
├── men_random_50/                # Image dataset
│     ├── image_man (1).jpg
│     ├── image_man (2).jpg
│     └── ...
│
└── core pipeline/
      └── vector_db/
            ├── embeddings.npy
            └── paths.txt
```

---

## Running the Application

### 1. Start Streamlit UI
```sh
streamlit run app.py
```

App opens at:
```
http://localhost:8501
```

---

### 2. Start FastAPI backend (optional)
```sh
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Available at:
```
http://127.0.0.1:8000/docs
```

---

## Regenerating the Vector Database

If images change:
```sh
python generate_vector_db.py
```

This recreates:
- embeddings.npy
- paths.txt

---

## Model Used

OpenCLIP:
```
laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

Chosen for:
- Speed
- Strong performance on lifestyle/dating images
- Reliable embedding quality

---

## Fixing Missing Image Path Errors

paths.txt must contain relative paths such as:

```
men_random_50/image_man (1).jpg
men_random_50/image_man (2).jpg
```

Loader converts them into:

```
core pipeline/men_random_50/image_man (1).jpg
```

Ensuring Windows compatibility.

---

## Credits

Built with:
- OpenCLIP
- PyTorch
- Streamlit
- FastAPI
- NumPy
- PIL
