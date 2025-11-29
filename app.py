import os
import io
import numpy as np
import torch
import open_clip
from PIL import Image
import streamlit as st
from pathlib import Path

# ------------------------------------
# CONFIG
# ------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DB_DIR = "core pipeline/vector_db"   # same as in your FastAPI app


# ------------------------------------
# CACHED LOADERS
# ------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    st.write("Loading CLIP model... (only runs once)")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(DEVICE)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer



@st.cache_resource
def load_vector_db(db_path=DB_DIR):
    emb_path = os.path.join(db_path, "embeddings.npy")
    txt_path = os.path.join(db_path, "paths.txt")

    # Load embeddings
    embeds_np = np.load(emb_path)
    embeds_t = torch.tensor(embeds_np, dtype=torch.float32).to(DEVICE)

    cleaned_paths = []

    with open(txt_path, "r") as f:
        for raw in f:
            p = raw.strip().replace("\\", "/")
            if not p:
                continue

            # FIX: prepend correct folder
            abs_p = Path("core pipeline") / p
            abs_p = abs_p.resolve()

            cleaned_paths.append(str(abs_p))   # ← USE abs_p, not abs_path

    return embeds_t, cleaned_paths

# def load_vector_db(db_path=DB_DIR):
#     emb_path = os.path.join(db_path, "embeddings.npy")
#     txt_path = os.path.join(db_path, "paths.txt")

#     # Load embeddings
#     embeds_np = np.load(emb_path)
#     embeds_t = torch.tensor(embeds_np, dtype=torch.float32).to(DEVICE)

#     # Load image paths
#     cleaned_paths = []
#     with open(txt_path, "r") as f:
#         for raw in f:
#             p = raw.strip()

#             # Skip empty lines
#             if not p:
#                 continue

#             # Replace win slashes & fix weird characters
#             p = p.replace("\\", "/")

#             # Convert to absolute path SAFELY
#             abs_p = os.path.abspath(os.path.join("core pipeline", p))
#             cleaned_paths.append(str(abs_path))


    return embeds_t, cleaned_paths




# def load_vector_db(db_path=DB_DIR):
#     emb_path = os.path.join(db_path, "embeddings.npy")
#     txt_path = os.path.join(db_path, "paths.txt")

#     if not os.path.exists(emb_path) or not os.path.exists(txt_path):
#         raise FileNotFoundError("Vector DB missing embeddings.npy or paths.txt")

#     embeds = np.load(emb_path)  # (N, D) numpy
#     with open(txt_path, "r") as f:
#         paths = [line.strip() for line in f.readlines()]

#     # convert to tensor on device
#     embeds_t = torch.tensor(embeds, dtype=torch.float32).to(DEVICE)
#     return embeds_t, paths




# ------------------------------------
# ENCODING HELPERS
# ------------------------------------
def encode_text(query: str, model, tokenizer):
    tok = tokenizer([query])
    with torch.no_grad():
        txt = model.encode_text(tok.to(DEVICE))
        txt /= txt.norm(dim=-1, keepdim=True)
    return txt  # shape (1, D)


def encode_image(pil_img: Image.Image, model, preprocess):
    img = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vec = model.encode_image(img)
        vec /= vec.norm(dim=-1, keepdim=True)
    return vec  # shape (1, D)


def cosine_topk(query_vec, vectors, k=5):
    # vectors: (N, D), query_vec: (1, D)
    sims = (vectors @ query_vec.T).squeeze(1)  # (N,)
    topk = torch.topk(sims, k)
    return topk.indices.cpu().tolist(), topk.values.cpu().tolist()


# ------------------------------------
# STREAMLIT UI
# ------------------------------------
def main():
    st.title("Dating Image Search App")

    model, preprocess, tokenizer = load_model_and_tokenizer()
    IMAGE_EMBEDDINGS, IMAGE_PATHS = load_vector_db()

    mode = st.radio("Choose search mode:", ["Text search", "Image search"])

    k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    # --------------------------------------
    # TEXT SEARCH
    # --------------------------------------
    if mode == "Text search":
        query = st.text_input(
            "Describe what you're looking for (e.g., 'guy with beard', 'couple sunset')"
        )

        run_btn = st.button("Run Text Search", key="btn_text")

        if run_btn:
            if not query.strip():
                st.warning("Please enter a valid query.")
            else:
                txt_vec = encode_text(query.strip(), model, tokenizer)
                idxs, scores = cosine_topk(txt_vec, IMAGE_EMBEDDINGS, k)

                st.subheader("Results")
                for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
                    img_path = IMAGE_PATHS[i]
                    try:
                        img = Image.open(img_path).convert("RGB")
                        st.image(
                            img,
                            caption=f"#{rank} | {img_path} | score={score:.3f}",
                            use_container_width =True,
                        )
                    except Exception as e:
                        st.write(f"Could not load image {img_path}: {e}")

    # --------------------------------------
    # IMAGE SEARCH
    # --------------------------------------
    else:
        uploaded = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded is not None:
            bytes_data = uploaded.read()
            pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            st.image(pil_img, caption="Query image", use_container_width =True)

        run_btn = st.button("Run Image Search", key="btn_image")

        if run_btn:
            if uploaded is None:
                st.warning("Please upload an image first.")
            else:
                img_vec = encode_image(pil_img, model, preprocess)
                idxs, scores = cosine_topk(img_vec, IMAGE_EMBEDDINGS, k)

                st.subheader("Results")
                for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
                    img_path = IMAGE_PATHS[i]
                    try:
                        img = Image.open(img_path).convert("RGB")
                        st.image(
                            img,
                            caption=f"#{rank} | {img_path} | score={score:.3f}",
                            use_container_width =True,
                        )
                    except Exception as e:
                        st.write(f"Could not load image {img_path}: {e}")


if __name__ == "__main__":
    main()
