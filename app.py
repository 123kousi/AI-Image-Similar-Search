from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64

from search import search_similar, search_by_text

app = FastAPI()

app.mount("/images", StaticFiles(directory="images"), name="images")


# 🌈 HOME PAGE
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>AI Image Search</title>
        <style>
            body {
                font-family: 'Segoe UI';
                margin: 0;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                text-align: center;
            }

            .container {
                margin-top: 100px;
            }

            h1 {
                font-size: 40px;
                margin-bottom: 20px;
            }

            input {
                padding: 12px;
                margin: 10px;
                border-radius: 10px;
                border: none;
                width: 250px;
            }

            button {
                padding: 12px 25px;
                border: none;
                border-radius: 10px;
                background: #ff7eb3;
                color: white;
                font-weight: bold;
                cursor: pointer;
                transition: 0.3s;
            }

            button:hover {
                background: #ff4f8b;
                transform: scale(1.05);
            }
        </style>
    </head>

    <body>

    <div class="container">
        <h1>🔍 AI Image Search</h1>

        <form action="/search" method="post" enctype="multipart/form-data">
            <input type="file" name="file"><br>
            <input type="text" name="query" placeholder="Search like: 3 stones, dog..."><br>
            <button type="submit">Search</button>
        </form>
    </div>

    </body>
    </html>
    """


# 🌈 SEARCH PAGE
@app.post("/search", response_class=HTMLResponse)
async def search(
    file: UploadFile = File(None),
    query: str = Form(None)
):
    results = []
    input_preview = ""

    # IMAGE SEARCH
    if file and file.filename != "":
        contents = await file.read()

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except:
            return "<h3>Invalid image file</h3>"

        results = search_similar(image)

        encoded = base64.b64encode(contents).decode()
        input_preview = f'<img src="data:image/jpeg;base64,{encoded}" class="preview"/>'

    # TEXT SEARCH
    elif query:
        results = search_by_text(query)
        input_preview = f"<p><b>Text:</b> {query}</p>"

    else:
        return "<h3>Please upload image or enter text</h3>"

    html = f"""
    <html>
    <head>
    <style>
    body {{
        font-family: 'Segoe UI';
        margin: 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }}

    .header {{
        text-align: center;
        padding: 20px;
    }}

    .container {{
        display: flex;
        gap: 20px;
        padding: 20px;
    }}

    .input-box {{
        width: 25%;
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }}

    .results {{
        width: 75%;
    }}

    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 15px;
    }}

    .card {{
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        overflow: hidden;
        backdrop-filter: blur(10px);
        transition: 0.3s;
    }}

    .card:hover {{
        transform: scale(1.05);
    }}

    img {{
        width: 100%;
        height: 200px;
        object-fit: cover;
    }}

    .preview {{
        width: 100%;
        border-radius: 10px;
    }}

    .score {{
        padding: 8px;
        text-align: center;
        font-weight: bold;
    }}

    a {{
        color: white;
        text-decoration: none;
        font-weight: bold;
    }}
    </style>
    </head>

    <body>

    <div class="header">
        <h2>✨ Results</h2>
        <a href="/">⬅ Back</a>
    </div>

    <div class="container">

        <div class="input-box">
            <h3>Input</h3>
            {input_preview}
        </div>

        <div class="results">
            <div class="grid">
    """

    for r in results:
        html += f"""
        <div class="card">
            <img src="/images/{r['image_path']}" />
            <div class="score">{r['score']:.2f}</div>
        </div>
        """

    html += """
            </div>
        </div>

    </div>

    </body>
    </html>
    """

    return html