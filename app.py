from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load dataset
file_path = "D:\\Diversion\\Training\\updated_file.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Define helper functions to extract structured information

def extract_rating(text):
    if pd.isna(text):
        return None
    match = re.search(r"(\d+(\.\d+)?)\/10", str(text))
    return float(match.group(1)) if match else None

def extract_pros(text):
    if pd.isna(text):
        return []
    match = re.search(r"Pros:\s*\n- (.*?)(\n\n|Cons:|$)", str(text), re.DOTALL)
    return [p.strip() for p in match.group(1).split("\n- ")] if match else []

def extract_cons(text):
    if pd.isna(text):
        return []
    match = re.search(r"Cons:\s*\n- (.*?)(\n\n|Verdict:|$)", str(text), re.DOTALL)
    return [c.strip() for c in match.group(1).split("\n- ")] if match else []

def extract_verdict(text):
    if pd.isna(text):
        return ""
    match = re.search(r"Verdict:\s*(.*)", str(text), re.DOTALL)
    return match.group(1).strip() if match else ""

# Apply extraction functions
for skin_type in ["oily_skin", "dry_skin", "combination_skin"]:
    df[f"{skin_type}_rating"] = df[skin_type].apply(extract_rating)
    df[f"{skin_type}_pros"] = df[skin_type].apply(extract_pros)
    df[f"{skin_type}_cons"] = df[skin_type].apply(extract_cons)
    df[f"{skin_type}_verdict"] = df[skin_type].apply(extract_verdict)

df.drop(columns=["oily_skin", "dry_skin", "combination_skin"], inplace=True)

# Fix: Replace inplace=True to avoid FutureWarning
for skin_type in ["oily_skin", "dry_skin", "combination_skin"]:
    df[f"{skin_type}_rating"] = df[f"{skin_type}_rating"].fillna(df[f"{skin_type}_rating"].mean())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["ingredients"].fillna("")).toarray()

# Train Random Forest models
y_oily = df["oily_skin_rating"].values
y_dry = df["dry_skin_rating"].values
y_comb = df["combination_skin_rating"].values

model_oily = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_oily)
model_dry = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_dry)
model_comb = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_comb)

# Define API Input Model
class ProductInput(BaseModel):
    ingredients: list[str]
    skin_type: str

# Define API Endpoint
@app.post("/predict")
def predict_product(data: ProductInput):
    input_features = vectorizer.transform([" ".join(data.ingredients)]).toarray()

    if data.skin_type.lower() == "oily":
        rating = round(model_oily.predict(input_features)[0], 2)
        pros_column, cons_column, verdict_column = "oily_skin_pros", "oily_skin_cons", "oily_skin_verdict"
    elif data.skin_type.lower() == "dry":
        rating = round(model_dry.predict(input_features)[0], 2)
        pros_column, cons_column, verdict_column = "dry_skin_pros", "dry_skin_cons", "dry_skin_verdict"
    elif data.skin_type.lower() == "combination":
        rating = round(model_comb.predict(input_features)[0], 2)
        pros_column, cons_column, verdict_column = "combination_skin_pros", "combination_skin_cons", "combination_skin_verdict"
    else:
        return {"error": "Invalid skin type"}

    similar_products = df[df["ingredients"].apply(lambda x: any(ing.lower() in x.lower() for ing in data.ingredients))]
    if similar_products.empty:
        return {
            "Predicted Rating": f"{rating} / 10",
            "Pros": ["No similar products found"],
            "Cons": ["No similar products found"],
            "Verdict": ["No verdict available"],
            "Final Suggestion": "Use with caution." if 5 <= rating <= 6 else "You can choose better alternatives." if rating < 5 else "Go for it!"
        }

    top_pros = similar_products[pros_column].explode().dropna().unique().tolist()[:5]
    top_cons = similar_products[cons_column].explode().dropna().unique().tolist()[:5]
    top_verdicts = similar_products[verdict_column].dropna().unique().tolist()[:1]

    final_suggestion = "Go for it!" if rating > 6 else "Use with caution." if rating in [5, 6] else "You can choose better alternatives."

    return {
        "Predicted Rating": f"{rating} / 10",
        "Pros": top_pros,
        "Cons": top_cons,
        "Verdict": top_verdicts,
        "Final Suggestion": final_suggestion
    }

# Ensure FastAPI doesn't restart unexpectedly
@app.on_event("startup")
async def startup_event():
    print("\nFastAPI server is running! Visit http://127.0.0.1:8000/docs to test the API.\n")

# Only run the server if the script is executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
