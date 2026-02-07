import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df1 = pd.read_csv("Training.csv")
df2 = pd.read_csv("Testing.csv")
df = pd.concat([df1, df2], ignore_index=True)

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model saved!")
