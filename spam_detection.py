# Importing necessary libraries for data processing, visualization, and modeling 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_curve, auc
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK stopwords (needed for preprocessing step)
nltk.download('stopwords')

# Step 1: Load the Dataset
dataset_path = "emails.csv"
emails_data = pd.read_csv(dataset_path)

print("\nOriginal dataset (first 5 rows):")
print(emails_data.head())

# Step 2.1: Check for null values
print("\nChecking for null values in the dataset...")
null_values = emails_data.isnull().sum()
print(null_values)

# Handle null values if present
if null_values.sum() > 0:
    print("\nNull values detected. Filling null values with 0.")
    emails_data = emails_data.fillna(0)
else:
    print("\nNo null values detected. Proceeding with the analysis.")

# Step 2: Preprocessing - Remove unnecessary columns
emails_data_cleaned = emails_data.drop(columns=["Email No."])

# Remove stopwords columns
stop_words = set(stopwords.words('english'))
stopword_columns = [col for col in emails_data_cleaned.columns if col in stop_words]
if stopword_columns:
    emails_data_cleaned = emails_data_cleaned.drop(columns=stopword_columns)

# Feature and Label Preparation
X = emails_data_cleaned.iloc[:, :-1]
y = emails_data_cleaned["Prediction"]
print("\nColumns after preprocessing:")
print(X.columns)

# Step 3: Visualize the Data Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=emails_data, x="Prediction")
plt.title("Email Distribution (Spam vs Non-Spam)")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks([0, 1], ["Non Spam", "Spam"])
plt.show()

# Step 4: Train MultinomialNB Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
multinomial_model = MultinomialNB(alpha=1.0)
multinomial_model.fit(X_train, y_train)
y_pred_multinomial = multinomial_model.predict(X_test)

# Evaluate the Model
conf_matrix_multinomial = confusion_matrix(y_test, y_pred_multinomial)
classification_rep_multinomial = classification_report(y_test, y_pred_multinomial)
f1_multinomial = f1_score(y_test, y_pred_multinomial)

print("\nMultinomialNB - Confusion Matrix:")
print(conf_matrix_multinomial)
print("\nMultinomialNB - Classification Report:")
print(classification_rep_multinomial)
print("\nMultinomialNB - F1-Score:", f1_multinomial)

# Step 5: Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_multinomial, annot=True, fmt="d", cmap="Greens", xticklabels=["Non Spam", "Spam"], yticklabels=["Non Spam", "Spam"])
plt.title("Confusion Matrix - MultinomialNB")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

# Step 6: Precision-Recall Curve
y_prob_multinomial = multinomial_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_prob_multinomial)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"Precision-Recall Curve (AUC = {pr_auc:.2f})")
plt.title("Precision-Recall Curve - MultinomialNB")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.show()







