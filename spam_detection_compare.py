# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download('stopwords')

# Step 1: Load the Dataset
# Assuming "emails.csv" is in the current working directory
dataset_path = "emails.csv"  # Replace with the correct path if needed
emails_data = pd.read_csv(dataset_path)

# Display a preview of the dataset
print("\nOriginal dataset (first 5 rows):")
print(emails_data.head())

# Display the column names
print("\nColumns in the dataset:")
print(emails_data.columns)

# Step 2: Preprocessing
# Remove unnecessary columns
if "Email No." in emails_data.columns:
    emails_data = emails_data.drop(columns=["Email No."])

# Display the dataset after removing unnecessary columns
print("\nDataset after removing 'Email No.' column (first 5 rows):")
print(emails_data.head())

# Visualize the initial distribution of Spam vs Non-Spam
plt.figure(figsize=(8, 6))
sns.countplot(data=emails_data, x="Prediction")
plt.title("Email Distribution (Spam vs Non Spam) - Before Preprocessing")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks([0, 1], ["Non Spam", "Spam"])
plt.show()



# Remove stopword columns (if they exist)
stop_words = set(stopwords.words('english'))
stopword_columns = [col for col in emails_data.columns if col in stop_words]
if stopword_columns:
    print(f"\nRemoving stopword columns: {stopword_columns}")
    emails_data = emails_data.drop(columns=stopword_columns)

# Step 3: Split Data into Features (X) and Target (y)
X = emails_data.iloc[:, :-1]
y = emails_data["Prediction"]

# Display the updated column list
print("\nColumns after preprocessing:")
print(X.columns)

# Step 4: Train and Evaluate NLTK Naive Bayes Model
# Prepare data for NLTK
nltk_dataset = [(row.to_dict(), label) for (_, row), label in zip(X.iterrows(), y)]

# Split into training and test sets
train_data, test_data = train_test_split(nltk_dataset, test_size=0.2, random_state=42)

# Train the NLTK Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_data)

# Evaluate the NLTK Naive Bayes model
y_true_nltk = [label for (_, label) in test_data]
y_pred_nltk = [classifier.classify(features) for (features, _) in test_data]

conf_matrix_nltk = confusion_matrix(y_true_nltk, y_pred_nltk)
classification_rep_nltk = classification_report(y_true_nltk, y_pred_nltk)
f1_nltk = f1_score(y_true_nltk, y_pred_nltk)

print("\nNLTK Naive Bayes - Confusion Matrix:")
print(conf_matrix_nltk)
print("\nNLTK Naive Bayes - Classification Report:")
print(classification_rep_nltk)
print("\nNLTK Naive Bayes - F1-Score:", f1_nltk)

# Step 5: Train and Evaluate MultinomialNB Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
multinomial_model = MultinomialNB(alpha=1.0)
multinomial_model.fit(X_train, y_train)
y_pred_multinomial = multinomial_model.predict(X_test)

conf_matrix_multinomial = confusion_matrix(y_test, y_pred_multinomial)
classification_rep_multinomial = classification_report(y_test, y_pred_multinomial)
f1_multinomial = f1_score(y_test, y_pred_multinomial)

print("\nMultinomialNB - Confusion Matrix:")
print(conf_matrix_multinomial)
print("\nMultinomialNB - Classification Report:")
print(classification_rep_multinomial)
print("\nMultinomialNB - F1-Score:", f1_multinomial)

# Step 6: Visualize Confusion Matrices
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nltk, annot=True, fmt="d", cmap="Blues", xticklabels=["Non Spam", "Spam"], yticklabels=["Non Spam", "Spam"])
plt.title("Confusion Matrix - NLTK Naive Bayes")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_multinomial, annot=True, fmt="d", cmap="Greens", xticklabels=["Non Spam", "Spam"], yticklabels=["Non Spam", "Spam"])
plt.title("Confusion Matrix - MultinomialNB")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

# Step 7: Compare Results
print("\nComparison of F1-Scores:")
print(f"NLTK Naive Bayes F1-Score: {f1_nltk}")
print(f"MultinomialNB F1-Score: {f1_multinomial}")
