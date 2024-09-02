# E-Commerce-Customer-Segmentation-and-Recommendation-System
Developed a system using Python, machine learning, and web development to segment customers and provide personalized recommendations.

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('customer_data.csv')

# Preprocessing
features = ['age', 'annual_income', 'spending_score']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.scatter(data['annual_income'], data['spending_score'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()

# Save the segmented data
data.to_csv('segmented_customer_data.csv', index=False)
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load your data
data = pd.read_csv('user_item_interactions.csv')

# Create a user-item matrix
user_item_matrix = data.pivot(index='user_id', columns='item_id', values='interaction').fillna(0)

# Compute cosine similarity
similarity_matrix = cosine_similarity(user_item_matrix)

# Convert to DataFrame for better readability
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to get recommendations for a user
def get_recommendations(user_id, num_recommendations=5):
    similar_users = similarity_df[user_id].sort_values(ascending=False).index[1:]  # Exclude the user itself
    recommendations = user_item_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)
    recommendations = recommendations[recommendations > 0]  # Filter out items that the user has already interacted with
    return recommendations.head(num_recommendations)

# Example usage
user_id = 1
print(f"Recommendations for User {user_id}:")
print(get_recommendations(user_id))
