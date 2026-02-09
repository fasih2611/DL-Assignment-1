import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def impute_data(df):
    df['neighbourhood_group'] = df['neighbourhood_group'].fillna(df['neighbourhood_group'].mode()[0])
    df['room_type'] = df['room_type'].fillna(df['room_type'].mode()[0])
    
    df['minimum_nights'] = df['minimum_nights'].fillna(df['minimum_nights'].median())
    df['amenity_score'] = df['amenity_score'].fillna(df['amenity_score'].median())
    df['availability_365'] = df['availability_365'].fillna(df['availability_365'].median())
    
    df['number_of_reviews'] = df['number_of_reviews'].fillna(0)
    return df

train_df = impute_data(train_df)
test_df = impute_data(test_df)

le_neighbourhood = LabelEncoder()
le_room = LabelEncoder()
le_neighbourhood.fit(train_df['neighbourhood_group'])
le_room.fit(train_df['room_type'])

train_df['neighbourhood_group'] = le_neighbourhood.transform(train_df['neighbourhood_group'])
train_df['room_type'] = le_room.transform(train_df['room_type'])

test_df['neighbourhood_group'] = le_neighbourhood.transform(test_df['neighbourhood_group'])
test_df['room_type'] = le_room.transform(test_df['room_type'])

plt.figure(figsize=(15, 10))

# Plot 1: Feature Correlation Heatmap 
plt.subplot(2, 2, 1)
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Correlation Matrix of Features")

# Plot 2: Class Balance
plt.subplot(2, 2, 2)
sns.countplot(x='price_class', data=train_df, palette='viridis', hue='price_class', legend=False)
plt.title("Distribution of Price Classes")

# Plot 3: Amenity Score by Price Class
plt.subplot(2, 2, 3)
sns.boxplot(x='price_class', y='amenity_score', data=train_df)
plt.title("Amenity Score by Price Class")

# Plot 4: Price Class by Neighborhood 
plt.subplot(2, 2, 4)
sns.countplot(x='neighbourhood_group', hue='price_class', data=train_df)
plt.title("Price Class Distribution by Neighborhood (Encoded)")

plt.tight_layout()
plt.show()

# Scale numerical features
scaler = StandardScaler()
feature_cols = ['minimum_nights', 'number_of_reviews', 'amenity_score', 'availability_365'] 
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

train_df.to_csv("train_encoded.csv", index=False)
test_df.to_csv("test_encoded.csv", index=False)

encoding_dict = {
    'neighbourhood_group': dict(enumerate(le_neighbourhood.classes_)),
    'room_type': dict(enumerate(le_room.classes_))
}
with open('encoding_dict.json', 'w') as f:
    json.dump(encoding_dict, f)
