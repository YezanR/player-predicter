from sklearn.externals import joblib
import pandas 
from data_transform import transformTestData, extractFeatures
from sklearn.model_selection import train_test_split


# Load the model we trained previously
model = joblib.load('player_classifier_model.pkl')

# scikit-learn assumes you want to predict the values for lots of houses at once, so it expects an array. 
df = pandas.read_csv("test_data.csv")

transformed_df = transformTestData(df)

features_df = extractFeatures(transformed_df)

feature_names = joblib.load('features.csv')

# add missing columns
for col in feature_names:
    if col not in features_df.columns:
        features_df[col] = 0

# reorder columns 
features_df = features_df[feature_names]

players_to_value = features_df.values

# Run the model and make a prediction for each house in the homes_to_value array
predicted_player_values = model.predict(players_to_value)

print("\n---------------------------------------------------")
print("\nThese players have the following estimated values: \n")
for i in range(len(predicted_player_values)):
    predicted_value = predicted_player_values[i]
    player_name = df.loc[[i]].Name.values[0]
    print(player_name + ":\t\t ${:,.2f}".format(predicted_value))

print("\n---------------------------------------------------\n")
    