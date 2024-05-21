import pandas as pd
import openai
import os

# Set your OpenAI API key
openai.api_key = ""

# Load the CSV file provided by the user
file_path = "/home/nilu/RAW_interactions.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Sample 20 reviews from the dataset
sampled_reviews = df.sample(n=20, random_state=1) if len(df) >= 20 else df

def get_absa_tuples(review):
    prompt = f"""
    According to the following sentiment elements definition:
    - The 'aspect term' refers to a specific feature, attribute, or aspect of a product or service that a user may express an opinion about.
    - The 'opinion term' refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service.
    - The 'sentiment polarity' refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities includes: 'positive', 'negative' and 'neutral'.

    Recognize all sentiment elements with their corresponding aspect terms, opinion terms and sentiment polarity in the following text with the format of [('aspect term', 'opinion term', 'sentiment polarity'), ...]: 

    Text: {review}
    Sentiment Elements: 
    """
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

# Get ABSA tuples for each sampled review
sampled_reviews["predicted_tuples"] = sampled_reviews["review"].apply(get_absa_tuples)

# Select the relevant columns
output_df = sampled_reviews[["user_id", "recipe_id", "review", "predicted_tuples"]]

# Save the results to a CSV file
output_file = "/home/nilu/predicted_absa_tuples.csv"
output_df.to_csv(output_file, index=False)

print(f"Predicted ABSA tuples saved to {output_file}")
