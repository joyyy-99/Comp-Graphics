import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import torch


def load_excel(file_path, sheet_name):
    """
    Load a specific sheet from an Excel file.
    :param file_path: Path to the Excel file.
    :param sheet_name: Sheet name to load.
    :return: DataFrame with the loaded data.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)


def split_name(full_name):
    """
    Split a full name into last name and first name.
    :param full_name: Name in the format 'Last, First Middle'
    :return: last_name, first_name
    """
    if pd.isna(full_name):
        return "", ""
    names = full_name.split(',')
    if len(names) == 2:
        last_name = names[0].strip()
        first_name = names[1].strip().split(' ')[0]  # First part of the first name
        return last_name, first_name
    return "", ""  # Return empty if the format is unexpected


def generate_email(first_name, last_name, used_emails):
    """
    Generate a unique email address based on first initial + last name.
    Ensures no special characters and email uniqueness.
    :param first_name: Student's first name.
    :param last_name: Student's last name.
    :param used_emails: Set to track unique emails.
    :return: Unique email string.
    """
    # Remove special characters from the last name
    last_name_clean = re.sub(r'\W+', '', last_name).lower()
    email = f"{first_name[0].lower()}{last_name_clean}@gmail.com"

    # Ensure email uniqueness
    counter = 1
    unique_email = email
    while unique_email in used_emails:
        unique_email = f"{first_name[0].lower()}{last_name_clean}{counter}@gmail.com"
        counter += 1

    used_emails.add(unique_email)
    return unique_email


def save_to_file(df, filename, file_type='csv'):
    """
    Save the DataFrame to a CSV or TSV file.
    :param df: DataFrame to save.
    :param filename: Name of the output file.
    :param file_type: Type of file to save (csv or tsv).
    """
    if file_type == 'csv':
        df.to_csv(f'{filename}.csv', index=False)
    elif file_type == 'tsv':
        df.to_csv(f'{filename}.tsv', sep='\t', index=False)
    print(f"File {filename}.{file_type} saved successfully!")


def log_special_character_names(df, logging):
    """
    Log names that contain special characters.
    :param df: DataFrame to check for special characters.
    :param logging: Logging instance to write to the log file.
    :return: DataFrame of names with special characters.
    """
    special_char_names = df[df['Last Name'].str.contains(r'\W+')]
    for _, row in special_char_names.iterrows():
        logging.info(f"Special Character in Name: {row['First Name']} {row['Last Name']}")
    return special_char_names


def merge_and_shuffle(df_a, df_b):
    """
    Merge two DataFrames, shuffle them, and save as JSON.
    :param df_a: DataFrame from sheet A.
    :param df_b: DataFrame from sheet B.
    :return: Merged and shuffled DataFrame.
    """
    merged_df = pd.concat([df_a, df_b]).sample(frac=1).reset_index(drop=True)
    return merged_df


def save_as_json(df, filename):
    """
    Save a DataFrame to a JSON file.
    :param df: DataFrame to save as JSON.
    :param filename: Output filename.
    """
    df.to_json(f"{filename}.json", orient='records', lines=True)
    print(f"JSON file {filename}.json saved successfully!")


def generate_gender_lists(df):
    """
    Generate two separate lists for male and female students.
    :param df: DataFrame containing student information.
    :return: Two DataFrames - one for male students, one for female students.
    """
    male_students = df[df['Gender'] == 'M']
    female_students = df[df['Gender'] == 'F']

    # Optionally save these lists to CSV files
    male_students.to_csv('male_students.csv', index=False)
    female_students.to_csv('female_students.csv', index=False)

    return male_students, female_students


def compute_name_similarity(male_names, female_names):
    """
    Compute a similarity matrix between male and female names using LaBSE embeddings.
    :param male_names: List of male names.
    :param female_names: List of female names.
    :return: A similarity matrix (torch.Tensor).
    """
    # Load the LaBSE model and tokenizer
    model_name = "sentence-transformers/LaBSE"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and get embeddings for male and female names
    male_inputs = tokenizer(male_names, return_tensors="pt", padding=True, truncation=True)
    female_inputs = tokenizer(female_names, return_tensors="pt", padding=True, truncation=True)

    # Compute embeddings
    with torch.no_grad():
        male_embeddings = model(**male_inputs).pooler_output
        female_embeddings = model(**female_inputs).pooler_output

    # Compute cosine similarity between male and female embeddings
    similarity_matrix = torch.nn.functional.cosine_similarity(male_embeddings.unsqueeze(1),
                                                              female_embeddings.unsqueeze(0), dim=-1)

    return similarity_matrix
