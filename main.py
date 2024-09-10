import pandas as pd
import logging
import json
from functions import load_excel, generate_email, save_to_file, log_special_character_names, merge_and_shuffle, \
    save_as_json
from functions import split_name, generate_gender_lists, compute_name_similarity
import constraints as const

# Set up logging
logging.basicConfig(filename=const.LOG_FILE_PATH, level=logging.INFO)


def log_student_counts(df, gender_column='Gender'):
    """
    Log the number of male and female students in the dataset.
    :param df: DataFrame of students.
    :param gender_column: Column that stores gender info.
    """
    male_count = len(df[df[gender_column] == 'M'])
    female_count = len(df[df[gender_column] == 'F'])

    logging.info(f"Number of Male Students: {male_count}")
    logging.info(f"Number of Female Students: {female_count}")
    return male_count, female_count


def main():
    print("Starting the program...")  # Debugging output

    # Load the Excel sheets
    print("Loading Excel sheets...")
    df_file_a = load_excel(const.EXCEL_FILE_PATH, const.SHEET_FILE_A)
    df_file_b = load_excel(const.EXCEL_FILE_PATH, const.SHEET_FILE_B)

    print("Splitting names...")
    # Split the Student Name into First and Last Name
    df_file_a[['Last Name', 'First Name']] = df_file_a['Student Name'].apply(lambda name: pd.Series(split_name(name)))
    df_file_b[['Last Name', 'First Name']] = df_file_b['Student Name'].apply(lambda name: pd.Series(split_name(name)))

    print("Generating emails...")
    # Generate emails and ensure uniqueness
    used_emails = set()  # Track unique emails
    df_file_a['Email'] = df_file_a.apply(lambda row: generate_email(row['First Name'], row['Last Name'], used_emails),
                                         axis=1)
    df_file_b['Email'] = df_file_b.apply(lambda row: generate_email(row['First Name'], row['Last Name'], used_emails),
                                         axis=1)

    print("Generating gender lists...")
    # Generate separate lists of male and female students
    male_students_a, female_students_a = generate_gender_lists(df_file_a)
    male_students_b, female_students_b = generate_gender_lists(df_file_b)

    print("Computing name similarities...")
    # Combine male and female names for similarity comparison
    male_names = male_students_a['First Name'].tolist() + male_students_b['First Name'].tolist()
    female_names = female_students_a['First Name'].tolist() + female_students_b['First Name'].tolist()

    if not male_names or not female_names:
        print("No male or female names found!")  # Debugging output
        return

    # Compute similarity between male and female names
    similarity_matrix = compute_name_similarity(male_names, female_names)
    print("Similarity Matrix:\n", similarity_matrix)

    # Save similarity results
    print("Saving name similarity to JSON...")
    with open('name_similarity.json', 'w') as f:
        similarity_results = {
            "male_names": male_names,
            "female_names": female_names,
            "similarity_matrix": similarity_matrix.tolist()
        }
        json.dump(similarity_results, f)
        print("name_similarity.json has been saved.")

    # Log student counts for both sheets
    log_student_counts(df_file_a)
    log_student_counts(df_file_b)

    # Log special character names for both sheets
    log_special_character_names(df_file_a, logging)
    log_special_character_names(df_file_b, logging)

    # Save the updated data to CSV and TSV
    save_to_file(df_file_a, 'file_a', 'csv')
    save_to_file(df_file_b, 'file_b', 'csv')
    save_to_file(df_file_a, 'file_a', 'tsv')
    save_to_file(df_file_b, 'file_b', 'tsv')

    # Merge, shuffle, and save as JSON
    merged_df = merge_and_shuffle(df_file_a, df_file_b)
    save_as_json(merged_df, 'merged_students')

    print("Processing complete!")


if __name__ == '__main__':
    main()
