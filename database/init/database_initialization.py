import sqlite3
import os


def load_sql(file_path):
    """
    Loads an SQL query from a file.
    
    Parameters:
    - file_path (str): Path to the SQL file.

    Returns:
    - str: SQL query as a string.
    """
    with open(file_path, 'r') as file:
        return file.read()


def initialize_database(db_path="database/causal_scenarios.db"):
    """
    Initializes the SQLite database and creates a table if it doesn't exist.
    """
    sql_query = load_sql("database/init/query_database_initialization.sql")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)


if __name__ == '__main__':
    proceed = True
    if os.path.exists("database/causal_scenarios.db"):
        input_delete = input("Are you sure you want to delete the existing database? (Yes/No): ")
        if input_delete.lower() == "yes":
            input_delete = input("If you continue, the databases will be deleted. Are you sure? (Yes/No): ")
            if input_delete.lower() == "yes":
                os.remove("database/causal_scenarios.db")
                os.remove("database/test_causal_scenarios.db")
        else:
            proceed = False
    if proceed:
        initialize_database("database/causal_scenarios.db")
        initialize_database("database/test_causal_scenarios.db")




        