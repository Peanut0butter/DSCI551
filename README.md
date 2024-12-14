
# ChatDB

ChatDB is a Python-based tool designed to simplify interactions with SQL databases using natural language inputs and automated SQL query generation. It provides functionality to explore database schemas, generate example `JOIN` queries, and execute SQL operations, with tabular visualization of results.

---

## Features

- **SQL Query Automation**: Generates SQL queries based on user inputs for common operations like `SELECT`, `WHERE`, `GROUP BY`, and `JOIN`.
- **Database Schema Analysis**:
  - Identifies tables and their columns in the database.
  - Matches columns across tables to generate example `JOIN` queries.
- **Result Visualization**: Uses the `tabulate` library to display query results in SQL-like tables for improved readability.
- **Support for MySQL Databases**: Interacts with MySQL databases using `PyMySQL`.
- **Interactive User Interface**: Command-line interface for seamless user interaction.

---

## Requirements

Make sure you have the following installed on your system:

- Python 3.x
- Required Python libraries:
  - `pymysql`
  - `pandas`
  - `tabulate`
  - `json`
  - `re`

You can install the dependencies using the following command:

```bash
pip install pymysql pandas tabulate
```

---

## Setup

### Clone the Repository:
```bash
git clone <repository-url>
cd ChatDB
```

### Configure MySQL Connection (Or finish this step during running the code):
Update the `ChatDB.py` file with your MySQL database connection details in the initialization section of the code:
```python
self.mysql_connection = pymysql.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database"
)
```

### Run the Application:
```bash
python3 ChatDB.py
```

---

## Usage

1. **Start ChatDB**: Run the script and follow the interactive prompts in the command-line interface.

2. **Features**:
   - **Upload Datasets**: Load datasets into your MySQL database for querying.
   - **View Tables/Columns**: Explore the database schema to understand available tables and columns.
   - **Query Database**:
     - Generate example queries.
     - Type custom SQL queries.
   - **Help**: Access documentation and usage instructions.
   - **Exit**: Terminate the program.

3. **Example Workflow**:
   - Start the program.
   - Use the "Upload Datasets" feature to load your data.
   - Use "View Tables/Collections" to check your database structure.
   - Generate `JOIN` examples or write your custom queries to explore and analyze the data.

---

## License

This project is licensed under the MIT license.
