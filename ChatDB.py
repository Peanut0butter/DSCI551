import os
import pymysql
from pymongo import MongoClient
import pandas as pd
import json
import random
import re
from tabulate import tabulate


class ChatDB:
    def __init__(self):
        self.mysql_connection = None
        self.mongodb_client = None
        self.table_metadata = {}  # MySQL table metadata
        self.collection_metadata = {}  # MongoDB collection metada
        self.current_database = None
        self.current_db_type = None
        self.running = True
        self.keyword_map = {
            "maximum": self.generate_max_query,
            "minimum": self.generate_min_query,
            "average": self.generate_avg_query,
            "sum": self.generate_sum_query,
            "count": self.generate_cnt_query,
            "top": self.generate_top_query,
        }

    def display_menu(self):
        print("\nChatDB Main Menu:")
        print("[1] Upload Datasets")
        print("[2] View Tables/Collections")
        print("[3] Query Database")
        print("[4] Help")
        print("[0] Exit")

    def handle_input(self):
        user_input = input("\nSelect an option: ").strip().lower()

        if user_input in {"help", "h"}:
            self.help_menu()
        elif user_input in {"exit", "quit", "q"}:
            self.exit_program()
        elif user_input == "1":
            self.upload_datasets()
        elif user_input == "2":
            self.view_database_details()
        elif user_input == "3":
            self.query_database()
        elif user_input == "4":
            self.help_menu()
        elif user_input == "0":
            self.exit_program()
        else:
            print("Invalid option. Please choose a valid menu option.")

    def connect_to_mysql(self, host="localhost", user="root", password="", database=None):
        try:
            self.mysql_connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            print(f"Connected to MySQL database '{database}'.")
            return True
        except Exception as e:
            print(f"Error connecting to MySQL: {e}")
            return False

    def connect_to_mongodb(self, host="localhost", port=27017, database=None):
        try:
            self.mongodb_client = MongoClient(host, port)
            print(f"Connected to MongoDB database '{database}'.")
            return True
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return False

    def upload_datasets(self):
        files = input("\nEnter the paths to your dataset files (separated by spaces): ").strip().split()
        files = [file.strip() for file in files]

        if not files or all(not os.path.exists(file) for file in files):
            print("No valid files provided. Please check the file paths and try again.")
            return

        db_type = None
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"Error: File '{file_path}' not found. Skipping.")
                continue

            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == ".csv":
                if db_type is None:
                    db_type = "mysql"
            elif file_extension == ".json":
                if db_type is None:
                    db_type = "mongodb"
            else:
                print(f"Error: Unsupported file type for '{file_path}'. Skipping.")
                continue

        if db_type == "mysql":
            self.setup_mysql()
        elif db_type == "mongodb":
            self.setup_mongodb()

        for file_path in files:
            file_extension = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path).split('.')[0]

            if db_type == "mysql" and file_extension == ".csv":
                self.process_mysql_file(file_path, file_name)
            elif db_type == "mongodb" and file_extension == ".json":
                self.process_mongodb_file(file_path, file_name)

    def populate_mysql_metadata(self):
        if self.current_db_type == "mysql":
            try:
                cursor = self.mysql_connection.cursor()
                cursor.execute("SHOW TABLES;")
                tables = [table[0] for table in cursor.fetchall()]

                for table in tables:
                    cursor.execute(f"DESCRIBE {table};")
                    columns = [col[0] for col in cursor.fetchall()]
                    self.table_metadata[table] = columns

                print("MySQL metadata populated.")
            except Exception as e:
                print(f"Error populating MySQL metadata: {e}")

    def populate_mongodb_metadata(self):
        if self.current_db_type == "mongodb":
            try:
                db = self.mongodb_client[self.current_database]
                collections = db.list_collection_names()

                for collection in collections:
                    sample_doc = db[collection].find_one()
                    if sample_doc:
                        self.collection_metadata[collection] = sample_doc.keys()

                print("MongoDB metadata populated.")
            except Exception as e:
                print(f"Error populating MongoDB metadata: {e}")



    def setup_mysql(self):
        print("Recommended database: MySQL (CSV files detected).")
        if not self.mysql_connection:
            host = input("Enter MySQL host (default: localhost): ").strip() or "localhost"
            user = input("Enter MySQL user (default: root): ").strip() or "root"
            password = input("Enter MySQL password: ").strip()
            if not self.connect_to_mysql(host, user, password):
                return

        cursor = self.mysql_connection.cursor()
        cursor.execute("SHOW DATABASES;")
        databases = [db[0] for db in cursor.fetchall()]
        print("\nExisting Databases:")
        for db in databases:
            print(f"- {db}")

        database = input("Enter database name to create/select: ").strip()
        if database not in databases:
            cursor.execute(f"CREATE DATABASE {database};")
            print(f"Database '{database}' created.")
        cursor.execute(f"USE {database};")
        self.current_database = database
        self.current_db_type = "mysql"
        self.populate_mysql_metadata()

    def setup_mongodb(self):
        print("Recommended database: MongoDB (JSON files detected).")
        if not self.mongodb_client:
            host = input("Enter MongoDB host (default: localhost): ").strip() or "localhost"
            port = int(input("Enter MongoDB port (default: 27017): ").strip() or "27017")
            if not self.connect_to_mongodb(host, port):
                return

        databases = self.mongodb_client.list_database_names()
        print("\nExisting Databases:")
        for db in databases:
            print(f"- {db}")

        database = input("Enter database name to create/select: ").strip()
        self.current_database = database
        self.current_db_type = "mongodb"
        self.populate_mongodb_metadata()


    def process_mysql_file(self, file_path, file_name):
        cursor = self.mysql_connection.cursor()
        data = pd.read_csv(file_path)
        table_name = file_name
        cols = ", ".join(data.columns)
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} VARCHAR(255)' for col in data.columns])});")
        for _, row in data.iterrows():
            values = "', '".join(map(str, row.values))
            cursor.execute(f"INSERT INTO {table_name} ({cols}) VALUES ('{values}');")
        self.mysql_connection.commit()
        print(f"File '{file_name}' uploaded as table '{table_name}' in MySQL database '{self.current_database}'.")

    def process_mongodb_file(self, file_path, file_name):
        db = self.mongodb_client[self.current_database]
        with open(file_path, 'r') as file:
            data = json.load(file)
        collection = db[file_name]
        collection.insert_many(data if isinstance(data, list) else [data])
        print(f"File '{file_name}' uploaded as collection in MongoDB database '{self.current_database}'.")

    def view_database_details(self):
        """
        Displays the details of the current database, such as tables/collections
        and a preview of the first few rows/documents in each.
        """
        if self.current_db_type == "mysql":
            try:
                cursor = self.mysql_connection.cursor()
                cursor.execute("SHOW TABLES;")
                tables = [table[0] for table in cursor.fetchall()]
                print(f"\nTables in MySQL Database '{self.current_database}':")
                for table in tables:
                    print(f" - {table}")
                    # Preview the first 5 rows
                    cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
                    rows = cursor.fetchall()
                    column_names = [col[0] for col in cursor.description]
                    print(f"Preview of {table}:")
                    print(tabulate(rows, headers=column_names, tablefmt="grid"))
            except Exception as e:
                print(f"Error fetching MySQL database details: {e}")

        elif self.current_db_type == "mongodb":
            try:
                db = self.mongodb_client[self.current_database]
                collections = db.list_collection_names()
                print(f"\nCollections in MongoDB Database '{self.current_database}':")
                for collection in collections:
                    print(f" - {collection}")
                    # Preview the first 5 documents
                    docs = list(db[collection].find().limit(5))
                    if docs:
                        fields = list(docs[0].keys())
                        print(f"Preview of {collection}:")
                        print(fields)
                        for doc in docs:
                            print(doc)
                    else:
                        print(f"Collection '{collection}' is empty.")
            except Exception as e:
                print(f"Error fetching MongoDB database details: {e}")

        else:
            print("No database is currently selected.")


    def query_database(self):
        """
        Handles querying the database with two options:
        1. Generate and display 5 example queries (at least 2 with GROUP BY).
        2. Accept natural language custom queries from the user.
        """
        print("\nQuery Options:")
        print("[1] Generate Example Queries")
        print("[2] Type Your Custom Query")
        option = input("Choose an option (1/2): ").strip()

        if option == "1":
            self.generate_example_queries()
        elif option == "2":
            self.handle_custom_query()
        else:
            print("Invalid option. Please choose either 1 or 2.")

    def generate_example_queries(self,groupby_only=False):
        """
        Dynamically generates and displays 5 sample queries.
        Ensures at least 2 queries use GROUP BY combined with aggregation functions.
        """
        groupby_queries = 0  # Track the number of GROUP BY queries generated

        print("\nGenerated Example Queries:")

        for i in range(5):
            if self.current_db_type == "mysql":
                cursor = self.mysql_connection.cursor()
                cursor.execute("SHOW TABLES;")
                tables = [table[0] for table in cursor.fetchall()]
                if not tables:
                    print("No tables found in the current MySQL database.")
                    return

                table = random.choice(tables) 
                cursor.execute(f"DESCRIBE {table};")
                columns_info = cursor.fetchall()
                columns = [col[0] for col in columns_info]

                numeric_columns = [col[0] for col in columns_info if "int" in col[1].lower() or "float" in col[1].lower()]
                category_columns = [col[0] for col in columns_info if "var" in col[1].lower()]

                primary_key = [col[0] for col in columns_info if col[3].lower() == "pri"]

                weighted_columns = []
                if primary_key:
                    weighted_columns.extend(primary_key * 3) 
                weighted_columns.extend(category_columns)
                weighted_columns.extend(columns)
 
                if weighted_columns:
                    group_column = random.choice(weighted_columns)
                else:
                    print(f"No suitable columns found in table {table}. Skipping...")
                    continue

                if not columns:
                    continue

                if groupby_only or groupby_queries < 2:  # Prioritize GROUP BY for the first two queries
                    query_type = "GROUP BY"
                else:
                    query_type = random.choice(["MAX", "SUM", "AVG", "COUNT", "GROUP BY"])

                if query_type == "GROUP BY":
                    group_column = random.choice(category_columns)
                    agg_column = random.choice(numeric_columns) if numeric_columns else None
                    agg_function = random.choice(["SUM", "MAX", "AVG", "COUNT"])

                    if agg_column:
                        query = f"SELECT {group_column}, {agg_function}({agg_column}) AS {agg_function.lower()}_value FROM {table} GROUP BY {group_column};"                         
                        if agg_function == "SUM":
                            explanation = f"Total '{agg_column}' each '{group_column}' in '{table}'."
                        if agg_function == "MAX":
                            explanation = f"Maximum '{agg_column}' each '{group_column}' in '{table}'."
                        if agg_function == "AVG":
                            explanation = f"Average '{agg_column}' each '{group_column}' in '{table}'."
                        if agg_function == "COUNT":
                            explanation = f"Number of '{agg_column}' each '{group_column}' in '{table}'."

                    else:
                        query = f"SELECT {group_column}, COUNT(*) AS count FROM `{table}` GROUP BY {group_column}"
                        explanation = f"Number of records each '{group_column}' in '{table}'."

                    groupby_queries += 1
               
                elif query_type in ["MAX","SUM","AVG"]:
                    if not numeric_columns:
                        continue
                    column = random.choice(numeric_columns)
                    query = f"SELECT {query_type}({column}) AS {query_type.lower()}_{column} FROM `{table}`;"                
                    explanation = f"{query_type} of column '{column} in '{table}'."

                elif query_type == "COUNT":
                    column = random.choice(columns)
                    query = f"SELECT {column}, COUNT(*) AS count FROM `{table}` GROUP BY {column};"
                    explanation = f"Number of records each '{column}' in '{table}'."
                else:
                    continue

                print(f"\nExplanation: {explanation}")
                print(f"Query: {query}")
                self.execute_mysql_query(query, result_key="example_output")

            elif self.current_db_type == "mongodb":
                db = self.mongodb_client[self.current_database]
                collections = db.list_collection_names()
                if not collections:
                    print("No collections found in the current MongoDB database.")
                    return

                collection = random.choice(collections)
                sample_doc = db[collection].find_one()
                if not sample_doc:
                    continue

                fields = list(sample_doc.keys())
                numeric_fields = [key for key, value in sample_doc.items() if isinstance(value, (int, float))]
                if not fields:
                    continue

                if groupby_only or groupby_queries < 2:  # Prioritize GROUP BY for the first two queries
                    query_type = "GROUP BY"
                else:
                    query_type = random.choice(["MAX", "SUM", "AVG", "COUNT", "GROUP BY"])

                if query_type == "GROUP BY":
                    group_field = random.choice(fields)
                    agg_field = random.choice(numeric_fields) if numeric_fields else None
                    agg_function = random.choice(["$sum", "$max", "$avg", "$count"])

                    if agg_field:
                        pipeline = [
                            {"$group": {"_id": f"${group_field}", f"{agg_function[1:]}_value": {agg_function: f"${agg_field}"}}},
                            {"$limit": 5}
                        ]
                        explanation = f"{agg_function[1:]} of '{agg_field}' by '{group_field}' in the '{collection}' collection."
                    else:
                        pipeline = [
                            {"$group": {"_id": f"${group_field}", "count": {"$sum": 1}}},
                            {"$limit": 5}
                        ]
                        explanation = f"Number od records each '{group_field}' in the '{collection}' collection ."
                    groupby_queries += 1

                elif query_type in ["MAX","SUM","AVG"] and numeric_fields:
                    field = random.choice(numeric_fields)
                    if query_type == "MAX":
                        pipeline = [{"$group": {"_id": None, "max_value": {"$max": f"${field}"}}}]
                        explanation = f"Maximum of '{field}' in the '{collection}' collection."
                    elif query_type == "SUM":
                        pipeline = [{"$group": {"_id": None, "total_sum": {"$sum": f"${field}"}}}]
                        explanation = f"Total '{field}' in the '{collection}' collection."
                    elif query_type == "AVG":
                        pipeline = [{"$group": {"_id": None, "avg_value": {"$avg": f"${field}"}}}]
                        explanation = f"Average '{field}' in the '{collection}' collection."
                elif query_type == "COUNT":
                    pipeline = [{"$group": {"_id": None, "row_count": {"$sum": 1}}}]
                    explanation = f"Number of records in the '{collection}' collection."
                else:
                    continue

                print(f"\nExplanation: {explanation}")
                print(f"Pipeline:\n{json.dumps(pipeline, indent=2)}")
                self.execute_mongodb_query(collection, pipeline, result_key="example_output")



    def handle_custom_query(self):
        """
        Handles custom user queries by interpreting natural language inputs, including filtering conditions.
        """
        user_query = input("\nType your query (e.g., 'total sales by day in LA', 'average grade by student where semester is Fall 2023'): ").strip().lower()

        # Check for examples of groupby or join
        if "examples of groupby" in user_query:
            self.generate_groupby_examples()
            return
        elif "examples of join" in user_query:
            self.generate_join_examples()
            return

        tokens = user_query.split()

        # Parse query intent (e.g., aggregation A by B)
        agg_function, agg_column, group_column, condition_column, condition_value = self.parse_query_intent(tokens)

        if not agg_function or not agg_column or not group_column:
            print("Could not parse the query. Please check your input.")
            return
        
        # Match columns and tables
        matched_columns, matched_tables = self.find_matching_columns_and_tables([agg_column, group_column])
        
        if len(matched_tables) == 1:
            # Single table query
            self.create_single_table_query(agg_function,agg_column, group_column, condition_column, condition_value, matched_tables[0])
        elif len(matched_tables) >= 2:
            # Two-table join query
            self.create_join_query(agg_function, agg_column, group_column, condition_column, condition_value, matched_tables)
        else:
            print("Could not determine appropriate tables for the query. Please refine it.")


    def parse_query_intent(self, tokens):
        """
        Parses user input to identify the aggregation function, group column, and filter.
        """
        agg_function_map = {
            "total": "SUM",
            "max": "MAX",
            "average": "AVG",
            "count": "COUNT",
        }

        agg_function = None
        agg_column = None
        group_column = None
        condition_column = None
        condition_value = None

        for i, token in enumerate(tokens):
            if token in agg_function_map:
                agg_function = agg_function_map[token]
                if i + 1 <len(tokens):
                    agg_column = tokens[i+1]
            if token == "by" and i + 1 < len(tokens):
                group_column = tokens[i + 1]
            if token == "in" and i + 1 < len(tokens):
                condition_value = tokens[i+1]
                if i >0:
                    condition_column = tokens[i - 1]
       
        return agg_function,agg_column, group_column, condition_column, condition_value

    def find_matching_columns_and_tables(self, columns):
        """
        Finds which tables contain the given columns.
        """
        matched_columns = {}
        matched_tables = set()

        if self.current_db_type == "mysql":
            cursor = self.mysql_connection.cursor()
            
            # Get a list of all tables
            cursor.execute("SHOW TABLES;")
            tables = [row[0].lower() for row in cursor.fetchall()]
            
            for table in tables:
                # Get columns for each table
                cursor.execute(f"DESCRIBE {table};")
                table_columns = [row[0].lower()  for row in cursor.fetchall()]
                
                # Check if all columns exist in a single table
                for column in columns:
                    if column in table_columns:
                        if column not in matched_columns:
                            matched_columns[column] = []
                        matched_columns[column].append(table)
                        matched_tables.add(table)
                
        # Check if all columns exist in the current table
        if all(column in table_columns for column in columns):
            print(f"All columns found in a single table: {table}")
            return {column: [table] for column in columns}, [table]

        elif self.current_db_type == "mongodb":
            for column in columns:
                for collection, collection_fields in self.collection_metadata.items():
                    if column in collection_fields:
                        if column not in matched_tables:
                            matched_columns[column] = []
                        matched_columns[column].append(collection)
                        matched_tables.add(collection)
            for collection, collection_fields in self.collection_metadata.items():
                if all(column in collection_fields for column in columns):
                    return {column: [collection] for column in columns}, [collection]

            # Get a list of all tables
            cursor.execute("SHOW TABLES;")
            tables = [row[0] for row in cursor.fetchall()]
            print(tables)
            for table in tables:
                # Get columns for each table
                cursor.execute(f"DESCRIBE {table};")
                table_columns = [row[0] for row in cursor.fetchall()]
                print(table_columns)
                # Check if all columns exist in a single table
                for column in columns:
                    if column in table_columns:
                        if column not in matched_columns:
                            matched_columns[column] = []
                        matched_columns[column].append(table)
                        matched_tables.add(table)
                print(matched_tables)
        return matched_columns, list(matched_tables)

    def generate_groupby_examples(self):
        """
        Generates three examples of groupby queries with aggregation functions.
        Ensures at least one example for MySQL and MongoDB if both are active.
        """
        if self.current_db_type == "mysql":
            print("\nMySQL GroupBy Examples:")
            self.generate_example_queries(groupby_only=True)
        elif self.current_db_type == "mongodb":
            print("\nMongoDB GroupBy Examples:")
            self.generate_example_queries(groupby_only=True)


    def generate_join_examples(self):
        """
        Generates examples of join queries by identifying common columns in multiple tables or collections.
        """
        if self.current_db_type == "mysql":
            print("\nMySQL Join Examples:")
            cursor = self.mysql_connection.cursor()
            cursor.execute("SHOW TABLES;")
            tables = [table[0] for table in cursor.fetchall()]

            common_columns = self.find_common_columns_mysql(tables)
            if not common_columns:
                print("No common columns found across tables for JOIN examples.")
                return

            for i in range(3):  # Generate 3 join examples
                table1, table2, column = random.choice(common_columns)
                query = f"""
                SELECT t1.*, t2.*
                FROM {table1} AS t1
                JOIN {table2} AS t2
                ON t1.{column} = t2.{column};
                """
                explanation = f"Join '{table1}' and '{table2}' on the common column '{column}'."
                print(f"\nExplanation: {explanation}")
                print(f"Query: {query}")
                self.execute_mysql_query(query, result_key="join_example")

        elif self.current_db_type == "mongodb":
            print("\nMongoDB Join Examples:")
            db = self.mongodb_client[self.current_database]
            collections = db.list_collection_names()

            common_columns = self.find_common_columns_mongodb(collections)
            if not common_columns:
                print("No common fields found across collections for JOIN examples.")
                return

            for i in range(3):  # Generate 3 join examples
                coll1, coll2, field = random.choice(common_columns)
                pipeline = [
                    {
                        "$lookup": {
                            "from": coll2,
                            "localField": field,
                            "foreignField": field,
                            "as": f"{coll2}_details"
                        }
                    },
                    {"$limit": 5}
                ]
                explanation = f"Join '{coll1}' and '{coll2}' using the common field '{field}'."
                print(f"\nExplanation: {explanation}")
                print(f"Pipeline:\n{json.dumps(pipeline, indent=2)}")
                self.execute_mongodb_query(coll1, pipeline, result_key="join_example")

    def find_common_columns_mysql(self, tables):
        """
        Finds common columns across MySQL tables to generate JOIN queries.
        """
        cursor = self.mysql_connection.cursor()
        column_map = {}

        for table in tables:
            cursor.execute(f"DESCRIBE {table};")
            columns = [col[0] for col in cursor.fetchall()]
            for column in columns:
                if column not in column_map:
                    column_map[column] = []
                column_map[column].append(table)

        common_columns = [
            (table1, table2, column)
            for column, tables in column_map.items()
            if len(tables) > 1
            for table1 in tables
            for table2 in tables
            if table1 != table2
        ]
        return common_columns

    def find_common_columns_mongodb(self, collections):
        """
        Finds common fields across MongoDB collections to generate JOIN examples.
        """
        db = self.mongodb_client[self.current_database]
        field_map = {}

        for coll in collections:
            sample_doc = db[coll].find_one()
            if not sample_doc:
                continue

            for field in sample_doc.keys():
                if field not in field_map:
                    field_map[field] = []
                field_map[field].append(coll)

        common_fields = [
            (coll1, coll2, field)
            for field, colls in field_map.items()
            if len(colls) > 1
            for coll1 in colls
            for coll2 in colls
            if coll1 != coll2
        ]
        return common_fields

    def create_join_query(self, agg_function, agg_column, group_column, condition_column, condition_value, tables):
        """
        Creates and executes a join query for both MySQL and MongoDB.
        """
        table_a, table_b = tables
        join_column = self.find_relationship_column(table_a, table_b)

        if not join_column:
            print(f"Could not determine a join relationship between {table_a} and {table_b}.")
            return

        if self.current_db_type == "mysql":
            # Build SQL join query
            query = f"""
                SELECT {table_b}.{group_column}, {agg_function}({table_a}.{agg_column}) AS {agg_function.lower()}_{agg_column}
                FROM {table_a}
                JOIN {table_b} ON {table_a}.{join_column} = {table_b}.{join_column}
            """
            if condition_column and condition_value:
                query += f" WHERE {table_b}.{condition_column} = '{condition_value}'"
            query += f" GROUP BY {table_b}.{group_column}"

            print(f"\nGenerated Query: {query}")
            self.execute_mysql_query(query)

        elif self.current_db_type == "mongodb":
            # MongoDB does not support direct joins; emulate with `$lookup`
            pipeline = [
                {
                    "$lookup": {
                        "from": table_b,
                        "localField": join_column,
                        "foreignField": join_column,
                        "as": "joined_table"
                    }
                },
                {"$unwind": "$joined_table"},
                {"$match": {f"joined_table.{condition_column}": condition_value}} if condition_column and condition_value else {},
                {"$group": {
                    "_id": f"$joined_table.{group_column}",
                    f"{agg_function.lower()}_{agg_column}": {f"${agg_function.lower()}": f"${agg_column}"}
                }}
            ]

            print(f"\nGenerated Aggregation Pipeline: {pipeline}")
            self.execute_mongodb_query(table_a, pipeline)


    def create_single_table_query(self, agg_function, agg_column, group_column, condition_column, condition_value, table_or_collection):
        """
        Generates and executes a query for a single table (MySQL) or collection (MongoDB).
        """
        if self.current_db_type == "mysql":
            # MySQL Query Construction
            select_clause = f" {group_column},{agg_function}({agg_column}) AS {agg_function.lower()}_{group_column}, {group_column}"
            where_clause = f"WHERE {condition_column} = '{condition_value}'" if condition_column and condition_value else ""
            group_by_clause = f"GROUP BY {group_column}"

            query = f"SELECT {select_clause} FROM {table_or_collection} {where_clause} {group_by_clause};"
            print("\nGenerated MySQL Query:")
            print(query)
            self.execute_mysql_query(query, result_key="single_table_result")

        elif self.current_db_type == "mongodb":
            # MongoDB Aggregation Pipeline Construction
            agg_operator_map = {"SUM": "$sum", "MAX": "$max", "AVG": "$avg", "COUNT": {"$sum": 1}}
            pipeline = []

            if condition_column and condition_value:
                    pipeline.append({"$match": {condition_column: condition_value}})

            pipeline.append({
                    "$group": {
                            "_id": f"${group_column}",
                            f"{agg_function.lower()}_{agg_column}": {agg_operator_map[agg_function]: f"${agg_column}"}
                    }
            })

            print("\nGenerated MongoDB Aggregation Pipeline:")
            print(pipeline)
            self.execute_mongodb_query(table_or_collection, pipeline, result_key="single_table_result")
        else:
            print("Unsupported database type.")


    def execute_mysql_query(self, query, result_key="result"):
        """
        Executes a MySQL query and fetches results from the server.
        """
        try:
            with self.mysql_connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                print(f"Results ({result_key}):")
                print(tabulate(result, headers=column_names, tablefmt="grid"))
        except Exception as e:
            print(f"Error executing MySQL query: {e}")

    def execute_mongodb_query(self, collection_name, pipeline, result_key="result"):
        """
        Executes a MongoDB aggregation pipeline and fetches results from the server.
        """
        try:
            db = self.mongodb_client[self.current_database]
            result = list(db[collection_name].aggregate(pipeline))
            print(f"Results ({result_key}):")
            if result:
                field_names = list(result[0].keys())
                print(field_names)
                for doc in result:
                    print(doc)
            else:
                print("No results found.")
        except Exception as e:
            print(f"Error executing MongoDB query: {e}")

    def generate_max_query(self, column, table):
        return f"SELECT MAX({column}) AS max_value FROM {table};"

    def generate_min_query(self, column, table):
        return f"SELECT MIN({column}) AS min_value FROM {table};"

    def generate_sum_query(self, column, table):
        return f"SELECT SUM({column}) AS sum_value FROM {table};"

    def generate_avg_query(self, column, table):
        return f"SELECT AVG({column}) AS avg_value FROM {table};"

    def generate_cnt_query(self, column, table):
        if column.lower() == "rows":
            return f"SELECT COUNT(*) AS row_count FROM {table};"
        else:
            return f"SELECT COUNT({column}) AS count_value FROM {table};"

    def generate_top_query(self, column, table, top_n=5):
        return f"SELECT * FROM {table} ORDER BY {column} DESC LIMIT {top_n};"

    def find_relationship_column(self, table_a, table_b):
        """
        Finds a common column (MySQL) or field (MongoDB) between two tables/collections.
        """
        if self.current_db_type == "mysql":
            # For MySQL
            try:
                columns_a = self.table_metadata[table_a]
                columns_b = self.table_metadata[table_b]
                # Find common columns
                common_columns = set(columns_a) & set(columns_b)
                if common_columns:
                    # Return the first common column as the relationship
                    return list(common_columns)[0]
                else:
                    print(f"No common columns found between {table_a} and {table_b}.")
                    return None
            except KeyError as e:
                print(f"Error accessing metadata for MySQL tables: {e}")
                return None

        elif self.current_db_type == "mongodb":
            # For MongoDB
            try:
                # Find the schema of the collections
                schema_a = self.collection_metadata.get(table_a, {})
                schema_b = self.collection_metadata.get(table_b, {})
                # Find common fields
                common_fields = set(schema_a.keys()) & set(schema_b.keys())
                if common_fields:
                    # Return the first common field as the relationship
                    return list(common_fields)[0]
                else:
                    print(f"No common fields found between {table_a} and {table_b}.")
                    return None
            except Exception as e:
                print(f"Error accessing metadata for MongoDB collections: {e}")
                return None

        else:
            print("Unsupported database type for finding relationships.")
            return None


    def help_menu(self):
        """
        Displays a help menu with a list of allowed questions or operations supported by the program.
        """
        print("\nHelp Menu: Allowed Questions")
        print("================================")
        print("1. What tables/collections exist in the database?")
        print("   - Choose option 2 in display menu.'")
        print("\n2. What data is in a specific table/collection?")
        print("   - Choose option 2 in display menu.'")
        print("\n3. Perform aggregation queries:")
        print( "   - Choose option 3, then choose option 2 in display menu.'")
        print("   a. MySQL:")
        print("      - Group data by a specific column and apply aggregation.")
        print("      - Example: 'Group enrollments by semester and count rows.'")
        print("   b. MongoDB:")
        print("      - Use aggregation pipelines to group and summarize data.")
        print("      - Example: 'Show the total sales per month.'")
        print("\n4. Basic statistical queries:")
        print("   - Calculate MAX, MIN, AVG, SUM, or COUNT for specific columns.")
        print("   - Example: 'Find the average grade in the `grades` table.'")
        print("\n5. Execute custom SQL or MongoDB queries:")
        print("   - Provide your own query or aggregation pipeline for execution.")
        print("\n6. Ask the program to generate sample queries.")
        print("   - Choose option 3, then option 1 in display menu.'")
        print("\n7. How to connect to a database?")
        print("   - Choose option 1.'")
        print("\nNote: for MySQL, passoword is required.")

    def run(self):
        """
        Runs the ChatDB main program loop.
        """
        print("Welcome to ChatDB!")
        while self.running:
            self.display_menu()
            self.handle_input()

    def exit_program(self):
        print("Exiting ChatDB. Goodbye!")
        self.running = False


if __name__ == "__main__":
    chatdb = ChatDB()
    chatdb.run()
