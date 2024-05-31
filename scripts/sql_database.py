import sqlite3
from sqlite3 import Error
import ast
import json

class DatabaseManager:
    def __init__(self, path_db_file, name_table=None):
        if name_table is None:
            name_table = path_db_file.split("\\")[-1].split(".")[0]
        self._name_table = name_table
        self._conn = self._create_connection(path_db_file)
        self._cursor = self._conn.cursor()

    def _create_connection(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)
        return conn

    def execute_query(self, query, params=None):
        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
            self._conn.commit()
        except Error as e:
            print(e)

    def fetch_all_records(self, query, params=None):
        try:
            self.execute_query(query, params)  # Call execute_query method to execute the query
            rows = self._cursor.fetchall()
            return rows
        except Error as e:
            print(e)

    def get_record_structure(self, name_table = None):
        if name_table is None:
            name_table = self._name_table
            
        if ' ' in name_table:
            raise ValueError(f"Invalid table name: {name_table}")
        
        query = f"PRAGMA table_info({name_table})"
        self.execute_query(query)
        columns = self._cursor.fetchall()
        return {column[1]: column[2] for column in columns}

    def delete_record(self, name_table, id):
        if ' ' in name_table:
            raise ValueError(f"Invalid table name: {name_table}")
        
        query = f"DELETE FROM {name_table} WHERE id = ?"
        self.execute_query(query, (id,))
        self._conn.commit()

    def get_all_tables_names(self):
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        self.execute_query(query)
        tables = self._cursor.fetchall()
        return [table[0] for table in tables]

    def close_connection(self):
        self._conn.close()

    def __del__(self):
        self._conn.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self._conn.close()      

class SingleTableManager(DatabaseManager):
    def __init__(self, db_file, name_table: str, table_structure: dict = {"id": "integer PRIMARY KEY", 
                                                                        "data1": "text NOT NULL", 
                                                                        "data2": "integer NOT NULL"}):
        self._name_table = name_table
        self._table_structure = self._process_tabel_struct(table_structure)
        
        super().__init__(db_file, self._name_table)
        self.create_table()

    def get_all_records(self) -> list:
        query = f"SELECT * FROM {self._name_table}"
        data = super().fetch_all_records(query)
        
        data = self._recover_json_data(data)
        return data

    def add_record(self, *args):
        if len(args) != len(self._table_structure) - 1:
            raise ValueError("Number of arguments is not equal to number of columns")
        
        columns = ', '.join(list(self._table_structure.keys())[1:])
        query = f"INSERT INTO {self._name_table} ({columns}) VALUES ({', '.join(['?'] * len(args))})"
        
        values = self._change_input_to_json(args)
        super().execute_query(query, values)

    def delete_record(self, id):
        super().delete_record(self._name_table, id)

    def get_record_structure(self):
        return super().get_record_structure(self._name_table)
    
    def create_table(self):
        columns = ', '.join([f"{column} {data['type']}" for column, data in self._table_structure.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {self._name_table} ({columns})"
        super().execute_query(create_table_query) 
        
    def get_num_of_records(self):
        query = f"SELECT COUNT(*) FROM {self._name_table}"
        return super().fetch_all_records(query)[0][0]
    
    def _process_tabel_struct(self, tabel_struct: dict[str, str]):
        
        struct = {}
        for key, value in tabel_struct.items():
            if "json" in value.lower():
                struct[key] = {'type':value.replace("json", "text").replace("JSON", "text"), 'is_json': True}
            else:
                struct[key] = {'type':value, 'is_json': False}
        
        return struct
    
    def _change_input_to_json(self, input_data):
        
        input_data_new = list(input_data)
        for ind, key in enumerate(self._table_structure, -1):
            if ind == -1: # Skip the first key which is the id
                continue
            
            if self._table_structure[key]['is_json']:
                input_data_new[ind] = json.dumps(input_data_new[ind])
            else:
                input_data_new[ind] = input_data_new[ind]
                
        return input_data_new
    
    def _recover_json_data(self, data):
        new_data = []
        for record in data:
            new_record = []
            for ind, field in enumerate(record):
                if self._table_structure[list(self._table_structure.keys())[ind]]['is_json']:
                    new_record.append(json.loads(field))
                else:
                    new_record.append(field)
            new_data.append(new_record)
        return new_data
    
class TableManagerTextures(SingleTableManager):
    def __init__(self, db_file):
        self._name_table = "texture_data"
        self._table_structure = {
                                 "id": "integer PRIMARY KEY", 
                                 "path_image_origin": "text NOT NULL", 
                                 "path_texture": "text NOT NULL",
                                 "box": "json NOT NULL"
                                 }
        super().__init__(db_file, self._name_table, self._table_structure)
        
    def add_record(self, path_image_origin: str, path_texture: str, box: list[int, int, int, int]):
        return super().add_record(path_image_origin, path_texture, box)
        
class TableManagerAnnotations(SingleTableManager):
    def __init__(self, db_file):
        self._name_table = "annotation_data"
        self._table_structure = {"id": "integer PRIMARY KEY", 
                                 "path_image": "text NOT NULL", 
                                 "boxes": "json NOT NULL"}
        super().__init__(db_file, self._name_table, self._table_structure)
        
    def add_record(self, path_image: str, boxes: list[list[int, int, int, int]]):
        return super().add_record(path_image, boxes)
    
    def get_all_image_paths(self):
        query = f"SELECT path_image FROM {self._name_table}"
        return super().fetch_all_records(query)
    
    def delete_annotations_of_image(self, path_image):
        query = f"DELETE FROM {self._name_table} WHERE path_image = ?"
        super().execute_query(query, (path_image,))
        self._conn.commit()

class TableManagerSegments(SingleTableManager):
    def __init__(self, db_file):
        self._name_table = "subsegments_data"
        self._table_structure = {"id": "integer PRIMARY KEY", 
                                    "path_texture": "text NOT NULL", 
                                    "path_segment": "text NOT NULL",
                                    "box": "json NOT NULL",
                                    "features": "json NOT NULL",
                                    "label": "integer NOT NULL"
                                    }
        super().__init__(db_file, self._name_table, self._table_structure)
        
    def add_record(self, path_texture: str, path_segment: str, box: list[int, int, int, int], features: list[float], label: int):
        return super().add_record(path_texture, path_segment, box, features, label)

class TextureDatabaseManager():
    def __init__(self, path_db_file):
        self.table_texture = TableManagerTextures(path_db_file)
        self.table_annotation = TableManagerAnnotations(path_db_file)
        self.table_subsegment = TableManagerSegments(path_db_file)
          
        
    
# Example of usage
if __name__ == '__main__':
    import os
    
    # Create a class to delete the database file after the tests
    class DatabaseDeleter:
        def __init__(self, path_database):
            self.path_database = path_database

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if os.path.exists(self.path_database):
                os.remove(self.path_database)

    path_database = "database.db"

    with DatabaseDeleter(path_database):
        db_texture = TableManagerTextures(path_database)
        print(db_texture.get_all_tables_names())
        print(db_texture.get_record_structure())

        db_texture.add_record("path_segment", "path_subsegment", [1, 2, 3, 4])
        db_texture.add_record("path_segment", "path_subsegment", [4, 3, 2, 1])

        print(db_texture.get_all_records())

        db_texture.delete_record(1)

        print("------------ After record delete -------------")
        print(db_texture.get_all_records())

        db_texture.close_connection()
        