import os
import json


class DataProcessor:
    """Handles loading and processing JSON data from a folder."""

    def __init__(self, folder_path):
        self.folder_path = folder_path

    def read_json_files(self):
        """Reads JSON files from the specified folder."""
        all_data = []

        for filename in os.listdir(self.folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    all_data.extend(data)
        return all_data
