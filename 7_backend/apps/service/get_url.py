import json

def get_product_url(test_result):
    file_path = "./apps/resource/product_url.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        return data[str(test_result)]