import os
import requests

def load_file_from_url(url, model_dir='weights', progress=True):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    file_path = os.path.join(model_dir, filename)
    if not os.path.isfile(file_path):
        r = requests.get(url, allow_redirects=True)
        with open(file_path, 'wb') as f:
            f.write(r.content)
    return file_path
