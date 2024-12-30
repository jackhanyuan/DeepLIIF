import os
import json
import base64
from io import BytesIO

import requests
from PIL import Image

# Use the sample images from the main DeepLIIF repo
images_dir = './Sample_Large_Tissues'
filename = 'ROI_1.png'

res = requests.post(
    url='https://deepliif.org/api/infer',
    files={
        'img': open(f'{images_dir}/{filename}', 'rb')
    },
    # optional param that can be 10x, 20x (default) or 40x
    params={
        'resolution': '40x'
    }
)

data = res.json()

def b64_to_pil(b):
    return Image.open(BytesIO(base64.b64decode(b.encode())))

for name, img in data['images'].items():
    output_filepath = f'{images_dir}/cloud/{os.path.splitext(filename)[0]}_{name}.png'
    with open(output_filepath, 'wb') as f:
        b64_to_pil(img).save(f, format='PNG')

print(json.dumps(data['scoring'], indent=2))