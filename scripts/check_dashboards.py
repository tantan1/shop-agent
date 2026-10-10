import requests
import re

r = requests.get('http://localhost:3000/dashboard/list')
print('Dashboard list:', r.status_code)
matches = re.findall(r'shop-agent.*?uid.*?"', r.text)
for m in matches[:20]:
    print(m)