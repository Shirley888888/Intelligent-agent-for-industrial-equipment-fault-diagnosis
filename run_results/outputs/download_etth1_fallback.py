import os
import urllib.request

urls = [
    'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/master/ETT/ETTh1.csv',
    'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT/ETTh1.csv',
]

dest = os.path.join('data','ETTh1.csv')
os.makedirs(os.path.dirname(dest), exist_ok=True)
for u in urls:
    try:
        print('Trying', u)
        resp = urllib.request.urlopen(u)
        data = resp.read()
        with open(dest, 'wb') as f:
            f.write(data)
        print('Downloaded to', dest)
        break
    except Exception as e:
        print('Failed:', e)
else:
    print('All attempts failed')
