import sys
c = open(r'E:\workspace\shop-agent\apps\shop-agent\src\main.py', 'r', encoding='utf-8').read()
m = c.find('if __name__ == \"__main__\":')
p = c.find('@app.get(\"/metrics\"')
print('main:', m, 'metrics:', p, 'after:', p > m)
