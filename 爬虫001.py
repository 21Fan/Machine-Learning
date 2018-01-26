import urllib.request#一个模块
response = urllib.request.urlopen("https://sale.jd.com/act/pt5AWzCMxTDgjV.html?cpdad=1DLSUE")
html = response.read()
html = html.decode("utf-8")#编码形式
print(html)
