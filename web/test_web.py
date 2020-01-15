import requests
import asyncio
from aiohttp import ClientSession

#r = requests.get("http://127.0.0.1:3600/add_task", {"id": "123", "info": "ko"})
#print(r.text)

#r = requests.get("http://127.0.0.1:3600/get_task", {"id": "123"})
#print(r.text)

'''files = {
    'file1': open('D:/s5/lena/lena.png', 'rb'),
    'file2': open('D:/s5/lena/103752.jpg', 'rb')
}
r = requests.post('http://127.0.0.1:3600/send_img', data={'info': 'ko123'}, files=files)
print(r.text)'''

async def hello(url):
    async with ClientSession() as session:
        async with session.post(url) as response:
            response = await response.read()
            print(eval(response))
    pass
loop = asyncio.get_event_loop()
loop.run_until_complete(hello("http://127.0.0.1:3600/test1"))
print("ko")
