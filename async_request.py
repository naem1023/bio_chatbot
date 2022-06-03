import aiohttp
import asyncio


	# async def get_boolqa_bot_response():
    # userText = request.args.get('msg')
    # print(userText)
    # data = {'data':userText}
    # async with aiohttp.ClientSession() as session:
    #     url = 'http://14.49.45.219:9091/chat_boolq'
    #     async with session.post(url, json=data) as resp:
    #         return ((await resp.json())['result'])

async def query():
    async with aiohttp.ClientSession() as session:
        chat_url = 'http://192.168.5.21:5000/predict'
        data = {"text": "My name is Chris, and what about you?"}
        async with session.get(chat_url, json=data) as resp:
            res = await resp.json()
            print(res)
            return res


# print(query())
asyncio.run(query())