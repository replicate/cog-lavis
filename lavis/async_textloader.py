import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

def _load_file(path):
    with open(path, 'r') as file:
        data = file.read()
    return data

async def function1(path1):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, _load_file, path1)
    return data

async def function2(path2):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, _load_file, path2)
    return data

async def function3(path1, path2):
    task1 = function1(path1)
    task2 = function2(path2)
    result1, result2 = await asyncio.gather(task1, task2)
    return result1, result2