import json
from typing import List, Dict

import requests


class ExecutorClient:
    def __init__(self, addr: str):
        self.addr = addr.rstrip('/')

    def execute(self, queries: List[str]) -> Dict:
        queries = [
            q.strip()
            for q in queries
        ]

        response = requests.post(
            f'{self.addr}/execute',
            dict(
                q='|||'.join(queries)
            )
        )

        response_dict = json.loads(response.text)

        return response_dict


def main():
    queries = []
    queries = [
        line.strip()
        for line in open('data/test_scholar.lf.txt')
    ]
    server_url = 'http://localhost:8081/execute'
    response = requests.post(
        server_url,
        dict(q='|||'.join(queries))
    )

    with open('data/executor.response.json', 'w') as f:
        f.write(json.dumps(json.loads(response.text)))


if __name__ == '__main__':
    main()
