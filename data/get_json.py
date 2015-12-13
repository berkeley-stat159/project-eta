import json
from data import get_hash_for_all


if __name__ == "__main__":
    hashes = get_hash_for_all('ds005')
    with open('ds005_hashes.json', 'w') as out_file:
        json.dump(hashes, out_file)
    