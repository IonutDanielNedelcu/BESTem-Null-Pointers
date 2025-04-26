import requests
from time import sleep
import random
import pred2

host = "http://10.41.186.9:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 10


def what_beats(word):
    prediction = pred2.predict(word)
    return prediction

def play_game(player_id):

    def get_round():
        response = requests.get(get_url)
        print(response.json())
        sys_word = response.json()['word']
        round_num = response.json()['round']
        return (sys_word, round_num)

    submitted_rounds = []
    round_num = 0

    while round_num != NUM_ROUNDS :
        print(submitted_rounds)
        sys_word, round_num = get_round()
        while round_num == 0 or round_num in submitted_rounds:
            sys_word, round_num = get_round()
            sleep(0.5)

        if round_num > 1:
            status = requests.post(status_url, json={"player_id": player_id}, timeout=2)
            print(status.json())

        choosen_word = what_beats(sys_word)
        data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_num}
        response = requests.post(post_url, json=data, timeout=5)
        submitted_rounds.append(round_num)
        print("POST: !!!!!!!!!!!!!!!!")
        print(response.json())


def register(player_id):
    register_url = f"{host}/register"
    data = {"player_id": player_id}
    response = requests.post(register_url, json=data)
    
    return response.json()
    


#register("abc123")
play_game("aeaiqDw3")