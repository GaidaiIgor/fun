"""Prints a table of the most recent CodinGame arena battles of a player: opponent, opponent rank, battle score and result."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from requests import Session
from requests.adapters import HTTPAdapter
from sys import stdout
from tabulate import tabulate
from time import sleep

SERVICES = "https://www.codingame.com/services"
THREADS = 4
RETRY_DELAYS = (2, 5, 15, 30)
SESSION = Session()
SESSION.mount("https://", HTTPAdapter(pool_maxsize=THREADS))

def main(game: str, user_id: int):
    """Prints the finished battles of the player user_id in the game arena, most recent first."""
    stdout.reconfigure(encoding="utf-8")
    me = call("Leaderboards", "getCodinGamerPuzzleRanking", [user_id, game])
    pseudo, my_rank = me["pseudo"], me["rank"]
    battles = [battle for battle in call("gamesPlayersRanking", "findLastBattlesByAgentId", [me["agentId"], None]) if battle["done"]]
    if not battles:
        raise SystemExit(f"the current {pseudo} submission has no finished battle yet")

    ranks = arena_ranks(game)
    unranked = sorted({player["userId"] for battle in battles for player in battle["players"] if player["userId"] not in ranks} - {user_id})
    with ThreadPoolExecutor(THREADS) as pool:
        scores = list(pool.map(battle_scores, [battle["gameId"] for battle in battles]))
        ranks.update(zip(unranked, pool.map(lambda player_id: opponent_rank(game, player_id), unranked)))

    rows = []
    for number, (battle, by_agent) in enumerate(zip(battles, scores), 1):
        mine = next(player for player in battle["players"] if player["userId"] == user_id)
        others = [player for player in battle["players"] if player is not mine]
        best = min(player["position"] for player in others)
        result = "win" if mine["position"] < best else "loss" if mine["position"] > best else "draw"
        names = ", ".join(player["nickname"] for player in others)
        opponent_ranks = ", ".join(str(ranks[player["userId"]]) for player in others)
        score = " - ".join("{:g}".format(by_agent[player["playerAgentId"]]) for player in [mine, *others])
        rows.append([number, names, opponent_ranks, score, result])

    tally = Counter(row[-1] for row in rows)
    wins, losses, draws = tally["win"], tally["loss"], tally["draw"]
    print(f"{pseudo} is rank {my_rank} in the {game} arena")
    print(tabulate(rows, headers=["#", "opponent", "rank", "score", "result"], tablefmt="simple_outline"))
    print(f"{len(rows)} battles, {wins} won, {losses} lost, {draws} drawn, {wins / len(rows):.0%} win rate")

def call(service: str, function: str, payload: list) -> dict | list:
    """Posts payload to the given CodinGame service function and returns its decoded JSON response, waiting out any rate limiting."""
    url = f"{SERVICES}/{service}/{function}"
    for delay in RETRY_DELAYS:
        response = SESSION.post(url, json=payload, timeout=60)
        if response.status_code != HTTPStatus.TOO_MANY_REQUESTS:
            response.raise_for_status()
            return response.json()
        sleep(delay)
    raise SystemExit("CodinGame keeps rate limiting the script, wait a minute and run it again")

def arena_ranks(game: str) -> dict[int, int]:
    """Returns the current global rank of every player on the first leaderboard page of the game arena, keyed by user id."""
    board = call("Leaderboards", "getFilteredPuzzleLeaderboard", [game, None, "global", {"active": False, "column": "", "filter": ""}])
    return {user["codingamer"]["userId"]: user["rank"] for user in board["users"]}

def battle_scores(game_id: int) -> dict[int, float]:
    """Returns the score every participating agent got in the battle game_id, keyed by agent id."""
    result = call("gameResult", "findByGameId", [game_id, None])
    return {agent["agentId"]: result["scores"][agent["index"]] for agent in result["agents"]}

def opponent_rank(game: str, user_id: int) -> int:
    """Returns the current global rank of the player user_id in the game arena."""
    return call("Leaderboards", "getCodinGamerPuzzleRanking", [user_id, game])["rank"]

if __name__ == "__main__":
    game = "code4life"
    user_id = 877390
    main(game, user_id)
