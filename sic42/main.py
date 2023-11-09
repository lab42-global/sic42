from sic42.sic import Tournament


if __name__ == '__main__':
    tournament = Tournament(
        config_path='deathmatch_config.json',
        competitors_path='competitors'
    )
    tournament.run_tournament()