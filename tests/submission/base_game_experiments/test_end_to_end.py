import ray


def test_meta_amtft_iasymbos():
    from submission.base_game_experiments.meta_amtft_various_env import main

    ray.shutdown()
    main(debug=True, env="IteratedAsymBoS", use_r2d2=True)


def test_amtft_ipd():
    from submission.base_game_experiments.amtft_various_env import main

    ray.shutdown()
    main(debug=True, env="IteratedPrisonersDilemma")


def test_amtft_ipd_with_r2d2():
    from submission.base_game_experiments.amtft_various_env import main

    ray.shutdown()
    main(debug=True, env="IteratedPrisonersDilemma", use_r2d2=True)


def test_amtft_iasymbos():
    from submission.base_game_experiments.amtft_various_env import main

    ray.shutdown()
    main(debug=True, env="IteratedAsymBoS", use_r2d2=True)


def test_amtft_iasymbos_and_pd():
    from submission.base_game_experiments.amtft_various_env import main

    ray.shutdown()
    main(debug=True, env="IteratedAsymBoSandPD", use_r2d2=True)


def test_amtft_coin_game():
    from submission.base_game_experiments.amtft_various_env import main

    ray.shutdown()
    main(debug=True, env="CoinGame", use_r2d2=True)


def test_amtft_mixed_motive_coin_game():
    from submission.base_game_experiments.amtft_various_env import main

    ray.shutdown()
    main(debug=True, env="ABCoinGame", use_r2d2=True)


def test_lola_pg_tune_class_api_coin_game():
    from submission.base_game_experiments.lola_pg_official import main

    ray.shutdown()
    main(debug=True, env="CoinGame")


def test_lola_pg_tune_class_api_mixed_motive_coin_game():
    from submission.base_game_experiments.lola_pg_official import main

    ray.shutdown()
    main(debug=True, env="ABCoinGame")


def test_lola_exact_tune_class_api():
    from submission.base_game_experiments.lola_exact_official import main

    ray.shutdown()
    main(debug=True)
