from scglm_v2.select_endpoint import MAX_WIKI_BPB_REGRESSION, MIN_PROXY_GAIN, decide


def _r(name, proxy, wiki):
    return {"run": name, "proxy_mean_acc": proxy, "bits_per_byte_selection_panels": {"wiki": wiki}}


def test_rule_edges():
    base = _r("main", 0.50, 1.00)
    assert decide(base, None)[0] is base
    assert decide(base, _r("anneal", 0.50 + MIN_PROXY_GAIN, 1.00 + MAX_WIKI_BPB_REGRESSION))[0]["run"] == "anneal"
    assert decide(base, _r("anneal", 0.50 + MIN_PROXY_GAIN - 1e-4, 0.90))[0] is base      # gain too small
    assert decide(base, _r("anneal", 0.60, 1.00 + MAX_WIKI_BPB_REGRESSION + 1e-4))[0] is base  # wiki regressed


def test_v1_vs_v2_rule_and_reference_values():
    import json
    from scglm_v2.v1_vs_v2 import V1_SUMMARY, decide, headline
    v1 = headline(json.loads(V1_SUMMARY.read_text()))
    assert abs(v1["four_task_mean_acc"] - 0.4695512839959672) < 1e-15   # registered value
    assert abs(v1["wikitext103_perplexity"] - 25.519837291770507) < 1e-12
    better = {"four_task_mean_acc": v1["four_task_mean_acc"] + 0.01, "wikitext103_perplexity": v1["wikitext103_perplexity"] - 1}
    worse = {"four_task_mean_acc": v1["four_task_mean_acc"] - 0.01, "wikitext103_perplexity": v1["wikitext103_perplexity"] + 1}
    assert decide(v1, better) == "V2" and decide(v1, worse) == "V1"
    assert decide(v1, {**better, "wikitext103_perplexity": worse["wikitext103_perplexity"]}) == "USER_DECIDES"
    assert decide(v1, {**worse, "wikitext103_perplexity": better["wikitext103_perplexity"]}) == "USER_DECIDES"
    tie = {"four_task_mean_acc": v1["four_task_mean_acc"], "wikitext103_perplexity": v1["wikitext103_perplexity"]}
    assert decide(v1, tie) == "USER_DECIDES"   # equal accuracy is not "better"; equal perplexity is "not worse"
