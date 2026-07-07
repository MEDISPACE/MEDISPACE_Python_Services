from main import recommendation_items


def test_recommendation_items_include_explanation_and_score():
    items = recommendation_items(["p1", "p2"], "Because", ["catalog"])

    assert items[0]["productId"] == "p1"
    assert items[0]["score"] > items[1]["score"]
    assert items[0]["reason"] == "Because"
    assert items[0]["evidence"] == ["catalog"]


def test_recommendation_items_preserve_model_native_score():
    items = recommendation_items(
        [{"productId": "p1", "score": 0.87654321}],
        "Because",
        ["catalog"],
    )

    assert items[0]["productId"] == "p1"
    assert items[0]["score"] == 0.876543
