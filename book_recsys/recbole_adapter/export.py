"""Map RecBole internal-id top-K results back to our (user, book) tokens."""


def recbole_predictions(user_internal, item_internal_matrix, uid2token,
                        iid2token) -> dict:
    """Build {user_token: [ranked book_tokens]} from RecBole internal ids.

    user_internal: sequence of internal user ids (length U).
    item_internal_matrix: U x K internal item ids (RecBole full_sort_topk output).
    uid2token / iid2token: RecBole internal-id -> original token maps.
    """
    return {
        uid2token[user]: [iid2token[item] for item in row]
        for user, row in zip(user_internal, item_internal_matrix)
    }
