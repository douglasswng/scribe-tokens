def compute_cer(pred_text: str, target_text: str) -> float:
    """
    Compute Character Error Rate using Levenshtein distance.

    CER = edit_distance(pred, target) / len(target)

    Args:
        pred_text: Predicted text string
        target_text: Ground truth text string

    Returns:
        Character Error Rate as a float
    """
    if len(target_text) == 0:
        return 1.0 if len(pred_text) > 0 else 0.0

    # Dynamic programming for edit distance
    m, n = len(pred_text), len(target_text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_text[i - 1] == target_text[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / len(target_text)
