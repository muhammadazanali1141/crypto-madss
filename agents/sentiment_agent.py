class SentimentAgent:
    def __init__(self):
        # Simple crypto-specific keyword lexicon (no paid API needed)
        self.positive_words = [
            "bullish", "surge", "rally", "soar", "gain", "breakout",
            "adoption", "partnership", "upgrade", "moon", "pump",
            "buy", "growth", "record high", "institutional", "positive",
            "outperform", "strong demand", "accumulation"
        ]
        self.negative_words = [
            "bearish", "crash", "plunge", "dump", "sell-off", "decline",
            "hack", "exploit", "regulation", "ban", "lawsuit", "fear",
            "sell", "correction", "fud", "negative", "collapse",
            "liquidation", "outflow", "weak demand"
        ]

    def analyze_text(self, text: str) -> dict:
        text_lower = text.lower()

        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)

        score = pos_count - neg_count

        if score > 0:
            sentiment = "positive"
        elif score < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": score,
            "positive_matches": pos_count,
            "negative_matches": neg_count,
        }

    def analyze_multiple(self, texts: list) -> dict:
        """Aggregate sentiment across multiple headlines/posts."""
        results = [self.analyze_text(t) for t in texts]

        total_score = sum(r["score"] for r in results)
        avg_score = total_score / len(results) if results else 0

        if avg_score > 0.2:
            overall = "positive"
        elif avg_score < -0.2:
            overall = "negative"
        else:
            overall = "neutral"

        return {
            "agent": "SentimentAgent",
            "overall_sentiment": overall,
            "average_score": round(avg_score, 2),
            "num_texts_analyzed": len(texts),
            "details": results,
        }


if __name__ == "__main__":
    agent = SentimentAgent()

    sample_headlines = [
        "Bitcoin surges as institutional adoption grows rapidly",
        "Major exchange hacked, investors fear further decline",
        "Analysts remain neutral ahead of Fed decision",
        "Crypto market shows strong bullish breakout pattern",
    ]

    result = agent.analyze_multiple(sample_headlines)

    print("Sentiment Analysis Result:")
    print(f"  Overall Sentiment: {result['overall_sentiment']}")
    print(f"  Average Score: {result['average_score']}")
    print(f"  Texts Analyzed: {result['num_texts_analyzed']}")
    print("\n  Details:")
    for i, d in enumerate(result["details"]):
        print(f"    [{i+1}] {sample_headlines[i][:50]}... -> {d['sentiment']} (score: {d['score']})")