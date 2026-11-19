"""Comprehensive test suite for Voting Panel Agent components and graph."""

import unittest
from tools import spam_signals, parse_vote, parse_reason, tally
from agent import create_voting_panel_graph


class TestVotingPanelTools(unittest.TestCase):
    def test_spam_signals_clean(self):
        msg = "Hi team, please find attached the meeting notes for today."
        signals = spam_signals(msg)
        self.assertEqual(signals["links"], 0)
        self.assertEqual(signals["urgent_words"], 0)
        self.assertLessEqual(signals["score"], 1)

    def test_spam_signals_spam(self):
        msg = "URGENT WINNER: You won $5000 FREE cash prize! Click here: http://claim-bonus.net/now"
        signals = spam_signals(msg)
        self.assertGreaterEqual(signals["links"], 1)
        self.assertGreaterEqual(signals["money_words"], 2)
        self.assertGreaterEqual(signals["score"], 4)

    def test_spam_signals_borderline(self):
        msg = "Urgent reminder: package delivery status at https://courier.com"
        signals = spam_signals(msg)
        self.assertIn(signals["score"], (2, 3))

    def test_parse_vote_formats(self):
        self.assertEqual(parse_vote("SPAM - urgency detected"), "SPAM")
        self.assertEqual(parse_vote("NOT_SPAM - looks legitimate"), "NOT_SPAM")
        self.assertEqual(parse_vote("UNSURE - score is 2"), "UNSURE")
        self.assertEqual(parse_vote("Vote: SPAM"), "SPAM")
        self.assertEqual(parse_vote("Vote: NOT_SPAM"), "NOT_SPAM")
        self.assertEqual(parse_vote("Vote: UNSURE"), "UNSURE")
        self.assertEqual(parse_vote("Reason: suspicious link\nVote: SPAM"), "SPAM")
        self.assertEqual(parse_vote("Reason: safe message\nVote: NOT_SPAM"), "NOT_SPAM")

    def test_tally_majority(self):
        winner, counts = tally(["SPAM", "SPAM", "NOT_SPAM"])
        self.assertEqual(winner, "SPAM")
        self.assertEqual(counts["SPAM"], 2)
        self.assertEqual(counts["NOT_SPAM"], 1)

    def test_tally_ignore_unsure(self):
        winner, counts = tally(["NOT_SPAM", "NOT_SPAM", "UNSURE"])
        self.assertEqual(winner, "NOT_SPAM")
        self.assertEqual(counts["NOT_SPAM"], 2)
        self.assertNotIn("UNSURE", counts)

    def test_tally_split_tie(self):
        winner, counts = tally(["SPAM", "NOT_SPAM", "UNSURE"])
        self.assertIsNone(winner)
        self.assertEqual(counts["SPAM"], 1)
        self.assertEqual(counts["NOT_SPAM"], 1)

    def test_tally_all_unsure(self):
        winner, counts = tally(["UNSURE", "UNSURE", "UNSURE"])
        self.assertIsNone(winner)
        self.assertEqual(len(counts), 0)


class TestVotingPanelGraph(unittest.TestCase):
    def setUp(self):
        self.graph = create_voting_panel_graph()

    def test_graph_majority_spam(self):
        msg = "URGENT! You won a FREE $1000 Walmart Gift Card! Click immediately: http://scam.me/win"
        res = self.graph.invoke({"message": msg})
        self.assertEqual(res["verdict"], "SPAM")
        self.assertEqual(res["decided_by"], "majority vote")
        self.assertIn("SPAM", res["strict"])
        self.assertIn("SPAM", res["friendly"])
        self.assertIn("SPAM", res["evidence"])

    def test_graph_majority_not_spam(self):
        msg = "Hey Sarah, hope you have a nice day! Are we still meeting for lunch tomorrow at 1?"
        res = self.graph.invoke({"message": msg})
        self.assertEqual(res["verdict"], "NOT_SPAM")
        self.assertEqual(res["decided_by"], "majority vote")
        self.assertIn("NOT_SPAM", res["strict"])
        self.assertIn("NOT_SPAM", res["friendly"])
        self.assertIn("NOT_SPAM", res["evidence"])

    def test_graph_split_judge_fallback(self):
        msg = "Urgent reminder: package delivery status at https://courier.com"
        res = self.graph.invoke({"message": msg})
        self.assertIn(res["verdict"], ("SPAM", "NOT_SPAM"))
        self.assertEqual(res["decided_by"], "judge (tie-break)")


if __name__ == "__main__":
    unittest.main()
