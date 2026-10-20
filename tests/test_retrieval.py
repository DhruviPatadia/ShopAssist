import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path("/content/drive/MyDrive/ShopAssist")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.retrieval import (
    load_policies,
    build_retrieval_index,
    search_policies
)


class TestShopAssistRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.documents_df = load_policies(
            PROJECT_DIR / "data" / "policies"
        )
        cls.model, cls.index = build_retrieval_index(cls.documents_df)

    def test_all_policy_documents_load(self):
        self.assertEqual(len(self.documents_df), 15)

    def test_policy_sources_are_markdown_files(self):
        self.assertTrue(
            all(source.endswith(".md")
                for source in self.documents_df["source"])
        )

    def test_damaged_item_query(self):
        results = search_policies(
            "My item arrived damaged. What should I do?",
            self.documents_df,
            self.model,
            self.index,
            top_k=3
        )
        self.assertFalse(results.empty)
        self.assertEqual(results.iloc[0]["source"], "damaged_items.md")

    def test_delivery_query(self):
        results = search_policies(
            "How long does standard delivery take?",
            self.documents_df,
            self.model,
            self.index,
            top_k=3
        )
        self.assertFalse(results.empty)
        self.assertEqual(results.iloc[0]["source"], "delivery_policy.md")

    def test_empty_query(self):
        results = search_policies(
            "",
            self.documents_df,
            self.model,
            self.index
        )
        self.assertTrue(results.empty)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
