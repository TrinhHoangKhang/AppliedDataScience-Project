import torch
from accelerate import Accelerator
from genrec.utils import get_config, init_seed, init_logger, get_dataset, get_tokenizer, get_model


class RecommenderInference:
    """
    A lightweight inference wrapper for a trained RPG model.

    Usage:
        rec = RecommenderInference(
            model_name='RPG',
            dataset_name='AmazonReviews2014',
            checkpoint_path='checkpoints/your_model.pth',
            config_dict={'category': 'Beauty', 'metadata': 'sentence', ...}
        )
        recommendations = rec.recommend(
            item_history=['ASIN001', 'ASIN002', 'ASIN003'],
            topk=10
        )
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        checkpoint_path: str,
        config_dict: dict = None,
        config_file: str = None,
    ):
        # Minimal accelerator setup (single process, no DDP)
        self.accelerator = Accelerator()
        base_config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        base_config['accelerator'] = self.accelerator
        base_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_config['use_ddp'] = False
        self.config = base_config

        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)

        # Load dataset
        print('[Inference] Loading dataset...')
        self.dataset = get_dataset(dataset_name)(self.config)
        self.dataset.split()  # needed so tokenizer can build training mask

        # Load tokenizer
        print('[Inference] Loading tokenizer...')
        self.tokenizer = get_tokenizer(model_name)(self.config, self.dataset)

        # Load model from checkpoint
        print(f'[Inference] Loading model from {checkpoint_path}...')
        self.model = get_model(model_name)(self.config, self.dataset, self.tokenizer)
        state_dict = torch.load(checkpoint_path, map_location=self.config['device'])
        self.model.load_state_dict(state_dict)
        self.model.to(self.config['device'])
        self.model.eval()
        print('[Inference] Model ready.')

    def recommend(self, item_history: list[str], topk: int = 10) -> list[dict]:
        """
        Given a list of item ASINs (interaction history), return the top-k recommended items.

        Args:
            item_history (list[str]): Ordered list of item ASINs the user has interacted with
                                      (chronological order, most recent last).
            topk (int): Number of recommendations to return.

        Returns:
            list[dict]: A list of dicts with keys:
                - 'asin': the item ASIN
                - 'item_id': the internal integer ID
                - 'meta': the metadata sentence (if available)
                - 'score_rank': rank of the recommendation (1 = top)
        """
        max_len = self.tokenizer.max_token_seq_len

        # Validate items are in the catalog
        valid_history = [item for item in item_history if item in self.dataset.item2id]
        if len(valid_history) == 0:
            raise ValueError('None of the provided items are in the dataset catalog.')
        if len(valid_history) < len(item_history):
            unknown = set(item_history) - set(valid_history)
            print(f'[Inference] Warning: {len(unknown)} items not in catalog, ignored: {unknown}')

        # Need at least 1 item in input (the last item is the "target" slot)
        # We feed the history as input and ask the model to predict what comes next
        # Pad with a dummy last item that we'll ignore — use the last item as both input[-1] and dummy target
        input_seq = valid_history[-(max_len + 1):]  # cap to max_len context items + 1 dummy

        # If only 1 item, duplicate it so tokenizer has an input+target pair
        if len(input_seq) == 1:
            input_seq = input_seq + input_seq

        # Tokenize: use the "later items" path (only last item is target, rest are input)
        input_ids = [self.dataset.item2id[item] for item in input_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_len = max_len - seq_lens
        input_ids_padded = input_ids + [0] * pad_len
        attention_mask_padded = attention_mask + [0] * pad_len

        batch = {
            'input_ids': torch.LongTensor([input_ids_padded]).to(self.config['device']),
            'attention_mask': torch.LongTensor([attention_mask_padded]).to(self.config['device']),
            'seq_lens': torch.LongTensor([seq_lens]).to(self.config['device']),
        }

        with torch.no_grad():
            preds = self.model.generate(batch, n_return_sequences=topk)  # (1, topk, 1)

        pred_item_ids = preds[0, :, 0].cpu().tolist()

        results = []
        for rank, item_id in enumerate(pred_item_ids, 1):
            asin = self.dataset.id_mapping['id2item'][item_id]
            meta = ''
            if self.dataset.item2meta is not None:
                meta = self.dataset.item2meta.get(asin, '')
            results.append({
                'score_rank': rank,
                'asin': asin,
                'item_id': item_id,
                'meta': meta,
            })

        return results

    def get_item_info(self, asin: str) -> dict:
        """Returns metadata for a given ASIN."""
        meta = ''
        if self.dataset.item2meta is not None:
            meta = self.dataset.item2meta.get(asin, '')
        item_id = self.dataset.item2id.get(asin, None)
        return {'asin': asin, 'item_id': item_id, 'meta': meta}

    def search_items_by_keyword(self, keyword: str, max_results: int = 20) -> list[dict]:
        """
        Search the catalog for items whose metadata contains the keyword.
        Useful for building a demo where the user searches for items by name.
        """
        keyword = keyword.lower()
        results = []
        if self.dataset.item2meta is None:
            return results
        for asin, meta in self.dataset.item2meta.items():
            if keyword in meta.lower():
                results.append({'asin': asin, 'meta': meta[:200]})
            if len(results) >= max_results:
                break
        return results
