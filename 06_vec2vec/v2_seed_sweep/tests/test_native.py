# Native (authors' text pipeline) mode with tiny offline stand-in encoders: tests OUR wrapper only
# (time budget, logging, checkpoint, resume), since Hugging Face is not reachable from this test machine.
import sys, os, types, random, json
sys.path.insert(0, os.path.abspath(".")); sys.path.insert(0, os.path.abspath("../ext/vec2vec"))
for name in ["beir", "beir.datasets", "beir.datasets.data_loader_hf"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["beir.datasets.data_loader_hf"].HFDataLoader = object
import torch, datasets
from transformers import BertConfig, BertModel, BertTokenizerFast
from sentence_transformers import SentenceTransformer, models
import train as T
from v2v_runner import run
BASE = os.path.abspath(".")
words = [f"w{i}" for i in range(500)]
vocab = os.path.abspath("t2/vocab.txt")
open(vocab, "w").write("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words))
def make(flag, dim):
    path = f"{BASE}/t2/tiny_{flag}"
    if not os.path.exists(path):
        torch.manual_seed(hash(flag) % 1000)
        BertModel(BertConfig(vocab_size=505, hidden_size=dim, num_hidden_layers=1, num_attention_heads=2,
                             intermediate_size=64, max_position_embeddings=128)).save_pretrained(path)
        BertTokenizerFast(vocab_file=vocab).save_pretrained(path)
    tr = models.Transformer(path, max_seq_length=64)
    return SentenceTransformer(modules=[tr, models.Pooling(dim, "mean")]).eval()
T.load_encoder = lambda flag, **k: make(flag, 64 if flag == "stella" else 48)
rng = random.Random(0)
texts = [" ".join(rng.choice(words) for _ in range(20)) for _ in range(9000)]
T.load_streaming_embeddings = lambda *a, **k: datasets.Dataset.from_dict({"text": texts})
resume = sys.argv[1] if len(sys.argv) > 1 else None
best, log = run("../ext/vec2vec", None, None, None, None, "t2/runs/native", budget_hours=float(sys.argv[2]) if resume else 0.02,
                seed=5, native=True, early_stopping=False, resume_from=resume,
                extra_args={"mixed_precision": "no", "num_points": 2000, "val_size": 512, "val_bs": 256,
                            "top_k_size": 256, "top_k_batches": 2})
for l in open(log):
    r = json.loads(l)
    if r["kind"] == "ours": print("EPOCH", r)
    if r["kind"] == "authors_val": print("VAL", {k: round(v, 4) for k, v in r.items() if "stella_gte_top_1" in k or "hours" in k})
