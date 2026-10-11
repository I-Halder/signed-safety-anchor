"""Functional test of prepare_fineweb_asa on a tiny subset of the (already downloaded) geodesic parquet files."""
import sys, os, glob, tempfile, shutil, numpy as np, pyarrow.parquet as pq
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import prepare_fineweb_asa as P
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.environ["TOK_DIR"], "tokenizer.json"))
files = sorted(glob.glob(os.path.join(os.environ["HF_GEODESIC_LOCAL"], "data", "*.parquet")))[:1]
total = pq.ParquetFile(files[0]).metadata.num_rows
mask = np.zeros(total, dtype=bool); mask[:300] = True
tmp = Path(tempfile.mkdtemp(dir=os.environ["DATA_ROOT"]))
spec, n, t = P.tokenise_selected(files, mask, tok, tmp, "test", batch=100, log_every=100)
print("tokenised", n, "docs", t, "tokens", flush=True)
P.write_shard(spec, n, tmp / "shard", 2)
from litdata.streaming import StreamingDataset, TokensLoader
ds = StreamingDataset(input_dir=str(tmp / "shard"), item_loader=TokensLoader(block_size=2049), shuffle=False, drop_last=True)
x = ds[0]; print("first block", tuple(x.shape), x.dtype, "| decoded start:", tok.decode(x[:12].tolist())[:80].replace("\n", " "), "| n blocks", len(ds))
shutil.rmtree(tmp); print("OK")
