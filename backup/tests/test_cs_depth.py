from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSFieldDecoder, CSFieldEncoder
from bubble_bi.models.ts_vqvae import TSVQVAE


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(d_model=32, heads=2, ff=64, **kw)


def test_cs_uses_its_own_depth_not_the_ts_depth():
    # TS is deep, CS is shallow -- they must not borrow each other's depth.
    cfg = _cfg(enc_layers=4, dec_layers=3, cs_enc_layers=1, cs_dec_layers=2)

    enc = CSFieldEncoder(cfg, d_in=6, n_stocks=5)
    dec = CSFieldDecoder(cfg, d_out=6, n_stocks=5)
    assert len(enc.enc.layers) == 1        # cs_enc_layers, NOT enc_layers=4
    assert len(dec.dec.layers) == 2        # cs_dec_layers, NOT dec_layers=3

    ts = TSVQVAE(cfg, d_in=6)
    assert len(ts.enc.enc.layers) == 4     # TS still uses its own enc_layers
    assert len(ts.dec.dec.layers) == 3     # ...and its own dec_layers


def test_cs_depth_defaults_preserve_the_old_shared_behaviour():
    # Defaults must reproduce what the model did before the split, so existing
    # checkpoints and results stay comparable.
    cfg = ModelConfig()
    assert cfg.cs_enc_layers == cfg.enc_layers
    assert cfg.cs_dec_layers == cfg.dec_layers
