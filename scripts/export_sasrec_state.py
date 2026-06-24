"""Extract SASRec's weights from the RecBole-pickled checkpoint into a clean, tensor-only
state file the UI can load without importing RecBole.

The shipped `artifacts/SASRec.pth` embeds a `recbole.config.Config` object, so a normal
`torch.load` needs RecBole installed. We unpickle with any `recbole.*` global stubbed out,
keep only `state_dict`, and re-save it. Run once after training:

    python scripts/export_sasrec_state.py
"""
import pickle

import torch

IN_PATH = "artifacts/SASRec.pth"
OUT_PATH = "artifacts/SASRec_state.pt"


class _Stub:

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _RecBoleStubUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module.startswith("recbole"):
            return _Stub
        return super().find_class(module, name)


class _PickleModule:
    Unpickler = _RecBoleStubUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _RecBoleStubUnpickler(file).load()


def main() -> None:
    ckpt = torch.load(IN_PATH, map_location="cpu", pickle_module=_PickleModule, weights_only=False)
    state = {k: v for k, v in ckpt["state_dict"].items()}
    torch.save(state, OUT_PATH)
    print(f"wrote {OUT_PATH}: {len(state)} tensors "
          f"(epoch {ckpt.get('epoch')}, best valid {ckpt.get('best_valid_score')})")


if __name__ == "__main__":
    main()
