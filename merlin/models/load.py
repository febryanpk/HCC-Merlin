import os

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from merlin.models.build import MerlinArchitecture
from merlin.utils import download_file


class Merlin(nn.Module):
    def __init__(
        self,
        classnames: list = ["negative", "hcc"],
        n_ctx: int = 16,
        csc: bool = False,
        use_coop: bool = False,
        ImageEmbedding: bool = False,
        HCCClassification: bool = False,
        PhenotypeCls: bool = False,
        rotate_flip: bool = False,
    ):
        super(Merlin, self).__init__()
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.csc = csc
        self.use_coop = use_coop
        self.ImageEmbedding = ImageEmbedding
        self.HCCClassification = HCCClassification
        self.PhenotypeCls = PhenotypeCls
        self.rotate_flip = rotate_flip
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(self.current_path, "checkpoints")
        self.checkpoint_name = (
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        )
        self.repo_id = "stanfordmimi/Merlin"
        self.model = self._load_model()
        self.targets = ["HCC"]
        # self.targets = pd.read_csv("documentation/phenotypes.csv")["phecode_str"].tolist()

    """
    Load the Merlin model with the initialized weights
    """

    def _load_model(self):
        self._download_checkpoint()
        model = MerlinArchitecture(
            classnames=self.classnames,
            n_ctx=self.n_ctx,
            csc=self.csc,
            use_coop=self.use_coop,
            ImageEmbedding=self.ImageEmbedding,
            HCCClassification=self.HCCClassification,
            PhenotypeCls=self.PhenotypeCls,
            rotate_flip=self.rotate_flip,
        )
        model.load_state_dict(
            torch.load(os.path.join(self.local_dir, self.checkpoint_name)), strict=False
        )
        if self.use_coop:
            # The prompt token embeddings were computed from the vanilla
            # Clinical-Longformer at construction; refresh them from Merlin's
            # fine-tuned text encoder that was just loaded.
            model.prompt_learner.build_token_buffers(model.encode_text.text_encoder)
        return model

    """ 
    Download the Merlin weights from the Hugging Face Hub
    """

    def _download_checkpoint(self):
        download_file(
            repo_id=self.repo_id,
            filename=self.checkpoint_name,
            local_dir=self.local_dir,
        )

    def forward(self, *input, **kwargs):
        return self.model(*input, **kwargs)
