import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from nltk.tokenize import wordpunct_tokenize
from torch import nn
from transformers import AutoModel, AutoTokenizer

from merlin.models import i3res


def sanitize_report(report):
    report = report.lower()
    return " ".join(wordpunct_tokenize(report))


class ImageEncoder(nn.Module):
    def __init__(
        self,
        ImageEmbedding: bool = False,
        HCCClassification: bool = False,
        PhenotypeCls: bool = False,
        rotate_flip=False,
    ):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.HCCClassification = HCCClassification
        self.PhenotypeCls = PhenotypeCls
        self.rotate_flip = rotate_flip
        resnet = torchvision.models.resnet152(pretrained=True)
        self.i3_resnet = i3res.I3ResNet(
            copy.deepcopy(resnet),
            class_nb=1692,
            conv_class=True,
            ImageEmbedding=self.ImageEmbedding,
            HCCClassification=self.HCCClassification,
            PhenotypeCls=self.PhenotypeCls,
            rotate_flip=self.rotate_flip,
        )

    def forward(self, image):
        if self.ImageEmbedding or self.HCCClassification or self.PhenotypeCls:
            return self.i3_resnet(image)
        else:
            contrastive_features, ehr_features, hcc_features = self.i3_resnet(image)
            return contrastive_features, ehr_features, hcc_features


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [sanitize_report(text) for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings


class PromptLearner(nn.Module):
    """CoOp prompt learner.

    Each class prompt is fed to the frozen text encoder as input embeddings:
        [<s>] [ctx_1 ... ctx_n] [class name tokens] [</s>] [<pad> ...]
    Only `ctx` is learnable; the <s>/class/</s> slots hold the text encoder's
    word embeddings (`token_prefix` / `token_suffix`).

    csc=False: one unified context shared by all classes, ctx (n_ctx, hidden).
    csc=True:  class-specific context, ctx (n_cls, n_ctx, hidden).
    """

    def __init__(
        self, classnames, n_ctx, tokenizer, text_encoder, csc=False, device="cuda"
    ):
        super().__init__()
        n_cls = len(classnames)
        hidden_size = text_encoder.config.hidden_size

        # Random init of the learnable context vectors
        if csc:
            ctx_vectors = torch.empty(n_cls, n_ctx, hidden_size)
        else:
            ctx_vectors = torch.empty(n_ctx, hidden_size)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors.to(device))
        print(f"Context: {'class-specific' if csc else 'unified'}")

        # "X" is a single BPE token for Clinical-Longformer, so each placeholder
        # occupies exactly one position and is replaced by one ctx vector
        prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]
        tokenized = tokenizer(
            prompts, truncation=True, padding=True, return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)  # (n_cls, seq_len)
        attention_mask = tokenized["attention_mask"].to(device)

        # Position ids exactly as the encoder derives them from input_ids
        # (pads -> padding_idx). With inputs_embeds it would otherwise fall back
        # to plain sequential positions.
        padding_idx = text_encoder.config.pad_token_id
        position_ids = (
            torch.cumsum(attention_mask, dim=1) * attention_mask + padding_idx
        ).long()

        # Derived from the class names and the frozen encoder, so they are
        # rebuilt rather than saved/loaded with checkpoints
        self.register_buffer("input_ids", input_ids, persistent=False)
        self.register_buffer("attention_mask", attention_mask, persistent=False)
        self.register_buffer("position_ids", position_ids, persistent=False)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.build_token_buffers(text_encoder)

    @torch.no_grad()
    def build_token_buffers(self, text_encoder):
        """(Re)compute the frozen <s> / class / </s> token embeddings.

        Must be called again after loading pretrained text-encoder weights,
        otherwise the buffers hold the embeddings of the old weights.
        """
        embedding = text_encoder.get_input_embeddings()(self.input_ids)
        self.register_buffer(
            "token_prefix", embedding[:, :1, :].clone(), persistent=False
        )  # <s>
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :].clone(), persistent=False
        )  # class name + </s> + pad

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:  # generic (shared) context
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # (n_cls, seq_len, hidden)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class CoCoOpPromptLearner(PromptLearner):
    def __init__(
        self, classnames, n_ctx, tokenizer, text_encoder, image_dim=512, device="cuda"
    ):
        super().__init__(classnames, n_ctx, tokenizer, text_encoder, device)

        hidden_size = text_encoder.config.hidden_size  # 768 for Clinical-Longformer
        self.meta_net = nn.Sequential(
            nn.Linear(image_dim, image_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(image_dim // 16, hidden_size),
        )

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)  # (seq_len, hidden_size)

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class MerlinArchitecture(nn.Module):
    def __init__(
        self,
        classnames: list = ["negative", "hcc"],
        n_ctx: int = 16,
        csc: bool = False,
        use_coop: bool = False,
        init_logit_scale: float = 1.0,
        ImageEmbedding: bool = False,
        HCCClassification: bool = False,
        PhenotypeCls: bool = False,
        rotate_flip: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.use_coop = use_coop
        self.ImageEmbedding = ImageEmbedding
        self.HCCClassification = HCCClassification
        self.PhenotypeCls = PhenotypeCls
        self.rotate_flip = rotate_flip

        # Initialize Image Encoder
        self.encode_image = ImageEncoder(
            ImageEmbedding=self.ImageEmbedding,
            HCCClassification=self.HCCClassification,
            PhenotypeCls=self.PhenotypeCls,
            rotate_flip=self.rotate_flip,
        ).to(device)

        # Same text tower in both modes; CoOp feeds it prompt embeddings instead
        # of tokenised reports (see encode_coop_text)
        self.encode_text = TextEncoder().to(device)

        # Configure architecture based on CoOp initialization choice
        if self.use_coop:
            assert classnames is not None, (
                "A list of classnames must be provided when use_coop=True"
            )

            # Freeze the original underlying transformer backbone weights
            for param in self.encode_text.text_encoder.parameters():
                param.requires_grad = False

            # Freeze the projection bottleneck layer
            for param in self.encode_text.linear_layer.parameters():
                param.requires_grad = False

            # Initialize Prompt Learner
            self.prompt_learner = PromptLearner(
                classnames=classnames,
                n_ctx=n_ctx,
                tokenizer=self.encode_text.tokenizer,
                text_encoder=self.encode_text.text_encoder,
                csc=csc,
                device=device,
            )

            # Initialize Prompt Learner
            # self.prompt_learner = CoCoOpPromptLearner(
            #     classnames=classnames,
            #     n_ctx=n_ctx,
            #     tokenizer=self.encode_text.tokenizer,
            #     text_encoder=self.encode_text.text_encoder,
            #     device=device
            # )

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def encode_coop_text(self):
        """Encode the learned class prompts with the frozen text tower.

        Mirrors TextEncoder.forward ([CLS] state -> linear_layer, no global
        attention) so the prompts live in Merlin's image-text space. The text
        encoder is frozen via requires_grad=False, not no_grad, so gradients
        still reach the context vectors.
        """
        prompts = self.prompt_learner()  # (n_cls, seq_len, hidden)
        text_outputs = self.encode_text.text_encoder(
            inputs_embeds=prompts,
            attention_mask=self.prompt_learner.attention_mask,
            position_ids=self.prompt_learner.position_ids,
        )
        cls_state = text_outputs.last_hidden_state[:, 0, :]
        return self.encode_text.linear_layer(cls_state)  # (n_cls, 512)

    def encode_cocoop_text(self, image_features):
        prompts = self.prompt_learner(image_features)
        B, C, L, H = prompts.shape

        prompts = prompts.reshape(B * C, L, H)

        attention_mask = (
            self.prompt_learner.attention_mask.unsqueeze(0)
            .expand(B, -1, -1)
            .reshape(B * C, L)
        )

        position_ids = (
            self.prompt_learner.position_ids.unsqueeze(0)
            .expand(B, -1, -1)
            .reshape(B * C, L)
        )

        text_outputs = self.encode_text.text_encoder(
            inputs_embeds=prompts,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        raw = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.encode_text.linear_layer(raw)
        return text_features.reshape(B, C, -1)  # (B, n_cls, 512)

    def forward(self, image, text=None):
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            return image_features
        elif self.HCCClassification and text is None:
            hcc_probs = self.encode_image(image)
            return hcc_probs
        elif self.PhenotypeCls and text is None:
            phenotype_probs = self.encode_image(image)
            return phenotype_probs
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif self.HCCClassification and text is not None:
            raise ValueError("Text input not required for HCC classification")
        elif self.PhenotypeCls and text is not None:
            raise ValueError("Text input not required for phenotype classification")
        elif text is None and not self.use_coop:
            raise ValueError("Text input required for Image and Text embedding")
        elif self.use_coop and text is not None:
            raise ValueError(
                "Text parameter should not be provided during CoOp forward passes."
            )

        image_features, ehr_features, hcc_features = self.encode_image(image)

        if self.use_coop:
            text_features = self.encode_coop_text()
            # text_features = self.encode_cocoop_text(image_features)
        else:
            text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, ehr_features, hcc_features, text_features
