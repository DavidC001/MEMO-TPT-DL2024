import torch
from torch import nn

import numpy as np


from clip import load, tokenize


class EasyPromptLearner(nn.Module):
    def __init__(self, device, clip, base_prompt="a photo of [CLS] with ground"):
        super().__init__()

        self.device = device
        self.base_prompt = base_prompt
        self.clip = clip
        self.tkn_embedder = clip.token_embedding

    def prepare_prompts(self, classnames):
        print("[PromptLearner] Preparing prompts")
        # get numbr of classes
        self.cls_num = len(classnames)

        # get prompt text prefix and suffix
        txt_prefix = self.base_prompt.split("[CLS]")[0]
        txt_suffix = self.base_prompt.split("[CLS]")[1]

        # tokenize the prefix and suffix
        tkn_prefix = tokenize(txt_prefix)
        tkn_suffix = tokenize(txt_suffix)
        tkn_pad = tokenize("")
        tkn_cls = tokenize(classnames)

        # get the index of the last element of the prefix and suffix
        idx = torch.arange(tkn_prefix.shape[1], 0, -1)
        self.indp = torch.argmax((tkn_prefix == 0) * idx, 1, keepdim=True)
        self.inds = torch.argmax((tkn_suffix == 0) * idx, 1, keepdim=True)

        # token length for each class
        self.indc = torch.argmax((tkn_cls == 0) * idx, 1, keepdim=True)

        # get the prefix, suffix, SOT and EOT
        self.tkn_sot = tkn_prefix[:, :1]
        self.tkn_prefix = tkn_prefix[:, 1 : self.indp - 1]
        self.tkn_suffix = tkn_suffix[:, 1 : self.inds - 1]
        self.tkn_eot = tkn_suffix[:, self.inds - 1 : self.inds]
        self.tkn_pad = tkn_pad[:, 2:]

        # load segments to CUDA, be ready to be embedded
        self.tkn_sot = self.tkn_sot.cuda(self.device)
        self.tkn_prefix = self.tkn_prefix.cuda(self.device)
        self.tkn_suffix = self.tkn_suffix.cuda(self.device)
        self.tkn_eot = self.tkn_eot.cuda(self.device)
        self.tkn_pad = self.tkn_pad.cuda(self.device)

        self.tkn_cls = tkn_cls.cuda(self.device)

        # gets the embeddings
        with torch.no_grad():
            self.emb_sot = self.tkn_embedder(self.tkn_sot)
            self.emb_prefix = self.tkn_embedder(self.tkn_prefix)
            self.emb_suffix = self.tkn_embedder(self.tkn_suffix)
            self.emb_eot = self.tkn_embedder(self.tkn_eot)
            self.emb_cls = self.tkn_embedder(self.tkn_cls)
            self.emb_pad = self.tkn_embedder(self.tkn_pad)

        # take out the embeddings of the class tokens (they are different lenghts)
        self.all_cls = []
        for i in range(self.cls_num):
            self.all_cls.append(self.emb_cls[i][1 : self.indc[i] - 1])

        # prepare the prompts, they are needed for text encoding
        txt_prompts = [self.base_prompt.replace("[CLS]", cls) for cls in classnames]
        self.tkn_prompts = tokenize(txt_prompts)

        # set the inital context, this will be reused at every new inference
        # this is the context that will be optimized
        self.emb_prefix = nn.Parameter(self.emb_prefix)
        self.emb_suffix = nn.Parameter(self.emb_suffix)

        self.pre_init_state = self.emb_prefix.detach().clone()
        self.suf_init_state = self.emb_suffix.detach().clone()

    def build_ctx(self):
        prompts = []
        for i in range(self.cls_num):
            pad_size = self.emb_cls.shape[1] - (
                self.emb_prefix.shape[1]
                + self.indc[i].item()
                + self.emb_suffix.shape[1]
            )

            prompt = torch.cat(
                (
                    self.emb_sot,
                    self.emb_prefix,
                    self.all_cls[i].unsqueeze(0),
                    self.emb_suffix,
                    self.emb_eot,
                    self.emb_pad[:, :pad_size],
                ),
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        return prompts

    def forward(self):

        return self.build_ctx()

    def reset(self):
        # ctx_vectors = self.init_ctx
        # self.ctx = ctx_vectors.detach().clone()
        pre_ctx = self.pre_init_state
        suf_ctx = self.suf_init_state
        # copy
        # self.emb_prefix.data = pre_ctx
        # self.emb_suffix.data = suf_ctx

        self.emb_prefix.copy_(pre_ctx)  # to be optimized
        self.emb_suffix.copy_(suf_ctx)  # to be optimized


class EasyTPT(nn.Module):
    def __init__(self, device, base_prompt="a photo of a [CLS]"):
        super(EasyTPT, self).__init__()
        self.device = device

        ###TODO: tobe parametrized
        arch = "RN50"
        DOWNLOAD_ROOT = "~/.cache/clip"
        ###

        self.base_prompt = base_prompt

        # Load clip
        clip, self.preprocess = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.clip = clip
        self.dtype = clip.dtype
        self.image_encoder = clip.encode_image
        self.text_encoder = clip.encode_text

        self.prompt_learner = EasyPromptLearner(device, clip, base_prompt)

    def forward(self, x):
        with torch.no_grad():
            image_feat = self.image_encoder(x)

        emb_prompts = self.prompt_learner()

        text_features = self.custom_encoder(
            emb_prompts, self.prompt_learner.tkn_prompts
        )
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_feat @ text_features.t()

        # breakpoint()
        return logits

    def cosine_similarity(self, images_z, texts_z):
        # Normalise the image and the text
        images_z /= images_z.norm(dim=-1, keepdim=True)
        texts_z /= texts_z.norm(dim=-1, keepdim=True)

        # Evaluate the cosine similarity between the sets of features
        similarity = images_z @ texts_z.T

        return similarity

    def custom_encoder(self, prompts, tokenized_prompts):
        x = prompts + self.clip.positional_embedding
        x = x.permute(1, 0, 2).type(self.dtype)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.clip.text_projection
        )

        return x

    def reset(self):
        self.prompt_learner.reset()