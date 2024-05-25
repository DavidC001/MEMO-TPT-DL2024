from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


from clip import load, tokenize

import sys

sys.path.append(".")
from EasyModel import EasyModel


class EasyPromptLearner(nn.Module):
    def __init__(
        self,
        device,
        clip,
        base_prompt="a photo of [CLS]",
        splt_ctx=False,
        classnames=None,
    ):
        super().__init__()

        self.device = device
        self.base_prompt = base_prompt
        self.tkn_embedder = clip.token_embedding
        # set requires_grad to False
        self.tkn_embedder.requires_grad_(False)

        self.split_ctx = splt_ctx

        self.prepare_prompts(classnames)

    def prepare_prompts(self, classnames):
        print("[PromptLearner] Preparing prompts")

        self.classnames = classnames
        # self.classnames = [cls.split(",")[0] for cls in self.classnames]

        # get numbr of classes
        self.cls_num = len(self.classnames)

        # get prompt text prefix and suffix
        txt_prefix = self.base_prompt.split("[CLS]")[0]
        txt_suffix = self.base_prompt.split("[CLS]")[1]

        # tokenize the prefix and suffix
        tkn_prefix = tokenize(txt_prefix)
        tkn_suffix = tokenize(txt_suffix)
        tkn_pad = tokenize("")
        tkn_cls = tokenize(self.classnames)

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
        self.tkn_sot = self.tkn_sot.to(self.device)
        self.tkn_prefix = self.tkn_prefix.to(self.device)
        self.tkn_suffix = self.tkn_suffix.to(self.device)
        self.tkn_eot = self.tkn_eot.to(self.device)
        self.tkn_pad = self.tkn_pad.to(self.device)

        self.tkn_cls = tkn_cls.to(self.device)

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
        self.txt_prompts = [
            self.base_prompt.replace("[CLS]", cls) for cls in self.classnames
        ]
        self.tkn_prompts = tokenize(self.txt_prompts)

        # set the inital context, this will be reused at every new inference
        # this is the context that will be optimized

        if self.split_ctx:
            self.pre_init_state = self.emb_prefix.detach().clone()
            self.suf_init_state = self.emb_suffix.detach().clone()
            self.emb_prefix = nn.Parameter(self.emb_prefix)
            self.emb_suffix = nn.Parameter(self.emb_suffix)
            self.register_parameter("emb_prefix", self.emb_prefix)
            self.register_parameter("emb_suffix", self.emb_suffix)
        else:
            self.ctx = torch.cat((self.emb_prefix, self.emb_suffix), dim=1)
            self.ctx_init_state = self.ctx.detach().clone()
            self.ctx = nn.Parameter(self.ctx)
            self.register_parameter("ctx", self.ctx)

    def build_ctx(self):
        prompts = []
        for i in range(self.cls_num):
            pad_size = self.emb_cls.shape[1] - (
                self.emb_prefix.shape[1]
                + self.indc[i].item()
                + self.emb_suffix.shape[1]
            )

            if self.split_ctx:
                prefix = self.emb_prefix
                suffix = self.emb_suffix
            else:
                prefix = self.ctx[:, : self.emb_prefix.shape[1]]
                suffix = self.ctx[:, self.emb_prefix.shape[1] :]

            prompt = torch.cat(
                (
                    self.emb_sot,
                    prefix,
                    self.all_cls[i].unsqueeze(0),
                    suffix,
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

        if self.split_ctx:
            self.emb_prefix.data.copy_(self.pre_init_state)  # to be optimized
            self.emb_suffix.data.copy_(self.suf_init_state)  # to be optimized
        else:
            self.ctx.data.copy_(self.ctx_init_state)  # to be optimized


class EasyTPT(EasyModel):
    def __init__(
        self,
        device,
        base_prompt="a photo of a [CLS]",
        arch="RN50",
        splt_ctx=False,
        classnames=None,
        ensemble=False,
        ttt_steps=1,
        lr=0.005,
        align_steps=0,
    ):
        super(EasyTPT, self).__init__()
        self.device = device

        ###TODO: tobe parametrized
        DOWNLOAD_ROOT = "~/.cache/clip"
        ###

        self.base_prompt = base_prompt
        self.ttt_steps = ttt_steps
        self.selected_idx = None
        self.ensemble = ensemble
        self.align_steps = align_steps
        # Load clip
        clip, self.preprocess = load(
            arch, device=device, download_root=DOWNLOAD_ROOT, jit=False
        )
        clip.float()
        self.clip = clip
        self.dtype = clip.dtype
        self.image_encoder = clip.encode_image
        # self.text_encoder = clip.encode_text

        # freeze the parameters
        for name, param in self.named_parameters():
            param.requires_grad_(False)

        # create the prompt learner
        self.prompt_learner = EasyPromptLearner(
            device, clip, base_prompt, splt_ctx, classnames
        )

        # create optimizer and save the state
        trainable_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"[EasyTPT TPT] Training parameter: {name}")
                trainable_param.append(param)
        self.optimizer = torch.optim.AdamW(trainable_param, lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())

        if align_steps > 0:

            emb_trainable_param = []
            # unfreeze the image encoder
            for name, param in self.clip.visual.named_parameters():
                # if parameter is not attnpoll
                if "attnpool" not in name:
                    param.requires_grad_(True)
                    emb_trainable_param.append(param)
                    print(f"[EasyTPT Emb] Training parameter: {name}")

            self.emb_optimizer = torch.optim.AdamW(emb_trainable_param, 0.0001)
            self.emb_optim_state = deepcopy(self.emb_optimizer.state_dict())
            self.clip_init_state = deepcopy(self.clip.visual.state_dict())

        if self.ensemble:
            print("[EasyTPT] Running TPT in Ensemble mode")

        if self.align_steps > 0:
            print("[EasyTPT] Running TPT with alignment")

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"[EasyTPT] Training parameter: {name}")

    def forward(self, x, top=0.10):
        """
        If x is a list of augmentations, run the confidence selection,
        otherwise just run the inference
        """

        self.eval()
        # breakpoint()
        if isinstance(x, list):
            x = torch.stack(x).to(self.device)

            logits = self.inference(x)
            if self.selected_idx is None:
                logits, self.selected_idx = self.select_confident_samples(logits, top)

                # self.selected_idx = self.select_closest_samples(x, top)
            else:
                logits = logits[self.selected_idx]
        else:
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x = x.to(self.device)

            logits = self.inference(x)

        # print (f"[EasyTPT] input shape: {x.shape}")
        # print("[EasyTPT] logits shape: ", logits.shape)
        return logits

    def inference(self, x):

        with torch.no_grad():
            image_feat = self.image_encoder(x)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        emb_prompts = self.prompt_learner()

        txt_features = self.custom_encoder(emb_prompts, self.prompt_learner.tkn_prompts)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_feat @ txt_features.t()

        return logits

    def align_emb_loss(self, image_feat):

        norm_feat = torch.nn.functional.normalize(image_feat, p=2, dim=1)

        cos_sim = torch.mm(norm_feat, norm_feat.T)

        # noself_mean = (cos_sim.sum() - torch.trace(cos_sim)) / (
        #     cos_sim.numel() - cos_sim.shape[0]
        # )
        loss = 1 - cos_sim.mean()

        return loss

    def align_embeddings(self, x):
        """
        Aligns the embeddings of the image encoder
        """

        self.forward(x)
        self.clip.visual.train()
        x = torch.stack(x).to(self.device)
        selected_augs = torch.index_select(x, 0, self.selected_idx)
        for _ in range(self.align_steps):
            image_feat = self.clip.visual(selected_augs.type(self.dtype))
            loss = self.align_emb_loss(image_feat)
            self.emb_optimizer.zero_grad()
            loss.backward()
            # print("distance before: ", loss.item())
            self.emb_optimizer.step()
        image_feat = self.clip.visual(selected_augs.type(self.dtype))
        loss = self.align_emb_loss(image_feat)
        # print("distance after: ", loss.item())
        self.clip.visual.eval()

    def custom_encoder(self, prompts, tokenized_prompts):
        """
        Custom clip text encoder, unlike the original clip encoder this one
        takes the prompts embeddings from the prompt learner
        """
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
        """
        Resets the optimizer and the prompt learner to their initial state,
        this has to be run before each new test
        """
        self.optimizer.load_state_dict(deepcopy(self.optim_state))
        self.prompt_learner.reset()
        self.selected_idx = None

        if self.align_steps > 0:
            # print("[EasyTPT] Resetting embeddings optimizer")
            self.emb_optimizer.load_state_dict(deepcopy(self.emb_optim_state))
            self.clip.visual.load_state_dict(deepcopy(self.clip_init_state))

    def select_closest_samples(self, x, top):

        with torch.no_grad():
            feat = self.clip.visual(x.type(self.dtype))
            feat = feat / feat.norm(dim=-1, keepdim=True)

            # Compute cosine similarities
            sims = F.cosine_similarity(feat[0].unsqueeze(0), feat[1:], dim=1)
            vals, idxs = torch.topk(sims, int(sims.shape[0] * top))

        return idxs

    def predict(self, images, niter=1):

        # self.reset()
        if self.ensemble:
            with torch.no_grad():
                out = self(images)
                marginal_prob = F.softmax(out, dim=1).mean(0)
                out_id = marginal_prob.argmax().item()
        else:
            if self.align_steps > 0:
                self.align_embeddings(images)

            for _ in range(niter):
                out = self(images)
                loss = self.avg_entropy(out)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                out = self(images[0])
                out_id = out.argmax(1).item()
                prediction = self.prompt_learner.classnames[out_id]

        # return out_id, prediction
        return out_id

    def get_optimizer(self):
        """
        Returns the optimizer

        Returns:
        - torch.optim: the optimizer
        """
        return self.optimizer
