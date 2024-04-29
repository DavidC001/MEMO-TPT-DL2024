from copy import deepcopy
import torch
from torch import nn

import numpy as np


from clip import load, tokenize


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
        self.clip = clip
        self.tkn_embedder = clip.token_embedding
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
        self.txt_prompts = [
            self.base_prompt.replace("[CLS]", cls) for cls in self.classnames
        ]
        self.tkn_prompts = tokenize(self.txt_prompts)

        # set the inital context, this will be reused at every new inference
        # this is the context that will be optimized

        if self.split_ctx:
            self.emb_prefix = nn.Parameter(self.emb_prefix)
            self.emb_suffix = nn.Parameter(self.emb_suffix)
            self.pre_init_state = self.emb_prefix.detach().clone()
            self.suf_init_state = self.emb_suffix.detach().clone()
        else:
            self.ctx = torch.cat((self.emb_prefix, self.emb_suffix), dim=1)
            self.ctx = nn.Parameter(self.ctx)
            self.ctx_init_state = self.ctx.detach().clone()

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
            self.emb_prefix.data.copy_(self.pre_init_state.data)  # to be optimized
            self.emb_suffix.data.copy_(self.suf_init_state.data)  # to be optimized
        else:
            self.ctx.data.copy_(self.ctx_init_state.data)  # to be optimized


class EasyTPT(nn.Module):
    def __init__(
        self,
        device,
        base_prompt="a photo of a [CLS]",
        arch="RN50",
        splt_ctx=False,
        classnames=None,
        ttt_steps=1,
        augs=64,
        lr=0.005,
    ):
        super(EasyTPT, self).__init__()
        self.device = device

        # switch to GPU if available
        if not torch.cuda.is_available():
            print("Using CPU this is no bueno")
        else:
            print("Switching to GPU, brace yourself!")
            torch.cuda.set_device(device)
            self.cuda(device)

        ###TODO: tobe parametrized
        DOWNLOAD_ROOT = "~/.cache/clip"
        ###

        self.base_prompt = base_prompt
        self.ttt_steps = ttt_steps
        self.augs = augs

        # Load clip
        clip, self.preprocess = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.clip = clip
        self.dtype = clip.dtype
        self.image_encoder = clip.encode_image
        self.text_encoder = clip.encode_text

        # freeze the parameters
        for name, param in self.named_parameters():
            param.requires_grad_(False)

        # create the prompt learner
        self.prompt_learner = EasyPromptLearner(
            device, clip, base_prompt, splt_ctx, classnames
        )

        # create optimizer and save the state
        trainable_param = self.prompt_learner.parameters()
        self.optimizer = torch.optim.AdamW(trainable_param, lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())

    def forward(self, x):

        self.eval()

        image = x[0].cuda()
        images = torch.cat(x, dim=0).cuda()

        self.test_time_tuning(images, ttt_steps=self.ttt_steps)

        with torch.no_grad():
            out = self.inference(image)

        return out

    def inference(self, x):

        with torch.no_grad():
            image_feat = self.image_encoder(x)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        emb_prompts = self.prompt_learner()

        txt_features = self.custom_encoder(emb_prompts, self.prompt_learner.tkn_prompts)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_feat @ txt_features.t()

        # breakpoint()
        return logits

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
        self.optimizer.load_state_dict(self.optim_state)
        self.prompt_learner.reset()

    def select_confident_samples(self, logits, top):
        """
        Performs confidence selection, will return the indexes of the
        augmentations with the highest confidence as well as the filtered
        logits

        Parameters:
        - logits (torch.Tensor): the logits of the model [NAUGS, NCLASSES]
        - top (float): the percentage of top augmentations to use
        """
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[
            : int(batch_entropy.size()[0] * top)
        ]
        return logits[idx], idx

    def avg_entropy(self, outputs):
        logits = outputs - outputs.logsumexp(
            dim=-1, keepdim=True
        )  # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(
            logits.shape[0]
        )  # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

    def test_time_tuning(self, images, ttt_steps=4):
        """
        Perform prompt tuning ttt_steps times

        Parameters:
        - images (torch.Tensor): the augmentations batch of size [NAUGS, 3, 224, 224]
        - ttt_steps (int): the number of tuning steps

        Doesn't return anything but updates the prompt context
        """
        selected_idx = None

        for j in range(ttt_steps):

            output = self.inference(images)
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = self.select_confident_samples(output, 0.10)

            loss = self.avg_entropy(output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def get_optimizer(self):
        """
        Returns the optimizer

        Returns:
        - torch.optim: the optimizer
        """
        return self.optimizer
