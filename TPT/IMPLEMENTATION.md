# Testtime Prompt Tuning


In the [paper](https://arxiv.org/pdf/2209.07511.pdf) implementation there's *a lot* of stuff happening, this is a very very trimmed down version with a fully functioning TPT implementation and much less boilerplate code. CoOp and CoCoOp are not implemented as they are not *needed* for the task and add another level of complexity and unreadeability to this already convoluted code. This poses a good base for both an eventual reimplementation of the full paper or a simpler version of TPT to be plugged in a more complex pipeline.

Now, documentation is non-existant so I'll try to explain what's happening in the code, should be correct but take it with a grain of salt as reverse engineering uncommented code is no fun.



## Initialization

### Dataset

In short, the dataloader will return a touple containing the images and the intex of the class. Images is a list of which the first element is the original image and the rest are augmentations. 

To create the transform for the agumentations, they use AugMixAgumenter which is a wrapper around AugMix. AugMix is a data augmentation technique that combines multiple augmentations into a single image. AugMixAugmenter applies base_transform *only* on the original image meanwhile *preprocess* is also applied to all the augmentations. 

*Now*, for some reason if the dataset name is composed of a single letter (in this imnplementation it includes all imagenet variations I/A/V/R/K) *augmix* is set to False so the only transofrmations (apart from preprocess) applied to the augmentations are *RandomResizedCrop* and *RandomHorizontalFlip*. Idk the reason for this but it's there, on the paper they mention how the 63 augmentations are in fact random resized crops, but I don't see why they would use AugMix's augmentations for the other datasets. Anyways if you want the random sampled augmentations you can set augmix to True.

### ClipTestTimeTuning

ClipTestTimeTuning is a wrapper around the clip model that allows for prompt tuning

It contains CLIP's text and image encoder and an instance of the PromptLearner class, most of the insiantiation parameters will be directly passed to the PromptLearner class.

### PromptLearner
- classnames:   list of all classnames of the dataset
- batch_size:   basically the number of agumentations + 1
- n_ctx=16:     context size (in tokens) for the prompt vectors. Keep in mind this will be overwritten if ctx_init is provided
- ctx_init:     initial context for the prompt vectors (o.g. "a photo of"). If provided, n_ctx will be set to its length.
- ctx_position: where the class name will be put in the prompt vector, it can be "end", "front" or "middle". If it's "middle" the class name will be put in the posotion of the '[CLS]' token
- learned_cls: this is for context-dependent visual reasoning so we don't really care.Instead of using a binary label "yes/no" or whatever it will learn the CLS token along with the prompt. For more info check the paper but this could be removed as well.

In PromptLearner there's some stuff happening.

First thing first the *ctx_init* is tokenized and run thru the embedder to get the prompt vector, which is saved in *ctx_vectors*. If *ctx_init* is not provided, *ctx_vectors* is randomly initialized, also if the token [CLS] is present the index position is saved, the token is removed and the *ctx_position* is set to "middle".

*ctx_vectors* are repeated to match the size of the batch (only if we aew doing batch-wise prompt tuning), after this they are saved both in *ctx_init_state* since at each inference we will need to reset the prompt vectors to their initial state and in *ctx* which is the actual prompt vector that will be *optimized*.

*ctx* will have shape [prompt_len, emb_size] where prompt_len is the lenght of the tokenized prompt and emb_size is the size of CLIP's embedding.

If *learned_cls* is False (so our case, we are not doing HOI) *name_lens* is set to the token lenght of each classname and *prompts* is set to plaintext prompts for each class.

*tokenized_prompts* is set to the tokenized version of the prompts and it's run thru the embedder to get the prompt vectors, which are saved in *embedding* and these are registered in the buffer. 

Now that we have the *generic* prompt we must prepare the context for the actual class names. This is done **separately** by calling *reset_classnames* after instantiating the model.

*n_cls* is set to the number of classes and the whitespaces in the classes are removed.

*name_lens* is set as the lenghts of the tokenized class names and *prompts* is set to the plaintext prompts for each class, after that the prompts are tokenized and run thru the embedder to get the prompt vectors, which are saved in *embedding*. Just like we did in the previouly in the constructor. It's the same procedure.

At this point we can save the *token_prefix* which should be the tokenized-embedded version of the start of string and the *token_suffix* which is the tokenized-embedded version of everything of what comes after the *ctx_init* prompt.

Basically if we have 200 classes, default CLIP's context length 77 and CLIP's embedding size 512, and a 4 token long prompt like "the photo of a" we will have these shapes:

- *self.ctx* [4, 512] the prompt vector (will be canged by the optim)
- *self.ctx_init_state* [4, 512] again the prompt vector (will stay untouched and used to reset ctx after each inference)
- *self.token_prefix* [200, 1, 512] the prefix is empty
- *self.token_suffix* [200, 72, 512] the suffix the part of the prompt that contains the class name and all the padding



## Classification

The actual inference happens after the prompt tuning procedure.

### Test Time Tuning

This is the **training loop** and it's run for *tta_steps* times. A *selected_idx* variable is instantiated, it will contain the indexes of the agumentations with the lowest entropy and are not discarded in the confidence selection step.

The batch of 64 images is passed thru image encoder with no_grad, we dont want to touch that one. The image encoder is wrapped as well with an additional linear layer that has as ouput size the number of classes we want to classify. 

The text features are extracted in the *get_text_features* function.
It starts with a forward pass of PromptLearner. 

------------------------------------
The first thing to do is to prepare the context vector for all the class names, remember in *self.ctx* we have previously saved the embedded prompt. *ctx* is expanded to match the number of classes in the dataset. At this point we have:

*prefix*&nbsp;&nbsp;&nbsp; [200, 1, 512]\
*ctx*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [200, 4, 512]\
*suffix*&nbsp;&nbsp;&nbsp;&nbsp; [200, 72, 512]

we can just concatenate them along the second dimension to get the full prompt vector for each class. This is saved in *prompts* and returned to the get_text_features function.

The propts are then ran thru CLIP's text encoder, they are normalized and returned.

Image features are normalized as well and the dot product between the image and text features is computed. The logits are then saved in *logits* and returned.

------------------------------------

The confidence selection step is run only once at the fist iteration, regardless of how many tta_steps we intend to do. It computes the batch entropy, sorts the values and slices the list at the desired percentage (10% or 0.1 by default), after that it returns both the filtered outputs and the indexes of the augmentations that are not discarded.

After that the loss is computed (it's the average entropy) and backpropagated.

### Inference

The prompt has now been tuned and it sits in *self.ctx* inside PromptLearner. What remains to do is just to run the orignal image inside the network and get the logits. The logits. The index of the highest value in the logits is the predicted class and the lookup can be done just by checking the value in the classnames list.