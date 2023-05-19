# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch 
from cog import BasePredictor, ConcatenateIterator, Input, Path
from lavis.models import load_model_and_preprocess
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # set device to cuda if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct", 
            model_type="vicuna13b", 
            is_eval=True, 
            device=self.device
    )

    def predict(
        self,
        prompt: str = Input(description=f"Text prompt to send to the model."),
        img: Path = Input(description='Image prompt to send to the model.'),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=512,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens. Defaults to 0 (no top-k sampling).",
            ge=0,
            le=500,
            default=0,
        ),
        penalty_alpha: float = Input(
            description="When > 0 and top_k > 1, penalizes new tokens based on their similarity to previous tokens. Can help minimize repitition while maintaining semantic coherence. Set to 0 to disable.",
            ge=0.0,
            le=1,
            default=0.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
        length_penalty: float = Input(
            description="Increasing the length_penalty parameter above 1.0 will cause the model to favor longer sequences, while decreasing it below 1.0 will cause the model to favor shorter sequences.",
            ge=0.01,
            le=5,
            default=1,
        ),
        no_repeat_ngram_size: int = Input(
            description="If set to int > 0, all ngrams of size no_repeat_ngram_size can only occur once.",
            ge=0,
            default=0,
        ),
        seed: int = Input(
            description="Set seed for reproducible outputs. Set to -1 for random seed.",
            ge=-1,
            default=-1,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator[str]:
        
        # set torch seed
        if seed == -1:
            seed = torch.seed()
            print(f"seed: {seed}")

        else:
            print(f'seed: {seed}')
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # prompts = [PROMPT_TEMPLATE.format(input_text=prompt)]
        raw_image = Image.open(img).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        samples = {"image": image, "prompt": prompt}

        # output = self.model.generate({"image": image, "prompt": "What is unusual about this image?"})

        top_k = top_k if top_k > 0 else None
        with torch.inference_mode():
            first_token_yielded = False
            prev_ids = []
            for output in self.model.generate(
                samples=samples,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                top_k = top_k, 
                penalty_alpha = penalty_alpha,
            ):
                cur_id = output.item()

                # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.model.llm_tokenizer.convert_ids_to_tokens(cur_id)

                # skip initial newline, which this almost always yields. hack - newline id = 13.
                if not first_token_yielded and not prev_ids and cur_id == 187:
                    continue

                # underscore means a space, means we yield previous tokens
                if cur_token.startswith("▁"):  # this is not a standard underscore.
                    # first token
                    if not prev_ids:
                        prev_ids = [cur_id]
                        continue

                    # there are tokens to yield
                    else:
                        token = self.model.llm_tokenizer.decode(prev_ids) + ' '
                        prev_ids = [cur_id]

                        if not first_token_yielded:
                            # no leading space for first token
                            token = token.strip() + ' '
                            first_token_yielded = True
                        yield token
                                # End token
                elif cur_token == self.model.llm_tokenizer.eos_token:
                    break
                
                else:
                    prev_ids.append(cur_id)
                    continue

            # remove any special tokens such as </s>
            token = self.model.llm_tokenizer.decode(prev_ids, skip_special_tokens=True)
            if not first_token_yielded:
                # no leading space for first token
                token = token.strip()
                first_token_yielded = True
            yield token

        if debug:
            # function that prints all parameters that were passed to generate
            def print_generate_params():
                print(f"max_length: {max_length}")
                print(f"temperature: {temperature}")
                print(f"top_p: {top_p}")
                print(f"top_k: {top_k}")
                print(f"repetition_penalty: {repetition_penalty}")
                print(f"length_penalty: {length_penalty}")
                print(f"no_repeat_ngram_size: {no_repeat_ngram_size}")
                print(f"penalty_alpha: {penalty_alpha}")
                print(f"seed: {seed}")
            
            print_generate_params()
            
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")
