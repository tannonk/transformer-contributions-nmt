import logging
import warnings
from functools import partial
from collections import defaultdict
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils


# From hub.utils
import copy
from typing import Any, Dict, Iterator, List
from omegaconf import open_dict
from fairseq import utils
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

    # uncomment for running in notebook
try:
    from wrappers.transformer_wrapper import FairseqTransformerHub
    from wrappers.interactive import *
except:
    # uncomment for testing, e.g. python wrappers/bart_wrapper.py
    from interactive import *
    from transformer_wrapper import FairseqTransformerHub 

# from fairseq.models.transformer import TransformerModel
from fairseq.models.bart import BARTModel

from einops import rearrange


class FairseqBartTransformerHub(FairseqTransformerHub):

    def __init__(self, cfg, task, models):
        super().__init__(cfg, task, models)
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def from_pretrained(cls, checkpoint_dir, checkpoint_file, data_name_or_path,
                         source_lang=None, target_lang=None, bpe=None, lang_pairs=None, fixed_dictionary=None):
        hub_interface = BARTModel.from_pretrained(checkpoint_dir, checkpoint_file, data_name_or_path,
                                                    source_lang=source_lang,
                                                    target_lang=target_lang,
                                                    )
        # hub_interface = checkpoint_dir, checkpoint_file, data_name_or_path)
        return cls(hub_interface.cfg, hub_interface.task, hub_interface.models)
    
    # def encode(self, sentence):
    #     return self.encode(sentence)

    def decode(self, tensor, as_string=False):
        toks = []
        # for token in torch.squeeze(tensor):
        for token_id in tensor.tolist(): 
            token_id_mapping = self.tgt_dict[token_id]
            try:
                token = self.bpe.decode(token_id_mapping)
            except ValueError: # token is a special token
                token = token_id_mapping
            
            toks.append(token)
            # tok.append(dictionary[token])
        # tok = dictionary.string(tensor,'sentencepiece').split()
        if as_string:
            return ''.join(toks).replace('▁', ' ')
        else:
            return toks

    def get_sample(self, split, index):

        if split not in self.task.datasets.keys():
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]['source']
        src_tok = self.decode(src_tensor, self.task.source_dictionary)
        src_sent = self.decode(src_tensor, self.task.source_dictionary, as_string=True)

        tgt_tensor = self.task.dataset(split)[index]['target']
        tgt_tok = self.decode(tgt_tensor, self.task.target_dictionary)
        tgt_sent = self.decode(tgt_tensor, self.task.target_dictionary, as_string=True)

        return {
                'src_tok': src_tok,
                'src_tensor': src_tensor,
                'tgt_tok': tgt_tok,
                'tgt_tensor': tgt_tensor,
                'src_sent': src_sent,
                'tgt_sent': tgt_sent
            }
    def get_interactive_sample(self, i, test_set_dir, src, tgt,
                                tokenizer, prepare_input_encoder,
                                prepare_input_decoder, hallucination=None):
        """Get interactive sample from tokenized and original word files."""
        test_src_bpe = f'{test_set_dir}/test.{tokenizer}.{src}'
        test_tgt_bpe = f'{test_set_dir}/test.{tokenizer}.{tgt}'
        test_src_word = f'{test_set_dir}/test.{src}'
        test_tgt_word = f'{test_set_dir}/test.{tgt}'

        with open(test_src_bpe, encoding="utf-8") as fbpe:
            # BPE source sentences
            src_bpe_sents = fbpe.readlines()
        with open(test_tgt_bpe, encoding="utf-8") as fbpe:
            # BPE target sentences
            tgt_bpe_sents = fbpe.readlines()
        with open(test_src_word, encoding="utf-8") as fword:
            # Original source sentences
            src_word_sents = fword.readlines()
        with open(test_tgt_word, encoding="utf-8") as fword:
            # Original target sentences
            tgt_word_sents = fword.readlines()

        src_word_sent = src_word_sents[i]
        tgt_word_sent = tgt_word_sents[i]

        src_tok_str = src_bpe_sents[i].strip() # removes leading and trailing whitespaces
        tgt_tok_str = tgt_bpe_sents[i].strip() # removes leading and trailing whitespaces

        src_tok, src_tensor = prepare_input_encoder(self, [src_tok_str])
        tgt_tok, tgt_tensor = prepare_input_decoder(self, tgt_tok_str)

        if test_src_word and test_tgt_word:
            src_word_sent = src_word_sents[i]
            tgt_word_sent = tgt_word_sents[i]
            return {
                'src_word_sent': src_word_sent,
                'src_tok': src_tok,
                'src_tok_str': src_tok_str,
                'src_tensor': src_tensor,
                'tgt_word_sent': tgt_word_sent,
                'tgt_tok': tgt_tok,
                'tgt_tok_str': tgt_tok_str,
                'tgt_tensor': tgt_tensor
            }

        return {
            'src_word_sent': None,
            'src_tok': src_tok,
            'src_tok_str': src_tok_str,
            'src_tensor': src_tensor,
            'tgt_word_sent': None,
            'tgt_tok': tgt_tok,
            'tgt_tok_str': tgt_tok_str,
            'tgt_tensor': tgt_tensor
        }

    def trace_forward(self, src_tensor, tgt_tensor):
        r"""Forward-pass through the model.
        Args:
            src_tensor (`tensor`):
                Source sentence tensor.
            tgt_tensor (`tensor`):
                Target sentence tensor (teacher forcing).
        Returns:
            model_output ('tuple'):
                output of the model.
            log_probs:
                log probabilities output by the model.
            encoder_output ('dict'):
                dictionary with 'encoder_out', 'encoder_padding_mask', 'encoder_embedding',
                                'encoder_states', 'src_tokens', 'src_lengths', 'attn_weights'.
            layer_inputs:
                dictionary with the input of the modeules of the model.
            layer_outputs:
                dictionary with the input of the modeules of the model.
        """
        with torch.no_grad():

            layer_inputs = defaultdict(list)
            layer_outputs = defaultdict(list)

            def save_activation(name, mod, inp, out):
                layer_inputs[name].append(inp)
                layer_outputs[name].append(out)

            handles = {}

            for name, layer in self.named_modules():
                handles[name] = layer.register_forward_hook(partial(save_activation, name))
            
            src_tensor = src_tensor.unsqueeze(0).to(self.device)
            # print('SOURCE TENSOR:', src_tensor)
            # prefix eos symbol at start
            # tgt_tensor = torch.cat([torch.tensor([self.task.target_dictionary.eos_index]), tgt_tensor[:-1]]).unsqueeze(0).to(self.device)
            tgt_tensor = tgt_tensor.unsqueeze(0).to(self.device)
            # print('TARGET TENSOR:', tgt_tensor)
            # breakpoint()
            model_output, encoder_out = self.models[0](src_tensor, src_tensor.size(-1), tgt_tensor, )
            # print('MODEL OUTPUT:', model_output)
            log_probs = self.models[0].get_normalized_probs(model_output, log_probs=True, sample=None)
            
            for k, v in handles.items():
                handles[k].remove()
            # breakpoint()
            return model_output, log_probs, encoder_out, layer_inputs, layer_outputs

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        prefix_allowed_tokens_fn=None,
        **kwargs
        ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            # recursive function call with packed input dim() == 2
            return self.generate(tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs)[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(
            self.models,
            gen_args,
            #prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        
        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                # src_str_with_unk = self.string(source_tokens)
                src_str_with_unk = self.decode(source_tokens, as_string=True)
                logger.info("S\t{}".format(src_str_with_unk))
                # import pdb;pdb.set_trace()
                print("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo["tokens"])
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    print("H\t{}\t{}".format(hypo["score"], hypo_str))
                    logger.info(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    print(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    if hypo["alignment"] is not None and getarg(
                        "print_alignment", False
                    ):
                        logger.info(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )
                        print(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )
        return outputs


if __name__ == '__main__':

    model_dir = '/scratch/tkew/ctrl_tokens/resources/models/muss_en_mined'
    checkpoint_file = 'model.pt'

    hub = FairseqBartTransformerHub.from_pretrained(
        model_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=model_dir,
        source_lang='complex',
        target_lang='simple',
    )
    
    NUM_LAYERS = hub.cfg.model.encoder_layers + hub.cfg.model.decoder_layers

    src_sentence = '<DEPENDENCYTREEDEPTHRATIO_0.4> <WORDRANKRATIO_0.75> <REPLACEONLYLEVENSHTEIN_0.65> <LENGTHRATIO_0.75> This is extremely hard to comprehend.'
    print(src_sentence)
    tgt_sentence = 'This is hard to comprehend.'
    print(tgt_sentence)
    breakpoint()
    src_tensor = hub.encode(src_sentence)
    print(src_tensor)
    tgt_tensor = hub.encode(tgt_sentence)
    # prefix target tensor with </s> token for BART model
    tgt_tensor = torch.concat([tgt_tensor[-1:], tgt_tensor[:-1]], dim=0)
    print(tgt_tensor)
    src_tokens = hub.decode(src_tensor)
    print(src_tokens)
    tgt_tokens = hub.decode(tgt_tensor)
    print(tgt_tokens)
    
    print('*** Testing generate()')
    # NOTE: calls to hub.generate() must use beam=1 to avoid error in subsequent calls to hub.forward_trace()
    outputs = hub.generate(src_tensor, beam=1, verbose=True)
    outputs = [hub.decode(hypo['tokens'], as_string=True) for hypo in outputs]
    print(outputs)

    # print('*** Testing generate()')
    # outputs = hub.generate(src_tensor, beam=4, verbose=True)
    # outputs = [hub.decode(hypo['tokens'], as_string=True) for hypo in outputs]
    # print(outputs)

    print('*** Testing trace_forward()')
    ft = hub.trace_forward(src_tensor, tgt_tensor)

    # # breakpoint()
    print('*** Testing get_contribution_rollout()')
    alti1 = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum', pre_layer_norm=False)
    print(alti1['encoder.self_attn'].shape)
    print(alti1['decoder.encoder_attn'].shape)
    print(alti1['total'].shape)
