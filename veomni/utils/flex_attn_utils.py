import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        and_masks,
        create_block_mask,
        flex_attention,
        or_masks,
    )
    flex_attention = torch.compile(flex_attention, dynamic=True, mode='max-autotune') #, mode="max-autotune-no-cudagraphs")
except ImportError:
    print("To enable flexattention, please install torch>=2.5.0")


def calculate_pad_length(seqlen, div_num):
    """
    calculate the min padlen,  make (seqlen + padlen) can be divisible by div_num

    :param seqlen: int, 序列长度
    :param div_num: int, 整数
    :return: int, 最小填充长度
    """
    if seqlen % div_num == 0:
        return 0
    else:
        padlen = div_num - (seqlen % div_num)
        return padlen


def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), counts)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def generate_doc_mask_mod(mask_mod: _mask_mod_signature, offsets: Tensor) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        q_logical = q_idx - offsets[document_id[q_idx]]
        kv_logical = kv_idx - offsets[document_id[kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask

    return doc_mask_mod


def generate_sliding_window(window_size: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size

    sliding_window_mask = and_masks(sliding_window, causal_mask)
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask


def create_flex_mask(document_ids, modality_indicators, slen):
    '''
    Current version:
    1. document attention
    2. within each document, causal attention. Within a same image, full attention
    '''

    def samedoc_mask(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx]

    def sameimg_mask(b, h, q_idx, kv_idx):
        is_image = modality_indicators[q_idx] > 0
        return is_image & (modality_indicators[q_idx] == modality_indicators[kv_idx])

    samedoc_causal_mask = and_masks(causal_mask, samedoc_mask)
    mask_mod = or_masks(samedoc_causal_mask, sameimg_mask)

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=slen,
        KV_LEN=slen,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask

def create_mmformer_full_flex_mask_padding(image_mask, div_num):
    # create a full attention mask for vision-only cases
    batchsz = image_mask.size(0)
    seq_len = image_mask.size(-1)
    # padlen = calculate_pad_length(seq_len, div_num)
    # if padlen > 0:
        # image_mask = F.pad(image_mask, (0, padlen), value=False)

    def full_mask(b, h, q_idx, kv_idx):
        return image_mask[b, kv_idx]  # image_mask [batch size, seqlen]
    
    block_mask = create_block_mask(
        full_mask,
        B=batchsz,
        H=None,
        Q_LEN=div_num,
        KV_LEN=div_num,
        BLOCK_SIZE=128,
        _compile=True,
    )

    return block_mask

def create_fast_flex_mask_padding(sample_ids, flex_indicators, div_num, block_size=64):
    """
    Builds a flexible attention mask with padding for a batch.

    1. Attention is limited to tokens within the same sample.
    2. Causal attention is applied within each sample.
    3. Full attention is applied to tokens from the same image/frame.
    4. Sequence length is padded to be divisible by `div_num`.

    Args:
        sample_ids: [B, L] -> Sample IDs for each sequence in the batch.
        flex_indicators: [B, L] -> Unique ID for each image/video frame, 0 for text.
        div_num: The number to make the sequence length a multiple of.
    """
    B, slen = sample_ids.shape
    padlen = div_num - slen
    
    # Pad the input tensors
    if padlen > 0:
        sample_ids = F.pad(sample_ids, (0, padlen), value=-1)
        flex_indicators = F.pad(flex_indicators, (0, padlen), value=-1)
    
    def same_sample_mask(b, h, q_idx, kv_idx):
        """Ensures attention is restricted to tokens within the same sample."""
        batch_sample_ids = sample_ids[b]
        valid_sample = batch_sample_ids[q_idx] >= 0
        return valid_sample & (batch_sample_ids[q_idx] == batch_sample_ids[kv_idx])
    
    def same_image_mask(b, h, q_idx, kv_idx):
        """Allows full attention for tokens from the same image."""
        batch_sample_ids = sample_ids[b]
        batch_flex_indicators = flex_indicators[b]
        
        is_vision = batch_flex_indicators[q_idx] > 0
        same_sample = batch_sample_ids[q_idx] == batch_sample_ids[kv_idx]
        same_image = batch_flex_indicators[q_idx] == batch_flex_indicators[kv_idx]
        return is_vision & same_image & same_sample

    def causal_mask(b, h, q_idx, kv_idx):
        """Standard causal mask."""
        return q_idx >= kv_idx
    
    # Combine masks: causal attention within a sample OR full attention for same image tokens
    same_sample_causal_mask = and_masks(causal_mask, same_sample_mask)
    final_mask = or_masks(same_sample_causal_mask, same_image_mask)
    
    # Create the block mask
    block_mask = create_block_mask(
        final_mask, 
        B=B,
        H=None, 
        Q_LEN=slen + padlen, 
        KV_LEN=slen + padlen, 
        BLOCK_SIZE=block_size,
        _compile=True,
    )
    
    return block_mask


def create_mmformer_flex_mask_padding(image_mask, atttention_mask, div_num):
    batchsz = image_mask.size(0)
    seq_len = image_mask.size(-1)
    padlen = calculate_pad_length(seq_len, div_num)
    eff_len = seq_len
    if padlen > 0:
        image_mask = F.pad(image_mask, (0, padlen), value=False)
        atttention_mask = F.pad(atttention_mask, (0, padlen), value=False)

    def full_mask(b, h, q_idx, kv_idx):
        return image_mask[b, kv_idx] & atttention_mask[b, q_idx] # image_mask [batch size, seqlen]

    def causal_mask(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & atttention_mask[b, q_idx]

    mask_mod = or_masks(full_mask, causal_mask)
    block_mask = create_block_mask(
        mask_mod,
        B=batchsz,
        H=None,
        Q_LEN=seq_len + padlen,
        KV_LEN=seq_len + padlen,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask

def create_mmformer_sptok_flex_mask_padding(image_mask, atttention_mask, cls_attention, special_tokens, div_num):
    # it is a variant which support one or some special tokens can attend to full seq content.
    # special_tokens: tokens able to attend to all tokens in sequence.
    batchsz = image_mask.size(0)
    seq_len = image_mask.size(-1)
    padlen = calculate_pad_length(seq_len, div_num)
    eff_len = seq_len
    if padlen > 0:
        image_mask = F.pad(image_mask, (0, padlen), value=False)
        atttention_mask = F.pad(atttention_mask, (0, padlen), value=False)
        special_tokens = F.pad(atttention_mask, (0, padlen), value=False)
        cls_attention = F.pad(cls_attention, (0, padlen), value=False)
    
    # we impl a comprehensive mm attention mask by summarying the following 3 masks
    def special_mask(b, h, q_idx, kv_idx):
        # return special_tokens[b, q_idx] & atttention_mask[b, kv_idx]
        return special_tokens[b, q_idx] & cls_attention[b, kv_idx]

    def full_mask(b, h, q_idx, kv_idx):
        return image_mask[b, kv_idx] & atttention_mask[b, q_idx] # image_mask [batch size, seqlen]

    def causal_mask(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & atttention_mask[b, q_idx]

    mask_mod = or_masks(full_mask, causal_mask)
    mask_mod = or_masks(mask_mod, special_mask)
    block_mask = create_block_mask(
        mask_mod,
        B=batchsz,
        H=None,
        Q_LEN=seq_len + padlen,
        KV_LEN=seq_len + padlen,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask

def create_vqa_flex_mask_padding(image_mask, vqa_ids, div_num):
    batchsz = image_mask.size(0)
    seq_len = image_mask.size(-1)
    padlen = calculate_pad_length(seq_len, div_num)
    eff_len = seq_len
    if padlen > 0:
        image_mask = F.pad(image_mask, (0, padlen), value=False)
        # atttention_mask = F.pad(atttention_mask, (0, padlen), value=False)
        vqa_ids = F.pad(vqa_ids, (0, padlen), value=0)

    def full_mask(b, h, q_idx, kv_idx):
        return image_mask[b, kv_idx] & (q_idx < eff_len) # image_mask [batch size, seqlen]
    
    def samevqa_mask(b, h, q_idx, kv_idx):
        return (vqa_ids[b, q_idx] == vqa_ids[b, kv_idx]) & (vqa_ids[b, q_idx] > 0) & (q_idx >= kv_idx)

    mask_mod = or_masks(full_mask, samevqa_mask)
    block_mask = create_block_mask(
        mask_mod,
        B=batchsz,
        H=None,
        Q_LEN=seq_len + padlen,
        KV_LEN=seq_len + padlen,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask

def calculate_pad_length(seqlen, div_num):
    if seqlen % div_num == 0:
        return 0
    else:
        padlen = div_num - (seqlen % div_num)
        return padlen

def create_mmformer_vqa_flex_mask_padding(
    image_mask,
    div_num,
    vqa_flex_indicator
):
    batchsz = image_mask.size(0)
    seq_len = image_mask.size(-1)
    padlen = calculate_pad_length(seq_len, div_num)
    # print(f"padlen {padlen}")
    if padlen > 0:
        vqa_flex_indicator = F.pad(vqa_flex_indicator, (0, padlen), value=-1)
        image_mask = F.pad(image_mask, (0, padlen), value=False)

    def qa_causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx & (vqa_flex_indicator[b, q_idx] == vqa_flex_indicator[b, kv_idx])
    
    def img_full_mask(b, h, q_idx, kv_idx):
        return image_mask[b, kv_idx]
    
    mask_mod = or_masks(img_full_mask, qa_causal_mask)
    # mask_mod = img_full_mask
    block_mask = create_block_mask(
        mask_mod,
        B=batchsz,
        H=None,
        Q_LEN=seq_len + padlen,
        KV_LEN=seq_len + padlen,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask

def create_pack_flex_attn_mask(
    dense_flex_mask,
    div_num
):
    batchsz, seq_len, seq_len = dense_flex_mask.shape
    padlen = calculate_pad_length(seq_len, div_num)

    if padlen > 0:
        dense_flex_mask = F.pad(dense_flex_mask, (0, padlen, 0, padlen), value=False)
    
    def get_pixel_mask(b, h, q_idx, kv_idx):
        return dense_flex_mask[b, q_idx, kv_idx]

    mask_mod = get_pixel_mask
    block_mask = create_block_mask(
        mask_mod,
        B=batchsz,
        H=None,
        Q_LEN=seq_len + padlen,
        KV_LEN=seq_len + padlen,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask

def create_flex_mask_padding(document_ids, modality_indicators, div_num, text_agnostic=False):
    '''
    Current version:
    1. document attention
    2. within each document, causal attention. Within a same image, full attention
    seqlen padded to divisable by some number
    document_ids [1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,4]-> a sample (v-t pair has a same id)
    modality_indicators [1,1,1,0,2,2,0,3,3,3,0,4,4,5,5,0] each frame has different modality indicator, text part is 0, each image or video frame has a unique modality indicator
    '''
    slen = document_ids.size(-1)
    padlen = calculate_pad_length(seqlen=slen, div_num=div_num)
    if padlen > 0:
        pad_doc_id = -100 # pad ignore id 
        document_ids = F.pad(document_ids, (0, padlen), value=pad_doc_id)
        modality_indicators = F.pad(modality_indicators, (0, padlen), value=-1)

    def samedoc_mask(b, h, q_idx, kv_idx):
        valid_doc = document_ids[q_idx] >= 0
        return valid_doc & (document_ids[q_idx] == document_ids[kv_idx]) # same sample pair

    def sameimg_mask(b, h, q_idx, kv_idx):
        is_image = modality_indicators[q_idx] >= 0 # vision
        same_doc = document_ids[q_idx] == document_ids[kv_idx] # same sample pair 
        return is_image & (modality_indicators[q_idx] == modality_indicators[kv_idx]) & same_doc # vision & same image or frame & same sample

    # error func
    def text_agnostic_mask(b, h, q_idx, kv_idx):
        is_image = modality_indicators[q_idx] >= 0
        return (~is_image) | (modality_indicators[kv_idx] == modality_indicators[q_idx]) | q_idx < kv_idx # text | same image or frame & causal

    def text_agnostic_mask_v2(b, h, q_idx, kv_idx):
        is_image = modality_indicators[q_idx] >= 0
        return (~is_image) & (q_idx < kv_idx) # text | same image or frame & causal

    samedoc_causal_mask = and_masks(causal_mask, samedoc_mask)
    mask_mod = or_masks(samedoc_causal_mask, sameimg_mask)

    if text_agnostic:
        mask_mod = and_masks(mask_mod, text_agnostic_mask_v2)

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=slen + padlen,
        KV_LEN=slen + padlen,
        BLOCK_SIZE=128,
        _compile=True,
    )
    # print(f"{block_mask}, {document_ids.max()}")
    return block_mask


def create_sparse_mask(segment_ids, indicators):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_mask(b, h, q_idx, kv_idx):
        return (indicators[q_idx] == indicators[kv_idx]) & (indicators[q_idx] >= 0)

    def sample_mask(b, h, q_idx, kv_idx):
        return segment_ids[q_idx] == segment_ids[kv_idx]

    return and_masks(or_masks(causal_mask, full_mask), sample_mask)

def dummy_full_sparse_attn_mask(
    batchsz,
    div_num,
    block_size=128,
):
    # create a almost fully sparse attn mask
    # batchsz = image_mask.size(0)
    # seq_len = image_mask.size(-1)

    def sparse_mask_gen(b, h, q_idx, kv_idx):
        return (q_idx // block_size) == (kv_idx // block_size)
    
    mask_mod = sparse_mask_gen

    block_mask = create_block_mask(
        mask_mod,
        B=batchsz,
        H=None,
        Q_LEN=div_num,
        KV_LEN=div_num,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask
    


