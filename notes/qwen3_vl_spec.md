# Qwen3-VL-Embedding-2B — native Bumblebee port spec

Goal: add `Bumblebee.Multimodal.Qwen3VL` (+ vision sub-config + featurizer) so
`Qwen/Qwen3-VL-Embedding-2B` loads natively and produces one shared 2048-d text+image space.
Golden fixtures: `test/fixtures/qwen3_vl/{golden.json,golden_image.png}` (text_norm=image_norm=1.0, dim 2048).
Reference source: transformers `models/qwen3_vl/modeling_qwen3_vl.py` + `vision_utils.py`; weights cached at
`~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B`.

## 2B config (USE THESE — class defaults are 4B/8B and WRONG)
Vision: hidden 1024, depth 24, heads 16 (head_dim 64), intermediate 4096, patch 16, temporal_patch 2,
spatial_merge 2, in_channels 3, num_position_embeddings 2304, out_hidden 2048, deepstack [5,11,17],
LayerNorm eps 1e-6, act gelu_pytorch_tanh, plain 2-layer MLP (NOT SwiGLU), vision RoPE theta 10000 (h/w only).
Text (Qwen3): hidden 2048, layers 28, heads 16, kv_heads 8 (GQA g=2), head_dim 128, intermediate 6144,
rms_eps 1e-6, rope_theta 5e6, act silu (SwiGLU), attention_bias false, vocab 151936, tie_word_embeddings TRUE,
QK-RMSNorm per head (eps 1e-6) before RoPE. mrope_section [24,20,20], mrope_interleaved TRUE.
Tokens: image_pad 151655, video_pad 151656, vision_start 151652, vision_end 151653, pad 151643, eos 151645.
Arch class: Qwen3VLForConditionalGeneration ; model_type qwen3_vl ; pooled dim 2048.

## Vision forward (single image, grid_thw=[1,32,32] -> 1024 patches)
1. pixel_values (Npatch, 3*2*16*16=1536); Conv3d(3->1024,k=(2,16,16),s=same) -> (1024,1024).
2. + bilinear-interp learned pos_embed: table Embedding(2304,1024), grid 48x48 (floor(sqrt(2304)));
   linspace(0,47,h)/w, 4 corners, weights; reorder to 2x2 block-major, repeat t. sum -> (1024,1024).
3. vision RoPE: dim head_dim//2=32, theta 1e4; pos_ids (seq,2)=(h_idx,w_idx) block-major; emb=cat(r,r)->(seq,64);
   apply to full head_dim=64 in fp32, rotate_half split@32.
4. x24 blocks pre-norm: h+=attn(LN(h)); h+=mlp(LN(h)). attn: qkv Linear(1024->3072,bias) ->(seq,3,16,64);
   RoPE q,k; scale 64^-.5; NON-causal, per-image via cu_seqlens; proj 1024->1024. mlp: fc1 1024->4096 bias,
   gelu_tanh, fc2 4096->1024 bias.
5. deepstack: after vision layers {5,11,17} run PatchMerger(use_postshuffle_norm=TRUE) on current h -> (256,2048) each.
6. main PatchMerger(post_shuffle=FALSE): x=LN(1024)(x); view(-1,4096); fc1 4096->4096; GELU(erf, NOT tanh);
   fc2 4096->2048 -> (256,2048). (deepstack merger: LN(4096) AFTER view; else same.)

## Fusion (Qwen3VLModel)
1. inputs_embeds = embed_tokens(input_ids) (bs,seq,2048).
2. image_embeds = merged (Nimg_tokens,2048); masked_scatter into positions where input_ids==151655.
3. position_ids: 3D (T,H,W) per get_rope_index; text spans: all axes arange+cur; image span: T=cur,
   H=cur+row, W=cur+col over llm_grid=(t, h/2, w/2); after image cur += max(gh,gw)//2. LM gets (4,bs,seq):
   axis0 = text pos (causal mask), axes1-3 = T/H/W for RoPE.
4. deepstack add: after LM layers 0,1,2 -> hidden[visual_pos] += deepstack[layer] (only image positions).

## mRoPE interleaved (mrope_section [24,20,20], theta 5e6, head_dim 128 -> 64 freqs)
freqs (3,bs,seq,64) fp32; apply_interleaved_mrope: start from T for all 64; slots 1,4,..,58 <- H;
slots 2,5,..,59 <- W; net pattern T,H,W repeat 20 triplets then T,T,T,T. emb=cat(f,f)->128; cos/sin.
Text-only tokens: T=H=W -> ordinary 1D RoPE (this is the fast path we need FIRST).

## Text decoder layer (Qwen3, x28) pre-norm
r=h; h=RMSNorm(h); h=attn(h); h=r+h; r=h; h=RMSNorm(h); h=SwiGLU_mlp(h); h=r+h.
attn: q 2048->2048, k/v 2048->1024 (bias false); q_norm/k_norm RMSNorm(128) per head BEFORE RoPE;
GQA repeat_kv n=2; scale 128^-.5; causal. mlp: down(silu(gate(h))*up(h)), gate/up 2048->6144, down 6144->2048.
final RMSNorm(2048) -> last_hidden_state (bs,seq,2048). NO lm_head for embeddings.

## Embedding head (sentence-transformers)
Prompt = chat template + system "Represent the user's input." + add_generation_prompt:
`<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n`
Image content = `<|vision_start|>` + N*`<|image_pad|>` (N = t*h*w/merge^2) + `<|vision_end|>` [+ optional text].
Image preprocess: convert_rgb, rescale 1/255, normalize mean/std [0.5]*3, patch 16, merge 2, temporal 2,
min_pixels 4096, max_pixels 1310720, bicubic. Processor Qwen2VLImageProcessorFast.
Pooling = LAST NON-PAD token: idx = (attention_mask first-0 pos, else seq_len) - 1, clamp>=0; gather. Then L2.

## Bumblebee wiring (from conventions agent)
- Register in lib/bumblebee.ex: @transformers_class_to_model "Qwen3VLForConditionalGeneration" -> {Bumblebee.Multimodal.Qwen3VL, :base};
  @model_type_to_featurizer "qwen3_vl" -> Bumblebee.Vision.Qwen3VLFeaturizer; @model_type_to_tokenizer_type "qwen3_vl" -> :qwen2.
- Reuse Bumblebee.Text.Qwen3 via its "input_embeddings" input (bypasses token embed) -> scatter vision tokens in.
  Its internals (core/decoder/embedder) are defp; either drive via input_embeddings or COPY core (~120 lines).
- Model must output rank-3 :hidden_state (for text_embedding + :last_token_pooling) AND rank-2 :embedding
  (image path: image_embedding serving has NO pooling, so do last-token select in model/1, emit rank-2).
- text_embedding: output_attribute: :hidden_state, output_pool: :last_token_pooling, embedding_processor: :l2_norm.
- image_embedding: output_attribute: :embedding, embedding_processor: :l2_norm; featurizer emits "pixel_values"
  + "grid_thw" (variable patch count -> clip_featurizer batch_template needs rework).
- params_mapping: delegate text via prefix "model.language_model"(verify!), vision via "model.visual"; merge top-level.
  VERIFY exact safetensors key prefixes against the index — most error-prone step.
- Config load: defimpl ...Transformers.Config, pop "vision_config"/"text_config".

## Build order (verify each against golden numerics)
1. TEXT-ONLY path first: Qwen3 decoder + interleaved-mRoPE-collapsing-to-1D + last-token + L2 -> match golden text_embedding.
   (text-only exercises no vision, no deepstack, no masked_scatter — smallest correct slice.)
2. Vision tower -> match intermediate merged (256,2048) against a torch dump.
3. Fusion + deepstack + 3D mRoPE -> match golden image_embedding.
FLAGGED: exact HF param prefixes; variable-patch featurizer batch_template; patch block-major flatten in processor.
