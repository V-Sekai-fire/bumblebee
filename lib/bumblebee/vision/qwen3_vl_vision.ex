defmodule Bumblebee.Vision.Qwen3VLVision do
  alias Bumblebee.Shared

  options =
    [
      in_channels: [default: 3, doc: "the number of channels in the input"],
      patch_size: [default: 16, doc: "the spatial patch size"],
      temporal_patch_size: [default: 2, doc: "the temporal patch size"],
      spatial_merge_size: [default: 2, doc: "the 2x2 patch merge size"],
      hidden_size: [default: 1024, doc: "the dimensionality of vision hidden layers"],
      num_blocks: [default: 24, doc: "the number of Transformer blocks"],
      num_attention_heads: [default: 16, doc: "the number of attention heads"],
      intermediate_size: [default: 4096, doc: "the dimensionality of the FFN intermediate layer"],
      out_hidden_size: [default: 2048, doc: "the merged output dimensionality (LM hidden size)"],
      num_position_embeddings: [default: 2304, doc: "the size of the learned position table (side^2)"],
      deepstack_visual_indexes: [
        default: [5, 11, 17],
        doc: "block indexes whose outputs are merged and returned as deepstack features"
      ],
      activation: [default: :gelu_approx_tanh, doc: "the vision MLP activation"],
      rotary_embedding_base: [default: 10_000.0, doc: "base for the 2-axis vision rotary embedding"],
      layer_norm_epsilon: [default: 1.0e-6, doc: "epsilon for layer normalization"]
    ]

  @moduledoc """
  The Qwen3-VL vision tower (SigLIP-lineage ViT + 2-axis RoPE + deepstack + 2x2 patch merger).

  ## Architectures

    * `:base` - the vision tower. Returns `:pooled_state` (the merged `{num_merged, out_hidden_size}`
      patch tokens that get scattered into the LM stream) plus `:deepstack_features` (a list, one per
      `deepstack_visual_indexes` entry, each `{num_merged, out_hidden_size}`).

  ## Inputs

  Position-dependent tensors are precomputed by `Bumblebee.Vision.Qwen3VLFeaturizer` (an exact port of
  `transformers.vision_utils`) and passed in, so the graph is pure tensor math over a packed single-image
  patch sequence:

    * `"pixel_values"` - `{num_patches, in_channels * temporal_patch_size * patch_size * patch_size}`
    * `"bilinear_indices"` - `{4, num_patches}` corner indices into the position table
    * `"bilinear_weights"` - `{4, num_patches}` bilinear weights
    * `"rotary_cos"` / `"rotary_sin"` - `{num_patches, head_dim}` precomputed vision-RoPE cos/sin

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts), do: Shared.put_config_attrs(spec, opts)

  @impl true
  def input_template(spec) do
    patch_dim = spec.in_channels * spec.temporal_patch_size * spec.patch_size * spec.patch_size

    %{
      "pixel_values" => Nx.template({4, patch_dim}, :f32),
      "bilinear_indices" => Nx.template({4, 4}, :s64),
      "bilinear_weights" => Nx.template({4, 4}, :f32),
      "rotary_cos" => Nx.template({4, head_dim(spec)}, :f32),
      "rotary_sin" => Nx.template({4, head_dim(spec)}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    Layers.output(%{
      pooled_state: outputs.pooled_state,
      deepstack_features: Axon.container(List.to_tuple(outputs.deepstack_features))
    })
  end

  defp head_dim(spec), do: div(spec.hidden_size, spec.num_attention_heads)

  defp inputs(spec) do
    patch_dim = spec.in_channels * spec.temporal_patch_size * spec.patch_size * spec.patch_size

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: {nil, patch_dim}),
      Axon.input("bilinear_indices", shape: {4, nil}),
      Axon.input("bilinear_weights", shape: {4, nil}),
      Axon.input("rotary_cos", shape: {nil, head_dim(spec)}),
      Axon.input("rotary_sin", shape: {nil, head_dim(spec)})
    ])
  end

  defp core(inputs, spec) do
    hidden_state =
      inputs["pixel_values"]
      |> Axon.dense(spec.hidden_size, name: "patch_embed")
      |> add_position_embeddings(inputs["bilinear_indices"], inputs["bilinear_weights"], spec)

    cos = inputs["rotary_cos"]
    sin = inputs["rotary_sin"]

    {hidden_state, deepstack} =
      Enum.reduce(0..(spec.num_blocks - 1), {hidden_state, []}, fn block, {hs, deep} ->
        hs = block(hs, cos, sin, spec, name: "blocks.#{block}")

        deep =
          case Enum.find_index(spec.deepstack_visual_indexes, &(&1 == block)) do
            nil -> deep
            di -> deep ++ [merger(hs, spec, postshuffle: true, name: "deepstack_merger_list.#{di}")]
          end

        {hs, deep}
      end)

    pooled_state = merger(hidden_state, spec, postshuffle: false, name: "merger")

    %{pooled_state: pooled_state, deepstack_features: deepstack}
  end

  # pos_embeds = sum_i table[bilinear_indices[i]] * bilinear_weights[i]
  defp add_position_embeddings(patch, indices, weights, spec) do
    table = Axon.param("pos_embed", fn _ -> {spec.num_position_embeddings, spec.hidden_size} end)

    Axon.layer(
      fn patch, indices, weights, table, _opts ->
        gathered = Nx.take(table, indices)
        Nx.add(patch, Nx.sum(Nx.multiply(gathered, Nx.new_axis(weights, -1)), axes: [0]))
      end,
      [patch, indices, weights, table],
      name: "pos_embed",
      op_name: :qwen3vl_pos_embed
    )
  end

  defp block(hidden_state, cos, sin, spec, opts) do
    name = opts[:name]

    attn =
      hidden_state
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm1"))
      |> attention(cos, sin, spec, name: join(name, "attn"))

    hidden_state = Axon.add(hidden_state, attn)

    ffn =
      hidden_state
      |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm2"))
      |> Axon.dense(spec.intermediate_size, name: join(name, "mlp.linear_fc1"))
      |> Layers.activation(spec.activation)
      |> Axon.dense(spec.hidden_size, name: join(name, "mlp.linear_fc2"))

    Axon.add(hidden_state, ffn)
  end

  defp attention(hidden_state, cos, sin, spec, opts) do
    name = opts[:name]
    heads = spec.num_attention_heads
    hd = head_dim(spec)

    qkv = Axon.dense(hidden_state, 3 * spec.hidden_size, name: join(name, "qkv"))

    out =
      Axon.layer(
        fn qkv, cos, sin, _opts ->
          qkv_attention(qkv, cos, sin, heads, hd)
        end,
        [qkv, cos, sin],
        op_name: :qwen3vl_vision_attention
      )

    Axon.dense(out, spec.hidden_size, name: join(name, "proj"))
  end

  # Packed single-image full attention; matches the verified probe.
  defp qkv_attention(qkv, cos, sin, heads, hd) do
    n = Nx.axis_size(qkv, 0)
    qkv = Nx.reshape(qkv, {n, 3, heads, hd})
    q = qkv[[.., 0]] |> rope(cos, sin, hd)
    k = qkv[[.., 1]] |> rope(cos, sin, hd)
    v = qkv[[.., 2]]
    # -> {heads, n, hd}
    q = Nx.transpose(q, axes: [1, 0, 2])
    k = Nx.transpose(k, axes: [1, 0, 2])
    v = Nx.transpose(v, axes: [1, 0, 2])
    scores = Nx.multiply(Nx.dot(q, [2], [0], k, [2], [0]), 1.0 / :math.sqrt(hd))
    attn = Axon.Activations.softmax(scores, axis: -1)
    out = Nx.dot(attn, [2], [0], v, [1], [0])
    out |> Nx.transpose(axes: [1, 0, 2]) |> Nx.reshape({n, heads * hd})
  end

  # rope over {n, heads, hd} with cos/sin {n, hd}
  defp rope(x, cos, sin, hd) do
    cos = Nx.new_axis(cos, 1)
    sin = Nx.new_axis(sin, 1)
    half = div(hd, 2)
    x1 = x[[.., .., 0..(half - 1)]]
    x2 = x[[.., .., half..(hd - 1)]]
    rotated = Nx.concatenate([Nx.negate(x2), x1], axis: -1)
    Nx.add(Nx.multiply(x, cos), Nx.multiply(rotated, sin))
  end

  # 2x2 patch merger -> out_hidden_size. postshuffle norms after the 4x concat; else before.
  defp merger(hidden_state, spec, opts) do
    name = opts[:name]
    postshuffle = opts[:postshuffle]
    merged_dim = spec.hidden_size * spec.spatial_merge_size * spec.spatial_merge_size

    normed =
      if postshuffle do
        hidden_state
        |> reshape_merge(merged_dim)
        |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
      else
        hidden_state
        |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
        |> reshape_merge(merged_dim)
      end

    normed
    |> Axon.dense(merged_dim, name: join(name, "linear_fc1"))
    |> Layers.activation(:gelu)
    |> Axon.dense(spec.out_hidden_size, name: join(name, "linear_fc2"))
  end

  defp reshape_merge(x, merged_dim) do
    Axon.nx(x, fn x ->
      n = Nx.axis_size(x, 0)
      cols = Nx.axis_size(x, 1)
      Nx.reshape(x, {div(n * cols, merged_dim), merged_dim})
    end)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, %{"vision_config" => data}), do: load(spec, data)

    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          in_channels: {"in_channels", number()},
          patch_size: {"patch_size", number()},
          temporal_patch_size: {"temporal_patch_size", number()},
          spatial_merge_size: {"spatial_merge_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"depth", number()},
          num_attention_heads: {"num_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          out_hidden_size: {"out_hidden_size", number()},
          num_position_embeddings: {"num_position_embeddings", number()},
          deepstack_visual_indexes: {"deepstack_visual_indexes", list(number())},
          activation: {"hidden_act", activation()}
        )

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        # Conv3d {out,in,t,h,w} -> dense kernel {in_flat, out}
        "patch_embed" => %{
          "kernel" => {
            [{"model.visual.patch_embed.proj", "weight"}],
            fn [w] ->
              {out, _, _, _, _} = Nx.shape(w)
              w |> Nx.reshape({out, :auto}) |> Nx.transpose()
            end
          },
          "bias" => "model.visual.patch_embed.proj.bias"
        },
        "pos_embed" => %{
          "pos_embed" => {[{"model.visual.pos_embed", "weight"}], fn [w] -> w end}
        },
        "blocks.{n}.norm1" => "model.visual.blocks.{n}.norm1",
        "blocks.{n}.norm2" => "model.visual.blocks.{n}.norm2",
        "blocks.{n}.attn.qkv" => "model.visual.blocks.{n}.attn.qkv",
        "blocks.{n}.attn.proj" => "model.visual.blocks.{n}.attn.proj",
        "blocks.{n}.mlp.linear_fc1" => "model.visual.blocks.{n}.mlp.linear_fc1",
        "blocks.{n}.mlp.linear_fc2" => "model.visual.blocks.{n}.mlp.linear_fc2",
        "merger.norm" => "model.visual.merger.norm",
        "merger.linear_fc1" => "model.visual.merger.linear_fc1",
        "merger.linear_fc2" => "model.visual.merger.linear_fc2",
        "deepstack_merger_list.{n}.norm" => "model.visual.deepstack_merger_list.{n}.norm",
        "deepstack_merger_list.{n}.linear_fc1" => "model.visual.deepstack_merger_list.{n}.linear_fc1",
        "deepstack_merger_list.{n}.linear_fc2" => "model.visual.deepstack_merger_list.{n}.linear_fc2"
      }
    end
  end
end
