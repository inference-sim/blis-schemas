package model

// Graph is a model expressed as cost primitives: a set of distinct layer kinds,
// a repetition pattern over them, and optional head and speculator stacks.
//
// The pattern is not a flat list of layers. A 72-layer uniform stack is
// {Pattern: ["attn_moe"], Repeat: 72}; a hybrid stack that interleaves three
// recurrent layers with one attention layer is
// {Pattern: ["mamba_moe","mamba_moe","mamba_moe","attn_moe"], Repeat: 10}. That
// mirrors how an engine itself resolves a hybrid model — a per-index type vector
// selecting a decoder-layer class — and keeps a 200-layer model a few lines rather
// than two hundred.
type Graph struct {
	Kind string `yaml:"kind"` // "ModelGraph"
	Name string `yaml:"name"` // matches the blis-catalog models/ directory

	// DerivedFrom records the vendor file this graph was translated from, and its
	// digest. The graph is a derived artifact, so it can fall out of step with its
	// source; the digest makes that detectable instead of silent.
	DerivedFrom Derivation `yaml:"derived_from"`

	Global     GlobalShape `yaml:"global"`
	LayerKinds []LayerKind `yaml:"layer_kinds"`
	Stack      Stack       `yaml:"stack"`
	Head       []Node      `yaml:"head,omitempty"`
	Speculator *Speculator `yaml:"speculator,omitempty"`

	// Modality records what a graph covers when its source describes more than a
	// language model. Several shipped configurations nest the text shapes under a
	// sub-object alongside a vision or audio tower; a graph derived from one of those
	// describes the decoder alone unless it says otherwise.
	//
	// The field exists so the omission is explicit. A cost model that silently
	// ignored an encoder would under-predict a multimodal request's prefill, and a
	// reader comparing a prediction against a measured run of the full model would
	// have no way to see why.
	Modality Modality `yaml:"modality,omitempty"`
}

// Modality states which towers a graph prices.
type Modality string

const (
	// ModalityTextOnly: the decoder stack only. This is the default when the field
	// is absent, because every graph this schema can currently express is a decoder.
	ModalityTextOnly Modality = "text_only"
	// ModalityTextDecoderOfMultimodal: the decoder of a model whose source also
	// describes an encoder that this graph does NOT price. A prediction from such a
	// graph is valid for text-only requests and understates any request carrying an
	// image or audio input.
	ModalityTextDecoderOfMultimodal Modality = "text_decoder_of_multimodal"
)

var modalities = map[Modality]bool{
	"": true, ModalityTextOnly: true, ModalityTextDecoderOfMultimodal: true,
}

// Valid reports whether m is a recognized modality. The empty value is valid and
// means text-only.
func (m Modality) Valid() bool { return modalities[m] }

// PricesEncoder reports whether the graph accounts for a non-text tower. It is
// always false today: the primitive set has no encoder term, so a graph can declare
// that it omits one but cannot declare that it includes one.
func (m Modality) PricesEncoder() bool { return false }

// Derivation identifies the vendor artifact a graph came from without adopting its
// vocabulary. Format names the dialect (for example "hf_config_json") so a reader
// knows which deriver produced this graph; the schema itself stays neutral.
//
// A deriver cannot assume a field exists just because it usually does. Depth is the
// clearest case: most configurations state num_hidden_layers, some spell it n_layer,
// and at least one shipped hybrid states neither — its depth is the length of a
// per-layer type vector, and a deriver reading only the usual key would produce a
// zero-layer model. The same applies to head dimension, which many configurations
// omit because it is hidden size over head count.
//
// This is why a graph carries a digest and a deriver version rather than being
// recomputed on demand: the translation has judgement in it, and a committed artifact
// can be reviewed where a silent recomputation cannot.
type Derivation struct {
	Format         string `yaml:"format"`
	Path           string `yaml:"path"`
	SHA256         string `yaml:"sha256"`
	DeriverVersion int    `yaml:"deriver_version"`
}

// GlobalShape holds the few quantities that are properties of the whole model
// rather than of one layer.
type GlobalShape struct {
	HiddenSize        int  `yaml:"hidden_size"`
	VocabSize         int  `yaml:"vocab_size"`
	TieWordEmbeddings bool `yaml:"tie_word_embeddings"`
	// WeightDType is the dtype parameters are stored in. It sets weight bytes and,
	// through them, both per-rank occupancy and decode-time HBM traffic.
	WeightDType DType `yaml:"weight_dtype"`
}

// DType is a numeric format. Only formats that change a byte count or a peak rate
// appear; a format that merely changes rounding does not belong in a cost schema.
type DType string

const (
	DTypeBF16 DType = "bf16"
	DTypeFP16 DType = "fp16"
	DTypeFP8  DType = "fp8"
	// DTypeNVFP4 is NVIDIA's 4-bit float: a 16-element block sharing an FP8 scale,
	// with native Tensor-Core support on Blackwell.
	DTypeNVFP4 DType = "nvfp4"
	// DTypeMXFP4 is the OCP microscaling 4-bit float: a 32-element block sharing an
	// E8M0 scale. It is a distinct format from NVFP4, not a spelling of it — the block
	// size and scale type differ, so the two reach different rates on the same part,
	// and a checkpoint in one cannot be priced as the other.
	DTypeMXFP4 DType = "mxfp4"
	DTypeINT8  DType = "int8"
	// DTypeINT4 is 4-bit integer weights with a per-group scale, the W4A16 form
	// compressed-tensors emits as num_bits 4 with type "int" (Kimi-K2.5 ships this way at
	// group_size 32). It is a distinct format from NVFP4 and MXFP4, not a spelling of
	// either: an integer grid dequantizes through a different path than a float one, so
	// the three reach different rates on the same part even where the payload width
	// matches. Storage is four bits of payload, as for the 4-bit floats.
	DTypeINT4 DType = "int4"
	DTypeFP32 DType = "fp32"
)

var dtypes = map[DType]bool{
	DTypeBF16: true, DTypeFP16: true, DTypeFP8: true, DTypeNVFP4: true,
	DTypeMXFP4: true, DTypeINT8: true, DTypeINT4: true, DTypeFP32: true,
}

// Valid reports whether d is a recognized dtype.
func (d DType) Valid() bool { return dtypes[d] }

// Bytes returns the storage width of one element. NVFP4 is four bits, so it
// returns a fractional value expressed in bytes; callers multiply by an element
// count and round at the end rather than per element.
func (d DType) Bytes() float64 {
	switch d {
	case DTypeFP32:
		return 4
	case DTypeBF16, DTypeFP16:
		return 2
	case DTypeFP8, DTypeINT8:
		return 1
	case DTypeNVFP4, DTypeMXFP4, DTypeINT4:
		// Four bits of payload. Each format adds a per-group scale — one FP8 byte per
		// sixteen elements for NVFP4, one E8M0 byte per thirty-two for MXFP4, and for INT4
		// one scale per group at whatever group_size the checkpoint declares — which a
		// caller needing exact stored bytes must add; this returns the payload width.
		return 0.5
	}
	return 0
}

// LayerKind is one distinct layer, as a small DAG over primitives.
type LayerKind struct {
	ID    string   `yaml:"id"`
	Nodes []Node   `yaml:"nodes"`
	Edges [][2]int `yaml:"edges"` // indices into Nodes; a DAG, checked acyclic
}

// Stack is the layer sequence. Three fields, because real stacks come in three
// shapes and forcing them into one would either lose information or bloat the file.
//
// Prologue is a literal run of layer kinds before the repeating part, and Epilogue a
// literal run after it. They exist because several shipped models are not periodic:
// one opens with three dense layers before seventy-five sparse ones, and another
// carries a 108-entry layer vector with no repeating unit at all. A schema offering
// only Pattern and Repeat would force the first into a 78-entry pattern and could
// not express the second without abandoning the compression entirely.
//
// A uniform stack is Pattern with Repeat and nothing else. A hybrid with a regular
// period is the same. A model with a prologue states it. A model with no periodicity
// puts its whole vector in Prologue and leaves Repeat at zero.
type Stack struct {
	Prologue []string `yaml:"prologue,omitempty"`
	Pattern  []string `yaml:"pattern,omitempty"`
	Repeat   int      `yaml:"repeat,omitempty"`
	Epilogue []string `yaml:"epilogue,omitempty"`
}

// Layers returns the total layer count the stack expands to.
func (s Stack) Layers() int {
	return len(s.Prologue) + len(s.Pattern)*s.Repeat + len(s.Epilogue)
}

// Expand returns the full layer-kind sequence. A cost model walks this rather than
// re-deriving the expansion, so the expansion lives in one place.
func (s Stack) Expand() []string {
	out := make([]string, 0, s.Layers())
	out = append(out, s.Prologue...)
	for i := 0; i < s.Repeat; i++ {
		out = append(out, s.Pattern...)
	}
	return append(out, s.Epilogue...)
}

// Speculator is a draft module, represented as its own stack rather than as a
// token count. For an MoE target the draft pass is a second MoE, and a scalar
// draft length hides that cost entirely.
type Speculator struct {
	// Method is the engine's speculative method name. It is carried as a string
	// because the set of methods is engine-version data, not a schema invariant;
	// a rules pack validates it against a declared engine version.
	Method  string `yaml:"method"`
	NumSpec int    `yaml:"num_spec_tokens"`
	Stack   Stack  `yaml:"stack"`
}

// EmitCondition names when a conditional node is part of the graph. A resolver maps each
// member to a decision from the resolved layout; the graph states which question to ask,
// never how to answer it.
type EmitCondition string

const (
	// EmitAlways is the zero value: the node is unconditional.
	EmitAlways EmitCondition = ""
	// EmitTensorParallel: present when the tensor-parallel width exceeds one. The
	// attention-output and mixer-output reductions.
	EmitTensorParallel EmitCondition = "tensor_parallel"
	// EmitExpertParallel: present when the expert-parallel width exceeds one. The MoE
	// dispatch and combine.
	EmitExpertParallel EmitCondition = "expert_parallel"
	// EmitTensorParallelUnlessSequenceParallelMoE: present when the tensor-parallel
	// width exceeds one AND the engine has not made the MoE input sequence-parallel.
	// Under sequence-parallel MoE the layer's reduction is replaced by a
	// reduce-scatter and all-gather pair, so this node and that pair are alternatives
	// rather than both running.
	EmitTensorParallelUnlessSequenceParallelMoE EmitCondition = "tensor_parallel_unless_sp_moe"
)

var emitConditions = map[EmitCondition]bool{
	EmitAlways: true, EmitTensorParallel: true, EmitExpertParallel: true,
	EmitTensorParallelUnlessSequenceParallelMoE: true,
}

// Valid reports whether c is a recognized condition.
func (c EmitCondition) Valid() bool { return emitConditions[c] }

// Node is one primitive with the shape parameters it prices. The fields are a
// union across ops: a GEMM reads N and K, an Attention reads the head counts, a
// RecurrentUpdate reads the state dimensions. Validation rejects a node carrying
// parameters its op does not price, so the union cannot hide a mistake.
type Node struct {
	Op   Op     `yaml:"op"`
	Role string `yaml:"role,omitempty"` // human label, e.g. "qkv_proj"

	// Emit names the condition under which this node exists. Empty means
	// unconditional. A collective node must carry one, because whether a collective
	// runs depends on the parallelism a deployment chooses rather than on the model.
	//
	// It is a closed enum rather than an expression. An earlier draft wrote predicates
	// as strings — "tp > 1 and not sp_moe" — which would have required the kernel to
	// carry an expression parser, and would have let a graph express a condition no
	// resolver knows how to evaluate. Every condition the catalog's models need is one
	// of three, and each is decided by the collective's role rather than by arbitrary
	// arithmetic, so the enum loses nothing and the resolver becomes a switch.
	Emit EmitCondition `yaml:"emit,omitempty"`

	// GEMM and GroupedGEMM.
	N int `yaml:"n,omitempty"`
	K int `yaml:"k,omitempty"`
	// WeightDType overrides GlobalShape.WeightDType for this node's parameters, for a
	// checkpoint that stores one part of itself at a different width from the rest.
	// Empty means the global dtype governs, which is the common case.
	//
	// Mixed-precision MoE is why this exists. DeepSeek-V4-Pro declares expert_dtype fp4
	// beside an fp8 quantization_config: the routed experts are 4-bit and everything else
	// is 8-bit. Pricing the experts at the global width doubled them -- 1,441 GiB against
	// 720 GiB for 384 experts -- which put a per-rank figure of 180 GiB on a 141 GiB part
	// and made a deployment InferenceX actually ran look impossible. A global dtype cannot
	// express that split, and inferring it from the quantization method would be a guess
	// about a width the checkpoint states.
	WeightDType DType `yaml:"weight_dtype,omitempty"`
	// GroupedGEMM only.
	Experts int `yaml:"experts,omitempty"`
	TopK    int `yaml:"top_k,omitempty"`
	// SharedExperts are experts every token passes through, in addition to the
	// TopK routed ones. They are a distinct cost: their work is dense rather than
	// routed, so it neither scales with TopK nor contributes to routing imbalance.
	// Whether an engine fuses them into the grouped GEMM or runs them separately
	// changes the kernel but not the byte count.
	SharedExperts int `yaml:"shared_experts,omitempty"`
	// SharedIntermediateSize is the shared experts' inner dimension where it differs
	// from N, as it commonly does. Zero means it equals N.
	SharedIntermediateSize int `yaml:"shared_intermediate_size,omitempty"`
	// LatentSize reduces the expert input dimension below the model's hidden size:
	// a projection to a narrower space runs before the routed experts, so K for the
	// expert matmul is this rather than HiddenSize. Zero means no reduction. It
	// changes both the expert FLOP count and the all-to-all volume, which is why it
	// is a node parameter rather than a global one.
	LatentSize int `yaml:"latent_size,omitempty"`

	// Attention.
	AttentionKind AttentionKind `yaml:"kind,omitempty"`
	NumQHeads     int           `yaml:"n_q,omitempty"`
	NumKVHeads    int           `yaml:"n_kv,omitempty"`
	HeadDim       int           `yaml:"d_h,omitempty"`
	// MLA-specific latent dimensions; zero for other kinds.
	KVLoRARank    int `yaml:"kv_lora_rank,omitempty"`
	QKRopeHeadDim int `yaml:"qk_rope_head_dim,omitempty"`
	// SWA window, in tokens; zero for other kinds.
	Window int `yaml:"window,omitempty"`
	// SparseMLA top-k index count; zero for other kinds.
	IndexTopK int `yaml:"index_topk,omitempty"`

	// RecurrentUpdate.
	RecurrentKind    RecurrentKind `yaml:"recurrent_kind,omitempty"`
	NumHeads         int           `yaml:"n_heads,omitempty"`
	StateSize        int           `yaml:"state_size,omitempty"`
	NumGroups        int           `yaml:"n_groups,omitempty"`
	ConvKernel       int           `yaml:"conv_kernel,omitempty"`
	IntermediateSize int           `yaml:"intermediate_size,omitempty"`
	// StateDType is the dtype the recurrent state is stored in, which is often
	// wider than the weight dtype.
	StateDType DType `yaml:"state_dtype,omitempty"`

	// Elementwise: bytes touched per token, when it is not derivable from
	// HiddenSize alone.
	BytesPerToken int `yaml:"bytes_per_token,omitempty"`
}
