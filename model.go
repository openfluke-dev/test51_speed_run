package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

// Policy I/O layout (ARC-AGI-3-shaped):
//
//	obs  = downsampled playfield + available-action mask
//	x    = concat(obs, think)
//	post = [action logits | think vector | coord x,y]
const (
	frameSize   = 64
	obsSide     = 8
	nActions    = 8 // RESET + ACTION1..ACTION7
	thinkDim    = 16
	coordDim    = 2
	obsPix      = obsSide * obsSide
	obsDim      = obsPix + nActions
	outDim      = nActions + thinkDim + coordDim
	inDim       = obsDim + thinkDim
	hiddenWide  = 64
	hiddenNarrow = 32
)

var policyWidths = []int{inDim, hiddenWide, hiddenWide, hiddenNarrow, outDim}

func policyWidthsFor(layer string) []int {
	switch strings.ToLower(strings.TrimSpace(layer)) {
	case "dense-wide":
		return []int{inDim, 96, 96, 48, outDim}
	case "dense-deep":
		return []int{inDim, 48, 48, 48, 48, 32, outDim}
	case "dense-deep-wide":
		return []int{inDim, 96, 96, 64, 64, 48, outDim}
	default:
		return policyWidths
	}
}

func buildPolicyNet(rng *rand.Rand, be core.Backend) (*parallel.Stack, error) {
	return buildPolicyNetEx(rng, be, "dense", core.DTypeFloat32, 1)
}

func buildPolicyNetEx(rng *rand.Rand, be core.Backend, layer string, dt core.DType, cams int) (*parallel.Stack, error) {
	if dt == 0 {
		dt = core.DTypeFloat32
	}
	if cams < 1 {
		cams = 1
	}
	if cams > 3 {
		cams = 3
	}
	widths := policyWidthsFor(layer)
	// Single cam: plain dense MLP (original think net).
	if cams == 1 {
		ops := make([]any, 0, len(widths)-1)
		for i := 0; i < len(widths)-1; i++ {
			in, out := widths[i], widths[i+1]
			act := core.ActivationTanh
			if i == len(widths)-2 {
				act = core.ActivationLinear
			}
			w := randScale(in*out, float32(1/math.Sqrt(float64(in))), rng)
			d, err := dense.NewConfigured(in, out, act, dt, quant.FormatNone, w)
			if err != nil {
				return nil, fmt.Errorf("dense %d (%s/%s): %w", i, layer, dt.String(), err)
			}
			ops = append(ops, d)
		}
		st, err := parallel.NewStack(ops...)
		if err != nil {
			return nil, err
		}
		st.Exec.Backend = be
		st.Exec.MultiCore = leafMultiCore
		st.SyncChildExec()
		return st, nil
	}

	// Multi-cam: Dense stem → N hemispheres (add) → Dense head.
	hidden := hiddenWide
	if len(widths) >= 3 {
		hidden = widths[1]
	}
	stemW := randScale(inDim*hidden, float32(1/math.Sqrt(float64(inDim))), rng)
	stem, err := dense.NewConfigured(inDim, hidden, core.ActivationTanh, dt, quant.FormatNone, stemW)
	if err != nil {
		return nil, fmt.Errorf("cam stem: %w", err)
	}
	branches := make([]any, cams)
	for i := 0; i < cams; i++ {
		bw := randScale(hidden*hidden, float32(1/math.Sqrt(float64(hidden))), rng)
		br, berr := dense.NewConfigured(hidden, hidden, core.ActivationTanh, dt, quant.FormatNone, bw)
		if berr != nil {
			return nil, fmt.Errorf("cam branch %d: %w", i, berr)
		}
		branches[i] = br
	}
	hemi, err := parallel.HemispheresFrom(parallel.Config{
		Dim: hidden, OutFeat: hidden, Branches: cams, Combine: parallel.CombineAdd,
	}, branches, nil)
	if err != nil {
		return nil, fmt.Errorf("cam hemispheres n=%d: %w", cams, err)
	}
	headW := randScale(hidden*outDim, float32(1/math.Sqrt(float64(hidden))), rng)
	head, err := dense.NewConfigured(hidden, outDim, core.ActivationLinear, dt, quant.FormatNone, headW)
	if err != nil {
		return nil, fmt.Errorf("cam head: %w", err)
	}
	st, err := parallel.Sandwich(stem, hemi, head)
	if err != nil {
		return nil, err
	}
	st.Exec.Backend = be
	st.Exec.MultiCore = leafMultiCore
	st.SyncChildExec()
	return st, nil
}

// leafMultiCore: false when many leaf-workers share the machine (1 core ≈ 1 leaf).
var leafMultiCore = true

func randScale(n int, scale float32, rng *rand.Rand) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = (rng.Float32()*2 - 1) * scale
	}
	return w
}

func argmax(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

func softmaxPTrue(logits []float32, lab int) float32 {
	if lab < 0 || lab >= len(logits) {
		return 0
	}
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	exps := make([]float32, len(logits))
	for i, v := range logits {
		exps[i] = float32(math.Exp(float64(v - max)))
		sum += exps[i]
	}
	if sum <= 0 {
		return 1 / float32(len(logits))
	}
	return exps[lab] / sum
}

func stackWeightBytes(s *parallel.Stack) int64 {
	var n int64
	for _, st := range dna.CollectStores(s) {
		if st == nil {
			continue
		}
		elems := int64(st.Rows * st.Cols)
		if elems <= 0 {
			f, err := st.FlattenF32()
			if err != nil || len(f) == 0 {
				continue
			}
			elems = int64(len(f))
		}
		bits := int64(st.DType.Bits())
		if bits <= 0 {
			bits = 32
		}
		n += (elems*bits + 7) / 8
	}
	return n
}

// weightSnapshot flattens all stores for LPD champ promotion.
func weightSnapshot(st *parallel.Stack) [][]float32 {
	stores := dna.CollectStores(st)
	out := make([][]float32, len(stores))
	for i, s := range stores {
		if s == nil {
			continue
		}
		f, err := s.FlattenF32()
		if err != nil {
			continue
		}
		cp := make([]float32, len(f))
		copy(cp, f)
		out[i] = cp
	}
	return out
}

func restoreWeights(st *parallel.Stack, snap [][]float32) error {
	stores := dna.CollectStores(st)
	if len(snap) != len(stores) {
		return fmt.Errorf("weight snap len %d != stores %d", len(snap), len(stores))
	}
	for i, s := range stores {
		if s == nil || snap[i] == nil {
			continue
		}
		if err := s.SetFromF32(snap[i]); err != nil {
			return err
		}
	}
	return nil
}
