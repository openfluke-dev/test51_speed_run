package main

import (
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/runtime/forward"
)

// encodeObs downsamples the playfield to obsSide×obsSide and appends an
// available-action mask (length nActions).
func encodeObs(fr Frame) []float32 {
	out := make([]float32, obsDim)
	if len(fr.Layers) == 0 || len(fr.Layers[0]) == 0 {
		return out
	}
	layer := fr.Layers[0]
	h := len(layer)
	w := len(layer[0])
	if h < 1 || w < 1 {
		return out
	}
	for oy := 0; oy < obsSide; oy++ {
		for ox := 0; ox < obsSide; ox++ {
			sy := oy * h / obsSide
			sx := ox * w / obsSide
			if sy >= h {
				sy = h - 1
			}
			if sx >= w {
				sx = w - 1
			}
			out[oy*obsSide+ox] = float32(layer[sy][sx]) / 9.0
		}
	}
	mask := make([]bool, nActions)
	if len(fr.AvailableActions) == 0 {
		for i := 1; i < nActions; i++ {
			mask[i] = true
		}
	} else {
		for _, a := range fr.AvailableActions {
			if a >= 0 && a < nActions {
				mask[a] = true
			}
		}
	}
	for i := 0; i < nActions; i++ {
		if mask[i] {
			out[obsPix+i] = 1
		}
	}
	return out
}

func concatObsThink(obs, think []float32) *core.Tensor[float32] {
	x := core.NewTensor[float32](1, inDim)
	copy(x.Data, obs)
	td := thinkDim
	if len(think) < td {
		td = len(think)
	}
	copy(x.Data[obsDim:obsDim+td], think[:td])
	return x
}

func extractThink(post []float32) []float32 {
	t := make([]float32, thinkDim)
	if len(post) < nActions+thinkDim {
		return t
	}
	copy(t, post[nActions:nActions+thinkDim])
	return t
}

func decodeAction(post []float32, fr Frame) Action {
	avail := map[int]bool{}
	if len(fr.AvailableActions) == 0 {
		for i := 1; i < nActions; i++ {
			avail[i] = true
		}
	} else {
		for _, a := range fr.AvailableActions {
			avail[a] = true
		}
	}
	best, bv := 1, float32(-1e9)
	lim := nActions
	if len(post) < lim {
		lim = len(post)
	}
	for i := 0; i < lim; i++ {
		if !avail[i] && i != 0 {
			continue
		}
		// Prefer non-RESET during play unless nothing else available.
		if i == 0 && len(fr.AvailableActions) > 0 {
			continue
		}
		if post[i] > bv {
			bv = post[i]
			best = i
		}
	}
	if !avail[best] {
		for _, a := range fr.AvailableActions {
			best = a
			break
		}
	}
	act := Action{ID: best}
	if best == 6 && len(post) >= nActions+thinkDim+coordDim {
		cx := post[nActions+thinkDim]
		cy := post[nActions+thinkDim+1]
		act.X = clamp(int((cx+1)*0.5*float32(frameSize-1)), 0, frameSize-1)
		act.Y = clamp(int((cy+1)*0.5*float32(frameSize-1)), 0, frameSize-1)
	}
	return act
}

// targetFromOracle builds MSE target: one-hot oracle action + zero think + goal coords.
func targetFromOracle(oracle int, fr Frame, gx, gy float32) *core.Tensor[float32] {
	y := core.NewTensor[float32](1, outDim)
	if oracle < 0 || oracle >= nActions {
		oracle = 5
	}
	y.Data[oracle] = 1
	// think target = zeros (compress thought after act)
	if len(y.Data) >= nActions+thinkDim+coordDim {
		y.Data[nActions+thinkDim] = gx
		y.Data[nActions+thinkDim+1] = gy
	}
	return y
}

func goalCoordsNorm(fr Frame) (float32, float32) {
	if len(fr.Layers) == 0 {
		return 0, 0
	}
	layer := fr.Layers[0]
	for y := range layer {
		for x := range layer[y] {
			if layer[y][x] == 2 {
				return float32(x)/float32(frameSize-1)*2 - 1, float32(y)/float32(frameSize-1)*2 - 1
			}
		}
	}
	return 0, 0
}

type thinkResult struct {
	X       *core.Tensor[float32]
	Post    *core.Tensor[float32]
	Tape    *parallel.SplitTape[float32]
	MeshFwd *forward.Result[float32]
	Action  Action
	ThinkK  int
}

// thinkThenAct runs K recurrent forwards feeding the think head back into
// the input — no env step during thinking.
func thinkThenAct(
	st *parallel.Stack,
	grid *architecture.Grid,
	mode parallel.TrainMode,
	fr Frame,
	k int,
) (thinkResult, error) {
	if k < 1 {
		k = 1
	}
	obs := encodeObs(fr)
	think := make([]float32, thinkDim)
	var last thinkResult
	for t := 0; t < k; t++ {
		x := concatObsThink(obs, think)
		var post *core.Tensor[float32]
		var tape *parallel.SplitTape[float32]
		var meshFwd *forward.Result[float32]
		switch {
		case mode == parallel.ModeMeshBP || mode == parallel.ModeMeshTween || mode == parallel.ModeMeshTweenChain:
			if grid == nil {
				return last, errNoGrid
			}
			fwd, ferr := forward.Forward(grid, x)
			if ferr != nil || fwd == nil || fwd.Output == nil {
				return last, ferr
			}
			meshFwd = fwd
			post = fwd.Output
		case mode.IsSplitFamily():
			tp, terr := parallel.OpenSplitTape(st, x)
			if terr != nil || tp == nil || tp.Post == nil {
				return last, terr
			}
			tape = tp
			post = tp.Post
		default:
			_, p, ferr := parallel.ForwardStack(st, x)
			if ferr != nil || p == nil {
				return last, ferr
			}
			post = p
		}
		think = extractThink(post.Data)
		last = thinkResult{X: x, Post: post, Tape: tape, MeshFwd: meshFwd, ThinkK: t + 1}
	}
	last.Action = decodeAction(last.Post.Data, fr)
	return last, nil
}

type simpleError string

func (e simpleError) Error() string { return string(e) }

const errNoGrid = simpleError("mesh mode requires grid")
